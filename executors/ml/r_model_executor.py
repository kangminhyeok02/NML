"""
r_model_executor.py
-------------------
R 기반 모델 학습 및 예측 실행기.

R 스크립트를 subprocess로 호출하여 학습/예측을 수행한다.
Python ↔ R 간 데이터 교환은 CSV 또는 RDS 파일을 통해 이루어진다.

사용 목적:
  - 기존 R 자산(통계모형, 스코어카드) 재사용
  - R 전용 라이브러리 활용 (glm, caret, survival, creditR 등)
  - 레거시 R 코드와의 통합 운영

실행 순서:
  1. Python에서 데이터 → CSV 임시파일 저장
  2. Rscript로 R 스크립트 실행 (--args로 경로 전달)
  3. R 스크립트 결과(모델 메타, 예측값) 읽기
  4. 결과 저장 및 반환
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)


class RModelExecutor(BaseExecutor):
    """
    R 모델 실행 executor.

    config 필수 키
    --------------
    r_script    : str   실행할 R 스크립트 경로 (FILE_ROOT_DIR 기준)
    mode        : str   "train" | "predict"
    model_id    : str   모델 저장 식별자

    [train 모드]
    train_path  : str   학습 데이터 경로 (.parquet)
    target_col  : str   타깃 컬럼명

    [predict 모드]
    input_path  : str   예측 대상 데이터 경로 (.parquet)

    config 선택 키
    --------------
    r_args      : dict  R 스크립트에 전달할 추가 인자
    rscript_cmd : str   Rscript 실행 명령 (기본: "Rscript")
    valid_path  : str   검증 데이터 경로 (train 모드)
    """

    def execute(self) -> dict:
        cfg  = self.config
        mode = cfg["mode"]

        if mode == "train":
            return self._run_train(cfg)
        elif mode == "predict":
            return self._run_predict(cfg)
        else:
            raise ExecutorException(f"지원하지 않는 mode: {mode}")

    # ------------------------------------------------------------------

    def _run_train(self, cfg: dict) -> dict:
        train_df = self._load_dataframe(cfg["train_path"])
        valid_df = self._load_dataframe(cfg["valid_path"]) if "valid_path" in cfg else None
        self._update_job_status(ExecutorStatus.RUNNING, progress=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            train_csv = os.path.join(tmpdir, "train.csv")
            train_df.to_csv(train_csv, index=False)

            valid_csv = ""
            if valid_df is not None:
                valid_csv = os.path.join(tmpdir, "valid.csv")
                valid_df.to_csv(valid_csv, index=False)

            model_dir  = str(self.file_root / f"models/{cfg['model_id']}")
            os.makedirs(model_dir, exist_ok=True)
            meta_path  = os.path.join(tmpdir, "meta.json")

            r_args = {
                "mode":       "train",
                "train_path": train_csv,
                "valid_path": valid_csv,
                "model_dir":  model_dir,
                "target_col": cfg["target_col"],
                "meta_path":  meta_path,
                **cfg.get("r_args", {}),
            }

            self._update_job_status(ExecutorStatus.RUNNING, progress=40)
            self._run_rscript(cfg["r_script"], r_args, cfg.get("rscript_cmd", "Rscript"))
            self._update_job_status(ExecutorStatus.RUNNING, progress=85)

            # R 스크립트가 생성한 메타 읽기
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            else:
                meta = {"model_id": cfg["model_id"], "model_dir": model_dir}

        meta["model_type"]  = "r"
        meta["r_script"]    = cfg["r_script"]
        meta["target_col"]  = cfg["target_col"]
        meta["model_path"]  = f"models/{cfg['model_id']}"
        self._save_json(meta, f"models/{cfg['model_id']}_meta.json")

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  meta,
            "message": f"R 모델 학습 완료  model_id={cfg['model_id']}",
        }

    def _run_predict(self, cfg: dict) -> dict:
        input_df = self._load_dataframe(cfg["input_path"])
        self._update_job_status(ExecutorStatus.RUNNING, progress=20)

        # 모델 메타 로드
        meta_path = self.file_root / f"models/{cfg['model_id']}_meta.json"
        if not meta_path.exists():
            raise ExecutorException(f"R 모델 메타 없음: {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv  = os.path.join(tmpdir, "input.csv")
            output_csv = os.path.join(tmpdir, "output.csv")
            input_df.to_csv(input_csv, index=False)

            r_args = {
                "mode":        "predict",
                "input_path":  input_csv,
                "output_path": output_csv,
                "model_dir":   str(self.file_root / meta["model_path"]),
                **cfg.get("r_args", {}),
            }

            self._update_job_status(ExecutorStatus.RUNNING, progress=50)
            self._run_rscript(meta["r_script"], r_args, cfg.get("rscript_cmd", "Rscript"))

            if not os.path.exists(output_csv):
                raise ExecutorException("R 스크립트가 예측 결과 파일을 생성하지 않았습니다.")

            pred_df = pd.read_csv(output_csv)

        # 결과 병합 및 저장
        result_df = pd.concat(
            [input_df.reset_index(drop=True), pred_df.reset_index(drop=True)],
            axis=1
        )
        output_path = f"predict/{cfg.get('output_id', cfg['model_id'])}_r_result.parquet"
        self._save_dataframe(result_df, output_path)
        self._update_job_status(ExecutorStatus.RUNNING, progress=90)

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result": {
                "output_path": output_path,
                "total_rows":  len(result_df),
                "pred_cols":   list(pred_df.columns),
            },
            "message": f"R 예측 완료  {len(result_df):,}건",
        }

    # ------------------------------------------------------------------

    def _run_rscript(self, script_rel_path: str, args: dict, rscript_cmd: str = "Rscript") -> None:
        """R 스크립트를 실행하고 stdout/stderr를 로깅한다."""
        script_path = self.file_root / script_rel_path
        if not script_path.exists():
            raise ExecutorException(f"R 스크립트 없음: {script_path}")

        # args를 JSON으로 전달
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(args, f)
            args_file = f.name

        cmd = [rscript_cmd, str(script_path), "--args_file", args_file]
        logger.info("Rscript 실행: %s", " ".join(cmd))

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        os.unlink(args_file)

        if proc.stdout:
            for line in proc.stdout.splitlines():
                logger.info("[R] %s", line)
        if proc.stderr:
            for line in proc.stderr.splitlines():
                logger.warning("[R STDERR] %s", line)

        if proc.returncode != 0:
            raise ExecutorException(
                f"R 스크립트 실행 실패  returncode={proc.returncode}\n{proc.stderr[-500:]}"
            )
