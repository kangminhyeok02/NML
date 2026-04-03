"""
predict_executor.py
-------------------
저장된 모델을 로드하여 신규 데이터에 대한 예측을 수행하는 실행기.

운영 환경에서 가장 빈번하게 호출되는 executor.
점수(score), 확률(probability), 등급(grade) 형태의 예측 결과를 생성한다.

실행 순서:
  1. 모델 메타 정보 로드
  2. 모델 파일 로드 (pickle / H2O MOJO / R RDS)
  3. 예측 대상 데이터 로드
  4. 피처 정렬 및 전처리
  5. 예측 수행 (score / probability / class)
  6. 후처리 (등급 부여, 정책 적용)
  7. 결과 저장
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)


class PredictExecutor(BaseExecutor):
    """
    예측 실행 executor.

    config 필수 키
    --------------
    model_id      : str   사용할 모델 식별자
    input_path    : str   예측 대상 데이터 상대 경로
    output_id     : str   결과 저장 식별자

    config 선택 키
    --------------
    score_col     : str   예측 점수 컬럼명 (기본: "score")
    grade_mapping : dict  점수 → 등급 매핑 구간 (예: {"A": [800, 1000], "B": [600, 800]})
    threshold     : float 이진 분류 임계값 (기본: 0.5)
    output_path   : str   결과 파일 저장 경로 (기본 자동 생성)
    model_type    : str   "python" | "h2o" | "r"  (기본: "python")
    """

    def execute(self) -> dict:
        cfg = self.config
        model_type = cfg.get("model_type", "python")

        # 1. 모델 메타 로드
        meta = self._load_model_meta(cfg["model_id"])

        # 2. 모델 로드
        self._update_job_status(ExecutorStatus.RUNNING, progress=15)
        model = self._load_model(meta, model_type)
        logger.info("model loaded  model_id=%s  type=%s", cfg["model_id"], model_type)

        # 3. 입력 데이터 로드
        df = self._load_dataframe(cfg["input_path"])
        logger.info("input data loaded  shape=%s", df.shape)
        self._update_job_status(ExecutorStatus.RUNNING, progress=35)

        # 4. 피처 정렬
        feature_cols = meta.get("feature_cols", [c for c in df.columns])
        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            raise ExecutorException(f"예측 데이터에 필요한 컬럼이 없습니다: {missing_cols}")
        X = df[feature_cols]

        # 5. 예측
        score_col = cfg.get("score_col", "score")
        result_df = df.copy()

        if model_type == "python":
            result_df = self._predict_python(model, X, result_df, score_col, cfg)
        elif model_type == "h2o":
            result_df = self._predict_h2o(model, X, result_df, score_col)
        elif model_type == "r":
            result_df = self._predict_r(meta, X, result_df, score_col)
        else:
            raise ExecutorException(f"지원하지 않는 model_type: {model_type}")

        self._update_job_status(ExecutorStatus.RUNNING, progress=70)

        # 6. 후처리 - 등급 부여
        grade_mapping = cfg.get("grade_mapping")
        if grade_mapping:
            result_df["grade"] = result_df[score_col].apply(
                lambda s: self._assign_grade(s, grade_mapping)
            )

        # 7. 저장
        output_path = cfg.get(
            "output_path",
            f"predict/{cfg['output_id']}_result.parquet"
        )
        self._save_dataframe(result_df, output_path)
        self._update_job_status(ExecutorStatus.RUNNING, progress=95)

        summary = {
            "output_id":    cfg["output_id"],
            "model_id":     cfg["model_id"],
            "total_rows":   len(result_df),
            "output_path":  output_path,
            "score_stats":  _series_stats(result_df[score_col]),
        }
        if "grade" in result_df.columns:
            summary["grade_dist"] = result_df["grade"].value_counts().to_dict()

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  summary,
            "message": f"예측 완료: {len(result_df):,}건  model={cfg['model_id']}",
        }

    # ------------------------------------------------------------------

    def _load_model_meta(self, model_id: str) -> dict:
        meta_path = self.file_root / f"models/{model_id}_meta.json"
        if not meta_path.exists():
            raise ExecutorException(f"모델 메타 파일이 없습니다: {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self, meta: dict, model_type: str):
        if model_type == "python":
            model_path = self.file_root / meta["model_path"]
            if not model_path.exists():
                raise ExecutorException(f"모델 파일이 없습니다: {model_path}")
            with open(model_path, "rb") as f:
                return pickle.load(f)
        elif model_type == "h2o":
            import h2o
            h2o.init()
            return h2o.import_mojo(str(self.file_root / meta["model_path"]))
        elif model_type == "r":
            # R 모델은 메타 정보만 필요; 실제 예측은 subprocess 호출
            return meta
        else:
            raise ExecutorException(f"지원하지 않는 model_type: {model_type}")

    def _predict_python(self, model, X, result_df, score_col, cfg):
        threshold = cfg.get("threshold", 0.5)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            result_df[score_col]    = np.round(proba, 6)
            result_df["pred_class"] = (proba >= threshold).astype(int)
        else:
            pred = model.predict(X)
            result_df[score_col] = np.round(pred, 6)
        return result_df

    def _predict_h2o(self, model, X, result_df, score_col):
        import h2o
        h2o_frame = h2o.H2OFrame(X)
        preds = model.predict(h2o_frame).as_data_frame()
        result_df[score_col] = preds.iloc[:, -1].values
        return result_df

    def _predict_r(self, meta, X, result_df, score_col):
        import subprocess, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_in:
            X.to_csv(tmp_in.name, index=False)
            tmp_input = tmp_in.name
        tmp_output = tmp_input.replace(".csv", "_pred.csv")

        r_script = meta.get("r_script_path", "r_scripts/predict.R")
        cmd = [
            "Rscript", str(self.file_root / r_script),
            "--input", tmp_input,
            "--model", str(self.file_root / meta["model_path"]),
            "--output", tmp_output,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ExecutorException(f"R 예측 실패: {proc.stderr}")

        preds = pd.read_csv(tmp_output)
        result_df[score_col] = preds.iloc[:, 0].values
        os.unlink(tmp_input)
        os.unlink(tmp_output)
        return result_df

    def _assign_grade(self, score: float, grade_mapping: dict) -> str:
        for grade, (low, high) in grade_mapping.items():
            if low <= score < high:
                return grade
        return "UNKNOWN"


def _series_stats(series: pd.Series) -> dict:
    return {
        "mean":   round(float(series.mean()), 4),
        "std":    round(float(series.std()), 4),
        "min":    round(float(series.min()), 4),
        "p25":    round(float(series.quantile(0.25)), 4),
        "p50":    round(float(series.quantile(0.50)), 4),
        "p75":    round(float(series.quantile(0.75)), 4),
        "max":    round(float(series.max()), 4),
    }
