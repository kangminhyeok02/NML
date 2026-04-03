"""
python_h2o_model_executor.py
----------------------------
Python 서비스 파이프라인과 H2O 모델을 통합하는 실행기.

h2o_model_executor.py가 순수 H2O 로직에 집중한다면,
이 executor는 Python 기반 전처리/후처리와 H2O 모델 추론을 결합한다.

대표 시나리오:
  1. Python으로 feature engineering 수행
  2. H2O MOJO 모델로 점수 산출
  3. Python으로 결과 후처리 (스케일링, 등급화, 마스킹)

즉, H2O 모델이 학습에 쓰였지만 추론 파이프라인은
Python 코드로 제어해야 할 때 사용한다.
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


class PythonH2OModelExecutor(BaseExecutor):
    """
    Python + H2O 통합 executor.

    config 필수 키
    --------------
    model_id    : str   H2O 모델 식별자 (MOJO 경로 포함된 메타)
    input_path  : str   입력 데이터 경로 (.parquet)
    output_id   : str   결과 저장 식별자

    config 선택 키
    --------------
    preprocess_steps   : list  Python 전처리 스텝 목록
    postprocess_steps  : list  Python 후처리 스텝 목록
    score_col          : str   점수 컬럼명 (기본: "score")
    h2o_ip             : str   H2O 서버 IP (기본: localhost)
    h2o_port           : int   H2O 서버 포트 (기본: 54321)
    use_mojo           : bool  MOJO 사용 여부 (기본: True)
    """

    def execute(self) -> dict:
        cfg       = self.config
        score_col = cfg.get("score_col", "score")

        # 메타 로드
        meta_path = self.file_root / f"models/{cfg['model_id']}_meta.json"
        if not meta_path.exists():
            raise ExecutorException(f"모델 메타 없음: {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        # 입력 데이터 로드
        df = self._load_dataframe(cfg["input_path"])
        logger.info("Python+H2O pipeline  rows=%d", len(df))
        self._update_job_status(ExecutorStatus.RUNNING, progress=15)

        # 1. Python 전처리
        pre_steps = cfg.get("preprocess_steps", [])
        df = self._apply_preprocess(df, pre_steps)
        self._update_job_status(ExecutorStatus.RUNNING, progress=35)

        # 2. H2O 추론
        feature_cols = meta.get("feature_cols") or list(df.columns)
        X = df[[c for c in feature_cols if c in df.columns]]

        use_mojo = cfg.get("use_mojo", True)
        if use_mojo:
            scores = self._predict_mojo(meta, X, cfg)
        else:
            scores = self._predict_h2o_live(meta, X, cfg)

        df[score_col] = np.round(scores, 6)
        self._update_job_status(ExecutorStatus.RUNNING, progress=70)

        # 3. Python 후처리
        post_steps = cfg.get("postprocess_steps", [])
        df = self._apply_postprocess(df, post_steps, score_col)
        self._update_job_status(ExecutorStatus.RUNNING, progress=90)

        # 저장
        output_path = f"predict/{cfg['output_id']}_py_h2o.parquet"
        self._save_dataframe(df, output_path)

        summary = {
            "output_id":   cfg["output_id"],
            "model_id":    cfg["model_id"],
            "total_rows":  len(df),
            "output_path": output_path,
            "score_stats": _stats(df[score_col]),
        }

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  summary,
            "message": f"Python+H2O 파이프라인 완료  {len(df):,}건",
        }

    # ------------------------------------------------------------------

    def _apply_preprocess(self, df: pd.DataFrame, steps: list) -> pd.DataFrame:
        """
        steps 예시:
          [{"type": "fillna",   "columns": ["col_a"], "value": 0},
           {"type": "clip",     "columns": ["col_b"], "lower": 0, "upper": 100},
           {"type": "log1p",    "columns": ["col_c"]},
           {"type": "eval",     "name": "ratio",     "expr": "col_a / (col_b + 1)"}]
        """
        for step in steps:
            step_type = step["type"]
            cols = step.get("columns", [])
            try:
                if step_type == "fillna":
                    df[cols] = df[cols].fillna(step["value"])
                elif step_type == "clip":
                    df[cols] = df[cols].clip(lower=step.get("lower"), upper=step.get("upper"))
                elif step_type == "log1p":
                    for col in cols:
                        df[col] = np.log1p(df[col].clip(lower=0))
                elif step_type == "eval":
                    df[step["name"]] = df.eval(step["expr"])
                elif step_type == "drop":
                    df = df.drop(columns=[c for c in cols if c in df.columns])
                else:
                    logger.warning("알 수 없는 전처리 step: %s", step_type)
            except Exception as exc:
                logger.warning("전처리 step 실패: %s  reason=%s", step_type, exc)
        return df

    def _apply_postprocess(self, df: pd.DataFrame, steps: list, score_col: str) -> pd.DataFrame:
        """
        steps 예시:
          [{"type": "scale",  "method": "minmax", "col": "score"},
           {"type": "grade",  "col": "score",     "map": {"A": [800, 1000], "B": [600, 800]}},
           {"type": "round",  "col": "score",     "decimals": 0}]
        """
        for step in steps:
            step_type = step["type"]
            col = step.get("col", score_col)
            try:
                if step_type == "scale" and col in df.columns:
                    method = step.get("method", "minmax")
                    if method == "minmax":
                        mn, mx = df[col].min(), df[col].max()
                        df[col] = (df[col] - mn) / (mx - mn + 1e-9)
                    elif method == "standard":
                        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
                elif step_type == "grade" and col in df.columns:
                    grade_map = step["map"]
                    df["grade"] = df[col].apply(
                        lambda s: next((g for g, (lo, hi) in grade_map.items() if lo <= s < hi), "UNKNOWN")
                    )
                elif step_type == "round" and col in df.columns:
                    df[col] = df[col].round(step.get("decimals", 0))
                else:
                    logger.warning("알 수 없는 후처리 step: %s", step_type)
            except Exception as exc:
                logger.warning("후처리 step 실패: %s  reason=%s", step_type, exc)
        return df

    def _predict_mojo(self, meta: dict, X: pd.DataFrame, cfg: dict) -> np.ndarray:
        """H2O MOJO를 EasyPredictModelWrapper로 실행 (h2o 서버 불필요)."""
        mojo_path = str(self.file_root / meta.get("mojo_path", f"models/{meta['model_id']}/model.zip"))
        try:
            import h2o
            from h2o.estimators import H2OEstimator
            h2o.init(ip=cfg.get("h2o_ip", "localhost"), port=cfg.get("h2o_port", 54321))
            model   = h2o.import_mojo(mojo_path)
            h2oframe = h2o.H2OFrame(X)
            preds   = model.predict(h2oframe).as_data_frame()
            return preds.iloc[:, -1].values
        except Exception as exc:
            raise ExecutorException(f"H2O MOJO 추론 실패: {exc}")

    def _predict_h2o_live(self, meta: dict, X: pd.DataFrame, cfg: dict) -> np.ndarray:
        """H2O 서버에 살아있는 모델로 실시간 추론."""
        import h2o
        h2o.init(ip=cfg.get("h2o_ip", "localhost"), port=cfg.get("h2o_port", 54321))
        model_key = meta.get("h2o_model_id")
        if not model_key:
            raise ExecutorException("h2o_model_id가 메타에 없습니다.")
        model    = h2o.get_model(model_key)
        h2oframe = h2o.H2OFrame(X)
        preds    = model.predict(h2oframe).as_data_frame()
        return preds.iloc[:, -1].values


def _stats(s: pd.Series) -> dict:
    return {k: round(float(v), 4) for k, v in {
        "mean": s.mean(), "min": s.min(), "max": s.max(), "std": s.std(),
    }.items()}
