"""
python_model_executor.py
------------------------
순수 Python 기반 ML 모델 학습/평가/저장 실행기.

scikit-learn, XGBoost, LightGBM, CatBoost 계열 모델을 지원한다.
모델 유형은 config의 model_type 키로 결정한다.

실행 순서:
  1. 학습/검증 데이터 로드
  2. 피처/타깃 분리
  3. 모델 인스턴스 생성
  4. 학습 (fit)
  5. 검증 세트 성능 평가
  6. 모델 및 메타 정보 저장
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, mean_squared_error, r2_score,
)

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)


# 지원 모델 레지스트리
def _build_model(model_type: str, params: dict):
    model_type = model_type.lower()

    if model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)

    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)

    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)

    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)

    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params)

    elif model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(**params)

    elif model_type == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**params)

    elif model_type == "linear_regression":
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**params)

    elif model_type == "random_forest_regressor":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)

    else:
        raise ExecutorException(f"지원하지 않는 model_type: {model_type}")


class PythonModelExecutor(BaseExecutor):
    """
    Python ML 모델 학습 executor.

    config 필수 키
    --------------
    model_type    : str   모델 유형 (예: "lightgbm", "xgboost", "logistic_regression")
    train_path    : str   학습 데이터 상대 경로 (.parquet)
    target_col    : str   타깃 컬럼명
    model_id      : str   저장할 모델 식별자

    config 선택 키
    --------------
    valid_path    : str   검증 데이터 경로 (없으면 학습 데이터의 20% 자동 분리)
    feature_cols  : list  사용할 피처 목록 (없으면 타깃 외 전체)
    model_params  : dict  모델 하이퍼파라미터
    task          : str   "classification" | "regression" (기본: "classification")
    """

    def execute(self) -> dict:
        cfg = self.config
        model_type  = cfg["model_type"]
        target_col  = cfg["target_col"]
        task        = cfg.get("task", "classification")
        model_params = cfg.get("model_params", {})

        # 1. 데이터 로드
        train_df = self._load_dataframe(cfg["train_path"])
        if "valid_path" in cfg:
            valid_df = self._load_dataframe(cfg["valid_path"])
        else:
            from sklearn.model_selection import train_test_split
            train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)

        self._update_job_status(ExecutorStatus.RUNNING, progress=20)

        # 2. 피처/타깃 분리
        feature_cols = cfg.get("feature_cols") or [c for c in train_df.columns if c != target_col]
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_valid = valid_df[feature_cols]
        y_valid = valid_df[target_col]

        # 3. 모델 생성
        model = _build_model(model_type, model_params)
        logger.info("training  model=%s  train_rows=%d  features=%d", model_type, len(X_train), len(feature_cols))
        self._update_job_status(ExecutorStatus.RUNNING, progress=40)

        # 4. 학습
        model.fit(X_train, y_train)
        self._update_job_status(ExecutorStatus.RUNNING, progress=75)

        # 5. 평가
        metrics = self._evaluate(model, X_valid, y_valid, task)
        logger.info("validation metrics: %s", metrics)

        # 6. 모델 저장
        model_path = f"models/{cfg['model_id']}.pkl"
        full_path = self.file_root / model_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "wb") as f:
            pickle.dump(model, f)

        # 메타 저장
        meta = {
            "model_id":     cfg["model_id"],
            "model_type":   model_type,
            "model_params": model_params,
            "feature_cols": feature_cols,
            "target_col":   target_col,
            "task":         task,
            "metrics":      metrics,
            "model_path":   model_path,
        }
        self._save_json(meta, f"models/{cfg['model_id']}_meta.json")
        self._update_job_status(ExecutorStatus.RUNNING, progress=95)

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  meta,
            "message": f"모델 학습 완료: {model_type}  {_metrics_summary(metrics)}",
        }

    def _evaluate(self, model, X: pd.DataFrame, y: pd.Series, task: str) -> dict:
        y_pred = model.predict(X)
        metrics: dict = {}

        if task == "classification":
            metrics["accuracy"]  = round(float(accuracy_score(y, y_pred)), 4)
            metrics["precision"] = round(float(precision_score(y, y_pred, average="binary", zero_division=0)), 4)
            metrics["recall"]    = round(float(recall_score(y, y_pred, average="binary", zero_division=0)), 4)
            metrics["f1"]        = round(float(f1_score(y, y_pred, average="binary", zero_division=0)), 4)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)[:, 1]
                metrics["auc"] = round(float(roc_auc_score(y, y_proba)), 4)
        else:
            metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y, y_pred))), 4)
            metrics["r2"]   = round(float(r2_score(y, y_pred)), 4)

        return metrics


def _metrics_summary(metrics: dict) -> str:
    if "auc" in metrics:
        return f"AUC={metrics['auc']}  F1={metrics.get('f1', 'N/A')}"
    if "rmse" in metrics:
        return f"RMSE={metrics['rmse']}  R2={metrics.get('r2', 'N/A')}"
    return str(metrics)
