"""
h2o_model_executor.py
---------------------
H2O 프레임워크 기반 모델 학습/예측 실행기.

H2O는 GBM, DRF, XGBoost, DeepLearning, GLM 등 다양한 알고리즘을 제공한다.
H2O 서버와 연동하여 대용량 데이터 학습과 빠른 예측을 지원한다.

실행 순서:
  1. H2O 클러스터 초기화/연결
  2. 데이터 → H2OFrame 변환
  3. train/valid 분리
  4. H2O 모델 학습
  5. 성능 평가
  6. 모델 MOJO 저장
  7. H2O 세션 종료 (선택)
"""

import logging
from typing import Optional

import pandas as pd

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)

H2O_ALGORITHMS = {
    "gbm":          "H2OGradientBoostingEstimator",
    "drf":          "H2ORandomForestEstimator",
    "xgboost":      "H2OXGBoostEstimator",
    "glm":          "H2OGeneralizedLinearEstimator",
    "deeplearning": "H2ODeepLearningEstimator",
    "automl":       "H2OAutoML",
}


class H2OModelExecutor(BaseExecutor):
    """
    H2O 모델 학습/예측 executor.

    config 필수 키
    --------------
    algorithm   : str   H2O 알고리즘 (gbm / drf / xgboost / glm / deeplearning / automl)
    train_path  : str   학습 데이터 경로 (.parquet)
    target_col  : str   타깃 컬럼명
    model_id    : str   저장 식별자

    config 선택 키
    --------------
    valid_path      : str   검증 데이터 경로
    feature_cols    : list  사용 피처 목록
    model_params    : dict  H2O 알고리즘 파라미터
    h2o_ip          : str   H2O 서버 IP (기본: localhost)
    h2o_port        : int   H2O 서버 포트 (기본: 54321)
    max_runtime_sec : int   AutoML 최대 실행 시간 (초)
    """

    def execute(self) -> dict:
        import h2o

        cfg       = self.config
        algorithm = cfg["algorithm"].lower()
        target    = cfg["target_col"]

        # 1. H2O 초기화
        h2o_ip   = cfg.get("h2o_ip", "localhost")
        h2o_port = cfg.get("h2o_port", 54321)
        try:
            h2o.init(ip=h2o_ip, port=h2o_port, nthreads=-1)
            logger.info("H2O cluster connected: %s:%d", h2o_ip, h2o_port)
        except Exception as exc:
            raise ExecutorException(f"H2O 클러스터 연결 실패: {exc}")

        self._update_job_status(ExecutorStatus.RUNNING, progress=15)

        # 2. 데이터 로드 → H2OFrame 변환
        train_df = self._load_dataframe(cfg["train_path"])
        train_h2o = h2o.H2OFrame(train_df)

        if "valid_path" in cfg:
            valid_df  = self._load_dataframe(cfg["valid_path"])
            valid_h2o = h2o.H2OFrame(valid_df)
        else:
            train_h2o, valid_h2o = train_h2o.split_frame(ratios=[0.8], seed=42)

        # 타깃 → factor (분류)
        train_h2o[target] = train_h2o[target].asfactor()
        valid_h2o[target] = valid_h2o[target].asfactor()

        feature_cols = cfg.get("feature_cols") or [c for c in train_h2o.columns if c != target]
        self._update_job_status(ExecutorStatus.RUNNING, progress=30)

        # 3. 모델 학습
        model_params = cfg.get("model_params", {})
        model = self._train_model(h2o, algorithm, feature_cols, target, train_h2o, valid_h2o, cfg, model_params)
        logger.info("H2O model trained: %s", model.model_id)
        self._update_job_status(ExecutorStatus.RUNNING, progress=70)

        # 4. 성능 평가
        perf   = model.model_performance(valid_h2o)
        metrics = {
            "auc":      round(float(perf.auc()), 4),
            "logloss":  round(float(perf.logloss()), 4),
        }
        try:
            metrics["ks"] = round(float(perf.kolmogorov_smirnov()), 4)
        except Exception:
            pass

        # 5. 모델 저장 (MOJO)
        mojo_dir = str(self.file_root / f"models/{cfg['model_id']}")
        import os
        os.makedirs(mojo_dir, exist_ok=True)
        mojo_path = model.save_mojo(mojo_dir)
        logger.info("MOJO saved: %s", mojo_path)

        # 6. 변수 중요도
        try:
            varimp = model.varimp(use_pandas=True)[["variable", "percentage"]].head(20)
            varimp_dict = varimp.set_index("variable")["percentage"].round(4).to_dict()
        except Exception:
            varimp_dict = {}

        self._update_job_status(ExecutorStatus.RUNNING, progress=90)

        meta = {
            "model_id":      cfg["model_id"],
            "algorithm":     algorithm,
            "h2o_model_id":  model.model_id,
            "mojo_path":     mojo_path,
            "feature_cols":  feature_cols,
            "target_col":    target,
            "metrics":       metrics,
            "varimp":        varimp_dict,
            "model_type":    "h2o",
        }
        self._save_json(meta, f"models/{cfg['model_id']}_meta.json")

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  meta,
            "message": f"H2O 모델 학습 완료  {algorithm}  AUC={metrics.get('auc', 'N/A')}",
        }

    # ------------------------------------------------------------------

    def _train_model(self, h2o, algorithm, x, y, train_h2o, valid_h2o, cfg, params):
        from h2o.estimators import (
            H2OGradientBoostingEstimator,
            H2ORandomForestEstimator,
            H2OXGBoostEstimator,
            H2OGeneralizedLinearEstimator,
            H2ODeepLearningEstimator,
        )
        from h2o.automl import H2OAutoML

        algo_map = {
            "gbm":          H2OGradientBoostingEstimator,
            "drf":          H2ORandomForestEstimator,
            "xgboost":      H2OXGBoostEstimator,
            "glm":          H2OGeneralizedLinearEstimator,
            "deeplearning": H2ODeepLearningEstimator,
        }

        if algorithm == "automl":
            max_rt = cfg.get("max_runtime_sec", 300)
            aml = H2OAutoML(max_runtime_secs=max_rt, seed=42, **params)
            aml.train(x=x, y=y, training_frame=train_h2o, leaderboard_frame=valid_h2o)
            return aml.leader

        if algorithm not in algo_map:
            raise ExecutorException(f"지원하지 않는 H2O 알고리즘: {algorithm}")

        estimator_cls = algo_map[algorithm]
        model = estimator_cls(**params)
        model.train(x=x, y=y, training_frame=train_h2o, validation_frame=valid_h2o)
        return model
