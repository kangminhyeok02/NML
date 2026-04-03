"""
automl_executor.py
------------------
자동 모델 탐색(AutoML) 실행기.

다양한 AutoML 프레임워크를 통합 지원하며,
여러 알고리즘 후보를 자동으로 탐색하고 최적 모델을 선택한다.

지원 프레임워크:
  - h2o_automl   : H2O AutoML (리더보드 기반)
  - autosklearn  : auto-sklearn (앙상블 탐색)
  - tpot         : TPOT (유전 알고리즘 기반 파이프라인)
  - optuna       : Optuna (베이지안 최적화 + 지정 알고리즘)
  - pycaret      : PyCaret (비교 실험 자동화)

실행 순서:
  1. 프레임워크 선택 및 초기화
  2. 학습/검증 데이터 로드
  3. AutoML 실행 (탐색 + 평가)
  4. 리더보드 생성
  5. 최적 모델 저장
"""

import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)

SUPPORTED_FRAMEWORKS = ["h2o_automl", "autosklearn", "tpot", "optuna", "pycaret"]


class AutoMLExecutor(BaseExecutor):
    """
    AutoML 실행 executor.

    config 필수 키
    --------------
    framework     : str   AutoML 프레임워크 ("h2o_automl" | "autosklearn" | "tpot" | "optuna" | "pycaret")
    train_path    : str   학습 데이터 경로 (.parquet)
    target_col    : str   타깃 컬럼명
    model_id      : str   결과 저장 식별자

    config 선택 키
    --------------
    valid_path      : str   검증 데이터 경로
    feature_cols    : list  사용 피처 목록
    max_runtime_sec : int   최대 탐색 시간(초) 기본 300
    n_trials        : int   Optuna 시도 횟수 (기본 50)
    metric          : str   최적화 지표 (기본 "auc")
    """

    def execute(self) -> dict:
        cfg       = self.config
        framework = cfg.get("framework", "optuna").lower()

        if framework not in SUPPORTED_FRAMEWORKS:
            raise ExecutorException(f"지원하지 않는 framework: {framework}  지원: {SUPPORTED_FRAMEWORKS}")

        # 데이터 로드
        train_df = self._load_dataframe(cfg["train_path"])
        if "valid_path" in cfg:
            valid_df = self._load_dataframe(cfg["valid_path"])
        else:
            train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)

        target_col   = cfg["target_col"]
        feature_cols = cfg.get("feature_cols") or [c for c in train_df.columns if c != target_col]
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_valid = valid_df[feature_cols]
        y_valid = valid_df[target_col]

        self._update_job_status(ExecutorStatus.RUNNING, progress=20)
        logger.info("AutoML 시작  framework=%s  train=%d  features=%d", framework, len(X_train), len(feature_cols))

        # 프레임워크별 실행
        dispatch = {
            "h2o_automl":  self._run_h2o_automl,
            "autosklearn": self._run_autosklearn,
            "tpot":        self._run_tpot,
            "optuna":      self._run_optuna,
            "pycaret":     self._run_pycaret,
        }
        best_model, leaderboard = dispatch[framework](cfg, X_train, y_train, X_valid, y_valid)

        self._update_job_status(ExecutorStatus.RUNNING, progress=80)

        # 최적 모델 저장
        model_path = f"models/{cfg['model_id']}_automl.pkl"
        full_path = self.file_root / model_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "wb") as f:
            pickle.dump(best_model, f)

        # 최종 검증 성능
        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(X_valid)[:, 1]
            final_auc = round(float(roc_auc_score(y_valid, y_proba)), 4)
        else:
            final_auc = None

        meta = {
            "model_id":    cfg["model_id"],
            "framework":   framework,
            "feature_cols": feature_cols,
            "target_col":  target_col,
            "final_auc":   final_auc,
            "leaderboard": leaderboard,
            "model_path":  model_path,
            "model_type":  "python",
        }
        self._save_json(meta, f"models/{cfg['model_id']}_meta.json")

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  meta,
            "message": f"AutoML 완료  framework={framework}  AUC={final_auc}  후보={len(leaderboard)}개",
        }

    # ------------------------------------------------------------------
    # 프레임워크별 실행 메서드
    # ------------------------------------------------------------------

    def _run_h2o_automl(self, cfg, X_train, y_train, X_valid, y_valid):
        import h2o
        from h2o.automl import H2OAutoML

        h2o.init(ip=cfg.get("h2o_ip", "localhost"), port=cfg.get("h2o_port", 54321))
        target = cfg["target_col"]
        train_df = pd.concat([X_train, y_train], axis=1)
        valid_df = pd.concat([X_valid, y_valid], axis=1)

        train_h2o = h2o.H2OFrame(train_df)
        valid_h2o = h2o.H2OFrame(valid_df)
        train_h2o[target] = train_h2o[target].asfactor()
        valid_h2o[target] = valid_h2o[target].asfactor()

        aml = H2OAutoML(max_runtime_secs=cfg.get("max_runtime_sec", 300), seed=42)
        aml.train(x=list(X_train.columns), y=target, training_frame=train_h2o, leaderboard_frame=valid_h2o)

        lb = aml.leaderboard.as_data_frame().to_dict(orient="records")

        # best model → sklearn-compatible wrapper
        best_h2o = aml.leader
        class H2OWrapper:
            def __init__(self, m): self.m = m
            def predict_proba(self, X):
                import h2o as _h2o
                frame = _h2o.H2OFrame(X)
                preds = self.m.predict(frame).as_data_frame()
                return np.column_stack([1 - preds.iloc[:, -1].values, preds.iloc[:, -1].values])
        return H2OWrapper(best_h2o), lb

    def _run_autosklearn(self, cfg, X_train, y_train, X_valid, y_valid):
        import autosklearn.classification as askl
        model = askl.AutoSklearnClassifier(
            time_left_for_this_task=cfg.get("max_runtime_sec", 300),
            per_run_time_limit=30,
            seed=42,
        )
        model.fit(X_train.values, y_train.values)
        leaderboard = [{"model": str(m), "weight": w} for m, w in model.get_models_with_weights()]
        return model, leaderboard

    def _run_tpot(self, cfg, X_train, y_train, X_valid, y_valid):
        from tpot import TPOTClassifier
        model = TPOTClassifier(
            max_time_mins=cfg.get("max_runtime_sec", 300) // 60,
            verbosity=2,
            random_state=42,
        )
        model.fit(X_train.values, y_train.values)
        leaderboard = [{"pipeline": str(model.fitted_pipeline_)}]
        return model.fitted_pipeline_, leaderboard

    def _run_optuna(self, cfg, X_train, y_train, X_valid, y_valid):
        import optuna
        from lightgbm import LGBMClassifier
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators":    trial.suggest_int("n_estimators", 50, 500),
                "max_depth":       trial.suggest_int("max_depth", 3, 12),
                "learning_rate":   trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves":      trial.suggest_int("num_leaves", 16, 128),
                "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "verbose": -1,
            }
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[])
            y_proba = model.predict_proba(X_valid)[:, 1]
            return roc_auc_score(y_valid, y_proba)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=cfg.get("n_trials", 50))

        best_params = study.best_params
        best_model  = LGBMClassifier(**best_params)
        best_model.fit(X_train, y_train)

        leaderboard = [
            {"rank": i + 1, "auc": round(t.value, 4), "params": t.params}
            for i, t in enumerate(
                sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:10]
            )
        ]
        return best_model, leaderboard

    def _run_pycaret(self, cfg, X_train, y_train, X_valid, y_valid):
        from pycaret.classification import setup, compare_models, get_config

        train_df = pd.concat([X_train, y_train.rename(cfg["target_col"])], axis=1)
        setup(data=train_df, target=cfg["target_col"], verbose=False, session_id=42)
        best = compare_models(sort="AUC", n_select=1)
        leaderboard = get_config("master_model_container")
        lb = [{"model": str(m)} for m in leaderboard[:10]]
        return best, lb
