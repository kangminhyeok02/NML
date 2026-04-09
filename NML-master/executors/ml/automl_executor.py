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


# =============================================================================
# Module-level functions
# =============================================================================


def fit_nice_auto_ml(
    service_db_info,
    file_server_host,
    file_server_port,
    h2o_file_server_host,
    h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf,
    done_file_path_faf,
    h2o_host,
    h2o_port,
    h2o_older_version_port,
    root_dir,
    json_obj,
) -> dict:
    """NICE AutoML 통합 학습."""
    import json
    import math
    import os
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from hyperopt import fmin, tpe, Trials, STATUS_OK, hp

    logger.info("fit_nice_auto_ml 시작  model_id=%s", json_obj.get("model_id"))

    root_path = Path(root_dir)

    # json_obj에서 파라미터 추출
    framework        = json_obj.get("framework", "optuna")
    train_path       = json_obj.get("train_path")
    target_col       = json_obj.get("target_col")
    feature_cols     = json_obj.get("feature_cols")
    model_id         = json_obj.get("model_id")
    max_runtime_sec  = int(json_obj.get("max_runtime_sec", 300))
    n_trials         = int(json_obj.get("n_trials", 50))

    # multitarget_properties 처리
    multitarget_props = json_obj.get("multitarget_properties", {})
    is_multitarget    = bool(multitarget_props)
    logger.debug("multitarget_properties=%s", multitarget_props)

    # parameters for saving / delete file / delete from DB
    model_dir = root_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"{model_id}_automl.pkl"
    meta_file  = model_dir / f"{model_id}_meta.json"

    # make parameter dictionary / variable selection
    if numpy_use_32bit_float_precision:
        float_dtype = np.float32
    else:
        float_dtype = np.float64

    # adjust fccc_result_json / input data / sampling
    train_suffix = Path(train_path).suffix.lower()
    if train_suffix == ".parquet":
        train_df = pd.read_parquet(train_path)
    else:
        train_df = pd.read_csv(train_path)

    if feature_cols is None:
        feature_cols = [c for c in train_df.columns if c != target_col]

    X = train_df[feature_cols].astype(float_dtype)
    y = train_df[target_col]

    # set h2o_frame / validation set (nfolds?)
    val_size   = json_obj.get("val_size", 0.2)
    nfolds     = int(json_obj.get("nfolds", 0))
    val_path   = json_obj.get("valid_path")

    if val_path and Path(val_path).exists():
        if Path(val_path).suffix.lower() == ".parquet":
            val_df = pd.read_parquet(val_path)
        else:
            val_df = pd.read_csv(val_path)
        X_val = val_df[feature_cols].astype(float_dtype)
        y_val = val_df[target_col]
        X_train, y_train = X, y
        use_nfolds = False
    elif nfolds > 0:
        from sklearn.model_selection import StratifiedKFold
        X_train, y_train = X, y
        X_val,   y_val   = None, None
        use_nfolds = True
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=42, stratify=y
        )
        use_nfolds = False

    logger.info("학습 데이터  train=%d  features=%d  nfolds=%d", len(X_train), len(feature_cols), nfolds)

    # define space & objective function
    search_space = json_obj.get("search_space", {})
    if search_space:
        hp_space = convert_list_to_hp(search_space)
    else:
        hp_space = {
            "n_estimators":     hp.choice("n_estimators",     [100, 200, 300, 500]),
            "max_depth":        hp.choice("max_depth",         [3, 5, 7, 9]),
            "learning_rate":    hp.uniform("learning_rate",    0.01, 0.3),
            "num_leaves":       hp.choice("num_leaves",        [31, 63, 127]),
            "subsample":        hp.uniform("subsample",        0.6, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        }

    trials   = Trials()
    history  = []
    best_ks_dev, best_ks_val = 0.0, 0.0

    # objective function / support stopping_rounds 0 when validation link exists
    def objective(params):
        from lightgbm import LGBMClassifier
        from sklearn.metrics import roc_auc_score

        params_int = {k: int(v) if k in ("n_estimators", "max_depth", "num_leaves") else v
                      for k, v in params.items()}
        params_int["verbose"] = -1

        model = LGBMClassifier(**params_int, random_state=42)

        if use_nfolds:
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
            cv_proba = cross_val_predict(model, X_train, y_train,
                                         cv=StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42),
                                         method="predict_proba")[:, 1]
            ks_dev = float(roc_auc_score(y_train, cv_proba))
            ks_val = ks_dev
        else:
            callbacks = []
            if X_val is not None:
                stopping_rounds = int(json_obj.get("stopping_rounds", 50))
                if stopping_rounds > 0:
                    from lightgbm import early_stopping as lgb_early_stopping
                    callbacks.append(lgb_early_stopping(stopping_rounds, verbose=False))
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          callbacks=callbacks)
            else:
                model.fit(X_train, y_train)

            train_proba = model.predict_proba(X_train)[:, 1]
            ks_dev      = float(roc_auc_score(y_train, train_proba))
            if X_val is not None:
                val_proba = model.predict_proba(X_val)[:, 1]
                ks_val    = float(roc_auc_score(y_val, val_proba))
            else:
                ks_val = ks_dev

        loss = get_automl_objective_loss(ks_dev, ks_val)

        # history
        history.append({
            "params": params_int,
            "ks_dev": round(ks_dev, 4),
            "ks_val": round(ks_val, 4),
            "loss":   round(loss, 6),
        })
        logger.debug("trial #%d  ks_dev=%.4f  ks_val=%.4f  loss=%.6f", len(history), ks_dev, ks_val, loss)
        return {"loss": loss, "status": STATUS_OK, "model": model, "ks_dev": ks_dev, "ks_val": ks_val}

    # hyperopt fmin()
    import time
    t_start = time.time()
    best_params = fmin(
        fn=objective,
        space=hp_space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=trials,
        timeout=max_runtime_sec,
        show_progressbar=False,
    )
    logger.info("hyperopt 완료  trials=%d  elapsed=%.1fs", len(trials.trials), time.time() - t_start)

    # DNN Hidden Layer
    hidden_layer_str = json_obj.get("hidden_layers")
    hidden_layer_cfg = convert_hidden_layer_to_dict(hidden_layer_str) if hidden_layer_str else []

    # best model by algorithm
    best_trial   = min(trials.trials, key=lambda t: t["result"]["loss"])
    best_model   = best_trial["result"]["model"]
    best_ks_dev  = best_trial["result"]["ks_dev"]
    best_ks_val  = best_trial["result"]["ks_val"]
    logger.info("best model  ks_dev=%.4f  ks_val=%.4f", best_ks_dev, best_ks_val)

    # predict / val perf / perf & score_distribution
    train_proba = best_model.predict_proba(X_train)[:, 1]
    score_col   = json_obj.get("score_col", "score")
    result_train = pd.DataFrame({score_col: train_proba, target_col: y_train.values})
    score_dist_train = result_train[score_col].describe().round(4).to_dict()

    if X_val is not None:
        val_proba   = best_model.predict_proba(X_val)[:, 1]
        result_val  = pd.DataFrame({score_col: val_proba, target_col: y_val.values})
        score_dist_val = result_val[score_col].describe().round(4).to_dict()
    else:
        score_dist_val = {}

    # Ensemble / ensemble dict
    n_ensemble = int(json_obj.get("ensemble_n_models", 3))
    ensemble_weights = generate_ensemble_weights(n_ensemble) if n_ensemble > 1 else [[1.0]]

    top_trials  = sorted(trials.trials, key=lambda t: t["result"]["loss"])[:n_ensemble]
    top_models  = [t["result"]["model"] for t in top_trials]

    ensemble_dict = {
        "n_models":       len(top_models),
        "weights":        ensemble_weights[0] if ensemble_weights else [1.0],
        "model_indices":  list(range(len(top_models))),
    }

    # Save files & DB insert
    with open(model_file, "wb") as f:
        pickle.dump({"best_model": best_model, "ensemble_models": top_models}, f)

    meta = {
        "model_id":        model_id,
        "framework":       framework,
        "feature_cols":    feature_cols,
        "target_col":      target_col,
        "ks_dev":          round(best_ks_dev, 4),
        "ks_val":          round(best_ks_val, 4),
        "n_trials":        n_trials,
        "hidden_layers":   hidden_layer_cfg,
        "ensemble":        ensemble_dict,
        "score_dist_train": score_dist_train,
        "score_dist_val":  score_dist_val,
        "history":         history[-20:],
        "model_path":      str(model_file),
        "model_type":      "python",
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # result_file_path_faf / done_file_path_faf
    result_path = Path(result_file_path_faf)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    done_path = Path(done_file_path_faf)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text("done", encoding="utf-8")

    logger.info("fit_nice_auto_ml 완료  model_id=%s  model_file=%s", model_id, model_file)
    return meta


def convert_hidden_layer_to_dict(hidden_layer_str) -> list:
    """문자열 형태의 hidden layer 설정을 dict 목록으로 변환.

    예: "64,32,16" -> [{"size": 64}, {"size": 32}, {"size": 16}]
    """
    if not hidden_layer_str:
        return []
    result = []
    for part in str(hidden_layer_str).split(","):
        part = part.strip()
        if part.isdigit():
            result.append({"size": int(part)})
        else:
            try:
                result.append({"size": int(float(part))})
            except ValueError:
                logger.warning("hidden layer 파싱 실패: '%s' — 건너뜀", part)
    return result


def convert_list_to_hp(d) -> dict:
    """list 파라미터를 hyperopt hp 공간으로 변환.

    list   -> hp.choice
    [min, max] (len==2, numeric range) -> hp.uniform
    """
    from hyperopt import hp as hyperopt_hp

    result = {}
    for key, val in d.items():
        if isinstance(val, list):
            # 길이 2이고 두 값 모두 숫자이면 연속 구간으로 해석
            if (
                len(val) == 2
                and all(isinstance(v, (int, float)) for v in val)
                and val[0] < val[1]
            ):
                result[key] = hyperopt_hp.uniform(key, val[0], val[1])
            else:
                result[key] = hyperopt_hp.choice(key, val)
        elif isinstance(val, dict):
            # {"min": x, "max": y} 형태
            lo = val.get("min", 0)
            hi = val.get("max", 1)
            result[key] = hyperopt_hp.uniform(key, lo, hi)
        else:
            # 단일 값은 고정 (choice with one element)
            result[key] = hyperopt_hp.choice(key, [val])
    return result


def get_automl_objective_loss(
    ks_dev: float,
    ks_val: float,
    best_model_multiplier: float = 5,
    val_perf_rate: float = 100,
) -> float:
    """AutoML 목적함수 손실값 계산.

    KS 개발/검증 성능과 multiplier, perf_rate로 종합 손실을 반환한다.
    AUC가 높을수록 손실이 낮아지고, 개발/검증 격차가 클수록 페널티가 부여된다.
    """
    import math

    if ks_dev <= 0:
        return 1.0

    # 과적합 격차 페널티
    gap_penalty = max(0.0, ks_dev - ks_val) * best_model_multiplier

    # 검증 성능 보상 (높을수록 낮은 손실)
    val_reward = ks_val * (val_perf_rate / 100.0)

    # 종합 손실: 1 - val_reward + gap_penalty (최소화 대상)
    loss = 1.0 - val_reward + gap_penalty

    # 수치 안정성 클리핑
    return float(max(0.0, min(loss, 10.0)))


def generate_ensemble_weights(n_models: int, inc: float = 0.1) -> list:
    """앙상블 가중치 조합 생성.

    n_models개 모델에 대해 inc 단위로 합이 1이 되는 가중치 조합 목록을 반환한다.
    """
    import math

    if n_models <= 0:
        return []
    if n_models == 1:
        return [[1.0]]

    steps = int(round(1.0 / inc))
    results = []

    def _recurse(remaining_models, remaining_sum, current):
        if remaining_models == 1:
            w = round(remaining_sum * inc, 10)
            if abs(w - round(w)) < 1e-9:
                results.append(current + [round(w, 4)])
            return
        for i in range(0, remaining_sum + 1):
            _recurse(remaining_models - 1, remaining_sum - i, current + [round(i * inc, 4)])

    _recurse(n_models, steps, [])
    logger.debug("generate_ensemble_weights  n_models=%d  combinations=%d", n_models, len(results))
    return results


def predict_nice_auto_ml(
    service_db_info,
    file_server_host,
    file_server_port,
    h2o_file_server_host,
    h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf,
    done_file_path_faf,
    h2o_host,
    h2o_port,
    h2o_older_version_port,
    root_dir,
    json_obj,
) -> dict:
    """NICE AutoML 예측."""
    import json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_nice_auto_ml 시작  model_id=%s", json_obj.get("model_id"))

    root_path  = Path(root_dir)
    model_id   = json_obj.get("model_id")
    input_path = json_obj.get("input_path")
    output_id  = json_obj.get("output_id")

    float_dtype = np.float32 if numpy_use_32bit_float_precision else np.float64

    # 메타 로드
    meta_file = root_path / "models" / f"{model_id}_meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"AutoML 메타 파일이 없습니다: {meta_file}")
    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    score_col    = json_obj.get("score_col", "score")

    # 모델 로드
    model_file = Path(meta["model_path"])
    if not model_file.exists():
        raise FileNotFoundError(f"AutoML 모델 파일이 없습니다: {model_file}")
    with open(model_file, "rb") as f:
        model_bundle = pickle.load(f)

    best_model      = model_bundle["best_model"]
    ensemble_models = model_bundle.get("ensemble_models", [best_model])

    # 입력 데이터 로드
    input_full = Path(input_path)
    if input_full.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_full)
    else:
        df = pd.read_csv(input_full)

    X = df[feature_cols].astype(float_dtype)

    # 앙상블 예측 (평균)
    ensemble_weights = meta.get("ensemble", {}).get("weights", [1.0])
    if len(ensemble_models) > 1 and len(ensemble_weights) == len(ensemble_models):
        proba_sum = np.zeros(len(X), dtype=np.float64)
        for model, w in zip(ensemble_models, ensemble_weights):
            proba_sum += model.predict_proba(X)[:, 1] * w
        proba = proba_sum
    else:
        proba = best_model.predict_proba(X)[:, 1]

    result_df           = df.copy()
    result_df[score_col] = np.round(proba, 6)

    # 출력 저장
    output_dir = root_path / "predict"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{output_id}_result.parquet"
    result_df.to_parquet(output_file, index=False)

    summary = {
        "model_id":    model_id,
        "output_id":   output_id,
        "total_rows":  len(result_df),
        "output_path": str(output_file),
        "score_mean":  round(float(proba.mean()), 4),
        "score_std":   round(float(proba.std()), 4),
    }

    result_path = Path(result_file_path_faf)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    done_path = Path(done_file_path_faf)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text("done", encoding="utf-8")

    logger.info("predict_nice_auto_ml 완료  output_id=%s  rows=%d", output_id, len(result_df))
    return summary


def apply_nice_auto_ml(
    service_db_info,
    file_server_host,
    file_server_port,
    root_dir,
    json_obj,
) -> dict:
    """NICE AutoML 모델 적용 (배포 환경에서 예측 수행)."""
    import json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("apply_nice_auto_ml 시작  model_id=%s", json_obj.get("model_id"))

    root_path  = Path(root_dir)
    model_id   = json_obj.get("model_id")
    input_path = json_obj.get("input_path")
    output_id  = json_obj.get("output_id", model_id)
    score_col  = json_obj.get("score_col", "score")

    # 메타 로드
    meta_file = root_path / "models" / f"{model_id}_meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"AutoML 메타 파일이 없습니다: {meta_file}")
    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    target_col   = meta.get("target_col")

    # 모델 로드
    model_file = Path(meta["model_path"])
    with open(model_file, "rb") as f:
        model_bundle = pickle.load(f)
    best_model = model_bundle["best_model"]

    # 입력 데이터 로드
    input_full = Path(input_path)
    if input_full.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_full)
    else:
        df = pd.read_csv(input_full)

    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"입력 데이터에 필요한 피처가 없습니다: {missing}")

    X = df[feature_cols]

    # 예측
    if hasattr(best_model, "predict_proba"):
        proba = best_model.predict_proba(X)[:, 1]
        result_df = df.copy()
        result_df[score_col] = np.round(proba, 6)
    else:
        pred = best_model.predict(X)
        result_df = df.copy()
        result_df[score_col] = np.round(pred.astype(float), 6)

    # 결과 저장
    output_dir = root_path / "apply"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{output_id}_apply.parquet"
    result_df.to_parquet(output_file, index=False)

    summary = {
        "model_id":    model_id,
        "output_id":   output_id,
        "total_rows":  len(result_df),
        "output_path": str(output_file),
        "score_mean":  round(float(result_df[score_col].mean()), 4),
    }

    logger.info("apply_nice_auto_ml 완료  output_id=%s  rows=%d", output_id, len(result_df))
    return summary
