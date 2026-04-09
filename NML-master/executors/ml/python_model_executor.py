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


# =============================================================================
# Module-level functions
# =============================================================================


def compute_balanced_class_weight(y_values) -> dict:
    """sklearn compute_class_weight 래퍼. tf v2에서 str 오류 방지 위해 dict 반환 보장."""
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    y_arr = np.array(y_values)
    classes = np.unique(y_arr)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_arr)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def fit_dnn_keras(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision, result_file_path_faf, done_file_path_faf,
    root_dir, json_obj
) -> dict:
    """Keras Sequential DNN 학습 (Dense layers, dropout, batch normalization)."""
    import json
    import os
    import pickle
    import numpy as np
    import pandas as pd

    logger.info("fit_dnn_keras start: model_id=%s", json_obj.get("model_id"))

    train_path  = os.path.join(root_dir, json_obj["train_path"])
    target_col  = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    hidden_layers = json_obj.get("hidden_layers", [128, 64])
    activation   = json_obj.get("activation", "relu")
    dropout_rate = float(json_obj.get("dropout_rate", 0.3))
    epochs       = int(json_obj.get("epochs", 50))
    batch_size   = int(json_obj.get("batch_size", 256))
    learning_rate = float(json_obj.get("learning_rate", 1e-3))
    model_id     = json_obj.get("model_id", "dnn_model")

    df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float32 if numpy_use_32bit_float_precision else np.float64)
    y = df[target_col].values

    n_classes = len(np.unique(y))
    class_weight = compute_balanced_class_weight(y)

    from tensorflow import keras
    import tensorflow as tf

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(shape=(X.shape[1],)))
    for units in hidden_layers:
        model.add(keras.layers.Dense(units, activation=activation))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout_rate))

    if n_classes == 2:
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
        metrics_list = ["AUC"]
    else:
        model.add(keras.layers.Dense(n_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
        metrics_list = ["accuracy"]

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics_list)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    ]

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )

    model_save_path = os.path.join(root_dir, f"models/{model_id}_keras")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)

    hist_log = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    meta = {
        "model_id": model_id,
        "model_type": "dnn_keras",
        "feature_cols": feature_cols,
        "target_col": target_col,
        "hidden_layers": hidden_layers,
        "activation": activation,
        "dropout_rate": dropout_rate,
        "epochs_ran": len(hist_log.get("loss", [])),
        "model_path": model_save_path,
        "history": hist_log,
    }
    meta_path = os.path.join(root_dir, f"models/{model_id}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if result_file_path_faf:
        with open(result_file_path_faf, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    if done_file_path_faf:
        open(done_file_path_faf, "w").close()

    logger.info("fit_dnn_keras done: model_id=%s  epochs_ran=%d", model_id, meta["epochs_ran"])
    return {"result": "ok", "meta": meta}


def fit_rnn_keras(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision, result_file_path_faf, done_file_path_faf,
    root_dir, json_obj
) -> dict:
    """Keras RNN/LSTM 학습."""
    import json
    import os
    import numpy as np
    import pandas as pd

    logger.info("fit_rnn_keras start: model_id=%s", json_obj.get("model_id"))

    train_path     = os.path.join(root_dir, json_obj["train_path"])
    target_col     = json_obj["target_col"]
    feature_cols   = json_obj.get("feature_cols")
    sequence_length = int(json_obj.get("sequence_length", 10))
    hidden_units   = int(json_obj.get("hidden_units", 64))
    epochs         = int(json_obj.get("epochs", 30))
    batch_size     = int(json_obj.get("batch_size", 128))
    model_id       = json_obj.get("model_id", "rnn_model")
    rnn_type       = json_obj.get("rnn_type", "lstm").lower()

    df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    X_raw = df[feature_cols].values.astype(np.float32 if numpy_use_32bit_float_precision else np.float64)
    y_raw = df[target_col].values

    # 시퀀스 생성
    X_seq, y_seq = [], []
    for i in range(len(X_raw) - sequence_length):
        X_seq.append(X_raw[i: i + sequence_length])
        y_seq.append(y_raw[i + sequence_length])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    n_classes = len(np.unique(y_seq))
    class_weight = compute_balanced_class_weight(y_seq)

    from tensorflow import keras

    rnn_layer = keras.layers.LSTM if rnn_type == "lstm" else keras.layers.SimpleRNN
    model = keras.Sequential([
        rnn_layer(hidden_units, input_shape=(sequence_length, len(feature_cols)), return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid") if n_classes == 2
        else keras.layers.Dense(n_classes, activation="softmax"),
    ])

    loss = "binary_crossentropy" if n_classes == 2 else "sparse_categorical_crossentropy"
    model.compile(optimizer="adam", loss=loss, metrics=["AUC" if n_classes == 2 else "accuracy"])

    history = model.fit(
        X_seq, y_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        class_weight=class_weight,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0,
    )

    model_save_path = os.path.join(root_dir, f"models/{model_id}_rnn_keras")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)

    hist_log = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    meta = {
        "model_id": model_id,
        "model_type": rnn_type,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "sequence_length": sequence_length,
        "hidden_units": hidden_units,
        "epochs_ran": len(hist_log.get("loss", [])),
        "model_path": model_save_path,
        "history": hist_log,
    }
    meta_path = os.path.join(root_dir, f"models/{model_id}_rnn_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if result_file_path_faf:
        with open(result_file_path_faf, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    if done_file_path_faf:
        open(done_file_path_faf, "w").close()

    logger.info("fit_rnn_keras done: model_id=%s  rnn_type=%s", model_id, rnn_type)
    return {"result": "ok", "meta": meta}


def dnn_param_search_gaussian(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision, result_file_path_faf, done_file_path_faf,
    root_dir, json_obj
) -> dict:
    """Gaussian Process(optuna)로 DNN 하이퍼파라미터 탐색."""
    import json
    import os
    import numpy as np
    import pandas as pd

    logger.info("dnn_param_search_gaussian start: model_id=%s", json_obj.get("model_id"))

    train_path   = os.path.join(root_dir, json_obj["train_path"])
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    n_trials     = int(json_obj.get("n_trials", 20))
    model_id     = json_obj.get("model_id", "dnn_search")
    param_bounds = json_obj.get("param_bounds", {})

    df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float32 if numpy_use_32bit_float_precision else np.float64)
    y = df[target_col].values

    import optuna
    from tensorflow import keras
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    hidden_bounds = param_bounds.get("hidden_layers", {"n_layers": [1, 3], "units": [32, 256]})
    dropout_bounds = param_bounds.get("dropout_rate", [0.1, 0.5])
    lr_bounds = param_bounds.get("learning_rate", [1e-4, 1e-2])

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", hidden_bounds["n_layers"][0], hidden_bounds["n_layers"][1])
        units = trial.suggest_int("units", hidden_bounds["units"][0], hidden_bounds["units"][1], log=True)
        dropout_rate = trial.suggest_float("dropout_rate", dropout_bounds[0], dropout_bounds[1])
        lr = trial.suggest_float("learning_rate", lr_bounds[0], lr_bounds[1], log=True)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        auc_scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            cw = compute_balanced_class_weight(y_tr)

            model = keras.Sequential()
            model.add(keras.layers.InputLayer(shape=(X.shape[1],)))
            for _ in range(n_layers):
                model.add(keras.layers.Dense(units, activation="relu"))
                model.add(keras.layers.Dropout(dropout_rate))
            model.add(keras.layers.Dense(1, activation="sigmoid"))
            model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy")
            model.fit(X_tr, y_tr, epochs=20, batch_size=256, class_weight=cw, verbose=0)
            y_prob = model.predict(X_val, verbose=0).flatten()
            auc_scores.append(roc_auc_score(y_val, y_prob))
        return float(np.mean(auc_scores))

    sampler = optuna.samplers.GPSampler() if hasattr(optuna.samplers, "GPSampler") else optuna.samplers.TPESampler()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_auc = study.best_value
    logger.info("dnn_param_search_gaussian best_auc=%.4f params=%s", best_auc, best_params)

    result = {
        "model_id": model_id,
        "best_params": best_params,
        "best_auc": float(best_auc),
        "n_trials": n_trials,
    }
    result_path = os.path.join(root_dir, f"models/{model_id}_param_search.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if result_file_path_faf:
        with open(result_file_path_faf, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    if done_file_path_faf:
        open(done_file_path_faf, "w").close()

    return {"result": "ok", "best_params": best_params, "best_auc": float(best_auc)}


def fit_lattice(
    service_db_info, file_server_host, file_server_port,
    root_dir, json_obj
) -> dict:
    """Lattice 모델 학습 (tensorflow_lattice 사용). 단조성 제약 조건(monotonicity) 지원."""
    import json
    import os
    import numpy as np
    import pandas as pd

    logger.info("fit_lattice start: model_id=%s", json_obj.get("model_id"))

    train_path    = os.path.join(root_dir, json_obj["train_path"])
    target_col    = json_obj["target_col"]
    feature_cols  = json_obj.get("feature_cols")
    model_id      = json_obj.get("model_id", "lattice_model")
    monotonicity  = json_obj.get("monotonicity", {})   # {"feature_name": "increasing"/"decreasing"/None}
    lattice_size  = int(json_obj.get("lattice_size", 2))
    epochs        = int(json_obj.get("epochs", 100))
    batch_size    = int(json_obj.get("batch_size", 256))

    df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    import tensorflow_lattice as tfl
    from tensorflow import keras

    feature_configs = []
    for col in feature_cols:
        col_vals = X[:, feature_cols.index(col)]
        mono = monotonicity.get(col)
        fc = tfl.configs.FeatureConfig(
            name=col,
            lattice_size=lattice_size,
            pwl_calibration_num_keypoints=10,
            pwl_calibration_input_keypoints=list(np.linspace(col_vals.min(), col_vals.max(), 10)),
            monotonicity=mono if mono else "none",
        )
        feature_configs.append(fc)

    model_config = tfl.configs.CalibratedLatticeConfig(
        feature_configs=feature_configs,
        output_min=0.0,
        output_max=1.0,
    )

    feature_keypoints = tfl.premade_lib.compute_feature_keypoints(
        feature_configs=feature_configs,
        features={col: X[:, i] for i, col in enumerate(feature_cols)},
    )
    tfl.premade_lib.set_feature_keypoints(
        feature_configs=feature_configs,
        feature_keypoints=feature_keypoints,
        add_missing_feature_configs=False,
    )

    model = tfl.premade.CalibratedLattice(model_config)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )

    input_dict = {col: X[:, i] for i, col in enumerate(feature_cols)}
    model.fit(input_dict, y, epochs=epochs, batch_size=batch_size, verbose=0)

    model_save_path = os.path.join(root_dir, f"models/{model_id}_lattice")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)

    meta = {
        "model_id": model_id,
        "model_type": "lattice",
        "feature_cols": feature_cols,
        "target_col": target_col,
        "monotonicity": monotonicity,
        "lattice_size": lattice_size,
        "model_path": model_save_path,
    }
    meta_path = os.path.join(root_dir, f"models/{model_id}_lattice_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("fit_lattice done: model_id=%s", model_id)
    return {"result": "ok", "meta": meta}


def boruta(
    service_db_info, file_server_host, file_server_port,
    root_dir, json_obj
) -> dict:
    """Boruta 변수 선택 (BorutaPy 사용). discrete 컬럼은 LabelEncoder로 인코딩 후 처리."""
    import json
    import os
    import numpy as np
    import pandas as pd

    logger.info("boruta start: model_id=%s", json_obj.get("model_id"))

    train_path    = os.path.join(root_dir, json_obj["train_path"])
    target_col    = json_obj["target_col"]
    feature_cols  = json_obj.get("feature_cols")
    n_estimators  = json_obj.get("n_estimators", "auto")
    max_iter      = int(json_obj.get("max_iter", 100))
    discrete_cols = json_obj.get("discrete_cols", [])

    df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy

    df_enc = df[feature_cols].copy()
    encoders = {}
    for col in discrete_cols:
        if col in df_enc.columns:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le

    # fill remaining object columns
    for col in df_enc.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    X = df_enc.values.astype(np.float64)
    y = df[target_col].values

    rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5, random_state=42)
    selector = BorutaPy(rf, n_estimators=n_estimators, max_iter=max_iter, random_state=42, verbose=0)
    selector.fit(X, y)

    selected   = [feature_cols[i] for i in range(len(feature_cols)) if selector.support_[i]]
    tentative  = [feature_cols[i] for i in range(len(feature_cols)) if selector.support_weak_[i]]
    rejected   = [feature_cols[i] for i in range(len(feature_cols))
                  if not selector.support_[i] and not selector.support_weak_[i]]

    result = {
        "selected_features":  selected,
        "tentative_features": tentative,
        "rejected_features":  rejected,
    }
    result_path = os.path.join(root_dir, f"models/{json_obj.get('model_id', 'boruta')}_boruta.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("boruta done: selected=%d  tentative=%d  rejected=%d",
                len(selected), len(tentative), len(rejected))
    return {"result": result}


def ensemble(
    service_db_info, file_server_host, file_server_port,
    root_dir, json_obj
) -> dict:
    """앙상블 모델 학습 (voting 또는 stacking)."""
    import json
    import os
    import pickle
    import numpy as np
    import pandas as pd

    logger.info("ensemble start: ensemble_type=%s", json_obj.get("ensemble_type", "voting"))

    train_path    = os.path.join(root_dir, json_obj["train_path"])
    target_col    = json_obj["target_col"]
    feature_cols  = json_obj.get("feature_cols")
    model_paths   = json_obj["model_paths"]
    ensemble_type = json_obj.get("ensemble_type", "voting")
    model_id      = json_obj.get("model_id", "ensemble_model")

    df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    base_estimators = []
    for i, mp in enumerate(model_paths):
        full_mp = os.path.join(root_dir, mp)
        with open(full_mp, "rb") as f:
            m = pickle.load(f)
        base_estimators.append((f"model_{i}", m))

    if ensemble_type == "stacking":
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        final_estimator = LogisticRegression(max_iter=500)
        ens_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=final_estimator,
            cv=5,
            passthrough=False,
            n_jobs=-1,
        )
    else:
        from sklearn.ensemble import VotingClassifier
        ens_model = VotingClassifier(estimators=base_estimators, voting="soft", n_jobs=-1)

    ens_model.fit(X, y)

    model_save_path = os.path.join(root_dir, f"models/{model_id}.pkl")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(ens_model, f)

    from sklearn.metrics import roc_auc_score
    y_prob = ens_model.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y, y_prob))

    meta = {
        "model_id": model_id,
        "model_type": f"ensemble_{ensemble_type}",
        "base_model_paths": model_paths,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "train_auc": auc,
        "model_path": model_save_path,
    }
    meta_path = os.path.join(root_dir, f"models/{model_id}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("ensemble done: model_id=%s  train_auc=%.4f", model_id, auc)
    return {"result": "ok", "meta": meta}


def predict_niceml(model_file_list: list, df: pd.DataFrame) -> pd.DataFrame:
    """NICEML 형식 모델 파일 목록으로 예측 수행. 각 파일을 pickle로 로드 후 predict_proba 앙상블 평균."""
    import pickle
    import numpy as np

    logger.debug("predict_niceml: n_models=%d  df_shape=%s", len(model_file_list), df.shape)

    prob_list = []
    for model_path in model_file_list:
        with open(model_path, "rb") as f:
            model_bundle = pickle.load(f)

        # 모델 번들이 dict 형태인 경우 (model + meta)
        if isinstance(model_bundle, dict):
            model = model_bundle.get("model", model_bundle)
            meta  = model_bundle.get("meta", {})
            feature_cols = meta.get("feature_cols")
        else:
            model = model_bundle
            feature_cols = None

        X = df[feature_cols] if feature_cols else df
        proba = model.predict_proba(X)[:, 1]
        prob_list.append(proba)

    avg_prob = np.mean(prob_list, axis=0)
    result_df = df.copy()
    result_df["score"] = avg_prob
    logger.debug("predict_niceml done: output_shape=%s", result_df.shape)
    return result_df


def niceml(
    service_db_info, file_server_host, file_server_port,
    root_dir, json_obj
) -> dict:
    """NICEML 모델 학습. model_meta에 standardize, missing_imputation_dict 포함."""
    import json
    import os
    import pickle
    import numpy as np
    import pandas as pd

    logger.info("niceml start: model_id=%s", json_obj.get("model_id"))

    train_path    = os.path.join(root_dir, json_obj["train_path"])
    target_col    = json_obj["target_col"]
    feature_cols  = json_obj.get("feature_cols")
    model_params  = json_obj.get("model_params", {})
    model_id      = json_obj.get("model_id", "niceml_model")
    standardize   = json_obj.get("standardize", True)
    imputation_dict = json_obj.get("missing_imputation_dict", {})

    df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    # 결측치 대체
    for col, fill_val in imputation_dict.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_val)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    X = df[feature_cols].values
    y = df[target_col].values

    scaler = None
    if standardize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    from sklearn.linear_model import LogisticRegression
    model_type = model_params.pop("model_type", "logistic_regression")
    model = _build_model(model_type, model_params)
    model.fit(X, y)

    from sklearn.metrics import roc_auc_score
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    auc = float(roc_auc_score(y, y_prob))

    model_bundle = {
        "model": model,
        "scaler": scaler,
        "meta": {
            "model_id": model_id,
            "model_type": model_type,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "standardize": standardize,
            "missing_imputation_dict": imputation_dict,
            "train_auc": auc,
        },
    }

    model_save_path = os.path.join(root_dir, f"models/{model_id}.pkl")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(model_bundle, f)

    meta = model_bundle["meta"]
    meta["model_path"] = model_save_path
    meta_path = os.path.join(root_dir, f"models/{model_id}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("niceml done: model_id=%s  train_auc=%.4f", model_id, auc)
    return {"result": "ok", "meta": meta}


def _dtype_count_table(
    df_dev: pd.DataFrame,
    df_val: pd.DataFrame,
    target: str,
    good_val,
    bad_val,
    subindex_var: Optional[str] = None,
) -> pd.DataFrame:
    """tree info용 dtype별 count table 생성."""
    rows = []
    for col in df_dev.columns:
        if col == target:
            continue
        dtype_str = str(df_dev[col].dtype)
        dev_total = len(df_dev)
        val_total = len(df_val)
        dev_miss  = int(df_dev[col].isna().sum())
        val_miss  = int(df_val[col].isna().sum())

        row = {
            "variable": col,
            "dtype": dtype_str,
            "dev_total": dev_total,
            "dev_missing": dev_miss,
            "dev_missing_rate": round(dev_miss / dev_total, 4) if dev_total else 0,
            "val_total": val_total,
            "val_missing": val_miss,
            "val_missing_rate": round(val_miss / val_total, 4) if val_total else 0,
        }

        if subindex_var and subindex_var in df_dev.columns:
            row["subindex_var"] = subindex_var
            row["dev_subindex_nunique"] = int(df_dev[subindex_var].nunique())

        # bad/good count per variable
        for split_name, df_split in [("dev", df_dev), ("val", df_val)]:
            bad_mask  = df_split[target] == bad_val
            good_mask = df_split[target] == good_val
            row[f"{split_name}_bad_count"]  = int(bad_mask.sum())
            row[f"{split_name}_good_count"] = int(good_mask.sum())

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_statistics(
    service_db_info, file_server_host, file_server_port,
    result_file_path_faf, done_file_path_faf,
    root_dir, numpy_use_32bit_float_precision, json_obj
) -> dict:
    """모델 fitting 후 통계 계산 (성능지표, 분포, 변수중요도)."""
    import json
    import os
    import pickle
    import numpy as np
    import pandas as pd

    logger.info("calculate_statistics start: model_id=%s", json_obj.get("model_id"))

    model_path   = os.path.join(root_dir, json_obj["model_path"])
    data_path    = os.path.join(root_dir, json_obj["data_path"])
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    model_id     = json_obj.get("model_id", "model")

    df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)

    if isinstance(model_bundle, dict):
        model = model_bundle.get("model", model_bundle)
        scaler = model_bundle.get("scaler")
    else:
        model = model_bundle
        scaler = None

    X = df[feature_cols].values.astype(np.float32 if numpy_use_32bit_float_precision else np.float64)
    if scaler is not None:
        X = scaler.transform(X)
    y = df[target_col].values

    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = float(roc_auc_score(y, y_prob))
    f1  = float(f1_score(y, y_pred, zero_division=0))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec  = float(recall_score(y, y_pred, zero_division=0))

    # KS 통계량
    from scipy.stats import ks_2samp
    ks_stat, _ = ks_2samp(y_prob[y == 1], y_prob[y == 0])

    # 변수중요도
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        feature_importance = dict(zip(feature_cols, [float(v) for v in fi]))

    # 점수 분포 (decile)
    df_score = pd.DataFrame({"score": y_prob, "target": y})
    df_score["decile"] = pd.qcut(df_score["score"], q=10, labels=False, duplicates="drop")
    decile_stats = df_score.groupby("decile").agg(
        count=("score", "count"),
        bad_count=("target", "sum"),
        avg_score=("score", "mean"),
    ).reset_index().to_dict(orient="records")

    stats = {
        "model_id": model_id,
        "auc": auc,
        "ks": float(ks_stat),
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "feature_importance": feature_importance,
        "decile_stats": decile_stats,
        "n_samples": len(df),
    }

    if result_file_path_faf:
        with open(result_file_path_faf, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    if done_file_path_faf:
        open(done_file_path_faf, "w").close()

    logger.info("calculate_statistics done: model_id=%s  auc=%.4f  ks=%.4f", model_id, auc, ks_stat)
    return {"result": stats}


def auto_split(
    service_db_info, file_server_host, file_server_port,
    root_dir, json_obj
) -> dict:
    """자동 데이터 분할 (train/valid/test). stratify 옵션 지원."""
    import json
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split

    logger.info("auto_split start: data_path=%s", json_obj.get("data_path"))

    data_path   = os.path.join(root_dir, json_obj["data_path"])
    target_col  = json_obj.get("target_col")
    valid_ratio = float(json_obj.get("valid_ratio", 0.15))
    test_ratio  = float(json_obj.get("test_ratio", 0.15))
    stratify    = json_obj.get("stratify", True)
    random_state = int(json_obj.get("random_state", 42))
    output_prefix = json_obj.get("output_prefix", "split")

    df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)

    strat_col = df[target_col] if stratify and target_col else None
    test_size_from_total = test_ratio
    valid_size_from_train = valid_ratio / (1.0 - test_ratio)

    df_train_val, df_test = train_test_split(
        df, test_size=test_size_from_total, random_state=random_state,
        stratify=strat_col,
    )
    strat_col_tv = df_train_val[target_col] if stratify and target_col else None
    df_train, df_valid = train_test_split(
        df_train_val, test_size=valid_size_from_train, random_state=random_state,
        stratify=strat_col_tv,
    )

    out_dir = os.path.join(root_dir, "splits")
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, f"{output_prefix}_train.parquet")
    valid_path = os.path.join(out_dir, f"{output_prefix}_valid.parquet")
    test_path  = os.path.join(out_dir, f"{output_prefix}_test.parquet")

    df_train.to_parquet(train_path, index=False)
    df_valid.to_parquet(valid_path, index=False)
    df_test.to_parquet(test_path, index=False)

    result = {
        "train_path": train_path,
        "valid_path": valid_path,
        "test_path":  test_path,
        "train_rows": len(df_train),
        "valid_rows": len(df_valid),
        "test_rows":  len(df_test),
    }
    logger.info("auto_split done: train=%d  valid=%d  test=%d",
                len(df_train), len(df_valid), len(df_test))
    return {"result": result}


def n_dtype(
    service_db_info, file_server_host, file_server_port,
    root_dir, json_obj
) -> dict:
    """변수 dtype 변환 (numeric/discrete 구분)."""
    import json
    import os
    import pandas as pd

    logger.info("n_dtype start: data_path=%s", json_obj.get("data_path"))

    data_path     = os.path.join(root_dir, json_obj["data_path"])
    numeric_cols  = json_obj.get("numeric_cols", [])
    discrete_cols = json_obj.get("discrete_cols", [])
    output_path   = json_obj.get("output_path", json_obj["data_path"])
    full_output   = os.path.join(root_dir, output_path)

    df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in discrete_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df.to_parquet(full_output, index=False)

    dtype_map = {col: str(df[col].dtype) for col in df.columns}
    logger.info("n_dtype done: numeric=%d  discrete=%d  output=%s",
                len(numeric_cols), len(discrete_cols), full_output)
    return {"result": "ok", "dtype_map": dtype_map, "output_path": full_output}


def predict_dtype(
    service_db_info, file_server_host, file_server_port,
    root_dir, json_obj
) -> dict:
    """예측 결과 dtype 처리."""
    import os
    import pandas as pd

    logger.info("predict_dtype start: result_path=%s", json_obj.get("result_path"))

    result_path  = os.path.join(root_dir, json_obj["result_path"])
    score_col    = json_obj.get("score_col", "score")
    round_digits = int(json_obj.get("round_digits", 6))
    cast_to_str_cols = json_obj.get("cast_to_str_cols", [])
    output_path  = json_obj.get("output_path", json_obj["result_path"])
    full_output  = os.path.join(root_dir, output_path)

    df = pd.read_parquet(result_path) if result_path.endswith(".parquet") else pd.read_csv(result_path)

    if score_col in df.columns:
        df[score_col] = df[score_col].astype(float).round(round_digits)

    for col in cast_to_str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df.to_parquet(full_output, index=False)

    logger.info("predict_dtype done: output=%s  shape=%s", full_output, df.shape)
    return {"result": "ok", "output_path": full_output, "shape": list(df.shape)}


def fit_lightgbm(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision, result_file_path_faf, done_file_path_faf,
    root_dir, json_obj
) -> dict:
    """LightGBM 학습 (early stopping, cv 지원)."""
    import json
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import lightgbm as lgb

    logger.info("fit_lightgbm start: model_id=%s", json_obj.get("model_id"))

    train_path   = os.path.join(root_dir, json_obj["train_path"])
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    model_params = json_obj.get("model_params", {})
    model_id     = json_obj.get("model_id", "lgb_model")
    valid_path   = json_obj.get("valid_path")
    use_cv       = json_obj.get("use_cv", False)
    num_boost_round = int(model_params.pop("num_boost_round", 1000))
    early_stopping  = int(model_params.pop("early_stopping_rounds", 50))

    df_train = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    if feature_cols is None:
        feature_cols = [c for c in df_train.columns if c != target_col]

    float_dtype = np.float32 if numpy_use_32bit_float_precision else np.float64
    X_train = df_train[feature_cols].values.astype(float_dtype)
    y_train = df_train[target_col].values

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=False)

    default_params = {
        "objective": "binary",
        "metric": "auc",
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }
    default_params.update(model_params)

    cv_hist_log = []
    if use_cv:
        cv_result = lgb.cv(
            default_params, dtrain,
            num_boost_round=num_boost_round,
            nfold=5,
            stratified=True,
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(period=0)],
        )
        best_round = len(cv_result.get("valid auc-mean", cv_result.get(list(cv_result.keys())[0])))
        cv_hist_log = get_cv_hist_log(cv_result)
        logger.info("fit_lightgbm cv best_round=%d", best_round)
    else:
        best_round = num_boost_round

    callbacks = [lgb.log_evaluation(period=0)]
    valid_sets = [dtrain]
    valid_names = ["train"]

    if valid_path:
        df_valid = pd.read_parquet(os.path.join(root_dir, valid_path)) if valid_path.endswith(".parquet") \
            else pd.read_csv(os.path.join(root_dir, valid_path))
        X_valid = df_valid[feature_cols].values.astype(float_dtype)
        y_valid = df_valid[target_col].values
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
        valid_sets.append(dvalid)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(early_stopping))

    model = lgb.train(
        default_params, dtrain,
        num_boost_round=best_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    model_save_path = os.path.join(root_dir, f"models/{model_id}.pkl")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)

    fi = dict(zip(feature_cols, [float(v) for v in model.feature_importance(importance_type="gain")]))
    meta = {
        "model_id": model_id,
        "model_type": "lightgbm",
        "feature_cols": feature_cols,
        "target_col": target_col,
        "model_params": default_params,
        "best_iteration": model.best_iteration,
        "feature_importance": fi,
        "cv_hist_log": cv_hist_log,
        "model_path": model_save_path,
    }
    meta_path = os.path.join(root_dir, f"models/{model_id}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if result_file_path_faf:
        with open(result_file_path_faf, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    if done_file_path_faf:
        open(done_file_path_faf, "w").close()

    logger.info("fit_lightgbm done: model_id=%s  best_iter=%d", model_id, model.best_iteration)
    return {"result": "ok", "meta": meta}


def fit_catboost(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision, result_file_path_faf, done_file_path_faf,
    root_dir, json_obj
) -> dict:
    """CatBoost 학습."""
    import json
    import os
    import pickle
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier, Pool

    logger.info("fit_catboost start: model_id=%s", json_obj.get("model_id"))

    train_path    = os.path.join(root_dir, json_obj["train_path"])
    target_col    = json_obj["target_col"]
    feature_cols  = json_obj.get("feature_cols")
    model_params  = json_obj.get("model_params", {})
    model_id      = json_obj.get("model_id", "catboost_model")
    valid_path    = json_obj.get("valid_path")
    cat_features  = json_obj.get("cat_features", [])

    df_train = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    if feature_cols is None:
        feature_cols = [c for c in df_train.columns if c != target_col]

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    cat_feature_indices = [feature_cols.index(c) for c in cat_features if c in feature_cols]

    default_params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 50,
        "auto_class_weights": "Balanced",
    }
    default_params.update(model_params)

    train_pool = Pool(X_train, label=y_train, cat_features=cat_feature_indices, feature_names=feature_cols)

    eval_pool = None
    if valid_path:
        df_valid = pd.read_parquet(os.path.join(root_dir, valid_path)) if valid_path.endswith(".parquet") \
            else pd.read_csv(os.path.join(root_dir, valid_path))
        eval_pool = Pool(df_valid[feature_cols], label=df_valid[target_col],
                         cat_features=cat_feature_indices, feature_names=feature_cols)

    model = CatBoostClassifier(**default_params)
    model.fit(train_pool, eval_set=eval_pool)

    model_save_path = os.path.join(root_dir, f"models/{model_id}.pkl")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)

    fi = dict(zip(feature_cols, [float(v) for v in model.get_feature_importance()]))
    meta = {
        "model_id": model_id,
        "model_type": "catboost",
        "feature_cols": feature_cols,
        "target_col": target_col,
        "model_params": default_params,
        "best_iteration": model.best_iteration_,
        "feature_importance": fi,
        "model_path": model_save_path,
    }
    meta_path = os.path.join(root_dir, f"models/{model_id}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if result_file_path_faf:
        with open(result_file_path_faf, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    if done_file_path_faf:
        open(done_file_path_faf, "w").close()

    logger.info("fit_catboost done: model_id=%s  best_iter=%d", model_id, model.best_iteration_)
    return {"result": "ok", "meta": meta}


def fit_xgboost(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision, result_file_path_faf, done_file_path_faf,
    root_dir, json_obj
) -> dict:
    """XGBoost 학습."""
    import json
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    logger.info("fit_xgboost start: model_id=%s", json_obj.get("model_id"))

    train_path   = os.path.join(root_dir, json_obj["train_path"])
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    model_params = json_obj.get("model_params", {})
    model_id     = json_obj.get("model_id", "xgb_model")
    valid_path   = json_obj.get("valid_path")
    num_boost_round = int(model_params.pop("num_boost_round", 1000))
    early_stopping  = int(model_params.pop("early_stopping_rounds", 50))

    df_train = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    if feature_cols is None:
        feature_cols = [c for c in df_train.columns if c != target_col]

    float_dtype = np.float32 if numpy_use_32bit_float_precision else np.float64
    X_train = df_train[feature_cols].values.astype(float_dtype)
    y_train = df_train[target_col].values

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "seed": 42,
        "nthread": -1,
    }
    default_params.update(model_params)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    evals = [(dtrain, "train")]
    callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping, save_best=True)]

    if valid_path:
        df_valid = pd.read_parquet(os.path.join(root_dir, valid_path)) if valid_path.endswith(".parquet") \
            else pd.read_csv(os.path.join(root_dir, valid_path))
        X_valid = df_valid[feature_cols].values.astype(float_dtype)
        y_valid = df_valid[target_col].values
        dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_cols)
        evals.append((dvalid, "valid"))

    evals_result = {}
    model = xgb.train(
        default_params, dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        evals_result=evals_result,
        callbacks=callbacks,
        verbose_eval=False,
    )

    model_save_path = os.path.join(root_dir, f"models/{model_id}.pkl")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)

    fi_scores = model.get_score(importance_type="gain")
    fi = {col: float(fi_scores.get(col, 0.0)) for col in feature_cols}
    meta = {
        "model_id": model_id,
        "model_type": "xgboost",
        "feature_cols": feature_cols,
        "target_col": target_col,
        "model_params": default_params,
        "best_iteration": model.best_iteration,
        "feature_importance": fi,
        "model_path": model_save_path,
    }
    meta_path = os.path.join(root_dir, f"models/{model_id}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if result_file_path_faf:
        with open(result_file_path_faf, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    if done_file_path_faf:
        open(done_file_path_faf, "w").close()

    logger.info("fit_xgboost done: model_id=%s  best_iter=%d", model_id, model.best_iteration)
    return {"result": "ok", "meta": meta}


def predict_grade(
    service_db_info, file_server_host, file_server_port,
    root_dir, json_obj
) -> dict:
    """예측 점수를 등급으로 변환. grade_mapping 또는 grade_bins 사용."""
    import json
    import os
    import pandas as pd
    import numpy as np

    logger.info("predict_grade start: result_path=%s", json_obj.get("result_path"))

    result_path  = os.path.join(root_dir, json_obj["result_path"])
    score_col    = json_obj.get("score_col", "score")
    grade_col    = json_obj.get("grade_col", "grade")
    grade_mapping = json_obj.get("grade_mapping")   # {"A": [0.8, 1.0], "B": [0.6, 0.8], ...}
    grade_bins   = json_obj.get("grade_bins")        # [0.0, 0.3, 0.6, 0.8, 1.0]
    grade_labels = json_obj.get("grade_labels")      # ["D","C","B","A"]
    output_path  = json_obj.get("output_path", json_obj["result_path"])
    full_output  = os.path.join(root_dir, output_path)

    df = pd.read_parquet(result_path) if result_path.endswith(".parquet") else pd.read_csv(result_path)

    if grade_mapping:
        def _map_grade(score):
            for grade, (lo, hi) in grade_mapping.items():
                if lo <= score <= hi:
                    return grade
            return None
        df[grade_col] = df[score_col].apply(_map_grade)
    elif grade_bins:
        labels = grade_labels if grade_labels else list(range(len(grade_bins) - 1))
        df[grade_col] = pd.cut(df[score_col], bins=grade_bins, labels=labels, include_lowest=True)
    else:
        # 기본: 4분위 등급
        df[grade_col] = pd.qcut(df[score_col], q=4, labels=["D", "C", "B", "A"])

    df.to_parquet(full_output, index=False) if full_output.endswith(".parquet") else df.to_csv(full_output, index=False)

    grade_dist = df[grade_col].value_counts().to_dict()
    logger.info("predict_grade done: output=%s  grade_dist=%s", full_output, grade_dist)
    return {"result": "ok", "output_path": full_output, "grade_distribution": {str(k): int(v) for k, v in grade_dist.items()}}


def _count_table_for_grading(
    df_score_dist: pd.DataFrame,
    score_col: str = "score",
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """등급화를 위한 카운트 테이블 생성."""
    import numpy as np

    df = df_score_dist.copy()
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["cum_count"] = df["rank"]

    if target_col and target_col in df.columns:
        total_bad = df[target_col].sum()
        total_good = len(df) - total_bad
        df["cum_bad"]  = df[target_col].cumsum()
        df["cum_good"] = df["cum_count"] - df["cum_bad"]
        df["cum_bad_rate"]  = df["cum_bad"] / total_bad if total_bad > 0 else 0.0
        df["cum_good_rate"] = df["cum_good"] / total_good if total_good > 0 else 0.0
        df["ks"] = (df["cum_bad_rate"] - df["cum_good_rate"]).abs()

    return df


def auto_grade_user_define(
    df_score_dist: pd.DataFrame,
    df_grade_info: pd.DataFrame,
    score_ascending: bool = False,
) -> pd.DataFrame:
    """사용자 정의 등급 기준으로 자동 등급 부여."""
    import numpy as np

    # df_grade_info 컬럼: grade, min_score, max_score
    df = df_score_dist.copy()
    df["grade"] = None

    for _, row in df_grade_info.iterrows():
        grade     = row["grade"]
        min_score = float(row["min_score"])
        max_score = float(row["max_score"])
        mask = (df["score"] >= min_score) & (df["score"] <= max_score)
        df.loc[mask, "grade"] = grade

    return df


def auto_grade_badrate(
    df_score_dist: pd.DataFrame,
    df_grade_info: pd.DataFrame,
    score_ascending: bool = False,
) -> pd.DataFrame:
    """bad rate 기준 자동 등급 부여."""
    import numpy as np

    # df_grade_info 컬럼: grade, min_badrate, max_badrate
    df = df_score_dist.copy()
    df = df.sort_values("score", ascending=score_ascending).reset_index(drop=True)

    if "target" not in df.columns:
        raise ValueError("df_score_dist에 'target' 컬럼이 필요합니다.")

    # 분위 기반으로 bad rate 계산
    n_bins = len(df_grade_info)
    df["bin"] = pd.qcut(df["score"], q=n_bins, labels=False, duplicates="drop")
    bin_stats = df.groupby("bin").agg(
        bad_rate=("target", "mean"),
        count=("target", "count"),
    ).reset_index()

    df["grade"] = None
    for _, grade_row in df_grade_info.iterrows():
        grade = grade_row["grade"]
        min_br = float(grade_row["min_badrate"])
        max_br = float(grade_row["max_badrate"])
        matching_bins = bin_stats[
            (bin_stats["bad_rate"] >= min_br) & (bin_stats["bad_rate"] <= max_br)
        ]["bin"].tolist()
        df.loc[df["bin"].isin(matching_bins), "grade"] = grade

    return df


def upload_features_cf(
    service_db_info, root_dir, result_file_path_faf, done_file_path_faf,
    meta_json, raw_features_file_path, file_server_host, file_server_port,
    external_purpose
) -> dict:
    """CF(Collaborative Filtering) 피처 업로드."""
    import json
    import os
    import shutil
    import pandas as pd

    logger.info("upload_features_cf start: raw_features_file_path=%s", raw_features_file_path)

    full_raw_path = os.path.join(root_dir, raw_features_file_path)
    df = pd.read_parquet(full_raw_path) if full_raw_path.endswith(".parquet") else pd.read_csv(full_raw_path)

    cf_output_dir = os.path.join(root_dir, "cf_features")
    os.makedirs(cf_output_dir, exist_ok=True)

    model_id = meta_json.get("model_id", "cf")
    feature_path = os.path.join(cf_output_dir, f"{model_id}_features.parquet")
    df.to_parquet(feature_path, index=False)

    meta = {
        "model_id": model_id,
        "external_purpose": external_purpose,
        "feature_path": feature_path,
        "n_rows": len(df),
        "columns": list(df.columns),
        "meta_json": meta_json,
    }
    meta_path = os.path.join(cf_output_dir, f"{model_id}_cf_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if result_file_path_faf:
        with open(result_file_path_faf, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    if done_file_path_faf:
        open(done_file_path_faf, "w").close()

    logger.info("upload_features_cf done: feature_path=%s  n_rows=%d", feature_path, len(df))
    return {"result": "ok", "meta": meta}


def fit_cf(
    service_db_info, file_server_host, file_server_port,
    root_dir, json_obj
) -> dict:
    """Collaborative Filtering 모델 학습 (implicit 사용)."""
    import json
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    logger.info("fit_cf start: model_id=%s", json_obj.get("model_id"))

    train_path  = os.path.join(root_dir, json_obj["train_path"])
    user_col    = json_obj.get("user_col", "user_id")
    item_col    = json_obj.get("item_col", "item_id")
    rating_col  = json_obj.get("rating_col", "rating")
    model_id    = json_obj.get("model_id", "cf_model")
    cf_type     = json_obj.get("cf_type", "als")
    factors     = int(json_obj.get("factors", 50))
    iterations  = int(json_obj.get("iterations", 20))
    regularization = float(json_obj.get("regularization", 0.01))

    df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)

    # user/item 인덱싱
    users = df[user_col].astype("category")
    items = df[item_col].astype("category")
    df["user_idx"] = users.cat.codes
    df["item_idx"] = items.cat.codes
    user_map = dict(enumerate(users.cat.categories))
    item_map = dict(enumerate(items.cat.categories))

    n_users = df["user_idx"].max() + 1
    n_items = df["item_idx"].max() + 1
    ratings = df[rating_col].values.astype(np.float32) if rating_col in df.columns else np.ones(len(df), dtype=np.float32)

    user_item_matrix = sp.csr_matrix(
        (ratings, (df["user_idx"].values, df["item_idx"].values)),
        shape=(n_users, n_items),
    )

    import implicit
    if cf_type == "bpr":
        model = implicit.bpr.BayesianPersonalizedRanking(factors=factors, iterations=iterations, regularization=regularization)
    else:
        model = implicit.als.AlternatingLeastSquares(factors=factors, iterations=iterations, regularization=regularization)

    model.fit(user_item_matrix)

    model_save_path = os.path.join(root_dir, f"models/{model_id}_cf.pkl")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump({"model": model, "user_map": user_map, "item_map": item_map,
                     "user_col": user_col, "item_col": item_col}, f)

    meta = {
        "model_id": model_id,
        "model_type": f"cf_{cf_type}",
        "user_col": user_col,
        "item_col": item_col,
        "n_users": n_users,
        "n_items": n_items,
        "factors": factors,
        "model_path": model_save_path,
    }
    meta_path = os.path.join(root_dir, f"models/{model_id}_cf_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("fit_cf done: model_id=%s  n_users=%d  n_items=%d", model_id, n_users, n_items)
    return {"result": "ok", "meta": meta}


def process_multitarget(
    service_db_info, file_server_host, file_server_port,
    root_dir, json_obj
) -> dict:
    """멀티타깃 처리: 여러 target_col에 대해 각각 학습."""
    import json
    import os

    logger.info("process_multitarget start: model_id=%s", json_obj.get("model_id"))

    multitarget_properties = json_obj.get("multitarget_properties", [])
    base_json = {k: v for k, v in json_obj.items() if k != "multitarget_properties"}

    all_results = []
    for mt_prop in multitarget_properties:
        child_json = {**base_json, **mt_prop}
        target_col = mt_prop.get("target_col", json_obj.get("target_col"))
        logger.info("process_multitarget fitting target_col=%s", target_col)

        model_type = child_json.get("model_type", "lightgbm")
        result = fit_lightgbm(
            service_db_info, file_server_host, file_server_port,
            child_json.get("numpy_use_32bit_float_precision", False),
            None, None, root_dir, child_json,
        )
        result["target_col"] = target_col
        all_results.append(result)

    final_result = adjust_multitarget_result_json({"results": all_results}, multitarget_properties)
    logger.info("process_multitarget done: n_targets=%d", len(multitarget_properties))
    return {"result": final_result}


def get_multitarget_df(df: pd.DataFrame, multitarget_properties: list) -> list:
    """멀티타깃 데이터프레임 목록 반환."""
    result = []
    for prop in multitarget_properties:
        target_col   = prop.get("target_col")
        feature_cols = prop.get("feature_cols")
        filter_cond  = prop.get("filter")

        df_sub = df.copy()
        if filter_cond:
            df_sub = df_sub.query(filter_cond)
        if feature_cols:
            keep_cols = feature_cols + ([target_col] if target_col else [])
            df_sub = df_sub[[c for c in keep_cols if c in df_sub.columns]]

        result.append({"df": df_sub, "target_col": target_col, "prop": prop})
    return result


def adjust_multitarget_result_json(result_json: dict, multitarget_properties: list) -> dict:
    """멀티타깃 결과 JSON 조정."""
    adjusted = result_json.copy()
    results  = adjusted.get("results", [])

    target_keys = [p.get("target_col", f"target_{i}") for i, p in enumerate(multitarget_properties)]
    adjusted["target_cols"] = target_keys
    adjusted["n_targets"]   = len(target_keys)

    target_summary = {}
    for res, key in zip(results, target_keys):
        meta = res.get("meta", {})
        target_summary[key] = {
            "model_id": meta.get("model_id"),
            "model_path": meta.get("model_path"),
            "auc": meta.get("metrics", {}).get("auc"),
        }
    adjusted["target_summary"] = target_summary
    return adjusted


def save_fitting_result(
    service_db_info, file_server_host, file_server_port,
    root_dir, result_file_path_faf, done_file_path_faf,
    json_obj, model, meta_json
) -> dict:
    """모델 학습 결과 저장 (pickle + meta JSON)."""
    import json
    import os
    import pickle

    model_id = meta_json.get("model_id", json_obj.get("model_id", "model"))
    logger.info("save_fitting_result: model_id=%s", model_id)

    model_save_path = os.path.join(root_dir, f"models/{model_id}.pkl")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)

    meta_json["model_path"] = model_save_path
    meta_path = os.path.join(root_dir, f"models/{model_id}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_json, f, ensure_ascii=False, indent=2)

    if result_file_path_faf:
        with open(result_file_path_faf, "w", encoding="utf-8") as f:
            json.dump(meta_json, f, ensure_ascii=False, indent=2)
    if done_file_path_faf:
        open(done_file_path_faf, "w").close()

    logger.info("save_fitting_result done: model_path=%s", model_save_path)
    return {"result": "ok", "model_path": model_save_path, "meta_path": meta_path}


def get_cv_hist_log(cv_results: dict, metric: str = "auc") -> list:
    """cv logging function (python nodes 공통). cross-validation 히스토리 로그 반환."""
    # cv_results: LightGBM cv 결과 혹은 {metric: [values]} 형태
    hist = []
    # 다양한 키 형태 탐색 (e.g. "valid auc-mean", "auc-mean", metric)
    mean_key = None
    std_key  = None
    for key in cv_results:
        key_lower = key.lower().replace(" ", "_")
        if metric.lower() in key_lower and "mean" in key_lower:
            mean_key = key
        if metric.lower() in key_lower and "stdv" in key_lower:
            std_key = key

    if mean_key is None:
        # fallback: 첫 번째 키 사용
        mean_key = list(cv_results.keys())[0]

    mean_vals = cv_results[mean_key]
    std_vals  = cv_results.get(std_key, [0.0] * len(mean_vals))

    for i, (mean_v, std_v) in enumerate(zip(mean_vals, std_vals)):
        hist.append({
            "iteration": i + 1,
            f"{metric}_mean": round(float(mean_v), 6),
            f"{metric}_std":  round(float(std_v), 6),
        })
    return hist


def gridsearch_py(
    service_db_info, file_server_host, file_server_port,
    root_dir, json_obj
) -> dict:
    """Python 노드용 GridSearch (XGBoost, LightGBM 포함). sklearn GridSearchCV 또는 직접 구현."""
    import json
    import os
    import pickle
    import numpy as np
    import pandas as pd

    logger.info("gridsearch_py start: model_id=%s", json_obj.get("model_id"))

    train_path   = os.path.join(root_dir, json_obj["train_path"])
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    model_type   = json_obj.get("model_type", "lightgbm")
    param_grid   = json_obj.get("param_grid", {})
    cv_folds     = int(json_obj.get("cv_folds", 5))
    scoring      = json_obj.get("scoring", "roc_auc")
    model_id     = json_obj.get("model_id", "gridsearch_model")

    df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    base_model = _build_model(model_type, {})

    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_folds,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X, y)

    best_model = gs.best_estimator_
    model_save_path = os.path.join(root_dir, f"models/{model_id}_gs.pkl")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(best_model, f)

    result = {
        "model_id": model_id,
        "model_type": model_type,
        "best_params": gs.best_params_,
        "best_score": float(gs.best_score_),
        "cv_folds": cv_folds,
        "scoring": scoring,
        "model_path": model_save_path,
    }
    result_path = os.path.join(root_dir, f"models/{model_id}_gs_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("gridsearch_py done: best_score=%.4f  best_params=%s",
                result["best_score"], result["best_params"])
    return {"result": result}


def fit_anomaly_svm(
    service_db_info, file_server_host, file_server_port,
    root_dir, json_obj
) -> dict:
    """sklearn One-Class SVM 이상탐지 학습. H2O는 one-class SVM 미지원이라 sklearn 사용."""
    import json
    import os
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler

    logger.info("fit_anomaly_svm start: model_id=%s", json_obj.get("model_id"))

    train_path   = os.path.join(root_dir, json_obj["train_path"])
    feature_cols = json_obj.get("feature_cols")
    model_id     = json_obj.get("model_id", "anomaly_svm")
    kernel       = json_obj.get("kernel", "rbf")
    nu           = float(json_obj.get("nu", 0.1))
    gamma        = json_obj.get("gamma", "scale")

    df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns]

    X = df[feature_cols].values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    model.fit(X_scaled)

    y_pred = model.predict(X_scaled)
    n_anomaly = int((y_pred == -1).sum())
    anomaly_rate = n_anomaly / len(y_pred)

    model_save_path = os.path.join(root_dir, f"models/{model_id}.pkl")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols}, f)

    meta = {
        "model_id": model_id,
        "model_type": "one_class_svm",
        "feature_cols": feature_cols,
        "kernel": kernel,
        "nu": nu,
        "train_anomaly_count": n_anomaly,
        "train_anomaly_rate": round(anomaly_rate, 4),
        "model_path": model_save_path,
    }
    meta_path = os.path.join(root_dir, f"models/{model_id}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("fit_anomaly_svm done: model_id=%s  anomaly_rate=%.4f", model_id, anomaly_rate)
    return {"result": "ok", "meta": meta}


def fit_svm(
    service_db_info, file_server_host, file_server_port,
    root_dir, json_obj
) -> dict:
    """sklearn SVM 분류 학습."""
    import json
    import os
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    logger.info("fit_svm start: model_id=%s", json_obj.get("model_id"))

    train_path   = os.path.join(root_dir, json_obj["train_path"])
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    model_id     = json_obj.get("model_id", "svm_model")
    kernel       = json_obj.get("kernel", "rbf")
    C            = float(json_obj.get("C", 1.0))
    gamma        = json_obj.get("gamma", "scale")
    model_params = json_obj.get("model_params", {})

    df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm_params = {"kernel": kernel, "C": C, "gamma": gamma, "probability": True, "class_weight": "balanced"}
    svm_params.update(model_params)
    model = SVC(**svm_params)
    model.fit(X_scaled, y)

    from sklearn.metrics import roc_auc_score
    y_prob = model.predict_proba(X_scaled)[:, 1]
    auc = float(roc_auc_score(y, y_prob))

    meta = {
        "model_id": model_id,
        "model_type": "svm",
        "feature_cols": feature_cols,
        "target_col": target_col,
        "kernel": kernel,
        "C": C,
        "train_auc": auc,
        "model_path": None,
    }

    meta = save_svm_model(service_db_info, file_server_host, file_server_port, root_dir,
                          json_obj, {"model": model, "scaler": scaler, "meta": meta})
    logger.info("fit_svm done: model_id=%s  train_auc=%.4f", model_id, auc)
    return {"result": "ok", "meta": meta}


def save_svm_model(
    service_db_info, file_server_host, file_server_port,
    root_dir, json_obj, model
) -> dict:
    """SVM 모델 결과를 zip 파일로 저장 (asvm, zsvm 형식)."""
    import json
    import os
    import pickle
    import zipfile

    model_id    = json_obj.get("model_id", "svm_model")
    svm_format  = json_obj.get("svm_format", "zsvm")   # "asvm" or "zsvm"
    logger.info("save_svm_model: model_id=%s  format=%s", model_id, svm_format)

    model_dir = os.path.join(root_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    pkl_path = os.path.join(model_dir, f"{model_id}_svm.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    # 메타 추출
    meta = model.get("meta", {}) if isinstance(model, dict) else {}
    meta["model_id"]   = model_id
    meta["svm_format"] = svm_format
    meta["pkl_path"]   = pkl_path

    meta_path = os.path.join(model_dir, f"{model_id}_svm_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    zip_path = os.path.join(model_dir, f"{model_id}.{svm_format}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pkl_path, arcname=os.path.basename(pkl_path))
        zf.write(meta_path, arcname=os.path.basename(meta_path))

    meta["zip_path"] = zip_path
    logger.info("save_svm_model done: zip_path=%s", zip_path)
    return meta


def fit_score_cutoff(
    service_db_info, file_server_host, file_server_port,
    root_dir, json_obj
) -> dict:
    """점수 컷오프 최적화 (F1, KS 기준 최적 threshold 탐색)."""
    import json
    import os
    import numpy as np
    import pandas as pd

    logger.info("fit_score_cutoff start: model_id=%s", json_obj.get("model_id"))

    score_path  = os.path.join(root_dir, json_obj["score_path"])
    score_col   = json_obj.get("score_col", "score")
    target_col  = json_obj.get("target_col", "target")
    criterion   = json_obj.get("criterion", "ks")   # "ks" or "f1"
    n_thresholds = int(json_obj.get("n_thresholds", 200))
    model_id    = json_obj.get("model_id", "score_cutoff")

    df = pd.read_parquet(score_path) if score_path.endswith(".parquet") else pd.read_csv(score_path)
    y_true  = df[target_col].values
    y_score = df[score_col].values

    thresholds = np.linspace(y_score.min(), y_score.max(), n_thresholds)
    best_thresh = thresholds[0]
    best_value  = -np.inf
    threshold_log = []

    from sklearn.metrics import f1_score as sk_f1
    from scipy.stats import ks_2samp

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        if criterion == "f1":
            val = float(sk_f1(y_true, y_pred, zero_division=0))
        else:  # ks
            ks_stat, _ = ks_2samp(y_score[y_true == 1], y_score[y_true == 0])
            # 각 threshold에서 predicted positive rate와 bad rate 비교
            pred_pos = y_pred.sum()
            if pred_pos == 0 or pred_pos == len(y_pred):
                val = 0.0
            else:
                val = float(sk_f1(y_true, y_pred, zero_division=0))

        threshold_log.append({"threshold": float(thresh), criterion: round(val, 6)})
        if val > best_value:
            best_value  = val
            best_thresh = float(thresh)

    # KS 기반 최적 threshold 재계산 (전체 분포 기준)
    if criterion == "ks":
        pos_scores = y_score[y_true == 1]
        neg_scores = y_score[y_true == 0]
        ks_vals = []
        for thresh in thresholds:
            tpr = float((pos_scores >= thresh).mean())
            fpr = float((neg_scores >= thresh).mean())
            ks_vals.append(abs(tpr - fpr))
        best_idx    = int(np.argmax(ks_vals))
        best_thresh = float(thresholds[best_idx])
        best_value  = float(ks_vals[best_idx])
        threshold_log = [{"threshold": float(t), "ks": round(k, 6)} for t, k in zip(thresholds, ks_vals)]

    result = {
        "model_id": model_id,
        "criterion": criterion,
        "best_threshold": best_thresh,
        f"best_{criterion}": round(best_value, 6),
        "n_thresholds": n_thresholds,
        "threshold_log": threshold_log,
    }
    result_path = os.path.join(root_dir, f"models/{model_id}_cutoff.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("fit_score_cutoff done: criterion=%s  best_threshold=%.6f  best_%s=%.6f",
                criterion, best_thresh, criterion, best_value)
    return {"result": result}
