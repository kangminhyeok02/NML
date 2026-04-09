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


# =============================================================================
# Module-level functions
# =============================================================================


def predict_python(
    service_db_info, file_server_host, file_server_port,
    root_dir, numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf, json_obj,
) -> dict:
    """Python(pickle) 모델 예측 — standardization, missing_imputation, reverse_prob 지원."""
    import json as _json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    logger.info("predict_python: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    target_col   = json_obj.get("target_col")
    score_col    = json_obj.get("score_col", "score")
    reverse_prob = json_obj.get("reverse_prob", True)
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    # 모델 메타 로드
    meta_file = root_path / "models" / f"{model_id}_meta.json"
    with open(meta_file, encoding="utf-8") as f:
        meta = _json.load(f)

    # 모델 로드
    model_path = root_path / meta["model_path"]
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # standardization mean & stddev
    std_dict    = meta.get("standardize", {})
    # model info json (discrete values...)
    missing_imp = meta.get("missing_imputation_dict", {})
    mining_type = meta.get("mining_type", "").upper()
    feature_cols = meta.get("feature_cols", [])

    # 데이터 로드
    full_path = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)

    # missing imputation
    for col, val in missing_imp.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    X = df[[c for c in feature_cols if c in df.columns]].fillna(0)

    # prevent converting int(NUM_DISC) to float (except lightgbm, XGBOOSTPY)
    FLOAT_EXEMPT = {"LIGHTGBM", "XGBOOSTPY", "LGBM"}
    if mining_type not in FLOAT_EXEMPT:
        X = X.astype(float_dtype)

    # standardization 적용
    if std_dict:
        for col in X.columns:
            if col in std_dict:
                mean_val = std_dict[col].get("mean", 0)
                std_val  = std_dict[col].get("std", 1) or 1
                X[col]   = (X[col] - mean_val) / std_val

    # 예측
    # in case of linear model prediction, disable reverse prob
    LINEAR_TYPES = {"LINEARREGRESSIONNEW", "DEEPLEARNINGLINEAR",
                    "RANDOMFORESTLINEAR", "GRADIENTBOOSTINGLINEAR",
                    "LINEARREGRESSION", "SVMLINEAR"}
    is_linear = mining_type in LINEAR_TYPES

    if hasattr(model, "predict_proba") and not is_linear:
        proba = model.predict_proba(X)[:, 1]
        if reverse_prob and not is_linear:
            proba = 1.0 - proba
    else:
        proba = model.predict(X).astype(float)

    df[score_col] = np.round(proba, 6)

    # writes output parameter to temp file (True when ensemble)
    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_predict.parquet"
    # save result to output parameter file (if exists, just overwrite)
    df.to_parquet(output_file, index=False)

    perf = {}
    if target_col and target_col in df.columns:
        y = df[target_col]
        try:
            perf["auc"] = round(float(roc_auc_score(y, proba)), 4)
        except Exception:
            pass

    # returned as a named list
    result = {
        "result":      "ok",
        "model_id":    model_id,
        "total_rows":  len(df),
        "perf":        perf,
        "output_file": str(output_file),
        "score_stats": _series_stats(df[score_col]),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def predict_h2o_r(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_older_version_port,
    h2o_script_obj, script_obj, root_dir, json_obj,
) -> dict:
    """H2O 및 R 모델 예측 — h2o mining types 지원."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_h2o_r: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    target_col   = json_obj.get("target_col")
    score_col    = json_obj.get("score_col", "score")
    mining_type  = json_obj.get("mining_type", "").upper()
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    # h2o mining types: H2O_MINING_TYPES / DISCRETE_MINING_TYPES / LINEAR_MINING_TYPES
    H2O_MINING_TYPES = {
        "RANDOMFOREST", "DEEPLEARNING", "GRADIENTBOOSTING", "GLM",
        "LINEARREGRESSIONNEW", "DEEPLEARNINGLINEAR", "RANDOMFORESTLINEAR",
        "GRADIENTBOOSTINGLINEAR", "XGBOOST", "AUTOML", "AUTOMLGLM", "AUTOMLDL",
        "KMEANS", "ANOMALY",
    }
    R_MINING_TYPES = {
        "LOGISTICREGRESSION", "LINEARREGRESSION", "DECISIONTREE_C50",
        "DECISIONTREE_RPART", "ANNLP", "ANNMLP", "SVM", "SVMLINEAR", "ANOMALYSVM",
    }

    meta_file = root_path / "models" / f"{model_id}_meta.json"
    with open(meta_file, encoding="utf-8") as f:
        meta = _json.load(f)

    feature_cols = meta.get("feature_cols", [])
    full_path    = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    X  = df[[c for c in feature_cols if c in df.columns]].fillna(0).astype(float_dtype)

    # writes output parameter to temp file (True when ensemble)
    if mining_type in H2O_MINING_TYPES:
        import h2o
        h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
        model = h2o.import_mojo(str(root_path / meta.get("model_path", f"models/{model_id}.zip")))
        h2o_frame = h2o.H2OFrame(X)
        preds = model.predict(h2o_frame).as_data_frame()
        proba = preds.iloc[:, -1].values
    elif mining_type in R_MINING_TYPES:
        # predict h2o models → returns performance
        import subprocess, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            X.to_csv(tmp.name, index=False)
            tmp_in = tmp.name
        tmp_out = tmp_in.replace(".csv", "_pred.csv")
        r_script = meta.get("r_script_path", "r_scripts/predict.R")
        cmd = ["Rscript", str(root_path / r_script),
               "--input", tmp_in, "--output", tmp_out,
               "--model", str(root_path / meta.get("model_path", ""))]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"R 예측 실패: {proc.stderr}")
        preds = pd.read_csv(tmp_out)
        proba = preds.iloc[:, 0].values
        os.unlink(tmp_in)
        os.unlink(tmp_out)
    else:
        import pickle
        with open(root_path / meta["model_path"], "rb") as f:
            model = pickle.load(f)
        proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)

    df[score_col] = np.round(proba, 6)
    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_predict.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result": "ok", "model_id": model_id, "total_rows": len(df),
        "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def predict_pretrained_ensemble(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_older_version_port,
    root_dir, json_obj,
) -> dict:
    """앙상블(auto-encoder / reverse_prob & weight) 예측.

    ensemble perf 결과와 pretrained ensemble perf 결과를 비슷하게 display하기 위해
    python에서 다시 계산.
    """
    import json as _json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    logger.info("predict_pretrained_ensemble: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    target_col   = json_obj.get("target_col")
    score_col    = json_obj.get("score_col", "score")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    # 앙상블 메타
    meta_file = root_path / "models" / f"{model_id}_meta.json"
    with open(meta_file, encoding="utf-8") as f:
        meta = _json.load(f)

    member_ids = meta.get("member_ids", [])
    weights    = meta.get("weights", [1.0 / max(len(member_ids), 1)] * len(member_ids))
    reverse_prob = meta.get("reverse_prob", True)

    full_path = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)

    ensemble_proba = np.zeros(len(df))
    for mid, w in zip(member_ids, weights):
        m_meta_file = root_path / "models" / f"{mid}_meta.json"
        if not m_meta_file.exists():
            continue
        with open(m_meta_file, encoding="utf-8") as f:
            m_meta = _json.load(f)
        feature_cols = m_meta.get("feature_cols", [])
        X = df[[c for c in feature_cols if c in df.columns]].fillna(0).astype(float_dtype)
        m_path = root_path / m_meta.get("model_path", f"models/{mid}.pkl")
        if not m_path.exists():
            continue
        with open(m_path, "rb") as f:
            model = pickle.load(f)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            proba = model.predict(X).astype(float)
        ensemble_proba += w * proba

    if reverse_prob:
        ensemble_proba = 1.0 - ensemble_proba

    df[score_col] = np.round(ensemble_proba, 6)

    # changed to calculate_perf_and_score_dist()
    perf = {}
    if target_col and target_col in df.columns:
        y = df[target_col]
        try:
            perf["auc"] = round(float(roc_auc_score(y, ensemble_proba)), 4)
        except Exception:
            pass

    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_ensemble_predict.parquet"
    # Upload OutParam → returned as a named list
    df.to_parquet(output_file, index=False)

    result = {
        "result": "ok", "model_id": model_id,
        "total_rows": len(df), "perf": perf, "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def predict_multitarget_validation_h2o_r(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_older_version_port,
    h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O/R 모델 멀티타겟 검증 예측 — supported mining types / load model."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_multitarget_validation_h2o_r: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    multitarget_props = json_obj.get("multitarget_properties", [])

    meta_file = root_path / "models" / f"{model_id}_meta.json"
    with open(meta_file, encoding="utf-8") as f:
        meta = _json.load(f)

    feature_cols = meta.get("feature_cols", [])
    full_path    = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    X  = df[[c for c in feature_cols if c in df.columns]].fillna(0).astype(float_dtype)

    # supported mining types — H2O 연결 및 예측
    import h2o
    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    model_path_str = str(root_path / meta.get("model_path", f"models/{model_id}.zip"))
    model = h2o.import_mojo(model_path_str)
    h2o_frame = h2o.H2OFrame(X)
    preds = model.predict(h2o_frame).as_data_frame()

    for i, prop in enumerate(multitarget_props):
        col_name = prop.get("score_col", f"score_{i}")
        proba_col = min(i, preds.shape[1] - 1)
        df[col_name] = np.round(preds.iloc[:, proba_col].values, 6)

    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_multitarget_val.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result": "ok", "model_id": model_id,
        "total_rows": len(df), "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def predict_multitarget_validation_h2o_py(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_older_version_port,
    root_dir, json_obj,
) -> dict:
    """Python H2O 모델 멀티타겟 검증 — lightGBM for h2o version 포함."""
    import json as _json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_multitarget_validation_h2o_py: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    # multitarget properties / multitarget weights
    multitarget_props   = json_obj.get("multitarget_properties", [])
    multitarget_weights = json_obj.get("multitarget_weights", [1.0] * len(multitarget_props))

    meta_file = root_path / "models" / f"{model_id}_meta.json"
    with open(meta_file, encoding="utf-8") as f:
        meta = _json.load(f)

    feature_cols = meta.get("feature_cols", [])
    mining_type  = meta.get("mining_type", "").upper()

    full_path = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    X  = df[[c for c in feature_cols if c in df.columns]].fillna(0)

    # supported mining types
    LIGHTGBM_TYPES = {"LIGHTGBM", "LGBM"}
    H2O_TYPES = {"RANDOMFOREST", "DEEPLEARNING", "GRADIENTBOOSTING", "GLM",
                 "XGBOOST", "AUTOML"}

    if mining_type in LIGHTGBM_TYPES:
        # lightGBM for h2o version
        X = X.astype(float_dtype)
        with open(root_path / meta["model_path"], "rb") as f:
            model = pickle.load(f)
        all_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)
    elif mining_type in H2O_TYPES:
        # h2o connection / python h2o prediction with h2o frame
        import h2o
        h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
        model = h2o.import_mojo(str(root_path / meta.get("model_path", "")))
        h2o_frame = h2o.H2OFrame(X.astype(float_dtype))
        preds = model.predict(h2o_frame).as_data_frame()
        all_proba = preds.values
    else:
        # auto-encoder
        X = X.astype(float_dtype)
        with open(root_path / meta["model_path"], "rb") as f:
            model = pickle.load(f)
        all_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X).reshape(-1, 1)

    for i, prop in enumerate(multitarget_props):
        col_name  = prop.get("score_col", f"score_{i}")
        col_idx   = min(i, all_proba.shape[1] - 1) if hasattr(all_proba, "shape") and len(all_proba.shape) > 1 else 0
        proba_col = all_proba[:, col_idx] if hasattr(all_proba, "shape") and len(all_proba.shape) > 1 else all_proba
        w = multitarget_weights[i] if i < len(multitarget_weights) else 1.0
        df[col_name] = np.round(proba_col * w, 6)

    # save OutParam of Predict Node
    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_multitarget_h2o_py.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result": "ok", "model_id": model_id,
        "total_rows": len(df), "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def predict_multitarget_validation_python(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    root_dir, json_obj,
) -> dict:
    """Python 모델 멀티타겟 검증 예측."""
    import json as _json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_multitarget_validation_python: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64
    multitarget_props = json_obj.get("multitarget_properties", [])

    meta_file = root_path / "models" / f"{model_id}_meta.json"
    with open(meta_file, encoding="utf-8") as f:
        meta = _json.load(f)

    feature_cols = meta.get("feature_cols", [])
    full_path    = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    X  = df[[c for c in feature_cols if c in df.columns]].fillna(0).astype(float_dtype)

    with open(root_path / meta["model_path"], "rb") as f:
        model = pickle.load(f)

    all_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X).reshape(-1, 1)

    for i, prop in enumerate(multitarget_props):
        col_name = prop.get("score_col", f"score_{i}")
        col_idx  = min(i + 1, all_proba.shape[1] - 1) if len(all_proba.shape) > 1 else 0
        proba    = all_proba[:, col_idx] if len(all_proba.shape) > 1 else all_proba
        df[col_name] = np.round(proba, 6)

    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_multitarget_py.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result": "ok", "model_id": model_id,
        "total_rows": len(df), "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def predict_h2o_autoencoder(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, root_dir, json_obj,
) -> dict:
    """H2O AutoEncoder 이상 탐지 예측 — 재구성 오차(MSE) 기반."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_h2o_autoencoder: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    score_col    = json_obj.get("score_col", "recon_error")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    meta_file = root_path / "models" / f"{model_id}_meta.json"
    with open(meta_file, encoding="utf-8") as f:
        meta = _json.load(f)

    feature_cols = meta.get("feature_cols", [])
    full_path    = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    X  = df[[c for c in feature_cols if c in df.columns]].fillna(0).astype(float_dtype)

    import h2o
    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    model = h2o.import_mojo(str(root_path / meta.get("model_path", "")))
    h2o_frame = h2o.H2OFrame(X)
    recon_err  = model.anomaly(h2o_frame).as_data_frame()
    df[score_col] = recon_err.iloc[:, 0].values.round(6)

    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_autoencoder_predict.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result": "ok", "model_id": model_id,
        "total_rows": len(df), "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def predict_under_ensemble(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    root_dir, json_obj,
) -> dict:
    """Undersampled 앙상블 예측 — 각 서브모델 평균."""
    import json as _json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_under_ensemble: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    score_col    = json_obj.get("score_col", "score")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    meta_file = root_path / "models" / f"{model_id}_meta.json"
    with open(meta_file, encoding="utf-8") as f:
        meta = _json.load(f)

    sub_model_ids = meta.get("sub_model_ids", [model_id])
    feature_cols  = meta.get("feature_cols", [])

    full_path = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    X  = df[[c for c in feature_cols if c in df.columns]].fillna(0).astype(float_dtype)

    # predict undersampled ensemble nodes
    ensemble_proba = np.zeros(len(df))
    n_valid = 0
    for sub_id in sub_model_ids:
        sub_path = root_path / "models" / f"{sub_id}.pkl"
        if not sub_path.exists():
            continue
        with open(sub_path, "rb") as f:
            sub_model = pickle.load(f)
        proba = sub_model.predict_proba(X)[:, 1] if hasattr(sub_model, "predict_proba") else sub_model.predict(X)
        ensemble_proba += proba
        n_valid += 1

    if n_valid > 0:
        ensemble_proba /= n_valid

    df[score_col] = np.round(ensemble_proba, 6)

    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_under_ensemble_predict.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result": "ok", "model_id": model_id,
        "total_rows": len(df), "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def predict_ensemble(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    root_dir, json_obj,
) -> dict:
    """가중 앙상블 예측."""
    import json as _json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_ensemble: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    score_col    = json_obj.get("score_col", "score")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    meta_file = root_path / "models" / f"{model_id}_meta.json"
    with open(meta_file, encoding="utf-8") as f:
        meta = _json.load(f)

    member_ids = meta.get("member_ids", [model_id])
    weights    = meta.get("weights", [1.0 / max(len(member_ids), 1)] * len(member_ids))

    full_path = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)

    ensemble_proba = np.zeros(len(df))
    for mid, w in zip(member_ids, weights):
        m_meta_file = root_path / "models" / f"{mid}_meta.json"
        if not m_meta_file.exists():
            continue
        with open(m_meta_file, encoding="utf-8") as f:
            m_meta = _json.load(f)
        feature_cols = m_meta.get("feature_cols", [])
        X = df[[c for c in feature_cols if c in df.columns]].fillna(0).astype(float_dtype)
        m_path = root_path / m_meta.get("model_path", f"models/{mid}.pkl")
        if not m_path.exists():
            continue
        with open(m_path, "rb") as f:
            model = pickle.load(f)
        proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
        ensemble_proba += w * proba

    df[score_col] = np.round(ensemble_proba, 6)

    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_ensemble_predict.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result": "ok", "model_id": model_id,
        "total_rows": len(df), "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def predict_score(
    service_db_info, file_server_host, file_server_port,
    result_file_path_faf, done_file_path_faf, root_dir, json_obj,
) -> dict:
    """확률 → 신용 점수 변환 (PDO 방식)."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_score: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    prob_col     = json_obj.get("prob_col", "score")
    score_col    = json_obj.get("output_score_col", "credit_score")
    pdo_value    = float(json_obj.get("pdo_value", 20))
    anchor_value = float(json_obj.get("anchor_value", 600))

    full_path = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)

    proba = df[prob_col].clip(1e-6, 1 - 1e-6)
    df[score_col] = make_score(pdo_value, anchor_value) if False else \
        np.round(anchor_value - pdo_value / np.log(2) * np.log(proba / (1 - proba)), 0).astype(int)

    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_score.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result": "ok", "model_id": model_id, "total_rows": len(df),
        "score_stats": _series_stats(df[score_col].astype(float)), "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def make_score(pdo_value: float, anchor_value: float):
    """PDO/Anchor 기반 점수 변환 함수 반환."""
    import numpy as np

    factor = pdo_value / np.log(2)

    def _score(prob: float) -> int:
        prob = max(min(prob, 1 - 1e-9), 1e-9)
        return int(round(anchor_value - factor * np.log(prob / (1 - prob))))

    return _score


def make_score2(
    x_pdo_value: float,
    anchor_value: float,
    prob_minimum: float,
    prob_maximum: float,
    score_minimum: int,
    score_maximum: int,
):
    """최소/최대 점수 범위로 클리핑하는 점수 변환 함수 반환."""
    import numpy as np

    base_fn = make_score(x_pdo_value, anchor_value)

    def _score2(prob: float) -> int:
        raw = base_fn(max(min(prob, prob_maximum), prob_minimum))
        return max(min(raw, score_maximum), score_minimum)

    return _score2


def predict_scorecard(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj,
    root_dir, json_obj,
) -> dict:
    """스코어카드 모델 예측 — 등급별 보고, summary_groupby_count 포함."""
    import json as _json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_scorecard: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    score_col    = json_obj.get("score_col", "score")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    # 스코어카드 모델 로드
    sc_file = root_path / "models" / f"{model_id}_scorecard.json"
    with open(sc_file, encoding="utf-8") as f:
        sc_data = _json.load(f)

    scorecard_df = pd.DataFrame(sc_data["scorecard"])
    selected_cols = sc_data.get("selected_cols", [])

    full_path = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)

    # 점수 산출
    scores = pd.Series(np.zeros(len(df)), index=df.index)
    for col in selected_cols:
        if col not in df.columns:
            continue
        try:
            bins = pd.qcut(df[col], q=10, duplicates="drop").astype(str)
        except Exception:
            try:
                bins = pd.cut(df[col], bins=10, duplicates="drop").astype(str)
            except Exception:
                continue
        score_map = scorecard_df[scorecard_df["variable"] == col].set_index("bin")["points"].to_dict()
        scores += bins.map(score_map).fillna(0)

    df[score_col] = scores.round(0).astype(int)

    # reporting for each grade / summary_groupby_count
    grade_mapping = json_obj.get("grade_mapping", {})
    if grade_mapping:
        def _grade(s):
            for g, (lo, hi) in grade_mapping.items():
                if lo <= s < hi:
                    return g
            return "UNKNOWN"
        df["grade"] = df[score_col].apply(_grade)
        grade_summary = df["grade"].value_counts().to_dict()
    else:
        grade_summary = {}

    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_scorecard_pred.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result": "ok", "model_id": model_id, "total_rows": len(df),
        "grade_summary": grade_summary, "output_file": str(output_file),
        "score_stats": _series_stats(df[score_col].astype(float)),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def predict_scorecard_py(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    root_dir, json_obj,
) -> dict:
    """ML 스코어카드 모델 Python 예측."""
    import json as _json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_scorecard_py: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    score_col    = json_obj.get("score_col", "score")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    model_file = root_path / "models" / f"{model_id}_scorecard_ml.pkl"
    with open(model_file, "rb") as f:
        model_data = pickle.load(f)

    model         = model_data["model"]
    selected_cols = model_data["selected_cols"]

    full_path = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    X  = df[[c for c in selected_cols if c in df.columns]].fillna(-9999).astype(float_dtype)

    proba = model.predict_proba(X)[:, 1]
    df[score_col] = np.round(proba * 1000, 0).astype(int)

    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_scorecard_py_pred.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result": "ok", "model_id": model_id, "total_rows": len(df),
        "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def predict_score_ranking(
    service_db_info, file_server_host, file_server_port,
    result_file_path_faf, done_file_path_faf, root_dir, json_obj,
) -> dict:
    """점수 분위(ranking) 산출 및 등급 할당."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_score_ranking: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    score_col    = json_obj.get("score_col", "score")
    rank_col     = json_obj.get("rank_col", "score_rank")
    n_ranks      = int(json_obj.get("n_ranks", 10))

    full_path = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)

    try:
        df[rank_col] = pd.qcut(df[score_col], q=n_ranks, labels=False, duplicates="drop") + 1
    except Exception:
        df[rank_col] = 1

    rank_dist = df[rank_col].value_counts().sort_index().to_dict()

    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_ranking.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result": "ok", "model_id": model_id, "total_rows": len(df),
        "rank_dist": {str(k): v for k, v in rank_dist.items()},
        "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def predict_h2o_autoencoder_py(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, root_dir, json_obj,
) -> dict:
    """H2O AutoEncoder Python 버전 예측 — h2o.import_mojo 사용."""
    return predict_h2o_autoencoder(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        numpy_use_32bit_float_precision,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, root_dir, json_obj,
    )


def predict_score_cutoff(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    root_dir, json_obj,
) -> dict:
    """Score Cutoff 기반 예측 분류 — 임계값으로 accept/reject 결정."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_score_cutoff: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    score_col    = json_obj.get("score_col", "score")
    cutoff_col   = json_obj.get("output_col", "decision")
    cutoff_value = float(json_obj.get("cutoff_value", 0.5))
    accept_label = json_obj.get("accept_label", "ACCEPT")
    reject_label = json_obj.get("reject_label", "REJECT")

    full_path = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)

    df[cutoff_col] = np.where(df[score_col] >= cutoff_value, accept_label, reject_label)
    accept_rate = float((df[cutoff_col] == accept_label).mean())

    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_score_cutoff.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result": "ok", "model_id": model_id, "total_rows": len(df),
        "cutoff_value": cutoff_value, "accept_rate": round(accept_rate, 4),
        "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result
