"""
pretrained_executor.py
----------------------
사전 학습된(pretrained) 모델을 활용한 추론(inference) 실행기.

재학습 없이 기존 모델을 로드하여 임베딩 추출, 피처 생성,
또는 최종 예측을 수행한다.

사용 사례:
  - 내부 리스크 팀이 이미 학습/검증한 모델을 운영에 배포
  - 외부 공개 pretrained 모델(HuggingFace, ONNX 등) 활용
  - Transfer Learning의 feature extractor 단계로 활용
  - A/B 테스트를 위한 챔피언/챌린저 모델 동시 배포

모델 형식 지원:
  - pickle  (.pkl): scikit-learn / XGBoost / LightGBM
  - onnx    (.onnx): ONNX Runtime 기반 추론
  - h2o     (MOJO): H2O MOJO
  - hugging : HuggingFace transformers
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


class PretrainedExecutor(BaseExecutor):
    """
    사전 학습 모델 추론 executor.

    config 필수 키
    --------------
    model_id    : str   모델 식별자 (메타 JSON 기준)
    input_path  : str   입력 데이터 경로 (.parquet)
    output_id   : str   결과 저장 식별자

    config 선택 키
    --------------
    model_format : str   "pickle" | "onnx" | "h2o" | "hugging" (기본: 메타에서 자동 감지)
    score_col    : str   예측 점수 컬럼명 (기본: "score")
    output_mode  : str   "score" | "embedding" | "both" (기본: "score")
    batch_size   : int   배치 추론 크기 (기본: 10000, ONNX/HuggingFace 사용 시)
    """

    def execute(self) -> dict:
        cfg = self.config

        # 모델 메타 로드
        meta = self._load_meta(cfg["model_id"])
        model_format = cfg.get("model_format") or meta.get("model_format", "pickle")
        output_mode  = cfg.get("output_mode", "score")
        score_col    = cfg.get("score_col", "score")

        self._update_job_status(ExecutorStatus.RUNNING, progress=15)

        # 데이터 로드
        df = self._load_dataframe(cfg["input_path"])
        feature_cols = meta.get("feature_cols") or list(df.columns)
        X = df[[c for c in feature_cols if c in df.columns]]
        logger.info("pretrained inference  format=%s  rows=%d  features=%d", model_format, len(X), len(X.columns))
        self._update_job_status(ExecutorStatus.RUNNING, progress=30)

        # 추론
        result_df = df.copy()
        if model_format == "pickle":
            result_df = self._infer_pickle(meta, X, result_df, score_col, output_mode)
        elif model_format == "onnx":
            result_df = self._infer_onnx(meta, X, result_df, score_col, cfg.get("batch_size", 10000))
        elif model_format == "h2o":
            result_df = self._infer_h2o(meta, X, result_df, score_col)
        elif model_format == "hugging":
            result_df = self._infer_hugging(meta, X, result_df, cfg.get("batch_size", 512))
        else:
            raise ExecutorException(f"지원하지 않는 model_format: {model_format}")

        self._update_job_status(ExecutorStatus.RUNNING, progress=85)

        # 저장
        output_path = f"predict/{cfg['output_id']}_pretrained.parquet"
        self._save_dataframe(result_df, output_path)

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result": {
                "output_id":    cfg["output_id"],
                "model_id":     cfg["model_id"],
                "model_format": model_format,
                "output_path":  output_path,
                "total_rows":   len(result_df),
            },
            "message": f"Pretrained 추론 완료  {len(result_df):,}건  format={model_format}",
        }

    # ------------------------------------------------------------------

    def _load_meta(self, model_id: str) -> dict:
        meta_path = self.file_root / f"models/{model_id}_meta.json"
        if not meta_path.exists():
            raise ExecutorException(f"모델 메타 없음: {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    def _infer_pickle(self, meta, X, result_df, score_col, output_mode) -> pd.DataFrame:
        model_path = self.file_root / meta["model_path"]
        if not model_path.exists():
            raise ExecutorException(f"모델 파일 없음: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        if output_mode in ("score", "both") and hasattr(model, "predict_proba"):
            result_df[score_col] = model.predict_proba(X)[:, 1].round(6)
        elif output_mode in ("score", "both"):
            result_df[score_col] = model.predict(X).round(6)

        if output_mode in ("embedding", "both") and hasattr(model, "transform"):
            embedding = model.transform(X)
            emb_cols  = [f"emb_{i}" for i in range(embedding.shape[1])]
            for i, col in enumerate(emb_cols):
                result_df[col] = embedding[:, i]

        return result_df

    def _infer_onnx(self, meta, X, result_df, score_col, batch_size) -> pd.DataFrame:
        import onnxruntime as ort

        model_path = str(self.file_root / meta["model_path"])
        sess = ort.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name

        scores = []
        for i in range(0, len(X), batch_size):
            batch = X.iloc[i: i + batch_size].values.astype(np.float32)
            preds = sess.run(None, {input_name: batch})
            # 분류 확률 또는 회귀값
            if len(preds) > 1 and preds[1] is not None:
                scores.extend(preds[1][:, 1].tolist())
            else:
                scores.extend(preds[0].flatten().tolist())

        result_df[score_col] = np.round(scores, 6)
        return result_df

    def _infer_h2o(self, meta, X, result_df, score_col) -> pd.DataFrame:
        import h2o
        h2o.init()
        model = h2o.import_mojo(str(self.file_root / meta["model_path"]))
        h2o_frame = h2o.H2OFrame(X)
        preds = model.predict(h2o_frame).as_data_frame()
        result_df[score_col] = preds.iloc[:, -1].values.round(6)
        return result_df

    def _infer_hugging(self, meta, X, result_df, batch_size) -> pd.DataFrame:
        from transformers import pipeline as hf_pipeline

        model_path = str(self.file_root / meta["model_path"])
        text_col   = meta.get("text_col", X.columns[0])
        pipe       = hf_pipeline("text-classification", model=model_path)

        texts  = X[text_col].fillna("").tolist()
        labels, scores = [], []
        for i in range(0, len(texts), batch_size):
            batch_res = pipe(texts[i: i + batch_size], truncation=True)
            for r in batch_res:
                labels.append(r["label"])
                scores.append(round(r["score"], 6))

        result_df["pred_label"] = labels
        result_df["pred_score"] = scores
        return result_df


# =============================================================================
# Module-level functions
# =============================================================================


def preload_mdl(json_obj: dict) -> dict:
    """MDL 파일 바이트를 디코딩하고 모델 타입 및 VarLayout을 반환."""
    import base64
    import pickle
    import io

    import pandas as pd

    logger.info("preload_mdl: model_type=%s", json_obj.get("ModelType", ""))

    mdl_file_bytes = base64.b64decode(json_obj["MdlFileBytes"])
    model_type     = json_obj.get("ModelType", "UNKNOWN")

    # VarLayout JSON → DataFrame
    var_layout_json = json_obj.get("VarLayout", "[]")
    try:
        if isinstance(var_layout_json, str):
            import json as _json
            layout_df = pd.read_json(io.StringIO(var_layout_json), orient="records")
        elif isinstance(var_layout_json, list):
            layout_df = pd.DataFrame(var_layout_json)
        else:
            layout_df = pd.DataFrame()
    except Exception as e:
        logger.warning("VarLayout 파싱 실패: %s", e)
        layout_df = pd.DataFrame()

    # 모델 바이트를 임시 언피클 (검증)
    try:
        model_obj = pickle.loads(mdl_file_bytes)
        logger.info("preload_mdl: model loaded  type=%s", type(model_obj).__name__)
    except Exception:
        model_obj = mdl_file_bytes  # 바이트 그대로 유지

    return {
        "ModelType":  model_type,
        "VarLayout":  layout_df.to_json(orient="records"),
        "ModelBytes": mdl_file_bytes,
        "ModelObj":   model_obj,
    }


def load_model_file(
    model_type: str,
    model_file_path: str,
    h2o_host: str,
    h2o_port: int,
    h2o_older_version_port: int,
    h2o_file_server_host: str,
    h2o_file_server_port: int,
) -> object:
    """모델 파일 경로에서 모델 객체를 로드한다."""
    import pickle
    from pathlib import Path

    model_type_upper = model_type.upper() if model_type else ""

    H2O_TYPES = {
        "RANDOMFOREST", "DEEPLEARNING", "GRADIENTBOOSTING", "GLM",
        "LINEARREGRESSIONNEW", "DEEPLEARNINGLINEAR", "RANDOMFORESTLINEAR",
        "GRADIENTBOOSTINGLINEAR", "XGBOOST", "AUTOML", "AUTOMLGLM",
        "AUTOMLDL", "KMEANS", "ANOMALY",
    }

    if model_type_upper in H2O_TYPES:
        try:
            import h2o
            h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
            model = h2o.import_mojo(model_file_path)
            logger.info("load_model_file: H2O MOJO loaded  path=%s", model_file_path)
            return model
        except Exception as e:
            logger.warning("H2O MOJO 로드 실패, fallback: %s", e)

    # pickle 기반 모델 (sklearn, LightGBM, XGBoost, CatBoost, etc.)
    model_path = Path(model_file_path)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_file_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info("load_model_file: pickle loaded  type=%s", type(model).__name__)
    return model


def load_pretrained_model(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    h2o_host: str,
    h2o_port: int,
    h2o_older_version_port: int,
    h2o_file_server_host: str,
    h2o_file_server_port: int,
    json_obj: dict,
) -> dict:
    """파일 서버에서 사전 학습된 모델을 로드하고 메타 정보를 반환."""
    import json as _json
    import pickle
    import requests
    from pathlib import Path

    logger.info("load_pretrained_model: model_id=%s", json_obj.get("model_id"))

    root_dir  = json_obj.get("root_dir", "/data")
    model_id  = json_obj["model_id"]
    model_key = json_obj.get("model_key", model_id)

    # 파일 서버에서 모델 메타 조회
    model_dir = Path(root_dir) / "models"
    meta_file = model_dir / f"{model_key}_meta.json"

    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            meta = _json.load(f)
    else:
        # 파일 서버 HTTP 조회 시도
        try:
            url  = f"http://{file_server_host}:{file_server_port}/model/{model_key}/meta"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            meta = resp.json()
        except Exception as e:
            logger.warning("파일 서버 메타 조회 실패: %s  fallback to empty meta", e)
            meta = {"model_id": model_id, "model_type": json_obj.get("model_type", "UNKNOWN")}

    model_type  = meta.get("model_type", json_obj.get("model_type", "UNKNOWN"))
    model_path  = meta.get("model_path", str(model_dir / f"{model_key}.pkl"))

    model = load_model_file(
        model_type, model_path,
        h2o_host, h2o_port, h2o_older_version_port,
        h2o_file_server_host, h2o_file_server_port,
    )

    return {
        "model":      model,
        "meta":       meta,
        "model_type": model_type,
        "model_id":   model_id,
    }


def predict_pretrained_model(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    h2o_host: str,
    h2o_port: int,
    h2o_older_version_port: int,
    h2o_file_server_host: str,
    h2o_file_server_port: int,
    result_file_path_faf: str,
    done_file_path_faf: str,
    numpy_use_32bit_float_precision: bool,
    root_dir: str,
    json_obj: dict,
) -> dict:
    """사전 학습 모델로 예측 수행 — ensemble 노드도 Python에서 수행."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    logger.info("predict_pretrained_model: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    target_col   = json_obj.get("target_col")
    score_col    = json_obj.get("score_col", "score")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    # 모델 로드
    loaded = load_pretrained_model(
        service_db_info, file_server_host, file_server_port,
        h2o_host, h2o_port, h2o_older_version_port,
        h2o_file_server_host, h2o_file_server_port,
        json_obj,
    )
    model      = loaded["model"]
    meta       = loaded["meta"]
    model_type = loaded["model_type"]

    # 데이터 로드
    full_path = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    feature_cols = meta.get("feature_cols") or [c for c in df.columns if c != target_col]
    X = df[[c for c in feature_cols if c in df.columns]].fillna(0).astype(float_dtype)

    # ensemble 노드도 python에서 수행하면서 ml_utils.calculate_performance() 이용
    reverse_prob = json_obj.get("reverse_prob", True)

    H2O_TYPES = {"RANDOMFOREST", "DEEPLEARNING", "GRADIENTBOOSTING", "GLM",
                 "XGBOOST", "AUTOML", "AUTOMLGLM", "AUTOMLDL"}

    if model_type.upper() in H2O_TYPES:
        import h2o
        h2o_frame = h2o.H2OFrame(X)
        preds = model.predict(h2o_frame).as_data_frame()
        proba = preds.iloc[:, -1].values
    else:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            proba = model.predict(X).astype(float)

    if reverse_prob:
        proba = 1.0 - proba

    df[score_col] = np.round(proba, 6)

    # ensemble perf 결과와 pretrained ensemble perf 결과를 비슷하게 display하기 위해 python에서 다시 계산
    perf = {}
    if target_col and target_col in df.columns:
        y = df[target_col]
        try:
            auc = round(float(roc_auc_score(y, proba)), 4)
            perf["auc"] = auc
        except Exception:
            pass

    # Upload OutParam — returned as a named list
    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_pretrained_predict.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result":      "ok",
        "model_id":    model_id,
        "total_rows":  len(df),
        "perf":        perf,
        "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def convert_ensemble_to_rclips_model(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    h2o_host: str,
    h2o_port: int,
    h2o_older_version_port: int,
    h2o_file_server_host: str,
    h2o_file_server_port: int,
    java_module_root_dir: str,
    java_home: str,
    json_obj: dict,
) -> dict:
    """앙상블 모델을 rclips 호환 모델 포맷으로 변환."""
    import json as _json
    import pickle
    from pathlib import Path

    logger.info("convert_ensemble_to_rclips_model: model_id=%s", json_obj.get("model_id"))

    root_dir  = json_obj.get("root_dir", "/data")
    model_id  = json_obj["model_id"]
    root_path = Path(root_dir)
    model_dir = root_path / "models"

    # 앙상블 구성 모델 로드
    ensemble_cfg  = json_obj.get("ensemble_config", {})
    member_ids    = ensemble_cfg.get("member_ids", [])
    weights       = ensemble_cfg.get("weights", [1.0 / max(len(member_ids), 1)] * len(member_ids))

    members = []
    for mid in member_ids:
        meta_file = model_dir / f"{mid}_meta.json"
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                meta = _json.load(f)
        else:
            meta = {"model_id": mid}
        members.append(meta)

    # rclips 포맷 직렬화
    rclips_model = {
        "model_id":  model_id,
        "type":      "ensemble",
        "members":   members,
        "weights":   weights,
        "java_home": java_home,
    }

    rclips_file = model_dir / f"{model_id}_rclips.pkl"
    with open(rclips_file, "wb") as f:
        pickle.dump(rclips_model, f)

    logger.info("convert_ensemble_to_rclips_model done: %s", rclips_file)
    return {
        "result":      "ok",
        "model_id":    model_id,
        "rclips_file": str(rclips_file),
        "n_members":   len(members),
    }
