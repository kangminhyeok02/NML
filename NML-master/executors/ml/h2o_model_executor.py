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


# =============================================================================
# Module-level functions
# =============================================================================


def _make_h2o_engine(db_info: dict):
    """SQLAlchemy engine 생성 (h2o_model_executor 내부용)."""
    from sqlalchemy import create_engine
    url = (
        f"postgresql+psycopg2://{db_info['user']}:{db_info['password']}"
        f"@{db_info['host']}:{db_info['port']}/{db_info['db']}"
    )
    return create_engine(url, pool_pre_ping=True)


def connect_h2o(h2o_host: str, h2o_port: int, no_progress: bool = True) -> None:
    """H2O 클러스터에 연결한다."""
    import h2o
    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    if no_progress:
        h2o.no_progress()
    logger.info("connect_h2o: connected to %s:%d", h2o_host, h2o_port)


def import_h2o_frame(
    data_path: str,
    h2o_host: str,
    h2o_port: int,
    target_col: str = None,
    mining_type: str = "",
) -> "h2o.H2OFrame":
    """파일에서 H2OFrame을 로드한다."""
    import h2o
    import pandas as pd

    connect_h2o(h2o_host, h2o_port)

    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
        h2o_frame = h2o.H2OFrame(df)
    else:
        h2o_frame = h2o.import_file(data_path)

    DISCRETE_TYPES = {"RANDOMFOREST", "DEEPLEARNING", "GRADIENTBOOSTING", "GLM",
                      "AUTOML", "AUTOMLGLM", "AUTOMLDL", "XGBOOST"}
    if target_col and target_col in h2o_frame.columns:
        if mining_type.upper() in DISCRETE_TYPES:
            h2o_frame[target_col] = h2o_frame[target_col].asfactor()

    return h2o_frame


def save_stg_obj_with_properties(
    company_db_info: dict,
    process_seq: str,
    stg_id: str,
    stg_name: str,
    stg_comment: str,
    stg_type: str,
    stg_obj: bytes,
    save_desc: str,
    user_id: str,
    properties: dict,
) -> dict:
    """전략 노드 직렬화 데이터 + properties를 DB에 저장(upsert)."""
    import base64
    import json as _json
    from datetime import datetime
    from sqlalchemy import text

    logger.info("save_stg_obj_with_properties: process_seq=%s  stg_id=%s", process_seq, stg_id)

    if isinstance(stg_obj, str):
        stg_obj = base64.b64decode(stg_obj)

    reg_dtm = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    engine  = _make_h2o_engine(company_db_info)

    with engine.begin() as conn:
        exists = conn.execute(
            text("SELECT COUNT(*) FROM tb_stg_obj WHERE process_seq = :ps AND stg_id = :si"),
            {"ps": process_seq, "si": stg_id},
        ).scalar()

        if exists:
            conn.execute(
                text(
                    "UPDATE tb_stg_obj SET stg_name=:sn, stg_comment=:sc, stg_type=:st, "
                    "stg_obj=:so, user_id=:ui, reg_dtm=:rd, properties=:pr "
                    "WHERE process_seq=:ps AND stg_id=:si"
                ),
                {"sn": stg_name, "sc": stg_comment, "st": stg_type, "so": stg_obj,
                 "ui": user_id, "rd": reg_dtm, "pr": _json.dumps(properties),
                 "ps": process_seq, "si": stg_id},
            )
        else:
            conn.execute(
                text(
                    "INSERT INTO tb_stg_obj "
                    "(process_seq, stg_id, stg_name, stg_comment, stg_type, stg_obj, user_id, reg_dtm, properties) "
                    "VALUES (:ps, :si, :sn, :sc, :st, :so, :ui, :rd, :pr)"
                ),
                {"ps": process_seq, "si": stg_id, "sn": stg_name, "sc": stg_comment,
                 "st": stg_type, "so": stg_obj, "ui": user_id, "rd": reg_dtm,
                 "pr": _json.dumps(properties)},
            )

    return {"result": "ok", "process_seq": process_seq, "stg_id": stg_id}


def save_mojo_pojo_model_file(
    service_db_info, file_server_host, file_server_port,
    company_code, process_seq,
    h2o_host, h2o_port, h2o_script_obj, root_dir,
    model_file_path_in_h2o_file_server,
    model_file_name, model_type="",
) -> dict:
    """H2O MOJO/POJO 모델 파일 저장 (h2o_model_executor 버전)."""
    import json as _json
    from pathlib import Path

    logger.info("save_mojo_pojo_model_file: %s  type=%s", model_file_name, model_type)

    connect_h2o(h2o_host, h2o_port)
    import h2o

    save_dir = Path(root_dir) / "models"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        model    = h2o.get_model(model_file_path_in_h2o_file_server)
        mojo_path = model.save_mojo(str(save_dir))
        logger.info("MOJO saved: %s", mojo_path)
        return {"result": "ok", "mojo_path": mojo_path, "model_type": model_type}
    except Exception as e:
        logger.warning("MOJO 저장 실패: %s", e)
        return {"result": "error", "message": str(e)}


def fit_gridSearch(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O Grid Search 학습."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("fit_gridSearch: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    algorithm    = json_obj.get("algorithm", "gbm").lower()
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    connect_h2o(h2o_host, h2o_port)
    import h2o
    from h2o.grid.grid_search import H2OGridSearch
    from h2o.estimators import (
        H2OGradientBoostingEstimator, H2ORandomForestEstimator,
        H2OXGBoostEstimator, H2ODeepLearningEstimator,
        H2OGeneralizedLinearEstimator,
    )

    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0).astype(float_dtype))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    algo_map = {
        "gbm":          H2OGradientBoostingEstimator,
        "drf":          H2ORandomForestEstimator,
        "xgboost":      H2OXGBoostEstimator,
        "deeplearning": H2ODeepLearningEstimator,
        "glm":          H2OGeneralizedLinearEstimator,
    }
    estimator_cls  = algo_map.get(algorithm, H2OGradientBoostingEstimator)
    hyper_params   = json_obj.get("hyper_params", {"max_depth": [3, 5, 7], "ntrees": [50, 100]})
    search_criteria = json_obj.get("search_criteria", {"strategy": "Cartesian"})

    grid = H2OGridSearch(
        model=estimator_cls(seed=42),
        hyper_params=hyper_params,
        search_criteria=search_criteria,
    )
    grid.train(x=feature_cols, y=target_col, training_frame=train_h2o)
    best_model = grid.get_grid(sort_by="auc", decreasing=True).models[0]

    return _save_h2o_result(best_model, json_obj, root_path, result_file_path_faf, done_file_path_faf)


def fit_glm(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O GLM 학습."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("fit_glm: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    connect_h2o(h2o_host, h2o_port)
    import h2o
    from h2o.estimators import H2OGeneralizedLinearEstimator

    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0).astype(float_dtype))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    glm = H2OGeneralizedLinearEstimator(
        family=json_obj.get("family", "binomial"),
        alpha=float(json_obj.get("alpha", 0.5)),
        lambda_search=json_obj.get("lambda_search", True),
        seed=42,
    )
    glm.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    return _save_h2o_result(glm, json_obj, root_path, result_file_path_faf, done_file_path_faf)


def fit_rf(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O Random Forest 학습."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("fit_rf (h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    connect_h2o(h2o_host, h2o_port)
    import h2o
    from h2o.estimators import H2ORandomForestEstimator

    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0).astype(float_dtype))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    model = H2ORandomForestEstimator(
        ntrees=int(json_obj.get("ntrees", 100)),
        max_depth=int(json_obj.get("max_depth", 20)),
        mtries=int(json_obj.get("mtries", -1)),
        seed=42,
    )
    model.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    return _save_h2o_result(model, json_obj, root_path, result_file_path_faf, done_file_path_faf)


def fit_gbm(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O GBM 학습."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("fit_gbm (h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    connect_h2o(h2o_host, h2o_port)
    import h2o
    from h2o.estimators import H2OGradientBoostingEstimator

    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0).astype(float_dtype))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    model = H2OGradientBoostingEstimator(
        ntrees=int(json_obj.get("ntrees", 100)),
        max_depth=int(json_obj.get("max_depth", 5)),
        learn_rate=float(json_obj.get("learn_rate", 0.1)),
        col_sample_rate=float(json_obj.get("col_sample_rate", 0.8)),
        seed=42,
    )
    model.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    return _save_h2o_result(model, json_obj, root_path, result_file_path_faf, done_file_path_faf)


def fit_xgboost(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O XGBoost 학습."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("fit_xgboost (h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    connect_h2o(h2o_host, h2o_port)
    import h2o
    from h2o.estimators import H2OXGBoostEstimator

    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0).astype(float_dtype))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    model = H2OXGBoostEstimator(
        ntrees=int(json_obj.get("ntrees", 100)),
        max_depth=int(json_obj.get("max_depth", 6)),
        learn_rate=float(json_obj.get("learn_rate", 0.1)),
        subsample=float(json_obj.get("subsample", 0.8)),
        seed=42,
    )
    model.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    return _save_h2o_result(model, json_obj, root_path, result_file_path_faf, done_file_path_faf)


def fit_kmeans(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O K-Means 클러스터링 학습."""
    import json as _json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("fit_kmeans: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    feature_cols = json_obj.get("feature_cols")
    k            = int(json_obj.get("k", 5))
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = list(df.select_dtypes(include=[float, int]).columns)

    connect_h2o(h2o_host, h2o_port)
    import h2o
    from h2o.estimators import H2OKMeansEstimator

    train_h2o = h2o.H2OFrame(df[feature_cols].fillna(0).astype(float_dtype))
    model = H2OKMeansEstimator(k=k, seed=42, standardize=True)
    model.train(x=feature_cols, training_frame=train_h2o)

    model_dir = root_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    mojo_path = model.save_mojo(str(model_dir))

    # 클러스터 중심
    try:
        centers = model.centers().as_data_frame().to_dict(orient="records")
    except Exception:
        centers = []

    meta = {
        "model_id":     model_id,
        "h2o_model_id": model.model_id,
        "mojo_path":    mojo_path,
        "feature_cols": feature_cols,
        "k":            k,
        "centers":      centers,
    }
    meta_file = model_dir / f"{model_id}_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        _json.dump(meta, f, ensure_ascii=False, default=str)

    result = {"result": "ok", "model_id": model_id, "k": k, "mojo_path": mojo_path}
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def paramsearch_rf(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O Random Forest 파라미터 탐색."""
    j = dict(json_obj)
    j.setdefault("algorithm", "drf")
    j.setdefault("hyper_params", {"ntrees": [50, 100, 200], "max_depth": [10, 20, 30]})
    return fit_gridSearch(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        numpy_use_32bit_float_precision, result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir, j,
    )


def paramsearch_gbm(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O GBM 파라미터 탐색."""
    j = dict(json_obj)
    j.setdefault("algorithm", "gbm")
    j.setdefault("hyper_params", {
        "ntrees": [50, 100, 200], "max_depth": [3, 5, 7], "learn_rate": [0.05, 0.1, 0.2],
    })
    return fit_gridSearch(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        numpy_use_32bit_float_precision, result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir, j,
    )


def paramsearch_xgboost(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O XGBoost 파라미터 탐색."""
    j = dict(json_obj)
    j.setdefault("algorithm", "xgboost")
    j.setdefault("hyper_params", {
        "ntrees": [50, 100, 200], "max_depth": [4, 6, 8], "learn_rate": [0.05, 0.1],
    })
    return fit_gridSearch(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        numpy_use_32bit_float_precision, result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir, j,
    )


def paramsearch_deep(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O DeepLearning 파라미터 탐색."""
    j = dict(json_obj)
    j.setdefault("algorithm", "deeplearning")
    j.setdefault("hyper_params", {
        "hidden": [[50, 50], [100, 100], [200, 200]], "epochs": [50, 100],
    })
    return fit_gridSearch(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        numpy_use_32bit_float_precision, result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir, j,
    )


def feature_ae(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O AutoEncoder 피처 추출 (h2o_model_executor 버전)."""
    import json as _json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("feature_ae (h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    feature_cols = json_obj.get("feature_cols")
    n_features   = int(json_obj.get("n_features", 10))
    hidden       = json_obj.get("hidden", [50, n_features, 50])
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = list(df.select_dtypes(include=[float, int]).columns)

    connect_h2o(h2o_host, h2o_port)
    import h2o
    from h2o.estimators.deeplearning import H2ODeepLearningEstimator

    train_h2o = h2o.H2OFrame(df[feature_cols].fillna(0).astype(float_dtype))
    ae = H2ODeepLearningEstimator(autoencoder=True, hidden=hidden, activation="Tanh", epochs=100, seed=42)
    ae.train(x=feature_cols, training_frame=train_h2o)

    encoded = ae.deepfeatures(train_h2o, layer=len(hidden) // 2).as_data_frame()
    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_ae_features.parquet"
    encoded.to_parquet(output_file, index=False)

    result = {"result": "ok", "model_id": model_id, "n_features": encoded.shape[1], "output_file": str(output_file)}
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def feature_glrm(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O GLRM 피처 추출 (h2o_model_executor 버전)."""
    import json as _json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("feature_glrm (h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    feature_cols = json_obj.get("feature_cols")
    k            = int(json_obj.get("k", 10))
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = list(df.select_dtypes(include=[float, int]).columns)

    connect_h2o(h2o_host, h2o_port)
    import h2o
    from h2o.estimators import H2OGeneralizedLowRankEstimator

    train_h2o = h2o.H2OFrame(df[feature_cols].fillna(0).astype(float_dtype))
    glrm = H2OGeneralizedLowRankEstimator(k=k, seed=42)
    glrm.train(x=feature_cols, training_frame=train_h2o)

    x_arch = glrm.score_archetype(train_h2o).as_data_frame()
    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_glrm_features.parquet"
    x_arch.to_parquet(output_file, index=False)

    result = {"result": "ok", "model_id": model_id, "k": k, "output_file": str(output_file)}
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def feature_svd(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O SVD 피처 추출 (h2o_model_executor 버전)."""
    import json as _json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("feature_svd (h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    feature_cols = json_obj.get("feature_cols")
    nv           = int(json_obj.get("nv", 10))
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = list(df.select_dtypes(include=[float, int]).columns)

    connect_h2o(h2o_host, h2o_port)
    import h2o
    from h2o.estimators import H2OSingularValueDecompositionEstimator

    train_h2o = h2o.H2OFrame(df[feature_cols].fillna(0).astype(float_dtype))
    svd = H2OSingularValueDecompositionEstimator(nv=nv, seed=42)
    svd.train(x=feature_cols, training_frame=train_h2o)

    u_frame = svd.predict(train_h2o).as_data_frame()
    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_svd_features.parquet"
    u_frame.to_parquet(output_file, index=False)

    result = {"result": "ok", "model_id": model_id, "nv": nv, "output_file": str(output_file)}
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def report_runcorr(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """상관관계 보고서 — 고상관 변수 쌍 탐지 (fire and forget 포함)."""
    import json as _json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("report_runcorr (h2o): model_id=%s", json_obj.get("model_id"))

    root_path   = Path(root_dir)
    data_path   = json_obj["data_path"]
    threshold   = float(json_obj.get("threshold", 0.8))
    feature_cols = json_obj.get("feature_cols")

    full_path = root_path / data_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = list(df.select_dtypes(include=[np.number]).columns)

    corr_matrix = df[feature_cols].corr()
    pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            c = abs(corr_matrix.iloc[i, j])
            if c >= threshold:
                pairs.append({"var1": feature_cols[i], "var2": feature_cols[j],
                              "corr": round(float(corr_matrix.iloc[i, j]), 4)})

    result = {"result": "ok", "n_high_corr": len(pairs), "pairs": pairs}
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def report_removecorr(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """고상관 변수 제거 — 상관계수 임계값 이상인 변수 쌍에서 하나 제거."""
    import json as _json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("report_removecorr: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    data_path    = json_obj["data_path"]
    threshold    = float(json_obj.get("threshold", 0.8))
    feature_cols = json_obj.get("feature_cols")

    full_path = root_path / data_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = list(df.select_dtypes(include=[np.number]).columns)

    corr_matrix = df[feature_cols].corr().abs()
    to_remove = set()
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if corr_matrix.iloc[i, j] >= threshold and feature_cols[j] not in to_remove:
                to_remove.add(feature_cols[j])

    remaining = [c for c in feature_cols if c not in to_remove]
    result = {
        "result":    "ok",
        "removed":   list(to_remove),
        "remaining": remaining,
        "n_removed": len(to_remove),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def adjust_multitarget_result_json(
    result_json: dict,
    multitarget_properties: list,
    h2o_host: str = None,
    h2o_port: int = None,
    h2o_script_obj=None,
    root_dir: str = None,
) -> dict:
    """멀티타겟 결과 JSON 조정 (h2o_model_executor 버전).

    add "h2o_host, h2o_port, h2o_script_obj, root_dir" for save_mojo_pojo_model_file()
    """
    import numpy as np

    adjusted = dict(result_json)
    scores   = adjusted.get("scores", {})

    for i, prop in enumerate(multitarget_properties):
        col = prop.get("score_col", f"score_{i}")
        if col in scores:
            arr  = np.array(scores[col])
            norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
            scores[col + "_norm"] = norm.round(6).tolist()

    adjusted["scores"]                = scores
    adjusted["multitarget_count"]     = len(multitarget_properties)
    adjusted["multitarget_properties"] = multitarget_properties
    if h2o_host:
        adjusted["h2o_host"] = h2o_host
    if h2o_port:
        adjusted["h2o_port"] = h2o_port
    return adjusted


def fit_dnn(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O DeepLearning 학습 (h2o_model_executor 버전)."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("fit_dnn (h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    connect_h2o(h2o_host, h2o_port)
    import h2o
    from h2o.estimators import H2ODeepLearningEstimator

    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0).astype(float_dtype))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    model = H2ODeepLearningEstimator(
        hidden=json_obj.get("hidden", [200, 200]),
        epochs=float(json_obj.get("epochs", 100)),
        activation=json_obj.get("activation", "Rectifier"),
        dropout_ratio=json_obj.get("dropout_ratio", None),
        seed=42,
    )
    model.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    return _save_h2o_result(model, json_obj, root_path, result_file_path_faf, done_file_path_faf)


def fit_under_ensemble(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O 언더샘플링 앙상블 학습 (h2o_model_executor 버전)."""
    import json as _json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("fit_under_ensemble (h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    n_models     = int(json_obj.get("n_models", 5))
    bad_val      = json_obj.get("bad_val", 1)
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    connect_h2o(h2o_host, h2o_port)
    import h2o
    from h2o.estimators import H2OGradientBoostingEstimator

    bad_df  = df[df[target_col] == bad_val]
    good_df = df[df[target_col] != bad_val]
    n_bad   = len(bad_df)

    mojo_paths = []
    model_dir  = root_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_models):
        sample_good = good_df.sample(n=min(n_bad * 3, len(good_df)), random_state=i)
        sub_df      = pd.concat([bad_df, sample_good]).sample(frac=1, random_state=i)
        train_h2o   = h2o.H2OFrame(sub_df[feature_cols + [target_col]].fillna(0).astype(float_dtype))
        train_h2o[target_col] = train_h2o[target_col].asfactor()
        sub_model = H2OGradientBoostingEstimator(ntrees=50, max_depth=5, seed=i)
        sub_model.train(x=feature_cols, y=target_col, training_frame=train_h2o)
        mojo_path = sub_model.save_mojo(str(model_dir))
        mojo_paths.append(mojo_path)

    meta = {
        "model_id": model_id, "mojo_paths": mojo_paths,
        "feature_cols": feature_cols, "n_models": n_models,
    }
    with open(model_dir / f"{model_id}_meta.json", "w", encoding="utf-8") as f:
        _json.dump(meta, f, ensure_ascii=False)

    result = {"result": "ok", "model_id": model_id, "n_models": n_models}
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def fit_automl(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O AutoML 학습 (h2o_model_executor 버전)."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("fit_automl (h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    max_runtime  = int(json_obj.get("max_runtime_secs", 300))
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    connect_h2o(h2o_host, h2o_port)
    import h2o
    from h2o.automl import H2OAutoML

    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0).astype(float_dtype))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    aml = H2OAutoML(max_runtime_secs=max_runtime, seed=42)
    aml.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    return _save_h2o_result(aml.leader, json_obj, root_path, result_file_path_faf, done_file_path_faf)


def fit_selected_automl_model(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """AutoML 리더보드에서 선택한 알고리즘 재학습 (h2o_model_executor 버전)."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("fit_selected_automl_model (h2o): model_id=%s", json_obj.get("model_id"))

    root_path     = Path(root_dir)
    train_path    = json_obj["train_path"]
    target_col    = json_obj["target_col"]
    selected_algo = json_obj.get("selected_algo", "GBM")
    feature_cols  = json_obj.get("feature_cols")
    float_dtype   = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    connect_h2o(h2o_host, h2o_port)
    import h2o
    from h2o.estimators import (
        H2OGradientBoostingEstimator, H2ORandomForestEstimator,
        H2OXGBoostEstimator, H2ODeepLearningEstimator,
        H2OGeneralizedLinearEstimator,
    )

    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0).astype(float_dtype))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    algo_map = {
        "GBM": H2OGradientBoostingEstimator, "DRF": H2ORandomForestEstimator,
        "XGBoost": H2OXGBoostEstimator, "DeepLearning": H2ODeepLearningEstimator,
        "GLM": H2OGeneralizedLinearEstimator,
    }
    cls    = algo_map.get(selected_algo, H2OGradientBoostingEstimator)
    extra  = json_obj.get("model_params", {})
    model  = cls(seed=42, **extra)
    model.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    return _save_h2o_result(model, json_obj, root_path, result_file_path_faf, done_file_path_faf)


def fit_anomaly(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_file_server_host, h2o_file_server_port,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O AutoEncoder 이상 탐지 학습 (h2o_model_executor 버전)."""
    import json as _json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("fit_anomaly (h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    feature_cols = json_obj.get("feature_cols")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = list(df.select_dtypes(include=[float, int]).columns)

    connect_h2o(h2o_host, h2o_port)
    import h2o
    from h2o.estimators.deeplearning import H2ODeepLearningEstimator

    train_h2o = h2o.H2OFrame(df[feature_cols].fillna(0).astype(float_dtype))
    hidden    = json_obj.get("hidden", [50, 25, 50])
    ae = H2ODeepLearningEstimator(
        autoencoder=True, hidden=hidden, activation="Tanh",
        epochs=float(json_obj.get("epochs", 100)), seed=42,
    )
    ae.train(x=feature_cols, training_frame=train_h2o)

    model_dir = root_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    mojo_path = ae.save_mojo(str(model_dir))

    recon_error = ae.anomaly(train_h2o).as_data_frame()
    threshold   = float(recon_error.iloc[:, 0].quantile(0.95))

    meta = {
        "model_id": model_id, "h2o_model_id": ae.model_id, "mojo_path": mojo_path,
        "feature_cols": feature_cols, "anomaly_threshold": threshold,
    }
    with open(model_dir / f"{model_id}_meta.json", "w", encoding="utf-8") as f:
        _json.dump(meta, f, ensure_ascii=False)

    result = {
        "result": "ok", "model_id": model_id,
        "mojo_path": mojo_path, "anomaly_threshold": threshold,
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def fit_pretrained(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """사전 학습 모델을 로드하고 재학습(fine-tuning) 후 저장."""
    import json as _json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("fit_pretrained: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    pretrained_id = json_obj.get("pretrained_model_id", model_id)
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    # 사전 학습 모델 메타 로드
    meta_file = root_path / "models" / f"{pretrained_id}_meta.json"
    with open(meta_file, encoding="utf-8") as f:
        pretrained_meta = _json.load(f)

    connect_h2o(h2o_host, h2o_port)
    import h2o

    pretrained_model = h2o.import_mojo(pretrained_meta.get("mojo_path", ""))

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = pretrained_meta.get("feature_cols",
                                           [c for c in df.columns if c != target_col])

    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0).astype(float_dtype))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    # fine-tuning: pretrained weights로 초기화
    from h2o.estimators import H2ODeepLearningEstimator
    model = H2ODeepLearningEstimator(
        pretrained_autoencoder=pretrained_model.model_id,
        hidden=json_obj.get("hidden", [200, 200]),
        epochs=float(json_obj.get("epochs", 50)),
        seed=42,
    )
    model.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    return _save_h2o_result(model, json_obj, root_path, result_file_path_faf, done_file_path_faf)


def upload_forced_model_file(
    service_db_info, file_server_host, file_server_port, json_obj,
) -> dict:
    """외부 모델 파일을 파일 서버에 강제 업로드."""
    import json as _json
    import base64
    from pathlib import Path

    logger.info("upload_forced_model_file: model_id=%s", json_obj.get("model_id"))

    model_id   = json_obj["model_id"]
    model_bytes = base64.b64decode(json_obj["model_bytes"]) if "model_bytes" in json_obj else b""
    root_dir   = json_obj.get("root_dir", "/data")
    model_type = json_obj.get("model_type", "UNKNOWN")

    save_dir = Path(root_dir) / "models"
    save_dir.mkdir(parents=True, exist_ok=True)

    ext        = json_obj.get("file_ext", "pkl")
    model_file = save_dir / f"{model_id}.{ext}"
    with open(model_file, "wb") as f:
        f.write(model_bytes)

    # 메타 저장
    meta = {
        "model_id":   model_id,
        "model_type": model_type,
        "model_path": str(model_file),
        "forced_upload": True,
    }
    with open(save_dir / f"{model_id}_meta.json", "w", encoding="utf-8") as f:
        _json.dump(meta, f, ensure_ascii=False)

    return {"result": "ok", "model_id": model_id, "model_file": str(model_file)}


def load_pretrained_ensemble(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    h2o_host, h2o_port, h2o_older_version_port,
    root_dir, json_obj,
) -> dict:
    """앙상블 구성 모델들을 로드하여 ensemble 객체 반환."""
    import json as _json
    import pickle
    from pathlib import Path

    logger.info("load_pretrained_ensemble: model_id=%s", json_obj.get("model_id"))

    root_path   = Path(root_dir)
    model_id    = json_obj["model_id"]
    member_ids  = json_obj.get("member_ids", [])

    meta_file = root_path / "models" / f"{model_id}_meta.json"
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            meta = _json.load(f)
        if not member_ids:
            member_ids = meta.get("member_ids", [])

    connect_h2o(h2o_host, h2o_port)
    import h2o

    members = []
    for mid in member_ids:
        m_meta_file = root_path / "models" / f"{mid}_meta.json"
        if not m_meta_file.exists():
            continue
        with open(m_meta_file, encoding="utf-8") as f:
            m_meta = _json.load(f)
        mojo_path = m_meta.get("mojo_path", "")
        try:
            model = h2o.import_mojo(mojo_path)
            members.append({"model_id": mid, "model": model, "meta": m_meta})
        except Exception as e:
            logger.warning("멤버 로드 실패: %s  %s", mid, e)

    return {"model_id": model_id, "members": members, "n_members": len(members)}


def fit_for_stacking(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """스태킹 앙상블을 위한 베이스 모델 학습 및 OOF 예측 저장."""
    import json as _json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold

    logger.info("fit_for_stacking: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    n_folds      = int(json_obj.get("n_folds", 5))
    algorithm    = json_obj.get("algorithm", "gbm").lower()
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    connect_h2o(h2o_host, h2o_port)
    import h2o
    from h2o.estimators import H2OGradientBoostingEstimator, H2ORandomForestEstimator

    X = df[feature_cols].fillna(0).astype(float_dtype)
    y = df[target_col]
    oof_preds = np.zeros(len(df))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    algo_cls = H2OGradientBoostingEstimator if algorithm == "gbm" else H2ORandomForestEstimator

    mojo_paths = []
    model_dir  = root_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        tr_df  = pd.concat([X_tr, y_tr], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)

        tr_h2o  = h2o.H2OFrame(tr_df)
        val_h2o = h2o.H2OFrame(val_df)
        tr_h2o[target_col]  = tr_h2o[target_col].asfactor()
        val_h2o[target_col] = val_h2o[target_col].asfactor()

        fold_model = algo_cls(ntrees=100, seed=fold_i)
        fold_model.train(x=feature_cols, y=target_col, training_frame=tr_h2o, validation_frame=val_h2o)

        preds = fold_model.predict(val_h2o).as_data_frame()
        oof_preds[val_idx] = preds.iloc[:, -1].values

        mojo_path = fold_model.save_mojo(str(model_dir))
        mojo_paths.append(mojo_path)

    df["oof_score"] = oof_preds
    oof_file = model_dir / f"{model_id}_oof.parquet"
    df[[target_col, "oof_score"]].to_parquet(oof_file, index=False)

    result = {
        "result": "ok", "model_id": model_id,
        "n_folds": n_folds, "mojo_paths": mojo_paths, "oof_file": str(oof_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def model_var_imp(model, feature_cols: list = None, top_n: int = 20) -> "pd.DataFrame":
    """H2O 모델 변수 중요도 반환 (h2o_model_executor 버전)."""
    import pandas as pd

    try:
        vi = model.varimp(use_pandas=True)
        if vi is not None:
            return vi.head(top_n)
    except Exception:
        pass
    return pd.DataFrame(columns=["variable", "importance", "percentage"])


def model_scoring_history(model) -> "pd.DataFrame":
    """H2O 모델 scoring history 반환 (h2o_model_executor 버전)."""
    import pandas as pd

    try:
        sh = model.scoring_history()
        return sh if sh is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def data_psm(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    root_dir, json_obj,
) -> dict:
    """성향 점수 매칭(PSM) — 처리군/대조군 매칭."""
    import json as _json
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("data_psm: model_id=%s", json_obj.get("model_id"))

    root_path      = Path(root_dir)
    model_id       = json_obj["model_id"]
    data_path      = json_obj["data_path"]
    treatment_col  = json_obj.get("treatment_col", "treatment")
    feature_cols   = json_obj.get("feature_cols")
    caliper        = float(json_obj.get("caliper", 0.1))
    float_dtype    = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / data_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != treatment_col]

    X = df[feature_cols].fillna(0).astype(float_dtype)
    y = df[treatment_col]

    # 로지스틱 회귀로 성향 점수 추정
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X, y)
    df["psm_score"] = lr.predict_proba(X)[:, 1]

    # 1:1 nearest-neighbor 매칭
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()

    matched_treat = []
    matched_ctrl  = []
    used_ctrl_idx = set()

    for _, t_row in treated.iterrows():
        ps_t = t_row["psm_score"]
        candidates = control[~control.index.isin(used_ctrl_idx)].copy()
        candidates["_dist"] = (candidates["psm_score"] - ps_t).abs()
        best = candidates.nsmallest(1, "_dist")
        if len(best) > 0 and best.iloc[0]["_dist"] <= caliper:
            matched_treat.append(t_row)
            matched_ctrl.append(best.iloc[0])
            used_ctrl_idx.add(best.index[0])

    matched_df = pd.concat(
        [pd.DataFrame(matched_treat), pd.DataFrame(matched_ctrl)], ignore_index=True
    )

    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_psm_matched.parquet"
    matched_df.to_parquet(output_file, index=False)

    result = {
        "result":     "ok",
        "model_id":   model_id,
        "n_matched":  len(matched_treat),
        "n_treated":  len(treated),
        "n_control":  len(control),
        "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def predict_contributions(
    model_id: str,
    h2o_host: str,
    h2o_port: int,
    pandas_df: "pd.DataFrame",
    var_layout_json: list,
) -> "pd.DataFrame":
    """SHAP contributions 산출 — connect_h2o(h2o_host, h2o_port) 사용."""
    import pandas as pd

    connect_h2o(h2o_host, h2o_port)
    import h2o

    var_layout = pd.DataFrame(var_layout_json) if isinstance(var_layout_json, list) else var_layout_json
    feature_cols = list(var_layout.get("VAR_NM", var_layout.iloc[:, 0]) if len(var_layout) > 0 else pandas_df.columns)

    try:
        model   = h2o.get_model(model_id)
        h2o_df  = h2o.H2OFrame(pandas_df[[c for c in feature_cols if c in pandas_df.columns]].fillna(0))
        contrib = model.predict_contributions(h2o_df).as_data_frame()
        return contrib
    except Exception as e:
        logger.warning("predict_contributions 실패: %s", e)
        return pd.DataFrame()


def get_glm_coef(
    model,
    feature_cols: list = None,
) -> "pd.DataFrame":
    """H2O GLM 계수 테이블 반환."""
    import pandas as pd

    try:
        coef_table = model.coef_norm()
        if isinstance(coef_table, dict):
            df = pd.DataFrame(list(coef_table.items()), columns=["variable", "standardized_coef"])
            return df
        return pd.DataFrame(coef_table)
    except Exception:
        pass

    if feature_cols and hasattr(model, "coef"):
        coef = model.coef()
        if isinstance(coef, dict):
            return pd.DataFrame(list(coef.items()), columns=["variable", "coef"])
    return pd.DataFrame(columns=["variable", "coef"])


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _save_h2o_result(model, json_obj: dict, root_path: "Path",
                     result_file_path_faf: str, done_file_path_faf: str) -> dict:
    """H2O 모델 학습 결과를 공통 포맷으로 저장."""
    import json as _json
    from pathlib import Path

    model_id  = json_obj["model_id"]
    model_dir = root_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # MOJO 저장
    try:
        mojo_path = model.save_mojo(str(model_dir))
    except Exception:
        mojo_path = ""

    # 성능 지표
    try:
        perf    = model.model_performance()
        metrics = {"auc": round(float(perf.auc()), 4), "logloss": round(float(perf.logloss()), 4)}
    except Exception:
        metrics = {}

    # 변수 중요도
    try:
        vi     = model.varimp(use_pandas=True)
        varimp = vi.set_index("variable")["percentage"].head(20).round(4).to_dict() if vi is not None else {}
    except Exception:
        varimp = {}

    meta = {
        "model_id":     model_id,
        "h2o_model_id": model.model_id,
        "mojo_path":    mojo_path,
        "metrics":      metrics,
        "varimp":       varimp,
        "feature_cols": json_obj.get("feature_cols", []),
        "target_col":   json_obj.get("target_col", ""),
    }
    meta_file = model_dir / f"{model_id}_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        _json.dump(meta, f, ensure_ascii=False)

    result = {"result": "ok", "model_id": model_id, "metrics": metrics, "mojo_path": mojo_path}
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()

    logger.info("_save_h2o_result: model_id=%s  metrics=%s", model_id, metrics)
    return result
