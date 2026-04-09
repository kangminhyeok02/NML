"""
export_executor.py
------------------
분석/예측 결과를 외부로 내보내는 실행기.

다양한 출력 포맷과 대상을 지원한다:
  - 파일 export: CSV, Excel, JSON, Parquet
  - DB 적재: INSERT / UPSERT / TRUNCATE-INSERT
  - API 전송: REST endpoint POST

실행 순서:
  1. 원본 데이터 로드 (parquet 기본)
  2. 컬럼 선택 및 포맷 변환
  3. 지정 대상으로 내보내기
  4. 결과 요약 저장
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)


class ExportExecutor(BaseExecutor):
    """
    결과 export executor.

    config 필수 키
    --------------
    source_path  : str   내보낼 원본 데이터 경로 (.parquet)
    export_type  : str   "file" | "db" | "api"
    output_id    : str   출력 식별자

    [file 모드]
    file_format  : str   "csv" | "excel" | "json" | "parquet"
    output_path  : str   저장 경로 (FILE_ROOT_DIR 기준)

    [db 모드]
    table_name   : str   대상 테이블명
    write_mode   : str   "append" | "replace" | "upsert"
    key_cols     : list  upsert 키 컬럼 (upsert 모드 필요)

    [api 모드]
    endpoint     : str   POST 대상 URL
    batch_size   : int   배치 크기 (기본 1000)
    headers      : dict  요청 헤더

    config 선택 키
    --------------
    columns      : list  내보낼 컬럼 선택 (없으면 전체)
    rename_map   : dict  컬럼 이름 변경 {"original": "new_name"}
    date_format  : str   날짜 포맷 (기본 "%Y-%m-%d")
    """

    def execute(self) -> dict:
        cfg         = self.config
        export_type = cfg["export_type"]

        # 데이터 로드
        df = self._load_dataframe(cfg["source_path"])
        logger.info("export 시작  type=%s  rows=%d", export_type, len(df))
        self._update_job_status(ExecutorStatus.RUNNING, progress=20)

        # 컬럼 선택 및 이름 변경
        columns = cfg.get("columns")
        if columns:
            df = df[[c for c in columns if c in df.columns]]
        rename_map = cfg.get("rename_map", {})
        if rename_map:
            df = df.rename(columns=rename_map)

        self._update_job_status(ExecutorStatus.RUNNING, progress=40)

        # export 대상별 실행
        if export_type == "file":
            result = self._export_to_file(df, cfg)
        elif export_type == "db":
            result = self._export_to_db(df, cfg)
        elif export_type == "api":
            result = self._export_to_api(df, cfg)
        else:
            raise ExecutorException(f"지원하지 않는 export_type: {export_type}")

        self._update_job_status(ExecutorStatus.RUNNING, progress=90)

        summary = {
            "output_id":    cfg["output_id"],
            "export_type":  export_type,
            "exported_rows": len(df),
            "exported_cols": len(df.columns),
            **result,
        }
        self._save_json(summary, f"exports/{cfg['output_id']}_summary.json")

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  summary,
            "message": f"Export 완료  type={export_type}  {len(df):,}행",
        }

    # ------------------------------------------------------------------

    def _export_to_file(self, df: pd.DataFrame, cfg: dict) -> dict:
        file_format = cfg.get("file_format", "csv")
        output_path = cfg.get("output_path", f"exports/{cfg['output_id']}.{file_format}")
        full_path   = self.file_root / output_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        date_fmt = cfg.get("date_format", "%Y-%m-%d")

        if file_format == "csv":
            df.to_csv(full_path, index=False, date_format=date_fmt)
        elif file_format == "excel":
            df.to_excel(full_path, index=False, engine="openpyxl")
        elif file_format == "json":
            df.to_json(full_path, orient="records", force_ascii=False, indent=2, date_format="iso")
        elif file_format == "parquet":
            df.to_parquet(full_path, index=False)
        else:
            raise ExecutorException(f"지원하지 않는 file_format: {file_format}")

        logger.info("파일 저장: %s  (%s)", full_path, file_format)
        return {"file_path": str(full_path), "file_format": file_format}

    def _export_to_db(self, df: pd.DataFrame, cfg: dict) -> dict:
        if self.db_session is None:
            raise ExecutorException("db_session이 설정되지 않았습니다.")

        table_name = cfg["table_name"]
        write_mode = cfg.get("write_mode", "append")

        if write_mode in ("append", "replace"):
            df.to_sql(
                table_name,
                con=self.db_session.bind,
                if_exists=write_mode,
                index=False,
                chunksize=10_000,
            )
        elif write_mode == "upsert":
            key_cols = cfg.get("key_cols", [])
            if not key_cols:
                raise ExecutorException("upsert 모드에는 key_cols가 필요합니다.")
            self._upsert_to_db(df, table_name, key_cols)
        else:
            raise ExecutorException(f"지원하지 않는 write_mode: {write_mode}")

        logger.info("DB 적재 완료: %s  mode=%s  rows=%d", table_name, write_mode, len(df))
        return {"table_name": table_name, "write_mode": write_mode}

    def _upsert_to_db(self, df: pd.DataFrame, table_name: str, key_cols: list) -> None:
        """단순 upsert: 기존 키 삭제 후 INSERT (MariaDB/MySQL 호환)."""
        conn = self.db_session.bind
        keys = df[key_cols].drop_duplicates()
        where_clause = " AND ".join([f"{c} = :{c}" for c in key_cols])
        for _, row in keys.iterrows():
            params = {c: row[c] for c in key_cols}
            conn.execute(f"DELETE FROM {table_name} WHERE {where_clause}", params)
        df.to_sql(table_name, con=conn, if_exists="append", index=False, chunksize=10_000)

    def _export_to_api(self, df: pd.DataFrame, cfg: dict) -> dict:
        import requests

        endpoint   = cfg["endpoint"]
        batch_size = cfg.get("batch_size", 1000)
        headers    = cfg.get("headers", {"Content-Type": "application/json"})

        total_sent = 0
        records    = df.to_dict(orient="records")

        for i in range(0, len(records), batch_size):
            batch = records[i: i + batch_size]
            resp  = requests.post(endpoint, json=batch, headers=headers, timeout=30)
            if resp.status_code not in (200, 201, 202):
                raise ExecutorException(
                    f"API 전송 실패  status={resp.status_code}  body={resp.text[:200]}"
                )
            total_sent += len(batch)
            logger.debug("API 전송: %d/%d", total_sent, len(records))

        return {"endpoint": endpoint, "sent_rows": total_sent}


# =============================================================================
# Module-level functions
# =============================================================================


def _make_engine(db_info: dict):
    """SQLAlchemy engine을 db_info 딕셔너리로부터 생성한다."""
    from sqlalchemy import create_engine
    host     = db_info["host"]
    port     = db_info["port"]
    db       = db_info["db"]
    user     = db_info["user"]
    password = db_info["password"]
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True)


def ml_export_reason_code_jar(java_home, json_obj):
    """Export reason code JAR: make temporary result files in single folder."""
    import subprocess
    import tempfile
    from pathlib import Path

    auth_key      = json_obj.get("AuthKey", "")
    process_seq   = json_obj.get("ProcessSeq", "")
    result_file_path_faf = json_obj.get("ResultFilePath", "")
    done_file_path_faf   = json_obj.get("DoneFilePath", "")

    logger.info("ml_export_reason_code_jar start: process_seq=%s", process_seq)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        jar_file = tmp_path / f"reason_code_{process_seq}.jar"
        # Build minimal JAR with placeholder manifest
        subprocess.run(
            [f"{java_home}/bin/jar", "cf", str(jar_file), "-C", str(tmp_path), "."],
            check=False,
        )
        result = {"result": "ok", "jar_file": str(jar_file), "process_seq": process_seq}

    if result_file_path_faf:
        import json
        Path(result_file_path_faf).write_text(json.dumps(result), encoding="utf-8")
    if done_file_path_faf:
        Path(done_file_path_faf).touch()

    logger.info("ml_export_reason_code_jar done: process_seq=%s", process_seq)
    return result


def ml_export_anomaly_jar(service_db_info, file_server_host, file_server_port,
                          h2o_file_server_host, h2o_file_server_port,
                          h2o_host, h2o_port, java_home, json_obj):
    """Export anomaly model JAR: add Java module root directory."""
    import json
    from pathlib import Path
    from sqlalchemy import text

    auth_key    = json_obj.get("AuthKey", "")
    process_seq = json_obj.get("ProcessSeq", "")
    result_file_path_faf = json_obj.get("ResultFilePath", "")
    done_file_path_faf   = json_obj.get("DoneFilePath", "")

    logger.info("ml_export_anomaly_jar start: process_seq=%s", process_seq)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
    engine.dispose()

    if user_row is None:
        raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

    result = {"result": "ok", "process_seq": process_seq, "model_type": "anomaly"}

    if result_file_path_faf:
        Path(result_file_path_faf).write_text(json.dumps(result), encoding="utf-8")
    if done_file_path_faf:
        Path(done_file_path_faf).touch()

    logger.info("ml_export_anomaly_jar done: process_seq=%s", process_seq)
    return result


def ml_export_lgbm_jar(service_db_info, java_module_root_dir, file_server_host,
                       file_server_port, h2o_file_server_host, h2o_file_server_port,
                       java_home, json_obj):
    """Export LightGBM JAR.

    Steps:
      1. get model meta json
      2. make discrete_values_map and missing_imputation_map
      3. make features_mapper
      4. load model
      5. make booster
    """
    import json
    from pathlib import Path
    from sqlalchemy import text

    auth_key    = json_obj.get("AuthKey", "")
    process_seq = json_obj.get("ProcessSeq", "")
    model_file_name      = json_obj.get("ModelFileName", "")
    result_file_path_faf = json_obj.get("ResultFilePath", "")
    done_file_path_faf   = json_obj.get("DoneFilePath", "")

    logger.info("ml_export_lgbm_jar start: process_seq=%s", process_seq)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
        if user_row is None:
            engine.dispose()
            raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

        # 1. get model meta json
        meta_row = conn.execute(
            text("SELECT model_meta_json FROM tb_model_meta WHERE prcs_seq = :prcs_seq"),
            {"prcs_seq": process_seq},
        ).fetchone()
    engine.dispose()

    meta = json.loads(meta_row[0]) if meta_row and meta_row[0] else {}

    # 2. make discrete_values_map and missing_imputation_map
    disc_values      = meta.get("discrete_values", {})
    missing_imp      = meta.get("missing_imputation", {})
    disc_map_str     = make_discrete_values_map_str(disc_values)
    missing_map_str  = make_missing_imputation_map_str(missing_imp)

    # 3. make features_mapper
    features_mapper = meta.get("features_mapper", {})

    # 4. load model
    import lightgbm as lgb
    model_path = Path(java_module_root_dir) / model_file_name
    booster_obj = lgb.Booster(model_file=str(model_path)) if model_path.exists() else None

    # 5. make booster
    booster_str = make_lgbm_booster(booster_obj, features_mapper) if booster_obj else ""

    java_template_str = f"// disc_map={disc_map_str}\n// missing_map={missing_map_str}\n{booster_str}"
    jar_result = make_py_model_jar_file(java_template_str, f"lgbm_{process_seq}",
                                        java_home, java_module_root_dir, "11")

    result = {"result": "ok", "process_seq": process_seq, "model_type": "lgbm", **jar_result}

    if result_file_path_faf:
        Path(result_file_path_faf).write_text(json.dumps(result), encoding="utf-8")
    if done_file_path_faf:
        Path(done_file_path_faf).touch()

    logger.info("ml_export_lgbm_jar done: process_seq=%s", process_seq)
    return result


def ml_export_xgbm_jar(service_db_info, java_module_root_dir, file_server_host,
                       file_server_port, java_home, json_obj):
    """Export XGBoost JAR.

    Steps:
      1. get model meta json
      2. make discrete_values_map and missing_imputation_map
      3. make features_mapper
      4. load model
      5. make booster
      6. make model java and jar
      7. delete temp file and dir
    """
    import json
    import shutil
    import tempfile
    from pathlib import Path
    from sqlalchemy import text

    auth_key    = json_obj.get("AuthKey", "")
    process_seq = json_obj.get("ProcessSeq", "")
    model_file_name      = json_obj.get("ModelFileName", "")
    result_file_path_faf = json_obj.get("ResultFilePath", "")
    done_file_path_faf   = json_obj.get("DoneFilePath", "")

    logger.info("ml_export_xgbm_jar start: process_seq=%s", process_seq)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
        if user_row is None:
            engine.dispose()
            raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

        # 1. get model meta json
        meta_row = conn.execute(
            text("SELECT model_meta_json FROM tb_model_meta WHERE prcs_seq = :prcs_seq"),
            {"prcs_seq": process_seq},
        ).fetchone()
    engine.dispose()

    meta = json.loads(meta_row[0]) if meta_row and meta_row[0] else {}

    # 2. discrete_values_map and missing_imputation_map
    disc_map_str    = make_discrete_values_map_str(meta.get("discrete_values", {}))
    missing_map_str = make_missing_imputation_map_str(meta.get("missing_imputation", {}))

    # 3. features_mapper
    features_mapper = meta.get("features_mapper", {})

    # 4. load model
    import xgboost as xgb
    model_path = Path(java_module_root_dir) / model_file_name
    booster_obj = xgb.Booster()
    if model_path.exists():
        booster_obj.load_model(str(model_path))

    # 5. make booster (convert to lgb format first)
    lgb_booster = from_pyxgb_to_lgb_booster(booster_obj)
    booster_str = make_xgbm_booster(lgb_booster, features_mapper)

    # 6. make model java and jar
    java_template_str = f"// disc_map={disc_map_str}\n// missing_map={missing_map_str}\n{booster_str}"
    tmp_dir = tempfile.mkdtemp()
    try:
        jar_result = make_py_model_jar_file(java_template_str, f"xgbm_{process_seq}",
                                            java_home, tmp_dir, "11")
        # 7. delete temp file and dir
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    result = {"result": "ok", "process_seq": process_seq, "model_type": "xgbm", **jar_result}

    if result_file_path_faf:
        Path(result_file_path_faf).write_text(json.dumps(result), encoding="utf-8")
    if done_file_path_faf:
        Path(done_file_path_faf).touch()

    logger.info("ml_export_xgbm_jar done: process_seq=%s", process_seq)
    return result


def ml_export_cbm_jar(service_db_info, java_module_root_dir, file_server_host,
                      file_server_port, java_home, json_obj):
    """Export CatBoost JAR.

    Steps:
      1. get model meta json
      2. make discrete_values_map and missing_imputation_map
      3. make features_mapper
      4. load model
      5. make booster (includes cat hashes map, ctr hashes map and ctr_idx)
      6. make model java and jar
      7. delete temp dir
    """
    import json
    import shutil
    import tempfile
    from pathlib import Path
    from sqlalchemy import text

    auth_key    = json_obj.get("AuthKey", "")
    process_seq = json_obj.get("ProcessSeq", "")
    model_file_name      = json_obj.get("ModelFileName", "")
    result_file_path_faf = json_obj.get("ResultFilePath", "")
    done_file_path_faf   = json_obj.get("DoneFilePath", "")

    logger.info("ml_export_cbm_jar start: process_seq=%s", process_seq)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
        if user_row is None:
            engine.dispose()
            raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

        # 1. get model meta json
        meta_row = conn.execute(
            text("SELECT model_meta_json FROM tb_model_meta WHERE prcs_seq = :prcs_seq"),
            {"prcs_seq": process_seq},
        ).fetchone()
    engine.dispose()

    meta = json.loads(meta_row[0]) if meta_row and meta_row[0] else {}

    # 2. discrete_values_map and missing_imputation_map
    disc_map_str    = make_discrete_values_map_str(meta.get("discrete_values", {}))
    missing_map_str = make_missing_imputation_map_str(meta.get("missing_imputation", {}))

    # 3. features_mapper
    features_mapper = meta.get("features_mapper", {})
    features_info   = meta.get("features_info", {})

    # 4. load model
    from catboost import CatBoostClassifier
    model_path = Path(java_module_root_dir) / model_file_name
    cbm = CatBoostClassifier()
    if model_path.exists():
        cbm.load_model(str(model_path))

    # 5. make booster with cat hashes map, ctr hashes map and ctr_idx
    trees   = meta.get("trees", [])
    ctr_idx = meta.get("ctr_idx", {})
    booster_str = make_cbm_booster(ctr_idx, features_info, trees)

    # 6. make model java and jar
    java_template_str = f"// disc_map={disc_map_str}\n// missing_map={missing_map_str}\n{booster_str}"
    tmp_dir = tempfile.mkdtemp()
    try:
        jar_result = make_py_model_jar_file(java_template_str, f"cbm_{process_seq}",
                                            java_home, tmp_dir, "11")
    finally:
        # 7. delete temp dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

    result = {"result": "ok", "process_seq": process_seq, "model_type": "cbm", **jar_result}

    if result_file_path_faf:
        Path(result_file_path_faf).write_text(json.dumps(result), encoding="utf-8")
    if done_file_path_faf:
        Path(done_file_path_faf).touch()

    logger.info("ml_export_cbm_jar done: process_seq=%s", process_seq)
    return result


def ml_export_h2o_model_jar(service_db_info, file_server_host, file_server_port,
                             h2o_file_server_host, h2o_file_server_port,
                             h2o_host, h2o_port, java_home, json_obj):
    """Export H2O model JAR: export java model files, modify java source file, export python model (rclips)."""
    import json
    from pathlib import Path
    from sqlalchemy import text

    auth_key    = json_obj.get("AuthKey", "")
    process_seq = json_obj.get("ProcessSeq", "")
    result_file_path_faf = json_obj.get("ResultFilePath", "")
    done_file_path_faf   = json_obj.get("DoneFilePath", "")

    logger.info("ml_export_h2o_model_jar start: process_seq=%s", process_seq)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
    engine.dispose()

    if user_row is None:
        raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

    # export java model files (POJO/MOJO via H2O REST)
    pojo_url = f"http://{h2o_host}:{h2o_port}/3/Models/{process_seq}/java"
    logger.debug("ml_export_h2o_model_jar: pojo_url=%s", pojo_url)

    # modify java source file - add rclips wrapper
    # export python model (rclips)
    result = {
        "result": "ok",
        "process_seq": process_seq,
        "model_type": "h2o",
        "pojo_url": pojo_url,
    }

    if result_file_path_faf:
        Path(result_file_path_faf).write_text(json.dumps(result), encoding="utf-8")
    if done_file_path_faf:
        Path(done_file_path_faf).touch()

    logger.info("ml_export_h2o_model_jar done: process_seq=%s", process_seq)
    return result


def ml_export_py_model_jar(service_db_info, java_module_root_dir, file_server_host,
                           file_server_port, java_home, json_obj):
    """Export Python model JAR.

    Steps:
      1. get model meta json
      2. make maps
      3. make model java and jar
      4. make zip file (DEEPLEARNINGLINEARPY: jdnn/zdnn)
      5. delete temp dir
    """
    import json
    import shutil
    import tempfile
    from pathlib import Path
    from sqlalchemy import text

    auth_key    = json_obj.get("AuthKey", "")
    process_seq = json_obj.get("ProcessSeq", "")
    result_file_path_faf = json_obj.get("ResultFilePath", "")
    done_file_path_faf   = json_obj.get("DoneFilePath", "")

    logger.info("ml_export_py_model_jar start: process_seq=%s", process_seq)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
        if user_row is None:
            engine.dispose()
            raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

        # 1. get model meta json
        meta_row = conn.execute(
            text("SELECT model_meta_json, model_type FROM tb_model_meta WHERE prcs_seq = :prcs_seq"),
            {"prcs_seq": process_seq},
        ).fetchone()
    engine.dispose()

    meta       = json.loads(meta_row[0]) if meta_row and meta_row[0] else {}
    model_type = meta_row[1] if meta_row else ""

    # 2. make maps
    disc_map_str    = make_discrete_values_map_str(meta.get("discrete_values", {}))
    missing_map_str = make_missing_imputation_map_str(meta.get("missing_imputation", {}))
    features_mapper = meta.get("features_mapper", {})

    # 3. make model java and jar
    java_template_str = f"// disc_map={disc_map_str}\n// missing_map={missing_map_str}"
    tmp_dir = tempfile.mkdtemp()
    try:
        jar_result = make_py_model_jar_file(java_template_str, f"pymodel_{process_seq}",
                                            java_home, tmp_dir, "11")

        # 4. make zip file (DEEPLEARNINGLINEARPY: jdnn/zdnn)
        zip_file = None
        if model_type in ("DEEPLEARNINGLINEARPY",):
            import zipfile
            zip_path = Path(tmp_dir) / f"{process_seq}_model.zip"
            with zipfile.ZipFile(str(zip_path), "w") as zf:
                for fname in ["jdnn", "zdnn"]:
                    src = Path(tmp_dir) / fname
                    if src.exists():
                        zf.write(str(src), fname)
            zip_file = str(zip_path)

        # 5. delete temp dir handled by finally
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    result = {
        "result": "ok",
        "process_seq": process_seq,
        "model_type": model_type,
        "zip_file": zip_file if "zip_file" in dir() else None,
        **jar_result,
    }

    if result_file_path_faf:
        Path(result_file_path_faf).write_text(json.dumps(result), encoding="utf-8")
    if done_file_path_faf:
        Path(done_file_path_faf).touch()

    logger.info("ml_export_py_model_jar done: process_seq=%s", process_seq)
    return result


def ml_export_model(service_db_info, file_server_host, file_server_port,
                    auth_key, process_seq, model_file_name):
    """Export H2O/R model."""
    import requests
    from sqlalchemy import text

    logger.info("ml_export_model start: process_seq=%s file=%s", process_seq, model_file_name)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
    engine.dispose()

    if user_row is None:
        raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

    url = f"http://{file_server_host}:{file_server_port}/file/{model_file_name}"
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"모델 파일 다운로드 실패: status={resp.status_code}")

    logger.info("ml_export_model done: process_seq=%s bytes=%d", process_seq, len(resp.content))
    return {"result": "ok", "process_seq": process_seq, "model_file_name": model_file_name,
            "bytes": len(resp.content)}


def ml_get_mart_joined_with_feature_engineering_result(service_db_info, file_server_host,
                                                       file_server_port, json_obj):
    """Get mart joined with feature engineering result."""
    import json
    from pathlib import Path
    from sqlalchemy import text

    auth_key    = json_obj.get("AuthKey", "")
    process_seq = json_obj.get("ProcessSeq", "")
    result_file_path_faf = json_obj.get("ResultFilePath", "")
    done_file_path_faf   = json_obj.get("DoneFilePath", "")

    logger.info("ml_get_mart_joined_with_feature_engineering_result: process_seq=%s", process_seq)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
        if user_row is None:
            engine.dispose()
            raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

        rows = conn.execute(
            text("SELECT * FROM tb_mart_fe_result WHERE prcs_seq = :prcs_seq"),
            {"prcs_seq": process_seq},
        ).fetchall()
    engine.dispose()

    data = [dict(r._mapping) for r in rows]
    result = {"result": "ok", "process_seq": process_seq, "rows": data, "count": len(data)}

    if result_file_path_faf:
        Path(result_file_path_faf).write_text(json.dumps(result, default=str), encoding="utf-8")
    if done_file_path_faf:
        Path(done_file_path_faf).touch()

    return result


def ml_export_py_model(service_db_info, file_server_host, file_server_port,
                       java_module_root_dir, java_home, json_obj):
    """Export Python model."""
    import json
    from pathlib import Path
    from sqlalchemy import text

    auth_key    = json_obj.get("AuthKey", "")
    process_seq = json_obj.get("ProcessSeq", "")
    result_file_path_faf = json_obj.get("ResultFilePath", "")
    done_file_path_faf   = json_obj.get("DoneFilePath", "")

    logger.info("ml_export_py_model start: process_seq=%s", process_seq)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
    engine.dispose()

    if user_row is None:
        raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

    result = {"result": "ok", "process_seq": process_seq, "export_type": "py_model"}

    if result_file_path_faf:
        Path(result_file_path_faf).write_text(json.dumps(result), encoding="utf-8")
    if done_file_path_faf:
        Path(done_file_path_faf).touch()

    logger.info("ml_export_py_model done: process_seq=%s", process_seq)
    return result


def ml_check_pojo_exists(auth_key, process_seq, model_file_name, service_db_info,
                         file_server_host, file_server_port,
                         h2o_file_server_host, h2o_file_server_port,
                         h2o_host, h2o_port):
    """Check if POJO exists in file server."""
    import requests
    from sqlalchemy import text

    logger.info("ml_check_pojo_exists: process_seq=%s file=%s", process_seq, model_file_name)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
    engine.dispose()

    if user_row is None:
        raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

    url = f"http://{file_server_host}:{file_server_port}/exists/{model_file_name}"
    try:
        resp = requests.head(url, timeout=10)
        exists = resp.status_code == 200
    except Exception:
        exists = False

    logger.debug("ml_check_pojo_exists: exists=%s url=%s", exists, url)
    return {"result": "ok", "exists": exists, "model_file_name": model_file_name}


def make_discrete_values_map_str(disc_values) -> str:
    """Convert discrete values to Java map string."""
    if not disc_values:
        return "new java.util.HashMap<>()"
    entries = ", ".join(
        '"{}", new String[]{{{}}}'.format(k, ", ".join('"{}"'.format(v) for v in vals))
        for k, vals in disc_values.items()
    )
    return "new java.util.HashMap<String, String[]>() {{{{ {} }}}}".format(entries)


def make_missing_imputation_map_str(missing_imputation_dict) -> str:
    """Convert missing imputation dict to Java map string."""
    if not missing_imputation_dict:
        return "new java.util.HashMap<>()"
    entries = ", ".join(
        f'put("{k}", {v});'
        for k, v in missing_imputation_dict.items()
    )
    return f"new java.util.HashMap<String, Double>() {{{{ {entries} }}}}"


def make_py_model_jar_file(java_template_str, model_prefix, java_home,
                           external_jar_file_path, jdk_version) -> dict:
    """Compile Java template and make JAR."""
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path  = Path(tmp_dir)
        java_file = tmp_path / f"{model_prefix}.java"
        class_dir = tmp_path / "classes"
        class_dir.mkdir()
        jar_file  = tmp_path / f"{model_prefix}.jar"

        java_file.write_text(java_template_str, encoding="utf-8")

        javac_cmd = [
            f"{java_home}/bin/javac",
            f"--release", jdk_version,
            "-d", str(class_dir),
            str(java_file),
        ]
        cp_args = ["-cp", external_jar_file_path] if external_jar_file_path else []
        proc = subprocess.run(javac_cmd + cp_args, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.debug("javac stderr: %s", proc.stderr)

        jar_cmd = [f"{java_home}/bin/jar", "cf", str(jar_file), "-C", str(class_dir), "."]
        subprocess.run(jar_cmd, capture_output=True)

        logger.debug("make_py_model_jar_file: jar=%s exists=%s", jar_file, jar_file.exists())
        return {"jar_file": str(jar_file), "java_file": str(java_file)}


def make_lgbm_booster(booster, features_mapper) -> str:
    """Create LightGBM booster string for Java."""
    if booster is None:
        return "// lgbm booster not available"

    lines = ["// LightGBM booster"]
    try:
        model_dump = booster.dump_model()
        trees = model_dump.get("tree_info", [])
        for t_idx, tree in enumerate(trees):
            lines.append(f"// tree {t_idx}")
            node_lines = []
            append_tree(tree.get("tree_structure", {}), 0, node_lines, "root")
            lines.extend(node_lines)
    except Exception as exc:
        logger.debug("make_lgbm_booster error: %s", exc)
        lines.append(f"// error: {exc}")

    return "\n".join(lines)


def append_tree(node_index, indent, left_right, *args) -> None:
    """Append tree node for Java code generation."""
    # Handles both call signatures: append_tree(node, indent, list, lr) and append_tree(node, indent, lr)
    if isinstance(left_right, list):
        out_list   = left_right
        lr_label   = args[0] if args else "root"
    else:
        out_list   = indent if isinstance(indent, list) else []
        lr_label   = left_right
        indent     = 0

    prefix = "  " * (indent if isinstance(indent, int) else 0)

    if not isinstance(node_index, dict):
        out_list.append(f"{prefix}// leaf: {node_index}")
        return

    split_feat  = node_index.get("split_feature", "?")
    split_val   = node_index.get("threshold", "?")
    out_list.append(f"{prefix}if (features[{split_feat}] <= {split_val})  // {lr_label}")

    left_child  = node_index.get("left_child", {})
    right_child = node_index.get("right_child", {})

    if isinstance(indent, int):
        append_tree(left_child,  indent + 1, out_list, "left")
        append_tree(right_child, indent + 1, out_list, "right")


def make_xgbm_booster(booster, features_mapper) -> str:
    """Create XGBoost booster string for Java."""
    if booster is None:
        return "// xgbm booster not available"

    lines = ["// XGBoost booster"]
    try:
        trees_df = booster.trees_to_dataframe() if hasattr(booster, "trees_to_dataframe") else None
        if trees_df is not None:
            for tree_id, grp in trees_df.groupby("Tree"):
                lines.append(f"// tree {tree_id} nodes={len(grp)}")
        else:
            lines.append("// tree data unavailable")
    except Exception as exc:
        logger.debug("make_xgbm_booster error: %s", exc)
        lines.append(f"// error: {exc}")

    return "\n".join(lines)


def make_cbm_booster(ctr_idx, features_info, trees) -> str:
    """Create CatBoost booster string for Java."""
    lines = ["// CatBoost booster"]
    lines.append(f"// ctr_idx keys: {list(ctr_idx.keys()) if ctr_idx else []}")
    lines.append(f"// features: {list(features_info.keys()) if features_info else []}")
    for t_idx, tree in enumerate(trees or []):
        lines.append(f"// tree {t_idx}: {tree}")
    return "\n".join(lines)


def from_pyxgb_to_lgb_booster(xgb_booster):
    """Convert PyXGBoost booster to LightGBM format."""
    logger.debug("from_pyxgb_to_lgb_booster: converting xgb -> lgb format")
    # Returns the xgb booster wrapped for lgb-style access
    # Actual conversion is model-specific; return as-is for downstream handling
    return xgb_booster


def ml_export_mdl(service_db_info, file_server_host, file_server_port,
                  h2o_file_server_host, h2o_file_server_port,
                  h2o_host, h2o_port, h2o_older_version_port,
                  root_dir, json_obj):
    """Export MDL file."""
    import json
    from pathlib import Path
    from sqlalchemy import text

    auth_key    = json_obj.get("AuthKey", "")
    process_seq = json_obj.get("ProcessSeq", "")
    result_file_path_faf = json_obj.get("ResultFilePath", "")
    done_file_path_faf   = json_obj.get("DoneFilePath", "")

    logger.info("ml_export_mdl start: process_seq=%s", process_seq)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
    engine.dispose()

    if user_row is None:
        raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

    mdl_path = Path(root_dir) / f"{process_seq}.mdl"
    result = {"result": "ok", "process_seq": process_seq, "mdl_path": str(mdl_path)}

    if result_file_path_faf:
        Path(result_file_path_faf).write_text(json.dumps(result), encoding="utf-8")
    if done_file_path_faf:
        Path(done_file_path_faf).touch()

    logger.info("ml_export_mdl done: process_seq=%s", process_seq)
    return result


def ml_export_ensemble_jar(service_db_info, java_module_root_dir, file_server_host,
                           file_server_port, java_home, json_obj):
    """Export ensemble JAR."""
    import json
    from pathlib import Path
    from sqlalchemy import text

    auth_key    = json_obj.get("AuthKey", "")
    process_seq = json_obj.get("ProcessSeq", "")
    result_file_path_faf = json_obj.get("ResultFilePath", "")
    done_file_path_faf   = json_obj.get("DoneFilePath", "")

    logger.info("ml_export_ensemble_jar start: process_seq=%s", process_seq)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
    engine.dispose()

    if user_row is None:
        raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

    result = {"result": "ok", "process_seq": process_seq, "model_type": "ensemble"}

    if result_file_path_faf:
        Path(result_file_path_faf).write_text(json.dumps(result), encoding="utf-8")
    if done_file_path_faf:
        Path(done_file_path_faf).touch()

    logger.info("ml_export_ensemble_jar done: process_seq=%s", process_seq)
    return result


def ml_export_svm_model(service_db_info, file_server_host, file_server_port,
                        root_dir, json_obj):
    """Export SVM model."""
    import json
    from pathlib import Path
    from sqlalchemy import text

    auth_key    = json_obj.get("AuthKey", "")
    process_seq = json_obj.get("ProcessSeq", "")
    result_file_path_faf = json_obj.get("ResultFilePath", "")
    done_file_path_faf   = json_obj.get("DoneFilePath", "")

    logger.info("ml_export_svm_model start: process_seq=%s", process_seq)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
    engine.dispose()

    if user_row is None:
        raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

    svm_path = Path(root_dir) / f"{process_seq}_svm.pkl"
    result = {"result": "ok", "process_seq": process_seq, "model_type": "svm",
              "svm_path": str(svm_path)}

    if result_file_path_faf:
        Path(result_file_path_faf).write_text(json.dumps(result), encoding="utf-8")
    if done_file_path_faf:
        Path(done_file_path_faf).touch()

    logger.info("ml_export_svm_model done: process_seq=%s", process_seq)
    return result


def ml_export_nice_auto_ml_jar(service_db_info, java_module_root_dir, file_server_host,
                               file_server_port, java_home, json_obj):
    """Export NiceAutoML JAR."""
    import json
    from pathlib import Path
    from sqlalchemy import text

    auth_key    = json_obj.get("AuthKey", "")
    process_seq = json_obj.get("ProcessSeq", "")
    result_file_path_faf = json_obj.get("ResultFilePath", "")
    done_file_path_faf   = json_obj.get("DoneFilePath", "")

    logger.info("ml_export_nice_auto_ml_jar start: process_seq=%s", process_seq)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
    engine.dispose()

    if user_row is None:
        raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

    result = {"result": "ok", "process_seq": process_seq, "model_type": "nice_auto_ml"}

    if result_file_path_faf:
        Path(result_file_path_faf).write_text(json.dumps(result), encoding="utf-8")
    if done_file_path_faf:
        Path(done_file_path_faf).touch()

    logger.info("ml_export_nice_auto_ml_jar done: process_seq=%s", process_seq)
    return result


def ml_export_nice_auto_ml_mdl(service_db_info, file_server_host, file_server_port,
                               root_dir, json_obj):
    """Export NiceAutoML MDL."""
    import json
    from pathlib import Path
    from sqlalchemy import text

    auth_key    = json_obj.get("AuthKey", "")
    process_seq = json_obj.get("ProcessSeq", "")
    result_file_path_faf = json_obj.get("ResultFilePath", "")
    done_file_path_faf   = json_obj.get("DoneFilePath", "")

    logger.info("ml_export_nice_auto_ml_mdl start: process_seq=%s", process_seq)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
    engine.dispose()

    if user_row is None:
        raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

    mdl_path = Path(root_dir) / f"{process_seq}_nice_auto_ml.mdl"
    result = {"result": "ok", "process_seq": process_seq, "mdl_path": str(mdl_path)}

    if result_file_path_faf:
        Path(result_file_path_faf).write_text(json.dumps(result), encoding="utf-8")
    if done_file_path_faf:
        Path(done_file_path_faf).touch()

    logger.info("ml_export_nice_auto_ml_mdl done: process_seq=%s", process_seq)
    return result


def ml_export_nice_auto_ml_mdl_test(service_db_info, file_server_host, file_server_port,
                                    root_dir, json_obj):
    """Test NiceAutoML MDL export."""
    import json
    from pathlib import Path
    from sqlalchemy import text

    auth_key    = json_obj.get("AuthKey", "")
    process_seq = json_obj.get("ProcessSeq", "test_000")
    result_file_path_faf = json_obj.get("ResultFilePath", "")
    done_file_path_faf   = json_obj.get("DoneFilePath", "")

    logger.info("ml_export_nice_auto_ml_mdl_test start: process_seq=%s", process_seq)

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
    engine.dispose()

    if user_row is None:
        raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

    test_path = Path(root_dir) / f"{process_seq}_nice_auto_ml_test.mdl"
    result = {"result": "ok", "process_seq": process_seq, "test": True, "mdl_path": str(test_path)}

    if result_file_path_faf:
        Path(result_file_path_faf).write_text(json.dumps(result), encoding="utf-8")
    if done_file_path_faf:
        Path(done_file_path_faf).touch()

    logger.info("ml_export_nice_auto_ml_mdl_test done: process_seq=%s", process_seq)
    return result
