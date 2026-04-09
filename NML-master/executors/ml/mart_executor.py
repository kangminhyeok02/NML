"""
mart_executor.py
----------------
모델링용 데이터 마트(mart) 생성 및 적재 실행기.

원천 DB 테이블 또는 파일에서 데이터를 읽어 feature engineering을 수행하고,
분석/모델 입력용 마트 데이터셋을 생성하여 File Server 또는 DB에 저장한다.

실행 순서:
  1. 원천 데이터 조회 (DB SQL 또는 파일)
  2. 타입 변환 / 결측 처리 / 이상값 클리핑
  3. 파생 변수 생성
  4. 학습/검증/예측 분리 (선택적)
  5. 마트 파일 저장 및 DB 메타 등록
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)


class MartExecutor(BaseExecutor):
    """
    데이터 마트 생성 executor.

    config 필수 키
    --------------
    source_query  : str   원천 데이터 SQL (db_session 필요) 또는 source_path 사용
    source_path   : str   원천 파일 상대 경로 (source_query 없을 때)
    target_id     : str   생성할 마트 식별자 (파일명 / DB 키)
    target_path   : str   저장 경로 (예: "mart/{target_id}.parquet")
    feature_rules : list  파생 변수 생성 규칙 목록 (선택)
    split         : dict  {"train": 0.7, "valid": 0.15, "test": 0.15} (선택)
    target_col    : str   타깃 컬럼명 (선택)
    """

    def execute(self) -> dict:
        cfg = self.config

        # 1. 원천 데이터 로드
        df = self._load_source(cfg)
        logger.info("source loaded  shape=%s", df.shape)
        self._update_job_status(ExecutorStatus.RUNNING, progress=20)

        # 2. 기본 전처리
        df = self._basic_preprocess(df, cfg)
        self._update_job_status(ExecutorStatus.RUNNING, progress=50)

        # 3. 파생 변수 생성
        feature_rules = cfg.get("feature_rules", [])
        if feature_rules:
            df = self._create_derived_features(df, feature_rules)
        self._update_job_status(ExecutorStatus.RUNNING, progress=70)

        # 4. 분할 (선택)
        split_cfg = cfg.get("split")
        saved_paths: dict[str, str] = {}
        if split_cfg:
            splits = self._split_dataframe(df, split_cfg, cfg.get("target_col"))
            for split_name, split_df in splits.items():
                path = cfg.get("target_path", "mart/{target_id}_{split}.parquet").format(
                    target_id=cfg["target_id"], split=split_name
                )
                saved_paths[split_name] = self._save_dataframe(split_df, path)
        else:
            path = cfg.get("target_path", f"mart/{cfg['target_id']}.parquet")
            saved_paths["full"] = self._save_dataframe(df, path)

        self._update_job_status(ExecutorStatus.RUNNING, progress=90)

        # 5. 메타 정보 저장
        meta = {
            "target_id":   cfg["target_id"],
            "shape":       list(df.shape),
            "columns":     list(df.columns),
            "saved_paths": saved_paths,
            "dtypes":      {c: str(t) for c, t in df.dtypes.items()},
        }
        meta_path = f"mart/{cfg['target_id']}_meta.json"
        self._save_json(meta, meta_path)

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  meta,
            "message": f"마트 생성 완료: {cfg['target_id']}  shape={df.shape}",
        }

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _load_source(self, cfg: dict) -> pd.DataFrame:
        if "source_query" in cfg and self.db_session is not None:
            return pd.read_sql(cfg["source_query"], self.db_session.bind)
        elif "source_path" in cfg:
            return self._load_dataframe(cfg["source_path"])
        else:
            raise ExecutorException("source_query 또는 source_path 중 하나가 필요합니다.")

    def _basic_preprocess(self, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        """타입 캐스팅, 결측 처리, 이상값 클리핑."""
        # 문자열 → 카테고리 자동 변환
        for col in df.select_dtypes(include="object").columns:
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype("category")

        # 수치형 결측 → 중앙값 대체
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        # 이상값 클리핑 (IQR 3배)
        clip_cols = cfg.get("clip_columns", list(num_cols))
        for col in clip_cols:
            q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q1, upper=q3)

        return df

    def _create_derived_features(self, df: pd.DataFrame, rules: list) -> pd.DataFrame:
        """
        rules 예시:
            [{"name": "ratio_a_b", "expr": "col_a / (col_b + 1)"},
             {"name": "log_c",     "expr": "log(col_c + 1)"}]
        """
        for rule in rules:
            try:
                df[rule["name"]] = df.eval(rule["expr"])
            except Exception as exc:
                logger.warning("파생변수 생성 실패: %s  reason=%s", rule["name"], exc)
        return df

    def _split_dataframe(
        self,
        df: pd.DataFrame,
        split_cfg: dict,
        target_col: Optional[str],
    ) -> dict[str, pd.DataFrame]:
        """비율에 따라 train/valid/test 분할."""
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df)
        train_end = int(n * split_cfg.get("train", 0.7))
        valid_end = train_end + int(n * split_cfg.get("valid", 0.15))

        return {
            "train": df.iloc[:train_end],
            "valid": df.iloc[train_end:valid_end],
            "test":  df.iloc[valid_end:],
        }


# =============================================================================
# Module-level functions
# =============================================================================


def _make_engine(db_info: dict):
    """PostgreSQL SQLAlchemy engine 생성 (postgresql+psycopg2)."""
    from sqlalchemy import create_engine
    url = (
        f"postgresql+psycopg2://{db_info['user']}:{db_info['password']}"
        f"@{db_info['host']}:{db_info['port']}/{db_info['db']}"
    )
    return create_engine(url, pool_pre_ping=True)


def get_common_code_index(service_db_info: dict, auth_key: str, code: str, code_role: str) -> dict:
    """공통코드 변수 인덱스 조회."""
    from sqlalchemy import text
    engine = _make_engine(service_db_info)
    sql = text(
        "SELECT code_index FROM common_code "
        "WHERE auth_key = :auth_key AND code = :code AND code_role = :code_role"
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"auth_key": auth_key, "code": code, "code_role": code_role}).fetchone()
    index = int(row[0]) if row else -1
    logger.debug("get_common_code_index: code=%s code_role=%s index=%s", code, code_role, index)
    return {"result": index, "code": code, "code_role": code_role}


def get_common_code_indexes(service_db_info: dict, auth_key: str, code: str) -> dict:
    """공통코드 인덱스 목록 조회."""
    from sqlalchemy import text
    engine = _make_engine(service_db_info)
    sql = text(
        "SELECT code_role, code_index FROM common_code "
        "WHERE auth_key = :auth_key AND code = :code ORDER BY code_index"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"auth_key": auth_key, "code": code}).fetchall()
    result = [{"code_role": r[0], "code_index": int(r[1])} for r in rows]
    logger.debug("get_common_code_indexes: code=%s count=%d", code, len(result))
    return {"result": result, "code": code}


def mart_exists(service_db_info: dict, auth_key: str, mart_name: str) -> bool:
    """마트 존재 여부 확인 (DB 조회)."""
    from sqlalchemy import text
    engine = _make_engine(service_db_info)
    sql = text(
        "SELECT COUNT(1) FROM mart_meta "
        "WHERE auth_key = :auth_key AND mart_name = :mart_name"
    )
    with engine.connect() as conn:
        cnt = conn.execute(sql, {"auth_key": auth_key, "mart_name": mart_name}).scalar()
    exists = int(cnt) > 0
    logger.debug("mart_exists: mart_name=%s exists=%s", mart_name, exists)
    return exists


def get_mart_domain(service_db_info: dict, auth_key: str, mart_name: str, add_whole_domain: bool = False) -> dict:
    """마트 도메인 조회."""
    from sqlalchemy import text
    engine = _make_engine(service_db_info)
    sql = text(
        "SELECT domain_name, domain_index, domain_type, domain_min, domain_max "
        "FROM mart_domain "
        "WHERE auth_key = :auth_key AND mart_name = :mart_name "
        "ORDER BY domain_index"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"auth_key": auth_key, "mart_name": mart_name}).fetchall()
    domains = [
        {
            "domain_name":  r[0],
            "domain_index": int(r[1]),
            "domain_type":  r[2],
            "domain_min":   float(r[3]) if r[3] is not None else None,
            "domain_max":   float(r[4]) if r[4] is not None else None,
        }
        for r in rows
    ]
    if add_whole_domain:
        domains.insert(0, {"domain_name": "전체", "domain_index": -1, "domain_type": "whole",
                           "domain_min": None, "domain_max": None})
    logger.debug("get_mart_domain: mart_name=%s domains=%d", mart_name, len(domains))
    return {"result": domains, "mart_name": mart_name}


def add_mart_meta(service_db_info: dict, file_server_host: str, file_server_port: int, json_obj: dict) -> dict:
    """마트 메타 등록 (변수 정보, binary 변수 저장 포함)."""
    import json as _json
    from sqlalchemy import text
    from pathlib import Path

    auth_key    = json_obj["auth_key"]
    mart_name   = json_obj["mart_name"]
    mart_path   = json_obj["mart_path"]
    variables   = json_obj.get("variables", [])
    description = json_obj.get("description", "")

    engine = _make_engine(service_db_info)

    # 변수 JSON을 binary로 직렬화하여 저장 (domain name length 예외 방지)
    for var in variables:
        if len(var.get("domain_name", "")) > 255:
            raise ValueError(f"도메인명이 255자를 초과합니다: {var.get('domain_name', '')[:30]}...")

    var_json_bytes = _json.dumps(variables, ensure_ascii=False).encode("utf-8")

    upsert_sql = text(
        "INSERT INTO mart_meta (auth_key, mart_name, mart_path, description, variable_json, created_at) "
        "VALUES (:auth_key, :mart_name, :mart_path, :description, :variable_json, NOW()) "
        "ON CONFLICT (auth_key, mart_name) DO UPDATE "
        "SET mart_path=EXCLUDED.mart_path, description=EXCLUDED.description, "
        "    variable_json=EXCLUDED.variable_json, updated_at=NOW()"
    )
    with engine.begin() as conn:
        conn.execute(upsert_sql, {
            "auth_key":      auth_key,
            "mart_name":     mart_name,
            "mart_path":     mart_path,
            "description":   description,
            "variable_json": var_json_bytes,
        })

    logger.info("add_mart_meta: mart_name=%s variables=%d", mart_name, len(variables))
    return {"result": "ok", "mart_name": mart_name, "variable_count": len(variables)}


def get_dtype_from_variable_json(variable_json: list, columns: list = None) -> dict:
    """변수 JSON에서 dtype 매핑 반환."""
    dtype_map = {
        "int":       "int64",
        "integer":   "int64",
        "float":     "float64",
        "double":    "float64",
        "string":    "object",
        "str":       "object",
        "category":  "category",
        "bool":      "bool",
        "boolean":   "bool",
        "datetime":  "datetime64[ns]",
        "date":      "datetime64[ns]",
    }
    result = {}
    for var in variable_json:
        name  = var.get("var_name") or var.get("name", "")
        vtype = str(var.get("var_type") or var.get("type", "")).lower()
        if columns is not None and name not in columns:
            continue
        result[name] = dtype_map.get(vtype, "object")
    logger.debug("get_dtype_from_variable_json: mapped %d variables", len(result))
    return result


def process_uploaded_mart_with_variable_json(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """parquet 업로드 시 variable_json으로 데이터 타입 직접 반영."""
    from pathlib import Path
    import json as _json

    mart_path     = json_obj["mart_path"]
    output_path   = json_obj.get("output_path", mart_path)
    variable_json = json_obj["variable_json"]

    df = pd.read_parquet(mart_path)
    dtype_map = get_dtype_from_variable_json(variable_json, columns=list(df.columns))

    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                if dtype == "datetime64[ns]":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif dtype == "category":
                    df[col] = df[col].astype("category")
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as exc:
                logger.warning("process_uploaded_mart_with_variable_json: col=%s dtype=%s err=%s", col, dtype, exc)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("process_uploaded_mart_with_variable_json: mart=%s rows=%d cols=%d", mart_path, len(df), len(df.columns))
    return {"result": "ok", "mart_path": output_path, "row_count": len(df), "col_count": len(df.columns)}


def process_uploaded_mart(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """업로드된 마트 처리 (csv/parquet 업로드 모드 포함)."""
    from pathlib import Path

    mart_path   = json_obj["mart_path"]
    output_path = json_obj.get("output_path", mart_path)
    upload_mode = json_obj.get("upload_mode", "parquet")   # "csv" or "parquet"
    encoding    = json_obj.get("encoding", "utf-8")
    sep         = json_obj.get("sep", ",")

    if upload_mode == "csv":
        df = pd.read_csv(mart_path, encoding=encoding, sep=sep, low_memory=False)
        # 문자열 컬럼 category 변환 (카디널리티 낮은 경우)
        for col in df.select_dtypes(include="object").columns:
            if df[col].nunique() / max(len(df), 1) < 0.5:
                df[col] = df[col].astype("category")
    else:
        df = pd.read_parquet(mart_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("process_uploaded_mart: mode=%s mart=%s rows=%d cols=%d", upload_mode, mart_path, len(df), len(df.columns))
    return {"result": "ok", "mart_path": output_path, "row_count": len(df), "col_count": len(df.columns)}


def process_uploaded_mart_to_db(service_db_info: dict, json_obj: dict) -> dict:
    """마트를 DB에 적재 (Windows local 플랫폼, uint32/string 스키마 사용)."""
    from sqlalchemy import text
    import pyarrow as pa
    import pyarrow.parquet as pq

    mart_path  = json_obj["mart_path"]
    table_name = json_obj["table_name"]
    auth_key   = json_obj.get("auth_key", "")

    # pyarrow로 읽어 uint32/string 스키마 강제
    table = pq.read_table(mart_path)
    schema_fields = []
    for field in table.schema:
        if pa.types.is_integer(field.type):
            schema_fields.append(pa.field(field.name, pa.uint32()))
        elif pa.types.is_large_string(field.type) or pa.types.is_string(field.type):
            schema_fields.append(pa.field(field.name, pa.string()))
        else:
            schema_fields.append(field)
    new_schema = pa.schema(schema_fields)
    table = table.cast(new_schema, safe=False)
    df = table.to_pandas()

    engine = _make_engine(service_db_info)
    df.to_sql(table_name, engine, if_exists="replace", index=False, chunksize=5000)
    logger.info("process_uploaded_mart_to_db: table=%s rows=%d", table_name, len(df))
    return {"result": "ok", "table_name": table_name, "row_count": len(df)}


def calculate_statistics_uploaded_mart(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """업로드 마트 통계 계산 + 프로파일링 결과 JSON 업로드."""
    import json as _json
    from pathlib import Path

    mart_path   = json_obj["mart_path"]
    auth_key    = json_obj.get("auth_key", "")
    mart_name   = json_obj.get("mart_name", "")
    root_dir    = json_obj.get("root_dir", "/data")
    result_file = json_obj.get("result_file", f"{mart_name}_profiling.json")

    df = pd.read_parquet(mart_path)
    profiling = {}
    for col in df.columns:
        s = df[col]
        info = {
            "dtype":         str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_rate":  round(float(s.isna().mean()), 4),
            "unique_count":  int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
            info.update({
                "mean": round(float(desc["mean"]), 4),
                "std":  round(float(desc["std"]), 4),
                "min":  round(float(desc["min"]), 4),
                "max":  round(float(desc["max"]), 4),
                "p01":  round(float(desc["1%"]), 4),
                "p25":  round(float(desc["25%"]), 4),
                "p50":  round(float(desc["50%"]), 4),
                "p75":  round(float(desc["75%"]), 4),
                "p99":  round(float(desc["99%"]), 4),
            })
        else:
            top = s.value_counts().head(10)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
        profiling[col] = info

    out_dir = Path(root_dir) / "mart_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    prof_path = out_dir / result_file
    with open(prof_path, "w", encoding="utf-8") as f:
        _json.dump(profiling, f, ensure_ascii=False)

    logger.info("calculate_statistics_uploaded_mart: mart=%s cols=%d profile=%s", mart_name, len(profiling), prof_path)
    return {"result": "ok", "mart_name": mart_name, "column_count": len(profiling), "profile_file": str(prof_path)}


def mart_concat_vertical(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """대용량 지원 수직 결합 (chunk 단위 parquet 읽기)."""
    from pathlib import Path
    import pyarrow as pa
    import pyarrow.parquet as pq

    file_paths  = json_obj["file_paths"]
    output_path = json_obj["output_path"]
    chunk_size  = int(json_obj.get("chunk_size", 100_000))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows = 0
    for fpath in file_paths:
        pf = pq.ParquetFile(fpath)
        for batch in pf.iter_batches(batch_size=chunk_size):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            total_rows += len(batch)
    if writer:
        writer.close()

    logger.info("mart_concat_vertical: files=%d total_rows=%d output=%s", len(file_paths), total_rows, output_path)
    return {"result": "ok", "file_count": len(file_paths), "row_count": total_rows, "output_path": output_path}


def upload_mart(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """마트 업로드 (fire-and-forget은 app server에서 처리)."""
    from pathlib import Path

    mart_path   = json_obj["mart_path"]
    output_path = json_obj.get("output_path", mart_path)
    auth_key    = json_obj.get("auth_key", "")
    mart_name   = json_obj.get("mart_name", Path(mart_path).stem)

    df = pd.read_parquet(mart_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info("upload_mart: mart_name=%s rows=%d output=%s", mart_name, len(df), output_path)
    return {"result": "ok", "mart_name": mart_name, "row_count": len(df), "output_path": output_path}


def upload_mart_to_db(service_db_info: dict, json_obj: dict) -> dict:
    """Windows 버전 DB에 마트 업로드."""
    return process_uploaded_mart_to_db(service_db_info, json_obj)


def get_auth_level(db_info: dict, user_id: str) -> int:
    """사용자 권한 레벨 조회."""
    from sqlalchemy import text
    engine = _make_engine(db_info)
    sql = text("SELECT auth_level FROM user_auth WHERE user_id = :user_id")
    with engine.connect() as conn:
        row = conn.execute(sql, {"user_id": user_id}).fetchone()
    level = int(row[0]) if row else 0
    logger.debug("get_auth_level: user_id=%s level=%d", user_id, level)
    return level


def get_mart_list(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    auth_key: str,
) -> dict:
    """마트 목록 조회."""
    from sqlalchemy import text
    engine = _make_engine(service_db_info)
    sql = text(
        "SELECT mart_name, mart_path, description, created_at, updated_at "
        "FROM mart_meta WHERE auth_key = :auth_key ORDER BY mart_name"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"auth_key": auth_key}).fetchall()
    result = [
        {
            "mart_name":   r[0],
            "mart_path":   r[1],
            "description": r[2],
            "created_at":  str(r[3]) if r[3] else None,
            "updated_at":  str(r[4]) if r[4] else None,
        }
        for r in rows
    ]
    logger.info("get_mart_list: auth_key=%s count=%d", auth_key, len(result))
    return {"result": result, "count": len(result)}


def get_export_mart_list(service_db_info: dict, auth_key: str, external_purpose: str) -> dict:
    """외부 배포용 마트 목록 조회."""
    from sqlalchemy import text
    engine = _make_engine(service_db_info)
    sql = text(
        "SELECT mart_name, mart_path, description, external_purpose "
        "FROM mart_meta "
        "WHERE auth_key = :auth_key AND external_purpose = :external_purpose "
        "ORDER BY mart_name"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"auth_key": auth_key, "external_purpose": external_purpose}).fetchall()
    result = [
        {"mart_name": r[0], "mart_path": r[1], "description": r[2], "external_purpose": r[3]}
        for r in rows
    ]
    logger.info("get_export_mart_list: auth_key=%s purpose=%s count=%d", auth_key, external_purpose, len(result))
    return {"result": result, "count": len(result)}


def delete_mart(service_db_info: dict, root_dir: str, auth_key: str, mart_name: str) -> dict:
    """마트 삭제."""
    from sqlalchemy import text
    from pathlib import Path

    engine = _make_engine(service_db_info)

    # DB에서 경로 조회 후 파일 삭제
    sel_sql = text("SELECT mart_path FROM mart_meta WHERE auth_key = :auth_key AND mart_name = :mart_name")
    with engine.connect() as conn:
        row = conn.execute(sel_sql, {"auth_key": auth_key, "mart_name": mart_name}).fetchone()

    deleted_file = False
    if row and row[0]:
        mart_path = Path(root_dir) / row[0] if not Path(row[0]).is_absolute() else Path(row[0])
        if mart_path.exists():
            mart_path.unlink()
            deleted_file = True

    del_sql = text("DELETE FROM mart_meta WHERE auth_key = :auth_key AND mart_name = :mart_name")
    with engine.begin() as conn:
        conn.execute(del_sql, {"auth_key": auth_key, "mart_name": mart_name})

    logger.info("delete_mart: mart_name=%s deleted_file=%s", mart_name, deleted_file)
    return {"result": "ok", "mart_name": mart_name, "deleted_file": deleted_file}


def get_mart_layout(service_db_info: dict, auth_key: str, mart_name: str) -> dict:
    """마트 레이아웃 조회."""
    from sqlalchemy import text
    engine = _make_engine(service_db_info)
    sql = text(
        "SELECT layout_json FROM mart_layout "
        "WHERE auth_key = :auth_key AND mart_name = :mart_name"
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"auth_key": auth_key, "mart_name": mart_name}).fetchone()

    import json as _json
    layout = _json.loads(row[0]) if row and row[0] else {}
    logger.debug("get_mart_layout: mart_name=%s", mart_name)
    return {"result": layout, "mart_name": mart_name}


def get_mart_layout_featuretools(service_db_info: dict, auth_key: str, mart_name: str) -> dict:
    """featuretools용 마트 레이아웃 조회."""
    layout_result = get_mart_layout(service_db_info, auth_key, mart_name)
    layout = layout_result.get("result", {})

    # featuretools 형식으로 변환: 각 엔티티/관계 정보 추출
    entities = layout.get("entities", {})
    relationships = layout.get("relationships", [])
    ft_layout = {
        "entities":      entities,
        "relationships": relationships,
    }
    logger.debug("get_mart_layout_featuretools: mart_name=%s entities=%d", mart_name, len(entities))
    return {"result": ft_layout, "mart_name": mart_name}


def get_mart_layout_for_excel(service_db_info: dict, auth_key: str, mart_name: str) -> dict:
    """Excel 출력용 마트 레이아웃 조회."""
    from sqlalchemy import text
    engine = _make_engine(service_db_info)
    sql = text(
        "SELECT v.var_name, v.var_type, v.description, v.domain_name, v.var_index "
        "FROM mart_variable v "
        "WHERE v.auth_key = :auth_key AND v.mart_name = :mart_name "
        "ORDER BY v.var_index"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"auth_key": auth_key, "mart_name": mart_name}).fetchall()
    result = [
        {
            "var_name":    r[0],
            "var_type":    r[1],
            "description": r[2],
            "domain_name": r[3],
            "var_index":   int(r[4]) if r[4] is not None else None,
        }
        for r in rows
    ]
    logger.debug("get_mart_layout_for_excel: mart_name=%s vars=%d", mart_name, len(result))
    return {"result": result, "mart_name": mart_name, "variable_count": len(result)}


def save_mart_layout(service_db_info: dict, json_obj: dict) -> dict:
    """마트 레이아웃 저장 (SVD값 포함)."""
    import json as _json
    from sqlalchemy import text

    auth_key    = json_obj["auth_key"]
    mart_name   = json_obj["mart_name"]
    layout      = json_obj.get("layout", {})
    svd_values  = json_obj.get("svd_values", {})

    layout_with_svd = dict(layout)
    if svd_values:
        layout_with_svd["svd_values"] = svd_values

    layout_json = _json.dumps(layout_with_svd, ensure_ascii=False)

    engine = _make_engine(service_db_info)
    upsert_sql = text(
        "INSERT INTO mart_layout (auth_key, mart_name, layout_json, updated_at) "
        "VALUES (:auth_key, :mart_name, :layout_json, NOW()) "
        "ON CONFLICT (auth_key, mart_name) DO UPDATE "
        "SET layout_json=EXCLUDED.layout_json, updated_at=NOW()"
    )
    with engine.begin() as conn:
        conn.execute(upsert_sql, {"auth_key": auth_key, "mart_name": mart_name, "layout_json": layout_json})

    logger.info("save_mart_layout: mart_name=%s svd=%s", mart_name, bool(svd_values))
    return {"result": "ok", "mart_name": mart_name}


def get_domain_index(service_db_info: dict, auth_key: str, mart_name: str, domain_name: str) -> dict:
    """도메인 인덱스 조회."""
    from sqlalchemy import text
    engine = _make_engine(service_db_info)
    sql = text(
        "SELECT domain_index FROM mart_domain "
        "WHERE auth_key = :auth_key AND mart_name = :mart_name AND domain_name = :domain_name"
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"auth_key": auth_key, "mart_name": mart_name, "domain_name": domain_name}).fetchone()
    index = int(row[0]) if row else -1
    logger.debug("get_domain_index: mart_name=%s domain_name=%s index=%d", mart_name, domain_name, index)
    return {"result": index, "mart_name": mart_name, "domain_name": domain_name}


def get_first_n_lines_in_server_repository(
    service_db_info: dict,
    root_dir: str,
    auth_key: str,
    mart_name: str,
    n: int = 5,
) -> dict:
    """서버 레포지토리 상위 N행 조회."""
    from pathlib import Path
    import json as _json

    repo_dir = Path(root_dir) / "server_repository" / auth_key
    # parquet 파일 우선, 없으면 csv
    candidates = list(repo_dir.glob(f"{mart_name}.parquet")) + list(repo_dir.glob(f"{mart_name}.csv"))
    if not candidates:
        raise FileNotFoundError(f"서버 레포지토리에 '{mart_name}' 파일이 없습니다: {repo_dir}")

    fpath = candidates[0]
    if fpath.suffix == ".parquet":
        df = pd.read_parquet(fpath)
    else:
        df = pd.read_csv(fpath, nrows=n)

    head = df.head(n)
    logger.debug("get_first_n_lines_in_server_repository: mart=%s n=%d", mart_name, n)
    return {
        "result":  head.where(head.notna(), other=None).to_dict(orient="records"),
        "columns": list(head.columns),
        "file":    str(fpath),
    }


def get_first_n_lines_in_serving_repository(
    service_db_info: dict,
    root_dir: str,
    auth_key: str,
    mart_name: str,
    n: int = 5,
) -> dict:
    """서빙 레포지토리 상위 N행 조회."""
    from pathlib import Path

    repo_dir   = Path(root_dir) / "serving_repository" / auth_key
    candidates = list(repo_dir.glob(f"{mart_name}.parquet")) + list(repo_dir.glob(f"{mart_name}.csv"))
    if not candidates:
        raise FileNotFoundError(f"서빙 레포지토리에 '{mart_name}' 파일이 없습니다: {repo_dir}")

    fpath = candidates[0]
    if fpath.suffix == ".parquet":
        df = pd.read_parquet(fpath)
    else:
        df = pd.read_csv(fpath, nrows=n)

    head = df.head(n)
    logger.debug("get_first_n_lines_in_serving_repository: mart=%s n=%d", mart_name, n)
    return {
        "result":  head.where(head.notna(), other=None).to_dict(orient="records"),
        "columns": list(head.columns),
        "file":    str(fpath),
    }


def upload_mart_from_serving(service_db_info: dict, root_dir: str, auth_key: str, mart_name: str) -> dict:
    """서빙 레포지토리에서 마트 업로드."""
    from pathlib import Path
    import shutil

    serving_dir = Path(root_dir) / "serving_repository" / auth_key
    target_dir  = Path(root_dir) / "mart" / auth_key

    candidates = list(serving_dir.glob(f"{mart_name}.parquet")) + list(serving_dir.glob(f"{mart_name}.csv"))
    if not candidates:
        raise FileNotFoundError(f"서빙 레포지토리에 '{mart_name}' 파일이 없습니다.")

    src  = candidates[0]
    dest = target_dir / src.name
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dest))

    logger.info("upload_mart_from_serving: mart_name=%s src=%s dest=%s", mart_name, src, dest)
    return {"result": "ok", "mart_name": mart_name, "source": str(src), "destination": str(dest)}


def validate_data(service_db_info: dict, auth_key: str, mart_name: str) -> dict:
    """데이터 유효성 검사."""
    from sqlalchemy import text

    engine = _make_engine(service_db_info)

    # 마트 메타 조회
    sel_sql = text(
        "SELECT mart_path, variable_json FROM mart_meta "
        "WHERE auth_key = :auth_key AND mart_name = :mart_name"
    )
    with engine.connect() as conn:
        row = conn.execute(sel_sql, {"auth_key": auth_key, "mart_name": mart_name}).fetchone()

    if not row:
        return {"result": "error", "message": f"마트가 존재하지 않습니다: {mart_name}"}

    mart_path = row[0]
    import json as _json
    variable_json = _json.loads(row[1]) if row[1] else []

    df = pd.read_parquet(mart_path)
    issues = []

    # 1. 컬럼 존재 여부 확인
    expected_cols = {v.get("var_name") or v.get("name") for v in variable_json}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        issues.append({"type": "missing_columns", "columns": list(missing_cols)})

    # 2. 결측률 100% 컬럼
    full_missing = [c for c in df.columns if df[c].isna().all()]
    if full_missing:
        issues.append({"type": "all_missing_columns", "columns": full_missing})

    # 3. 중복 행 검사
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append({"type": "duplicate_rows", "count": dup_count})

    is_valid = len(issues) == 0
    logger.info("validate_data: mart_name=%s valid=%s issues=%d", mart_name, is_valid, len(issues))
    return {"result": "ok", "mart_name": mart_name, "is_valid": is_valid, "issues": issues, "row_count": len(df)}


def get_server_repository_dirs(service_db_info: dict, root_dir: str, auth_key: str) -> dict:
    """서버 레포지토리 디렉토리 목록 조회."""
    from pathlib import Path

    repo_dir = Path(root_dir) / "server_repository" / auth_key
    repo_dir.mkdir(parents=True, exist_ok=True)

    dirs = sorted([d.name for d in repo_dir.iterdir() if d.is_dir()])
    files = sorted([f.name for f in repo_dir.iterdir() if f.is_file()])

    logger.debug("get_server_repository_dirs: auth_key=%s dirs=%d files=%d", auth_key, len(dirs), len(files))
    return {"result": {"dirs": dirs, "files": files}, "path": str(repo_dir)}


def get_svd_values(service_db_info: dict, auth_key: str, mart_name: str) -> dict:
    """SVD 값 조회."""
    from sqlalchemy import text
    import json as _json

    engine = _make_engine(service_db_info)
    sql = text(
        "SELECT svd_json FROM mart_svd "
        "WHERE auth_key = :auth_key AND mart_name = :mart_name"
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"auth_key": auth_key, "mart_name": mart_name}).fetchone()

    svd = _json.loads(row[0]) if row and row[0] else {}
    logger.debug("get_svd_values: mart_name=%s keys=%d", mart_name, len(svd))
    return {"result": svd, "mart_name": mart_name}


def get_variable_svd_values(service_db_info: dict, auth_key: str, mart_name: str, var_name: str) -> dict:
    """변수별 SVD 값 조회."""
    svd_result = get_svd_values(service_db_info, auth_key, mart_name)
    svd = svd_result.get("result", {})
    var_svd = svd.get(var_name, {})
    logger.debug("get_variable_svd_values: mart_name=%s var_name=%s", mart_name, var_name)
    return {"result": var_svd, "mart_name": mart_name, "var_name": var_name}


def save_svd(service_db_info: dict, json_obj: dict) -> dict:
    """SVD 값 저장."""
    import json as _json
    from sqlalchemy import text

    auth_key   = json_obj["auth_key"]
    mart_name  = json_obj["mart_name"]
    svd_values = json_obj.get("svd_values", {})

    svd_json = _json.dumps(svd_values, ensure_ascii=False)
    engine = _make_engine(service_db_info)
    upsert_sql = text(
        "INSERT INTO mart_svd (auth_key, mart_name, svd_json, updated_at) "
        "VALUES (:auth_key, :mart_name, :svd_json, NOW()) "
        "ON CONFLICT (auth_key, mart_name) DO UPDATE "
        "SET svd_json=EXCLUDED.svd_json, updated_at=NOW()"
    )
    with engine.begin() as conn:
        conn.execute(upsert_sql, {"auth_key": auth_key, "mart_name": mart_name, "svd_json": svd_json})

    logger.info("save_svd: mart_name=%s keys=%d", mart_name, len(svd_values))
    return {"result": "ok", "mart_name": mart_name, "svd_key_count": len(svd_values)}


def get_mart_var_info(
    service_db_info: dict,
    auth_key: str,
    mart_name: str,
    return_all_columns: bool = False,
) -> dict:
    """마트 변수 정보 조회."""
    from sqlalchemy import text

    engine = _make_engine(service_db_info)
    base_cols = "var_name, var_type, description, domain_name, var_index"
    if return_all_columns:
        base_cols = "*"

    sql = text(
        f"SELECT {base_cols} FROM mart_variable "
        "WHERE auth_key = :auth_key AND mart_name = :mart_name "
        "ORDER BY var_index"
    )
    with engine.connect() as conn:
        cursor = conn.execute(sql, {"auth_key": auth_key, "mart_name": mart_name})
        keys   = list(cursor.keys())
        rows   = cursor.fetchall()

    result = [dict(zip(keys, row)) for row in rows]
    logger.debug("get_mart_var_info: mart_name=%s vars=%d all_cols=%s", mart_name, len(result), return_all_columns)
    return {"result": result, "mart_name": mart_name, "variable_count": len(result)}


def create_variable_from_other_mart(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """다른 마트에서 변수 생성."""
    from pathlib import Path

    src_mart_path  = json_obj["src_mart_path"]
    dest_mart_path = json_obj["dest_mart_path"]
    src_var        = json_obj["src_var"]
    dest_var       = json_obj.get("dest_var", src_var)
    key_col        = json_obj.get("key_col")

    df_src  = pd.read_parquet(src_mart_path)
    df_dest = pd.read_parquet(dest_mart_path)

    if src_var not in df_src.columns:
        raise ValueError(f"소스 마트에 변수가 없습니다: {src_var}")

    if key_col and key_col in df_src.columns and key_col in df_dest.columns:
        mapping = df_src.set_index(key_col)[src_var]
        df_dest[dest_var] = df_dest[key_col].map(mapping)
    else:
        if len(df_src) == len(df_dest):
            df_dest[dest_var] = df_src[src_var].values
        else:
            raise ValueError("key_col 없이는 두 마트의 행 수가 같아야 합니다.")

    df_dest.to_parquet(dest_mart_path, index=False)
    logger.info("create_variable_from_other_mart: src_var=%s dest_var=%s", src_var, dest_var)
    return {"result": "ok", "dest_var": dest_var, "row_count": len(df_dest)}


def get_variable_length(service_db_info: dict, auth_key: str, mart_name: str, var_name: str) -> dict:
    """변수 길이 조회 (변수명 기준)."""
    from sqlalchemy import text

    engine = _make_engine(service_db_info)
    sql = text(
        "SELECT var_length FROM mart_variable "
        "WHERE auth_key = :auth_key AND mart_name = :mart_name AND var_name = :var_name"
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"auth_key": auth_key, "mart_name": mart_name, "var_name": var_name}).fetchone()

    length = int(row[0]) if row and row[0] is not None else None
    logger.debug("get_variable_length: mart_name=%s var_name=%s length=%s", mart_name, var_name, length)
    return {"result": length, "mart_name": mart_name, "var_name": var_name}


def create_feature_name(service_db_info: dict, json_obj: dict) -> dict:
    """피처명 생성."""
    from sqlalchemy import text

    auth_key     = json_obj["auth_key"]
    mart_name    = json_obj["mart_name"]
    base_name    = json_obj["base_name"]
    suffix_rules = json_obj.get("suffix_rules", [])

    # 기존 변수명 목록 조회 (중복 방지)
    engine = _make_engine(service_db_info)
    sql = text("SELECT var_name FROM mart_variable WHERE auth_key = :auth_key AND mart_name = :mart_name")
    with engine.connect() as conn:
        existing = {row[0] for row in conn.execute(sql, {"auth_key": auth_key, "mart_name": mart_name}).fetchall()}

    # 후보 생성
    candidates = [base_name] + [f"{base_name}_{r}" for r in suffix_rules]
    feature_name = None
    for candidate in candidates:
        if candidate not in existing:
            feature_name = candidate
            break
    if feature_name is None:
        import uuid
        feature_name = f"{base_name}_{uuid.uuid4().hex[:6]}"

    logger.debug("create_feature_name: base=%s feature_name=%s", base_name, feature_name)
    return {"result": feature_name, "mart_name": mart_name, "base_name": base_name}


def get_constants(service_db_info: dict, auth_key: str) -> dict:
    """상수 목록 JSON 반환."""
    from sqlalchemy import text

    engine = _make_engine(service_db_info)
    sql = text(
        "SELECT const_name, const_value, const_type, description "
        "FROM system_constants WHERE auth_key = :auth_key ORDER BY const_name"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"auth_key": auth_key}).fetchall()

    result = {
        r[0]: {"value": r[1], "type": r[2], "description": r[3]}
        for r in rows
    }
    logger.debug("get_constants: auth_key=%s count=%d", auth_key, len(result))
    return {"result": result, "count": len(result)}


def table_to_server(service_db_info: dict, root_dir: str, auth_key: str, table_name: str) -> dict:
    """테이블을 서버로 업로드."""
    from pathlib import Path

    engine = _make_engine(service_db_info)
    df = pd.read_sql_table(table_name, engine)

    out_dir = Path(root_dir) / "server_repository" / auth_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{table_name}.parquet"
    df.to_parquet(str(out_path), index=False)

    logger.info("table_to_server: table=%s rows=%d output=%s", table_name, len(df), out_path)
    return {"result": "ok", "table_name": table_name, "row_count": len(df), "output_path": str(out_path)}


def create_user_auth(service_db_info: dict, json_obj: dict) -> dict:
    """사용자 권한 생성."""
    from sqlalchemy import text

    auth_key   = json_obj["auth_key"]
    table_id   = json_obj["table_id"]
    user_id    = json_obj["user_id"]
    auth_level = int(json_obj.get("auth_level", 1))

    engine = _make_engine(service_db_info)
    upsert_sql = text(
        "INSERT INTO mart_user_auth (auth_key, table_id, user_id, auth_level, created_at) "
        "VALUES (:auth_key, :table_id, :user_id, :auth_level, NOW()) "
        "ON CONFLICT (auth_key, table_id, user_id) DO UPDATE "
        "SET auth_level=EXCLUDED.auth_level, updated_at=NOW()"
    )
    with engine.begin() as conn:
        conn.execute(upsert_sql, {
            "auth_key":   auth_key,
            "table_id":   table_id,
            "user_id":    user_id,
            "auth_level": auth_level,
        })

    logger.info("create_user_auth: table_id=%s user_id=%s auth_level=%d", table_id, user_id, auth_level)
    return {"result": "ok", "table_id": table_id, "user_id": user_id, "auth_level": auth_level}


def get_user_authorities(
    service_db_info: dict,
    auth_key: str,
    table_id: str,
    user_ids: list = [],
) -> dict:
    """마트 사용자 권한 조회."""
    from sqlalchemy import text

    engine = _make_engine(service_db_info)
    if user_ids:
        sql = text(
            "SELECT user_id, auth_level FROM mart_user_auth "
            "WHERE auth_key = :auth_key AND table_id = :table_id AND user_id = ANY(:user_ids)"
        )
        with engine.connect() as conn:
            rows = conn.execute(sql, {"auth_key": auth_key, "table_id": table_id, "user_ids": user_ids}).fetchall()
    else:
        sql = text(
            "SELECT user_id, auth_level FROM mart_user_auth "
            "WHERE auth_key = :auth_key AND table_id = :table_id"
        )
        with engine.connect() as conn:
            rows = conn.execute(sql, {"auth_key": auth_key, "table_id": table_id}).fetchall()

    result = [{"user_id": r[0], "auth_level": int(r[1])} for r in rows]
    logger.debug("get_user_authorities: table_id=%s count=%d", table_id, len(result))
    return {"result": result, "table_id": table_id, "count": len(result)}


def get_user_variable(service_db_info: dict, auth_key: str, mart_name: str) -> dict:
    """사용자 변수 조회."""
    from sqlalchemy import text

    engine = _make_engine(service_db_info)
    sql = text(
        "SELECT var_name, var_type, expression, description, created_at "
        "FROM mart_user_variable "
        "WHERE auth_key = :auth_key AND mart_name = :mart_name "
        "ORDER BY var_name"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"auth_key": auth_key, "mart_name": mart_name}).fetchall()

    result = [
        {
            "var_name":    r[0],
            "var_type":    r[1],
            "expression":  r[2],
            "description": r[3],
            "created_at":  str(r[4]) if r[4] else None,
        }
        for r in rows
    ]
    logger.debug("get_user_variable: mart_name=%s count=%d", mart_name, len(result))
    return {"result": result, "mart_name": mart_name, "count": len(result)}


def get_user_variable_operator(service_db_info: dict, auth_key: str) -> dict:
    """사용자 변수 연산자 조회."""
    from sqlalchemy import text

    engine = _make_engine(service_db_info)
    sql = text(
        "SELECT operator_name, operator_symbol, operator_type, description "
        "FROM variable_operators ORDER BY operator_name"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()

    result = [
        {"operator_name": r[0], "operator_symbol": r[1], "operator_type": r[2], "description": r[3]}
        for r in rows
    ]
    logger.debug("get_user_variable_operator: count=%d", len(result))
    return {"result": result, "count": len(result)}


def get_min_max_count_scale(service_db_info: dict, auth_key: str, mart_name: str) -> dict:
    """min/max/count/scale 조회 (domain_index 포함)."""
    from sqlalchemy import text

    engine = _make_engine(service_db_info)
    sql = text(
        "SELECT v.var_name, v.var_min, v.var_max, v.var_count, v.var_scale, d.domain_index "
        "FROM mart_variable v "
        "LEFT JOIN mart_domain d "
        "  ON v.auth_key = d.auth_key AND v.mart_name = d.mart_name AND v.domain_name = d.domain_name "
        "WHERE v.auth_key = :auth_key AND v.mart_name = :mart_name "
        "ORDER BY v.var_name"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"auth_key": auth_key, "mart_name": mart_name}).fetchall()

    result = [
        {
            "var_name":     r[0],
            "var_min":      float(r[1]) if r[1] is not None else None,
            "var_max":      float(r[2]) if r[2] is not None else None,
            "var_count":    int(r[3]) if r[3] is not None else None,
            "var_scale":    float(r[4]) if r[4] is not None else None,
            "domain_index": int(r[5]) if r[5] is not None else None,
        }
        for r in rows
    ]
    logger.debug("get_min_max_count_scale: mart_name=%s vars=%d", mart_name, len(result))
    return {"result": result, "mart_name": mart_name, "count": len(result)}


def create_user_variable(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """사용자 변수 생성."""
    from sqlalchemy import text
    from pathlib import Path

    auth_key    = json_obj["auth_key"]
    mart_name   = json_obj["mart_name"]
    mart_path   = json_obj["mart_path"]
    var_name    = json_obj["var_name"]
    expression  = json_obj["expression"]
    var_type    = json_obj.get("var_type", "float")
    description = json_obj.get("description", "")

    df = pd.read_parquet(mart_path)
    try:
        df[var_name] = df.eval(expression)
    except Exception as exc:
        raise ValueError(f"사용자 변수 수식 오류: {expression}  reason={exc}") from exc

    df.to_parquet(mart_path, index=False)

    engine = _make_engine(service_db_info)
    upsert_sql = text(
        "INSERT INTO mart_user_variable (auth_key, mart_name, var_name, var_type, expression, description, created_at) "
        "VALUES (:auth_key, :mart_name, :var_name, :var_type, :expression, :description, NOW()) "
        "ON CONFLICT (auth_key, mart_name, var_name) DO UPDATE "
        "SET var_type=EXCLUDED.var_type, expression=EXCLUDED.expression, "
        "    description=EXCLUDED.description, updated_at=NOW()"
    )
    with engine.begin() as conn:
        conn.execute(upsert_sql, {
            "auth_key": auth_key, "mart_name": mart_name, "var_name": var_name,
            "var_type": var_type, "expression": expression, "description": description,
        })

    logger.info("create_user_variable: mart_name=%s var_name=%s", mart_name, var_name)
    return {"result": "ok", "mart_name": mart_name, "var_name": var_name, "row_count": len(df)}


def create_user_variable_encryption_test(service_db_info: dict, json_obj: dict) -> dict:
    """사용자 변수 암호화 테스트."""
    import hashlib

    var_name   = json_obj["var_name"]
    test_value = str(json_obj.get("test_value", ""))
    algorithm  = json_obj.get("algorithm", "sha256")

    if algorithm == "sha256":
        encrypted = hashlib.sha256(test_value.encode("utf-8")).hexdigest()
    elif algorithm == "md5":
        encrypted = hashlib.md5(test_value.encode("utf-8")).hexdigest()
    elif algorithm == "sha512":
        encrypted = hashlib.sha512(test_value.encode("utf-8")).hexdigest()
    else:
        raise ValueError(f"지원하지 않는 암호화 알고리즘: {algorithm}")

    logger.debug("create_user_variable_encryption_test: var_name=%s algorithm=%s", var_name, algorithm)
    return {"result": encrypted, "var_name": var_name, "algorithm": algorithm}


def delete_user_variable(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """사용자 변수 삭제 (list 허용, fire-and-forget)."""
    from sqlalchemy import text

    auth_key  = json_obj["auth_key"]
    mart_name = json_obj["mart_name"]
    mart_path = json_obj["mart_path"]
    var_names = json_obj["var_names"]
    if isinstance(var_names, str):
        var_names = [var_names]

    # parquet에서 컬럼 삭제
    df = pd.read_parquet(mart_path)
    cols_to_drop = [c for c in var_names if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        df.to_parquet(mart_path, index=False)

    # DB에서 삭제
    engine = _make_engine(service_db_info)
    del_sql = text(
        "DELETE FROM mart_user_variable "
        "WHERE auth_key = :auth_key AND mart_name = :mart_name AND var_name = ANY(:var_names)"
    )
    with engine.begin() as conn:
        conn.execute(del_sql, {"auth_key": auth_key, "mart_name": mart_name, "var_names": var_names})

    logger.info("delete_user_variable: mart_name=%s vars=%s", mart_name, var_names)
    return {"result": "ok", "mart_name": mart_name, "deleted_vars": cols_to_drop}


def replace_user_variable(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """사용자 변수 교체."""
    old_var_name = json_obj.get("old_var_name")
    new_json_obj = dict(json_obj)

    # 기존 변수 삭제
    if old_var_name:
        del_obj = {
            "auth_key":  json_obj["auth_key"],
            "mart_name": json_obj["mart_name"],
            "mart_path": json_obj["mart_path"],
            "var_names": [old_var_name],
        }
        delete_user_variable(service_db_info, file_server_host, file_server_port, del_obj)

    # 새 변수 생성
    result = create_user_variable(service_db_info, file_server_host, file_server_port, new_json_obj)
    logger.info("replace_user_variable: old=%s new=%s", old_var_name, json_obj.get("var_name"))
    return result


def chips_to_server(service_db_info: dict, root_dir: str, auth_key: str, chips_data: dict) -> dict:
    """CHIPS 데이터를 서버로 전송."""
    import json as _json
    from pathlib import Path

    chips_name = chips_data.get("chips_name", "chips")
    payload    = chips_data.get("data", {})

    out_dir = Path(root_dir) / "server_repository" / auth_key / "chips"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{chips_name}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False)

    logger.info("chips_to_server: chips_name=%s output=%s", chips_name, out_path)
    return {"result": "ok", "chips_name": chips_name, "output_path": str(out_path)}


def delete_server_repository_file_name(
    service_db_info: dict,
    root_dir: str,
    auth_key: str,
    file_name: str,
) -> dict:
    """서버 레포지토리 파일 삭제."""
    from pathlib import Path

    repo_dir  = Path(root_dir) / "server_repository" / auth_key
    file_path = repo_dir / file_name

    if file_path.exists():
        file_path.unlink()
        logger.info("delete_server_repository_file_name: deleted %s", file_path)
        deleted = True
    else:
        logger.warning("delete_server_repository_file_name: not found %s", file_path)
        deleted = False

    return {"result": "ok", "file_name": file_name, "deleted": deleted}


def get_server_repository_var_layout(
    service_db_info: dict,
    root_dir: str,
    auth_key: str,
    file_name: str,
) -> dict:
    """서버 레포지토리 변수 레이아웃 조회."""
    from pathlib import Path
    import pyarrow.parquet as pq

    repo_dir  = Path(root_dir) / "server_repository" / auth_key
    file_path = repo_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {file_path}")

    if file_path.suffix == ".parquet":
        schema = pq.read_schema(str(file_path))
        layout = [
            {"var_name": field.name, "var_type": str(field.type)}
            for field in schema
        ]
    elif file_path.suffix == ".csv":
        df = pd.read_csv(str(file_path), nrows=0)
        layout = [{"var_name": c, "var_type": "object"} for c in df.columns]
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {file_path.suffix}")

    logger.debug("get_server_repository_var_layout: file=%s vars=%d", file_name, len(layout))
    return {"result": layout, "file_name": file_name, "variable_count": len(layout)}


def mono_bin(X: "pd.Series", n: int = 20) -> "pd.DataFrame":
    """IV 계산용 monotonic binning.

    단조증가/감소 구간으로 연속형 변수를 구간화한다.
    """
    import numpy as np

    r = 0.0
    d = pd.DataFrame({"X": X, "Y": X})  # placeholder; caller should pass target separately
    # monotonic binning via correlation enforcement
    while np.abs(r) < 1.0:
        try:
            d = pd.DataFrame({"X": X, "Bucket": pd.qcut(X, n)})
            d = d.groupby("Bucket", as_index=True)["X"].agg(["min", "max", "count"]).reset_index()
            r = np.corrcoef(d["min"], range(len(d)))[0, 1]
        except Exception:
            pass
        n -= 1
        if n < 2:
            break

    logger.debug("mono_bin: bins=%d n=%d", len(d), n)
    return d


def data_vars(df1: "pd.DataFrame", target: str) -> "pd.DataFrame":
    """변수 중요도/IV 계산.

    각 변수에 대해 Information Value(IV)를 계산하고 중요도 순으로 정렬한다.
    """
    import numpy as np

    rows = []
    y = df1[target]
    total_good = int((y == 0).sum())
    total_bad  = int((y == 1).sum())

    for col in df1.columns:
        if col == target:
            continue
        s = df1[col].fillna("__missing__")
        if pd.api.types.is_numeric_dtype(df1[col]):
            try:
                bins = pd.qcut(df1[col], q=10, duplicates="drop")
            except Exception:
                bins = pd.cut(df1[col], bins=5)
            s = bins.astype(str).fillna("__missing__")
        iv_total = 0.0
        for val, grp in df1.groupby(s):
            g = int((grp[target] == 0).sum())
            b = int((grp[target] == 1).sum())
            if total_good == 0 or total_bad == 0 or g == 0 or b == 0:
                continue
            dist_g = g / total_good
            dist_b = b / total_bad
            woe     = np.log(dist_g / dist_b)
            iv_total += (dist_g - dist_b) * woe
        rows.append({"variable": col, "iv": round(iv_total, 4)})

    result_df = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    logger.debug("data_vars: target=%s vars=%d", target, len(result_df))
    return result_df
