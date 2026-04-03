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
