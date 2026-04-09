"""
process_executor.py
-------------------
전체 ML 파이프라인을 순차적으로 제어하는 오케스트레이션 실행기.

개별 executor를 단계별로 호출하고 각 단계의 결과를 다음 단계로 전달한다.
파이프라인 진행 상황을 잡 상태 파일에 기록하며, 특정 단계 실패 시
중단/계속/스킵 정책에 따라 동작한다.

파이프라인 예시:
  mart → data_analysis → python_model → scorecard → predict → stg → report → export

실행 순서:
  1. 파이프라인 설정 파싱
  2. 단계별 executor 인스턴스 생성
  3. 순차 실행 및 단계 결과 수집
  4. 전체 파이프라인 결과 요약
"""

import importlib
import logging
from typing import Any, Optional

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)

# executor 모듈 레지스트리 (module_path, class_name)
EXECUTOR_REGISTRY: dict[str, tuple[str, str]] = {
    "mart":             ("executors.ml.mart_executor",          "MartExecutor"),
    "data_analysis":    ("executors.ml.data_analysis_executor", "DataAnalysisExecutor"),
    "python_model":     ("executors.ml.python_model_executor",  "PythonModelExecutor"),
    "h2o_model":        ("executors.ml.h2o_model_executor",     "H2OModelExecutor"),
    "r_model":          ("executors.ml.r_model_executor",       "RModelExecutor"),
    "automl":           ("executors.ml.automl_executor",        "AutoMLExecutor"),
    "scorecard":        ("executors.ml.scorecard_executor",     "ScorecardExecutor"),
    "predict":          ("executors.ml.predict_executor",       "PredictExecutor"),
    "pretrained":       ("executors.ml.pretrained_executor",    "PretrainedExecutor"),
    "report":           ("executors.ml.report_executor",        "ReportExecutor"),
    "export":           ("executors.ml.export_executor",        "ExportExecutor"),
    "rulesearch":       ("executors.ml.rulesearch_executor",    "RuleSearchExecutor"),
    "stg":              ("executors.ml.stg_executor",           "StrategyExecutor"),
    "rl":               ("executors.ml.rl_executor",            "RLExecutor"),
}


class ProcessExecutor(BaseExecutor):
    """
    ML 파이프라인 오케스트레이션 executor.

    config 필수 키
    --------------
    pipeline : list  실행할 단계 목록
      각 항목:
        - name     : str   단계 이름 (고유)
        - executor : str   executor 유형 (EXECUTOR_REGISTRY 키)
        - config   : dict  해당 executor의 config
        - on_error : str   "stop" | "skip" | "continue" (기본: "stop")
        - input_from: str  이전 단계 결과를 이 단계 config에 병합할 필드명 (선택)

    config 선택 키
    --------------
    stop_on_first_failure : bool  첫 번째 실패 시 중단 (기본 True)
    """

    def execute(self) -> dict:
        pipeline = self.config.get("pipeline", [])
        if not pipeline:
            raise ExecutorException("pipeline이 비어 있습니다.")

        stop_on_fail = self.config.get("stop_on_first_failure", True)
        step_results: list[dict] = []
        context: dict[str, Any] = {}   # 단계 간 결과 공유

        total_steps = len(pipeline)

        for step_idx, step in enumerate(pipeline):
            step_name     = step["name"]
            executor_type = step["executor"]
            step_config   = dict(step.get("config", {}))

            # 이전 단계 결과 주입
            input_from = step.get("input_from")
            if input_from and input_from in context:
                step_config.update(context[input_from])

            # job_id / service_id 전파
            step_config.setdefault("job_id",     f"{self.job_id}__{step_name}")
            step_config.setdefault("service_id", self.service_id)
            step_config.setdefault("project_id", self.project_id)

            progress_start = int(step_idx / total_steps * 90)
            self._update_job_status(
                ExecutorStatus.RUNNING,
                progress=float(progress_start),
                message=f"실행 중: [{step_idx+1}/{total_steps}] {step_name}",
            )
            logger.info("[Pipeline] step %d/%d: %s (%s)", step_idx + 1, total_steps, step_name, executor_type)

            # executor 인스턴스 생성 및 실행
            try:
                executor = self._build_executor(executor_type, step_config)
                step_result = executor.run()
            except Exception as exc:
                step_result = {
                    "status":  ExecutorStatus.FAILED,
                    "job_id":  step_config["job_id"],
                    "result":  {},
                    "message": str(exc),
                }

            step_results.append({"step": step_name, **step_result})

            # 결과를 context에 저장
            context[step_name] = step_result.get("result", {})

            # 실패 처리
            if step_result["status"] == ExecutorStatus.FAILED:
                on_error = step.get("on_error", "stop")
                logger.warning("[Pipeline] step failed: %s  policy=%s", step_name, on_error)
                if on_error == "stop" and stop_on_fail:
                    break
                elif on_error == "skip":
                    continue

        # 전체 결과 요약
        failed_steps  = [r["step"] for r in step_results if r["status"] == ExecutorStatus.FAILED]
        overall_status = ExecutorStatus.FAILED if failed_steps else ExecutorStatus.COMPLETED

        return {
            "status":        overall_status,
            "result": {
                "pipeline_name": self.config.get("pipeline_name", "unnamed"),
                "total_steps":   total_steps,
                "executed":      len(step_results),
                "failed_steps":  failed_steps,
                "step_results":  step_results,
            },
            "message": (
                f"파이프라인 완료  {len(step_results)}/{total_steps}단계"
                + (f"  실패={failed_steps}" if failed_steps else "")
            ),
        }

    # ------------------------------------------------------------------

    def _build_executor(self, executor_type: str, config: dict) -> BaseExecutor:
        if executor_type not in EXECUTOR_REGISTRY:
            raise ExecutorException(f"등록되지 않은 executor: {executor_type}  등록 목록: {list(EXECUTOR_REGISTRY)}")

        module_path, class_name = EXECUTOR_REGISTRY[executor_type]
        module = importlib.import_module(module_path)
        cls    = getattr(module, class_name)
        return cls(config=config, db_session=self.db_session, file_root_dir=str(self.file_root))


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


def _check_process_name_already_exists(company_db_info: dict, prcs_name: str) -> bool:
    """프로세스 이름 중복 여부를 확인한다. 이미 존재하면 True, 없으면 False 반환."""
    from sqlalchemy import text
    engine = _make_engine(company_db_info)
    sql = text("SELECT COUNT(*) FROM tb_process WHERE prcs_name = :prcs_name")
    with engine.connect() as conn:
        result = conn.execute(sql, {"prcs_name": prcs_name})
        count = result.scalar()
    engine.dispose()
    return (count or 0) > 0


def get_system_auth_id(service_db_info: dict, auth_key: str, id_gb_cd: str) -> dict:
    """
    시스템 인증 ID를 조회한다.
    auth_key로 사용자를 검증하고, id_gb_cd로 구분 코드를 필터링한다.
    """
    from sqlalchemy import text
    engine = _make_engine(service_db_info)
    # 1) auth_key 로 사용자 검증
    user_sql = text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key")
    with engine.connect() as conn:
        user_row = conn.execute(user_sql, {"auth_key": auth_key}).fetchone()
        if user_row is None:
            engine.dispose()
            raise ValueError(f"유효하지 않은 auth_key: {auth_key}")
        user_id = user_row[0]

        # 2) 시스템 인증 ID 조회
        sql = text(
            "SELECT id_gb_cd, sys_auth_id, sys_auth_pw, user_id, reg_dtm "
            "FROM tb_sys_auth_id "
            "WHERE id_gb_cd = :id_gb_cd AND user_id = :user_id"
        )
        rows = conn.execute(sql, {"id_gb_cd": id_gb_cd, "user_id": user_id}).fetchall()

    engine.dispose()
    result = [dict(row._mapping) for row in rows]
    logger.debug("get_system_auth_id: id_gb_cd=%s rows=%d", id_gb_cd, len(result))
    return {"result": result}


def get_process_list(service_db_info: dict, json_obj: dict) -> dict:
    """
    전체 프로세스 목록을 반환한다.
    json_obj에서 auth_key, page, page_size를 추출하여 페이징 처리한다.
    """
    from sqlalchemy import text
    auth_key  = json_obj.get("auth_key", "")
    page      = int(json_obj.get("page", 1))
    page_size = int(json_obj.get("page_size", 20))
    offset    = (page - 1) * page_size

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        # 사용자 검증
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
        if user_row is None:
            engine.dispose()
            raise ValueError(f"유효하지 않은 auth_key: {auth_key}")
        user_id = user_row[0]

        count_sql = text(
            "SELECT COUNT(*) FROM tb_process p "
            "LEFT JOIN tb_process_auth pa ON p.prcs_seq = pa.prcs_seq "
            "WHERE pa.user_id = :user_id OR p.owner_id = :user_id"
        )
        total = conn.execute(count_sql, {"user_id": user_id}).scalar() or 0

        list_sql = text(
            "SELECT p.prcs_seq, p.prcs_name, p.prcs_desc, p.owner_id, p.reg_dtm, p.upd_dtm "
            "FROM tb_process p "
            "LEFT JOIN tb_process_auth pa ON p.prcs_seq = pa.prcs_seq "
            "WHERE pa.user_id = :user_id OR p.owner_id = :user_id "
            "ORDER BY p.upd_dtm DESC "
            "LIMIT :limit OFFSET :offset"
        )
        rows = conn.execute(list_sql, {"user_id": user_id, "limit": page_size, "offset": offset}).fetchall()

    engine.dispose()
    process_list = [dict(r._mapping) for r in rows]
    logger.debug("get_process_list: user_id=%s total=%d", user_id, total)
    return {"total": total, "page": page, "page_size": page_size, "result": process_list}


def get_export_process_list(service_db_info: dict, json_obj: dict) -> dict:
    """외부 내보내기(export)가 가능한 프로세스 목록을 반환한다."""
    from sqlalchemy import text
    auth_key = json_obj.get("auth_key", "")
    engine   = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
        if user_row is None:
            engine.dispose()
            raise ValueError(f"유효하지 않은 auth_key: {auth_key}")
        user_id = user_row[0]

        sql = text(
            "SELECT p.prcs_seq, p.prcs_name, p.prcs_desc, p.owner_id, p.reg_dtm "
            "FROM tb_process p "
            "LEFT JOIN tb_process_auth pa ON p.prcs_seq = pa.prcs_seq "
            "WHERE (pa.user_id = :user_id OR p.owner_id = :user_id) "
            "  AND p.export_yn = 'Y' "
            "ORDER BY p.prcs_name"
        )
        rows = conn.execute(sql, {"user_id": user_id}).fetchall()

    engine.dispose()
    result = [dict(r._mapping) for r in rows]
    logger.debug("get_export_process_list: user_id=%s count=%d", user_id, len(result))
    return {"result": result}


def modify_process(service_db_info: dict, json_obj: dict) -> dict:
    """프로세스 메타 정보(이름, 설명 등)를 수정한다."""
    from sqlalchemy import text
    from datetime import datetime
    prcs_seq  = json_obj["prcs_seq"]
    prcs_name = json_obj.get("prcs_name")
    prcs_desc = json_obj.get("prcs_desc", "")
    upd_dtm   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    engine = _make_engine(service_db_info)
    with engine.begin() as conn:
        sql = text(
            "UPDATE tb_process "
            "SET prcs_name = :prcs_name, prcs_desc = :prcs_desc, upd_dtm = :upd_dtm "
            "WHERE prcs_seq = :prcs_seq"
        )
        conn.execute(sql, {
            "prcs_name": prcs_name,
            "prcs_desc": prcs_desc,
            "upd_dtm":   upd_dtm,
            "prcs_seq":  prcs_seq,
        })

    engine.dispose()
    logger.info("modify_process: prcs_seq=%s prcs_name=%s", prcs_seq, prcs_name)
    return {"result": "ok", "prcs_seq": prcs_seq}


def create_process(service_db_info: dict, root_dir: str, json_obj: dict) -> dict:
    """
    신규 프로세스를 생성하고 관련 디렉토리를 구성한다.
    중복 이름이 존재하면 예외를 발생시킨다.
    """
    import uuid
    from pathlib import Path
    from datetime import datetime
    from sqlalchemy import text

    auth_key  = json_obj.get("auth_key", "")
    prcs_name = json_obj["prcs_name"]
    prcs_desc = json_obj.get("prcs_desc", "")

    # 중복 체크
    if _check_process_name_already_exists(service_db_info, prcs_name):
        raise ValueError(f"이미 존재하는 프로세스 이름입니다: {prcs_name}")

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
        if user_row is None:
            engine.dispose()
            raise ValueError(f"유효하지 않은 auth_key: {auth_key}")
        user_id = user_row[0]

    prcs_seq  = str(uuid.uuid4()).replace("-", "")[:16]
    reg_dtm   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO tb_process (prcs_seq, prcs_name, prcs_desc, owner_id, reg_dtm, upd_dtm) "
                "VALUES (:prcs_seq, :prcs_name, :prcs_desc, :owner_id, :reg_dtm, :upd_dtm)"
            ),
            {
                "prcs_seq":  prcs_seq,
                "prcs_name": prcs_name,
                "prcs_desc": prcs_desc,
                "owner_id":  user_id,
                "reg_dtm":   reg_dtm,
                "upd_dtm":   reg_dtm,
            },
        )

    # 디렉토리 구성
    prcs_dir = Path(root_dir) / "processes" / prcs_seq
    prcs_dir.mkdir(parents=True, exist_ok=True)
    (prcs_dir / "models").mkdir(exist_ok=True)
    (prcs_dir / "output").mkdir(exist_ok=True)

    engine.dispose()
    logger.info("create_process: prcs_seq=%s prcs_name=%s dir=%s", prcs_seq, prcs_name, prcs_dir)
    return {"result": "ok", "prcs_seq": prcs_seq, "prcs_dir": str(prcs_dir)}


def apply_process(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """
    프로세스를 복사(apply)한다. 파일 서버에 프로세스 파일을 복사하고 DB 레코드를 갱신한다.
    """
    import shutil
    from pathlib import Path
    from datetime import datetime
    from sqlalchemy import text

    src_prcs_seq  = json_obj["src_prcs_seq"]
    dest_prcs_seq = json_obj["dest_prcs_seq"]
    root_dir      = json_obj.get("root_dir", "/data")
    upd_dtm       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    src_dir  = Path(root_dir) / "processes" / src_prcs_seq
    dest_dir = Path(root_dir) / "processes" / dest_prcs_seq

    if src_dir.exists():
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(str(src_dir), str(dest_dir))
        logger.info("apply_process: copied %s → %s", src_dir, dest_dir)
    else:
        logger.warning("apply_process: src_dir not found: %s", src_dir)

    engine = _make_engine(service_db_info)
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE tb_process SET upd_dtm = :upd_dtm WHERE prcs_seq = :prcs_seq"),
            {"upd_dtm": upd_dtm, "prcs_seq": dest_prcs_seq},
        )
    engine.dispose()

    return {
        "result":        "ok",
        "src_prcs_seq":  src_prcs_seq,
        "dest_prcs_seq": dest_prcs_seq,
        "file_server":   f"{file_server_host}:{file_server_port}",
    }


def delete_process(service_db_info: dict, root_dir: str, json_obj: dict) -> dict:
    """프로세스 관련 파일 및 DB 레코드를 삭제한다."""
    import shutil
    from pathlib import Path
    from sqlalchemy import text

    prcs_seq = json_obj["prcs_seq"]
    engine   = _make_engine(service_db_info)

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM tb_process_auth WHERE prcs_seq = :prcs_seq"),
            {"prcs_seq": prcs_seq},
        )
        conn.execute(
            text("DELETE FROM tb_process WHERE prcs_seq = :prcs_seq"),
            {"prcs_seq": prcs_seq},
        )

    engine.dispose()

    prcs_dir = Path(root_dir) / "processes" / prcs_seq
    if prcs_dir.exists():
        shutil.rmtree(prcs_dir)
        logger.info("delete_process: removed dir %s", prcs_dir)

    logger.info("delete_process: prcs_seq=%s", prcs_seq)
    return {"result": "ok", "prcs_seq": prcs_seq}


def save_process(service_db_info: dict, meta_json: dict, process_bytes: bytes) -> dict:
    """
    프로세스 직렬화(bytes) 데이터와 메타 정보를 DB에 저장한다.
    이미 레코드가 있으면 UPDATE, 없으면 INSERT(upsert).
    """
    import json as _json
    from datetime import datetime
    from sqlalchemy import text

    prcs_seq = meta_json["prcs_seq"]
    reg_dtm  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_str = _json.dumps(meta_json, ensure_ascii=False)

    engine = _make_engine(service_db_info)
    with engine.begin() as conn:
        exists = conn.execute(
            text("SELECT COUNT(*) FROM tb_process_obj WHERE prcs_seq = :prcs_seq"),
            {"prcs_seq": prcs_seq},
        ).scalar()

        if exists:
            conn.execute(
                text(
                    "UPDATE tb_process_obj "
                    "SET process_obj = :process_obj, meta_json = :meta_json, reg_dtm = :reg_dtm "
                    "WHERE prcs_seq = :prcs_seq"
                ),
                {
                    "process_obj": process_bytes,
                    "meta_json":   meta_str,
                    "reg_dtm":     reg_dtm,
                    "prcs_seq":    prcs_seq,
                },
            )
        else:
            conn.execute(
                text(
                    "INSERT INTO tb_process_obj (prcs_seq, process_obj, meta_json, reg_dtm) "
                    "VALUES (:prcs_seq, :process_obj, :meta_json, :reg_dtm)"
                ),
                {
                    "prcs_seq":    prcs_seq,
                    "process_obj": process_bytes,
                    "meta_json":   meta_str,
                    "reg_dtm":     reg_dtm,
                },
            )

    engine.dispose()
    logger.info("save_process: prcs_seq=%s size=%d bytes", prcs_seq, len(process_bytes))
    return {"result": "ok", "prcs_seq": prcs_seq}


def get_process_obj(service_db_info: dict, auth_key: str, process_seq: str) -> dict:
    """
    프로세스 직렬화 객체를 조회한다.
    auth_key로 사용자를 검증한 뒤 process_seq에 해당하는 바이너리 객체를 반환한다.
    """
    import base64
    from sqlalchemy import text

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
        if user_row is None:
            engine.dispose()
            raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

        row = conn.execute(
            text(
                "SELECT process_obj, meta_json, reg_dtm "
                "FROM tb_process_obj WHERE prcs_seq = :prcs_seq"
            ),
            {"prcs_seq": process_seq},
        ).fetchone()

    engine.dispose()
    if row is None:
        raise ValueError(f"프로세스 객체를 찾을 수 없습니다: {process_seq}")

    process_bytes = row[0]
    meta_json     = row[1]
    reg_dtm       = row[2]

    # bytes → base64 문자열로 직렬화하여 반환
    obj_b64 = base64.b64encode(process_bytes).decode("utf-8") if process_bytes else None
    logger.debug("get_process_obj: prcs_seq=%s reg_dtm=%s", process_seq, reg_dtm)
    return {"prcs_seq": process_seq, "process_obj": obj_b64, "meta_json": meta_json, "reg_dtm": str(reg_dtm)}


def insert_system_auth_id(service_db_info: dict, json_obj: dict) -> dict:
    """시스템 인증 ID를 등록한다."""
    from datetime import datetime
    from sqlalchemy import text

    id_gb_cd    = json_obj["id_gb_cd"]
    sys_auth_id = json_obj["sys_auth_id"]
    sys_auth_pw = json_obj.get("sys_auth_pw", "")
    user_id     = json_obj["user_id"]
    reg_dtm     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    engine = _make_engine(service_db_info)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO tb_sys_auth_id (id_gb_cd, sys_auth_id, sys_auth_pw, user_id, reg_dtm) "
                "VALUES (:id_gb_cd, :sys_auth_id, :sys_auth_pw, :user_id, :reg_dtm)"
            ),
            {
                "id_gb_cd":    id_gb_cd,
                "sys_auth_id": sys_auth_id,
                "sys_auth_pw": sys_auth_pw,
                "user_id":     user_id,
                "reg_dtm":     reg_dtm,
            },
        )
    engine.dispose()
    logger.info("insert_system_auth_id: id_gb_cd=%s user_id=%s", id_gb_cd, user_id)
    return {"result": "ok"}


def delete_system_auth_id(
    service_db_info: dict,
    auth_key: str,
    id_gb_cd: str,
    user_name: str,
) -> dict:
    """시스템 인증 ID를 삭제한다."""
    from sqlalchemy import text

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
        if user_row is None:
            engine.dispose()
            raise ValueError(f"유효하지 않은 auth_key: {auth_key}")
        user_id = user_row[0]

    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM tb_sys_auth_id "
                "WHERE id_gb_cd = :id_gb_cd AND user_id = :user_id"
            ),
            {"id_gb_cd": id_gb_cd, "user_id": user_id},
        )
    engine.dispose()
    logger.info("delete_system_auth_id: id_gb_cd=%s user_name=%s", id_gb_cd, user_name)
    return {"result": "ok"}


def get_process_history_meta(
    service_db_info: dict,
    auth_key: str,
    process_seq: str,
) -> dict:
    """프로세스 변경 이력의 메타 정보를 조회한다."""
    from sqlalchemy import text

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
            text(
                "SELECT prcs_seq, reg_dtm, user_id, change_desc "
                "FROM tb_process_history "
                "WHERE prcs_seq = :prcs_seq "
                "ORDER BY reg_dtm DESC"
            ),
            {"prcs_seq": process_seq},
        ).fetchall()

    engine.dispose()
    result = [dict(r._mapping) for r in rows]
    logger.debug("get_process_history_meta: prcs_seq=%s count=%d", process_seq, len(result))
    return {"prcs_seq": process_seq, "history": result}


def get_process_history_obj(
    service_db_info: dict,
    auth_key: str,
    process_seq: str,
    reg_dtm: str,
) -> dict:
    """특정 시점의 프로세스 이력 객체를 조회한다."""
    import base64
    from sqlalchemy import text

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
        if user_row is None:
            engine.dispose()
            raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

        row = conn.execute(
            text(
                "SELECT process_obj, meta_json "
                "FROM tb_process_history "
                "WHERE prcs_seq = :prcs_seq AND reg_dtm = :reg_dtm"
            ),
            {"prcs_seq": process_seq, "reg_dtm": reg_dtm},
        ).fetchone()

    engine.dispose()
    if row is None:
        raise ValueError(f"이력 객체를 찾을 수 없습니다: prcs_seq={process_seq} reg_dtm={reg_dtm}")

    process_bytes = row[0]
    meta_json     = row[1]
    obj_b64 = base64.b64encode(process_bytes).decode("utf-8") if process_bytes else None
    logger.debug("get_process_history_obj: prcs_seq=%s reg_dtm=%s", process_seq, reg_dtm)
    return {"prcs_seq": process_seq, "reg_dtm": reg_dtm, "process_obj": obj_b64, "meta_json": meta_json}


def create_user_auth(service_db_info: dict, json_obj: dict) -> dict:
    """프로세스에 대한 사용자 권한을 등록한다."""
    from datetime import datetime
    from sqlalchemy import text

    prcs_seq   = json_obj["prcs_seq"]
    user_id    = json_obj["user_id"]
    auth_level = json_obj.get("auth_level", 1)
    reg_dtm    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    engine = _make_engine(service_db_info)
    with engine.begin() as conn:
        # 기존 권한이 있으면 UPDATE, 없으면 INSERT
        exists = conn.execute(
            text(
                "SELECT COUNT(*) FROM tb_process_auth "
                "WHERE prcs_seq = :prcs_seq AND user_id = :user_id"
            ),
            {"prcs_seq": prcs_seq, "user_id": user_id},
        ).scalar()

        if exists:
            conn.execute(
                text(
                    "UPDATE tb_process_auth SET auth_level = :auth_level, upd_dtm = :reg_dtm "
                    "WHERE prcs_seq = :prcs_seq AND user_id = :user_id"
                ),
                {"auth_level": auth_level, "reg_dtm": reg_dtm, "prcs_seq": prcs_seq, "user_id": user_id},
            )
        else:
            conn.execute(
                text(
                    "INSERT INTO tb_process_auth (prcs_seq, user_id, auth_level, reg_dtm) "
                    "VALUES (:prcs_seq, :user_id, :auth_level, :reg_dtm)"
                ),
                {"prcs_seq": prcs_seq, "user_id": user_id, "auth_level": auth_level, "reg_dtm": reg_dtm},
            )

    engine.dispose()
    logger.info("create_user_auth: prcs_seq=%s user_id=%s auth_level=%s", prcs_seq, user_id, auth_level)
    return {"result": "ok", "prcs_seq": prcs_seq, "user_id": user_id}


def get_user_authorities(
    service_db_info: dict,
    auth_key: str,
    prcs_seq: str,
    user_ids: list = [],
) -> dict:
    """프로세스에 대한 사용자 권한 목록을 조회한다."""
    from sqlalchemy import text

    engine = _make_engine(service_db_info)
    with engine.connect() as conn:
        user_row = conn.execute(
            text("SELECT user_id FROM tb_user_auth WHERE auth_key = :auth_key"),
            {"auth_key": auth_key},
        ).fetchone()
        if user_row is None:
            engine.dispose()
            raise ValueError(f"유효하지 않은 auth_key: {auth_key}")

        if user_ids:
            placeholders = ", ".join(f":uid_{i}" for i in range(len(user_ids)))
            params = {"prcs_seq": prcs_seq}
            params.update({f"uid_{i}": uid for i, uid in enumerate(user_ids)})
            sql = text(
                f"SELECT user_id, auth_level, reg_dtm FROM tb_process_auth "
                f"WHERE prcs_seq = :prcs_seq AND user_id IN ({placeholders})"
            )
        else:
            params = {"prcs_seq": prcs_seq}
            sql = text(
                "SELECT user_id, auth_level, reg_dtm FROM tb_process_auth "
                "WHERE prcs_seq = :prcs_seq"
            )

        rows = conn.execute(sql, params).fetchall()

    engine.dispose()
    result = [dict(r._mapping) for r in rows]
    logger.debug("get_user_authorities: prcs_seq=%s count=%d", prcs_seq, len(result))
    return {"prcs_seq": prcs_seq, "authorities": result}


def overwrite_template(service_db_info: dict, root_dir: str, json_obj: dict) -> dict:
    """프로세스 템플릿 파일을 덮어쓴다."""
    import shutil
    from pathlib import Path
    from datetime import datetime
    from sqlalchemy import text

    prcs_seq      = json_obj["prcs_seq"]
    template_name = json_obj.get("template_name", "default")
    template_dir  = Path(root_dir) / "templates" / template_name
    prcs_dir      = Path(root_dir) / "processes" / prcs_seq

    if not template_dir.exists():
        raise FileNotFoundError(f"템플릿 디렉토리가 없습니다: {template_dir}")

    prcs_dir.mkdir(parents=True, exist_ok=True)
    for item in template_dir.iterdir():
        dest = prcs_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(str(item), str(dest))
        else:
            shutil.copy2(str(item), str(dest))

    upd_dtm = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    engine  = _make_engine(service_db_info)
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE tb_process SET upd_dtm = :upd_dtm WHERE prcs_seq = :prcs_seq"),
            {"upd_dtm": upd_dtm, "prcs_seq": prcs_seq},
        )
    engine.dispose()

    logger.info("overwrite_template: prcs_seq=%s template=%s", prcs_seq, template_name)
    return {"result": "ok", "prcs_seq": prcs_seq, "template_name": template_name}
