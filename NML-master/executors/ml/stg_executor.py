"""
stg_executor.py  (Strategy Executor)
-------------------------------------
모델 예측 결과를 실제 업무 전략(Strategy)으로 변환하는 실행기.

금융/리스크 도메인에서 모델이 산출한 점수/확률값을 바탕으로
승인/거절/한도/등급 등의 업무 의사결정을 자동화한다.

전략 유형:
  - grade_strategy    : 점수 → 등급 (A/B/C/D/E) 매핑
  - threshold_strategy: 임계값 기반 승인/거절
  - tiered_strategy   : 다단계 정책 (등급별 한도/조건)
  - matrix_strategy   : 2차원 매트릭스 (score × 기존등급)
  - override_rule     : 정책 룰에 의한 강제 오버라이드

실행 순서:
  1. 예측 점수 데이터 로드
  2. 전략 설정 파싱
  3. 전략 적용 (등급화 / 승인결정 / 한도산출)
  4. 오버라이드 룰 적용
  5. 결과 저장 및 요약
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)


class StrategyExecutor(BaseExecutor):
    """
    업무 전략 적용 executor.

    config 필수 키
    --------------
    input_path     : str   예측 점수가 포함된 데이터 경로 (.parquet)
    score_col      : str   점수 컬럼명
    strategy_type  : str   "grade" | "threshold" | "tiered" | "matrix"
    output_id      : str   결과 저장 식별자

    config 선택 키
    --------------
    grade_map      : dict  등급별 점수 구간 {"A": [800,1000], "B": [600,800], ...}
    threshold      : float 이진 승인/거절 임계값
    tiered_rules   : list  다단계 정책 목록
    matrix_rules   : dict  2차원 매트릭스 정의
    override_rules : list  오버라이드 룰 목록
    key_cols       : list  결과에 포함할 키 컬럼 목록
    """

    def execute(self) -> dict:
        cfg = self.config
        strategy_type = cfg["strategy_type"]
        score_col     = cfg["score_col"]

        df = self._load_dataframe(cfg["input_path"])
        if score_col not in df.columns:
            raise ExecutorException(f"점수 컬럼이 없습니다: {score_col}")

        logger.info("전략 적용 시작  type=%s  rows=%d", strategy_type, len(df))
        self._update_job_status(ExecutorStatus.RUNNING, progress=20)

        # 전략 적용
        if strategy_type == "grade":
            df = self._apply_grade_strategy(df, score_col, cfg["grade_map"])
        elif strategy_type == "threshold":
            df = self._apply_threshold_strategy(df, score_col, cfg["threshold"])
        elif strategy_type == "tiered":
            df = self._apply_tiered_strategy(df, score_col, cfg["tiered_rules"])
        elif strategy_type == "matrix":
            df = self._apply_matrix_strategy(df, score_col, cfg["matrix_rules"])
        else:
            raise ExecutorException(f"지원하지 않는 strategy_type: {strategy_type}")

        self._update_job_status(ExecutorStatus.RUNNING, progress=65)

        # 오버라이드 룰 적용
        override_rules = cfg.get("override_rules", [])
        if override_rules:
            df, override_cnt = self._apply_overrides(df, override_rules)
            logger.info("오버라이드 적용: %d건", override_cnt)

        self._update_job_status(ExecutorStatus.RUNNING, progress=85)

        # 저장
        key_cols = cfg.get("key_cols", [])
        save_cols = key_cols + [score_col] + [
            c for c in ["grade", "decision", "limit_amt", "override_flag"]
            if c in df.columns
        ]
        save_df = df[[c for c in save_cols if c in df.columns]]
        output_path = f"strategy/{cfg['output_id']}_result.parquet"
        self._save_dataframe(save_df, output_path)

        summary = self._build_summary(save_df, cfg)
        self._save_json(summary, f"strategy/{cfg['output_id']}_summary.json")

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  {
                "output_path": output_path,
                "total_rows":  len(df),
                "summary":     summary,
            },
            "message": f"전략 적용 완료  type={strategy_type}  {len(df):,}건",
        }

    # ------------------------------------------------------------------

    def _apply_grade_strategy(self, df, score_col, grade_map) -> pd.DataFrame:
        """
        grade_map 예시:
          {"A": [800, 1000], "B": [600, 800], "C": [400, 600], "D": [0, 400]}
        """
        def _grade(score):
            for grade, (low, high) in grade_map.items():
                if low <= score < high:
                    return grade
            return "UNKNOWN"
        df["grade"]    = df[score_col].apply(_grade)
        df["decision"] = df["grade"].apply(lambda g: "APPROVE" if g in ["A", "B"] else "REJECT")
        return df

    def _apply_threshold_strategy(self, df, score_col, threshold: float) -> pd.DataFrame:
        df["decision"] = np.where(df[score_col] >= threshold, "APPROVE", "REJECT")
        return df

    def _apply_tiered_strategy(self, df, score_col, tiered_rules: list) -> pd.DataFrame:
        """
        tiered_rules 예시:
          [{"score_min": 800, "grade": "A", "limit_pct": 1.0, "rate": 3.5},
           {"score_min": 600, "grade": "B", "limit_pct": 0.7, "rate": 5.0},
           {"score_min": 0,   "grade": "C", "limit_pct": 0.0, "rate": None, "reject": True}]
        """
        rules_sorted = sorted(tiered_rules, key=lambda r: r["score_min"], reverse=True)

        grades, decisions, limit_pcts = [], [], []
        for score in df[score_col]:
            applied = False
            for rule in rules_sorted:
                if score >= rule["score_min"]:
                    grades.append(rule.get("grade", "UNKNOWN"))
                    decisions.append("REJECT" if rule.get("reject") else "APPROVE")
                    limit_pcts.append(rule.get("limit_pct", 0.0))
                    applied = True
                    break
            if not applied:
                grades.append("REJECT_ALL")
                decisions.append("REJECT")
                limit_pcts.append(0.0)

        df["grade"]      = grades
        df["decision"]   = decisions
        df["limit_pct"]  = limit_pcts
        return df

    def _apply_matrix_strategy(self, df, score_col, matrix_rules: dict) -> pd.DataFrame:
        """
        matrix_rules 예시:
          {"existing_grade_col": "crif_grade",
           "matrix": {"A": {"700+": "APPROVE_FULL", "500-700": "APPROVE_PARTIAL"},
                      "B": {"700+": "APPROVE_PARTIAL", "500-700": "REVIEW"}}}
        """
        existing_col = matrix_rules["existing_grade_col"]
        matrix       = matrix_rules["matrix"]

        def _get_score_band(score):
            if score >= 700: return "700+"
            elif score >= 500: return "500-700"
            else: return "500-"

        if existing_col not in df.columns:
            raise ExecutorException(f"기존 등급 컬럼 없음: {existing_col}")

        decisions = []
        for _, row in df.iterrows():
            ex_grade   = str(row[existing_col])
            score_band = _get_score_band(row[score_col])
            decision   = matrix.get(ex_grade, {}).get(score_band, "REVIEW")
            decisions.append(decision)
        df["decision"] = decisions
        return df

    def _apply_overrides(self, df, override_rules: list):
        """
        override_rules 예시:
          [{"condition": "debt_ratio > 0.9", "decision": "REJECT", "reason": "고DSR"},
           {"condition": "fraud_flag == 1",  "decision": "REJECT", "reason": "사기이력"}]
        """
        if "override_flag" not in df.columns:
            df["override_flag"]   = 0
            df["override_reason"] = ""

        override_cnt = 0
        for rule in override_rules:
            try:
                mask = df.eval(rule["condition"])
                before_cnt = df.loc[mask, "decision"].eq(rule["decision"]).sum()
                df.loc[mask, "decision"]       = rule["decision"]
                df.loc[mask, "override_flag"]  = 1
                df.loc[mask, "override_reason"] = rule.get("reason", rule["condition"])
                override_cnt += int(mask.sum()) - int(before_cnt)
            except Exception as exc:
                logger.warning("오버라이드 룰 적용 실패: %s  reason=%s", rule["condition"], exc)
        return df, override_cnt

    def _build_summary(self, df, cfg) -> dict:
        summary: dict = {"strategy_type": cfg["strategy_type"], "total": len(df)}
        if "decision" in df.columns:
            summary["decision_dist"] = df["decision"].value_counts().to_dict()
            approve_cnt = (df["decision"].str.startswith("APPROVE")).sum()
            summary["approve_rate"] = round(float(approve_cnt) / len(df), 4)
        if "grade" in df.columns:
            summary["grade_dist"] = df["grade"].value_counts().to_dict()
        if "override_flag" in df.columns:
            summary["override_count"] = int(df["override_flag"].sum())
        return summary


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


def init_output_parameter(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """
    노드 출력 파라미터 파일을 빈 상태로 초기화한다.
    파일 서버의 출력 파라미터 파일을 비우거나 헤더만 남긴 상태로 재생성한다.
    """
    from pathlib import Path
    import json as _json

    process_seq = json_obj["process_seq"]
    stg_id      = json_obj["stg_id"]
    root_dir    = json_obj.get("root_dir", "/data")

    output_dir = Path(root_dir) / "processes" / process_seq / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{stg_id}_output_param.json"
    empty_payload = {"process_seq": process_seq, "stg_id": stg_id, "params": []}
    with open(output_file, "w", encoding="utf-8") as f:
        _json.dump(empty_payload, f, ensure_ascii=False)

    logger.info(
        "init_output_parameter: process_seq=%s stg_id=%s file=%s",
        process_seq, stg_id, output_file,
    )
    return {"result": "ok", "output_file": str(output_file)}


def save_stg_obj(service_db_info: dict, json_obj: dict) -> dict:
    """전략 노드 직렬화 데이터를 DB에 저장(upsert)한다."""
    import json as _json
    from datetime import datetime
    from sqlalchemy import text

    process_seq = json_obj["process_seq"]
    stg_id      = json_obj["stg_id"]
    stg_seq     = json_obj.get("stg_seq")
    stg_name    = json_obj.get("stg_name", "")
    stg_comment = json_obj.get("stg_comment", "")
    stg_type    = json_obj.get("stg_type", "")
    stg_obj     = json_obj.get("stg_obj")          # bytes or base64 string
    user_id     = json_obj.get("user_id", "")
    reg_dtm     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # base64 문자열이면 bytes로 디코딩
    import base64
    if isinstance(stg_obj, str):
        stg_obj = base64.b64decode(stg_obj)

    engine = _make_engine(service_db_info)
    with engine.begin() as conn:
        exists = conn.execute(
            text(
                "SELECT COUNT(*) FROM tb_stg_obj "
                "WHERE process_seq = :process_seq AND stg_id = :stg_id"
            ),
            {"process_seq": process_seq, "stg_id": stg_id},
        ).scalar()

        if exists:
            conn.execute(
                text(
                    "UPDATE tb_stg_obj "
                    "SET stg_name = :stg_name, stg_comment = :stg_comment, "
                    "    stg_type = :stg_type, stg_obj = :stg_obj, "
                    "    user_id = :user_id, reg_dtm = :reg_dtm "
                    "WHERE process_seq = :process_seq AND stg_id = :stg_id"
                ),
                {
                    "stg_name":    stg_name,
                    "stg_comment": stg_comment,
                    "stg_type":    stg_type,
                    "stg_obj":     stg_obj,
                    "user_id":     user_id,
                    "reg_dtm":     reg_dtm,
                    "process_seq": process_seq,
                    "stg_id":      stg_id,
                },
            )
        else:
            conn.execute(
                text(
                    "INSERT INTO tb_stg_obj "
                    "(process_seq, stg_id, stg_name, stg_comment, stg_type, stg_obj, user_id, reg_dtm) "
                    "VALUES (:process_seq, :stg_id, :stg_name, :stg_comment, "
                    "        :stg_type, :stg_obj, :user_id, :reg_dtm)"
                ),
                {
                    "process_seq": process_seq,
                    "stg_id":      stg_id,
                    "stg_name":    stg_name,
                    "stg_comment": stg_comment,
                    "stg_type":    stg_type,
                    "stg_obj":     stg_obj,
                    "user_id":     user_id,
                    "reg_dtm":     reg_dtm,
                },
            )
    engine.dispose()
    logger.info("save_stg_obj: process_seq=%s stg_id=%s", process_seq, stg_id)
    return {"result": "ok", "process_seq": process_seq, "stg_id": stg_id}


def save_stg_obj_with_properties(service_db_info: dict, json_obj: dict) -> dict:
    """전략 노드 직렬화 데이터와 속성(properties JSON)을 함께 저장한다."""
    import json as _json
    from datetime import datetime
    from sqlalchemy import text

    process_seq = json_obj["process_seq"]
    stg_id      = json_obj["stg_id"]
    stg_name    = json_obj.get("stg_name", "")
    stg_comment = json_obj.get("stg_comment", "")
    stg_type    = json_obj.get("stg_type", "")
    stg_obj     = json_obj.get("stg_obj")
    properties  = json_obj.get("properties", {})
    user_id     = json_obj.get("user_id", "")
    reg_dtm     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    props_str   = _json.dumps(properties, ensure_ascii=False)

    import base64
    if isinstance(stg_obj, str):
        stg_obj = base64.b64decode(stg_obj)

    engine = _make_engine(service_db_info)
    with engine.begin() as conn:
        exists = conn.execute(
            text(
                "SELECT COUNT(*) FROM tb_stg_obj "
                "WHERE process_seq = :process_seq AND stg_id = :stg_id"
            ),
            {"process_seq": process_seq, "stg_id": stg_id},
        ).scalar()

        if exists:
            conn.execute(
                text(
                    "UPDATE tb_stg_obj "
                    "SET stg_name = :stg_name, stg_comment = :stg_comment, "
                    "    stg_type = :stg_type, stg_obj = :stg_obj, "
                    "    properties = :properties, user_id = :user_id, reg_dtm = :reg_dtm "
                    "WHERE process_seq = :process_seq AND stg_id = :stg_id"
                ),
                {
                    "stg_name":    stg_name,
                    "stg_comment": stg_comment,
                    "stg_type":    stg_type,
                    "stg_obj":     stg_obj,
                    "properties":  props_str,
                    "user_id":     user_id,
                    "reg_dtm":     reg_dtm,
                    "process_seq": process_seq,
                    "stg_id":      stg_id,
                },
            )
        else:
            conn.execute(
                text(
                    "INSERT INTO tb_stg_obj "
                    "(process_seq, stg_id, stg_name, stg_comment, stg_type, "
                    " stg_obj, properties, user_id, reg_dtm) "
                    "VALUES (:process_seq, :stg_id, :stg_name, :stg_comment, :stg_type, "
                    "        :stg_obj, :properties, :user_id, :reg_dtm)"
                ),
                {
                    "process_seq": process_seq,
                    "stg_id":      stg_id,
                    "stg_name":    stg_name,
                    "stg_comment": stg_comment,
                    "stg_type":    stg_type,
                    "stg_obj":     stg_obj,
                    "properties":  props_str,
                    "user_id":     user_id,
                    "reg_dtm":     reg_dtm,
                },
            )
    engine.dispose()
    logger.info("save_stg_obj_with_properties: process_seq=%s stg_id=%s", process_seq, stg_id)
    return {"result": "ok", "process_seq": process_seq, "stg_id": stg_id}


def update_stg_description(service_db_info: dict, json_obj: dict) -> dict:
    """전략 노드의 설명(description/comment)을 업데이트한다."""
    from datetime import datetime
    from sqlalchemy import text

    process_seq = json_obj["process_seq"]
    stg_id      = json_obj["stg_id"]
    stg_comment = json_obj.get("stg_comment", "")
    reg_dtm     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    engine = _make_engine(service_db_info)
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE tb_stg_obj "
                "SET stg_comment = :stg_comment, reg_dtm = :reg_dtm "
                "WHERE process_seq = :process_seq AND stg_id = :stg_id"
            ),
            {
                "stg_comment": stg_comment,
                "reg_dtm":     reg_dtm,
                "process_seq": process_seq,
                "stg_id":      stg_id,
            },
        )
    engine.dispose()
    logger.info("update_stg_description: process_seq=%s stg_id=%s", process_seq, stg_id)
    return {"result": "ok"}


def update_stg_properties(service_db_info: dict, json_obj: dict) -> dict:
    """전략 노드의 속성(properties JSON)을 업데이트한다."""
    import json as _json
    from datetime import datetime
    from sqlalchemy import text

    process_seq = json_obj["process_seq"]
    stg_id      = json_obj["stg_id"]
    properties  = json_obj.get("properties", {})
    props_str   = _json.dumps(properties, ensure_ascii=False)
    reg_dtm     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    engine = _make_engine(service_db_info)
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE tb_stg_obj "
                "SET properties = :properties, reg_dtm = :reg_dtm "
                "WHERE process_seq = :process_seq AND stg_id = :stg_id"
            ),
            {
                "properties":  props_str,
                "reg_dtm":     reg_dtm,
                "process_seq": process_seq,
                "stg_id":      stg_id,
            },
        )
    engine.dispose()
    logger.info("update_stg_properties: process_seq=%s stg_id=%s", process_seq, stg_id)
    return {"result": "ok"}


def get_stg_history_properties(
    service_db_info: dict,
    auth_key: str,
    process_seq: str,
    stg_id: str,
) -> dict:
    """전략 노드의 변경 이력에서 속성(properties) 목록을 조회한다."""
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
                "SELECT stg_seq, reg_dtm, user_id, properties "
                "FROM tb_stg_history "
                "WHERE process_seq = :process_seq AND stg_id = :stg_id "
                "ORDER BY reg_dtm DESC"
            ),
            {"process_seq": process_seq, "stg_id": stg_id},
        ).fetchall()

    engine.dispose()
    result = [dict(r._mapping) for r in rows]
    logger.debug(
        "get_stg_history_properties: process_seq=%s stg_id=%s count=%d",
        process_seq, stg_id, len(result),
    )
    return {"process_seq": process_seq, "stg_id": stg_id, "history": result}


def get_stg_history_object(
    service_db_info: dict,
    auth_key: str,
    process_seq: str,
    stg_id: str,
    stg_seq: int,
) -> dict:
    """특정 시퀀스(stg_seq)의 전략 노드 이력 객체를 조회한다."""
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
                "SELECT stg_obj, properties, reg_dtm "
                "FROM tb_stg_history "
                "WHERE process_seq = :process_seq AND stg_id = :stg_id AND stg_seq = :stg_seq"
            ),
            {"process_seq": process_seq, "stg_id": stg_id, "stg_seq": stg_seq},
        ).fetchone()

    engine.dispose()
    if row is None:
        raise ValueError(
            f"이력 객체를 찾을 수 없습니다: process_seq={process_seq} stg_id={stg_id} stg_seq={stg_seq}"
        )

    stg_obj_bytes = row[0]
    properties    = row[1]
    reg_dtm       = row[2]
    obj_b64 = base64.b64encode(stg_obj_bytes).decode("utf-8") if stg_obj_bytes else None
    logger.debug(
        "get_stg_history_object: process_seq=%s stg_id=%s stg_seq=%s",
        process_seq, stg_id, stg_seq,
    )
    return {
        "process_seq": process_seq,
        "stg_id":      stg_id,
        "stg_seq":     stg_seq,
        "stg_obj":     obj_b64,
        "properties":  properties,
        "reg_dtm":     str(reg_dtm),
    }


def get_stg_history(
    service_db_info: dict,
    auth_key: str,
    process_seq: str,
    stg_id: str,
) -> dict:
    """전략 노드의 전체 변경 이력 목록(메타+객체 포함)을 조회한다."""
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

        rows = conn.execute(
            text(
                "SELECT stg_seq, stg_name, stg_comment, stg_type, "
                "       stg_obj, properties, user_id, reg_dtm "
                "FROM tb_stg_history "
                "WHERE process_seq = :process_seq AND stg_id = :stg_id "
                "ORDER BY reg_dtm DESC"
            ),
            {"process_seq": process_seq, "stg_id": stg_id},
        ).fetchall()

    engine.dispose()

    history = []
    for r in rows:
        entry = dict(r._mapping)
        raw_obj = entry.get("stg_obj")
        if isinstance(raw_obj, (bytes, bytearray)):
            entry["stg_obj"] = base64.b64encode(raw_obj).decode("utf-8")
        history.append(entry)

    logger.debug(
        "get_stg_history: process_seq=%s stg_id=%s count=%d",
        process_seq, stg_id, len(history),
    )
    return {"process_seq": process_seq, "stg_id": stg_id, "history": history}


def get_mart_data_joined_with_output_parameter(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """
    마트 데이터와 출력 파라미터(예측 결과) 파일을 키 컬럼 기준으로 JOIN하여 반환한다.
    """
    import json as _json
    from pathlib import Path
    import pandas as pd

    process_seq  = json_obj["process_seq"]
    stg_id       = json_obj["stg_id"]
    mart_path    = json_obj["mart_path"]          # parquet 절대 경로
    key_col      = json_obj.get("key_col", "id")
    root_dir     = json_obj.get("root_dir", "/data")
    page         = int(json_obj.get("page", 1))
    page_size    = int(json_obj.get("page_size", 100))

    # 마트 로드
    mart_df = pd.read_parquet(mart_path)

    # 출력 파라미터 로드
    output_file = Path(root_dir) / "processes" / process_seq / "output" / f"{stg_id}_output_param.json"
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            out_data = _json.load(f)
        params_list = out_data.get("params", [])
        if params_list:
            out_df = pd.DataFrame(params_list)
            merged = mart_df.merge(out_df, on=key_col, how="left")
        else:
            merged = mart_df.copy()
    else:
        merged = mart_df.copy()

    total = len(merged)
    offset = (page - 1) * page_size
    paged  = merged.iloc[offset: offset + page_size]

    logger.debug(
        "get_mart_data_joined_with_output_parameter: process_seq=%s stg_id=%s total=%d",
        process_seq, stg_id, total,
    )
    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "result":    paged.to_dict(orient="records"),
    }


def delete_stg(
    service_db_info: dict,
    auth_key: str,
    process_seq: str,
    stg_id: str,
    stg_seq: int,
) -> dict:
    """전략 노드 이력 항목을 삭제한다. stg_seq == 0 이면 현재 노드 자체를 삭제한다."""
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

    with engine.begin() as conn:
        if stg_seq == 0:
            # 현재 노드 및 모든 이력 삭제
            conn.execute(
                text("DELETE FROM tb_stg_history WHERE process_seq = :process_seq AND stg_id = :stg_id"),
                {"process_seq": process_seq, "stg_id": stg_id},
            )
            conn.execute(
                text("DELETE FROM tb_stg_obj WHERE process_seq = :process_seq AND stg_id = :stg_id"),
                {"process_seq": process_seq, "stg_id": stg_id},
            )
        else:
            conn.execute(
                text(
                    "DELETE FROM tb_stg_history "
                    "WHERE process_seq = :process_seq AND stg_id = :stg_id AND stg_seq = :stg_seq"
                ),
                {"process_seq": process_seq, "stg_id": stg_id, "stg_seq": stg_seq},
            )

    engine.dispose()
    logger.info("delete_stg: process_seq=%s stg_id=%s stg_seq=%s", process_seq, stg_id, stg_seq)
    return {"result": "ok", "process_seq": process_seq, "stg_id": stg_id, "stg_seq": stg_seq}


def get_number_of_rows_in_node(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """노드에 연결된 마트/출력 파라미터 데이터의 행 수를 반환한다."""
    from pathlib import Path
    import pandas as pd

    process_seq = json_obj["process_seq"]
    stg_id      = json_obj["stg_id"]
    mart_path   = json_obj.get("mart_path")
    root_dir    = json_obj.get("root_dir", "/data")

    row_count = 0
    if mart_path:
        p = Path(mart_path)
        if p.exists():
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            row_count = len(df)
        else:
            logger.warning("get_number_of_rows_in_node: mart_path not found: %s", mart_path)
    else:
        # 출력 파라미터 파일에서 행 수 확인
        output_file = (
            Path(root_dir) / "processes" / process_seq / "output" / f"{stg_id}_output_param.json"
        )
        if output_file.exists():
            import json as _json
            with open(output_file, "r", encoding="utf-8") as f:
                data = _json.load(f)
            row_count = len(data.get("params", []))

    logger.debug(
        "get_number_of_rows_in_node: process_seq=%s stg_id=%s rows=%d",
        process_seq, stg_id, row_count,
    )
    return {"process_seq": process_seq, "stg_id": stg_id, "row_count": row_count}


def get_mart_data_joined_with_output_parameter_and_condition(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """
    마트 데이터와 출력 파라미터를 JOIN하고 조건(condition)으로 필터링하여 반환한다.

    - 외부용 앱(external_app=True)이고 사용자 권한 레벨이 2~3이면 데이터 내보내기를 비활성화한다.
    - Windows 로컬 버전에서는 별도 경로 처리를 포함한다.
    """
    import sys
    import json as _json
    from pathlib import Path
    import pandas as pd
    from sqlalchemy import text

    process_seq   = json_obj["process_seq"]
    stg_id        = json_obj["stg_id"]
    mart_path     = json_obj["mart_path"]
    key_col       = json_obj.get("key_col", "id")
    condition     = json_obj.get("condition", "")       # pandas query 문자열
    auth_key      = json_obj.get("auth_key", "")
    external_app  = bool(json_obj.get("external_app", False))
    root_dir      = json_obj.get("root_dir", "/data")
    page          = int(json_obj.get("page", 1))
    page_size     = int(json_obj.get("page_size", 100))

    # Windows 로컬 버전 체크
    is_windows_local = sys.platform.startswith("win") and json_obj.get("local_version", False)

    # 외부 앱인 경우 사용자 권한 레벨 확인
    export_disabled = False
    if external_app and auth_key:
        engine = _make_engine(service_db_info)
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT pa.auth_level FROM tb_process_auth pa "
                    "JOIN tb_user_auth ua ON pa.user_id = ua.user_id "
                    "WHERE ua.auth_key = :auth_key AND pa.prcs_seq = :prcs_seq"
                ),
                {"auth_key": auth_key, "prcs_seq": process_seq},
            ).fetchone()
            if row is not None and row[0] in (2, 3):
                export_disabled = True
        engine.dispose()

    if export_disabled:
        logger.warning(
            "get_mart_data_joined_with_output_parameter_and_condition: "
            "export disabled for external app (auth_level 2 or 3)"
        )
        return {
            "export_disabled": True,
            "total":           0,
            "page":            page,
            "page_size":       page_size,
            "result":          [],
        }

    # Windows 로컬 버전: 경로 구분자 보정
    if is_windows_local:
        mart_path = mart_path.replace("/", "\\")

    # 마트 로드
    mart_p = Path(mart_path)
    if not mart_p.exists():
        raise FileNotFoundError(f"마트 파일을 찾을 수 없습니다: {mart_path}")
    mart_df = pd.read_parquet(mart_p) if mart_p.suffix == ".parquet" else pd.read_csv(mart_p)

    # 출력 파라미터 JOIN
    out_dir     = Path(root_dir) / "processes" / process_seq / "output"
    output_file = out_dir / f"{stg_id}_output_param.json"
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            out_data = _json.load(f)
        params_list = out_data.get("params", [])
        if params_list:
            out_df  = pd.DataFrame(params_list)
            merged  = mart_df.merge(out_df, on=key_col, how="left")
        else:
            merged = mart_df.copy()
    else:
        merged = mart_df.copy()

    # 조건 필터링
    if condition:
        try:
            merged = merged.query(condition)
        except Exception as exc:
            logger.warning(
                "get_mart_data_joined_with_output_parameter_and_condition: "
                "condition query failed: %s  reason=%s",
                condition, exc,
            )

    # 최종 내보내기 컬럼
    export_cols = json_obj.get("export_cols")
    if export_cols:
        valid_cols = [c for c in export_cols if c in merged.columns]
        merged = merged[valid_cols]

    total  = len(merged)
    offset = (page - 1) * page_size
    paged  = merged.iloc[offset: offset + page_size]

    logger.debug(
        "get_mart_data_joined_with_output_parameter_and_condition: "
        "process_seq=%s stg_id=%s total=%d is_windows_local=%s",
        process_seq, stg_id, total, is_windows_local,
    )
    return {
        "export_disabled": False,
        "total":           total,
        "page":            page,
        "page_size":       page_size,
        "result":          paged.to_dict(orient="records"),
    }
