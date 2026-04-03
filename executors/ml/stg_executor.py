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
