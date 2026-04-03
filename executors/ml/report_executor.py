"""
report_executor.py
------------------
분석/모델링 결과를 정형화된 리포트로 생성하는 실행기.

출력 형식: JSON (구조화 데이터), Excel (.xlsx), HTML (선택)

리포트 유형:
  - model_performance : 모델 성능 요약 (AUC, KS, 혼동행렬, ROC 데이터)
  - eda_report        : 데이터 분석 요약
  - scorecard_report  : 스코어카드 변수 요약표 및 성능
  - prediction_report : 예측 결과 분포 요약
  - combined          : 위 항목을 통합한 종합 리포트
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)


class ReportExecutor(BaseExecutor):
    """
    리포트 생성 executor.

    config 필수 키
    --------------
    report_type  : str   리포트 유형 ("model_performance" | "eda_report" | "scorecard_report" | "prediction_report" | "combined")
    output_id    : str   출력 식별자
    report_name  : str   리포트 제목

    config 선택 키
    --------------
    model_meta_path   : str   모델 메타 JSON 경로 (model_performance용)
    eda_result_path   : str   EDA 결과 JSON 경로 (eda_report용)
    scorecard_path    : str   스코어카드 결과 JSON 경로
    prediction_path   : str   예측 결과 parquet 경로
    score_col         : str   점수 컬럼명 (기본: "score")
    target_col        : str   타깃 컬럼명
    output_format     : list  출력 포맷 목록 ["json", "excel"] (기본: ["json", "excel"])
    """

    def execute(self) -> dict:
        cfg         = self.config
        report_type = cfg["report_type"]
        output_fmt  = cfg.get("output_format", ["json", "excel"])

        self._update_job_status(ExecutorStatus.RUNNING, progress=15)

        # 리포트 데이터 수집
        if report_type == "model_performance":
            report_data = self._build_model_performance_report(cfg)
        elif report_type == "eda_report":
            report_data = self._build_eda_report(cfg)
        elif report_type == "scorecard_report":
            report_data = self._build_scorecard_report(cfg)
        elif report_type == "prediction_report":
            report_data = self._build_prediction_report(cfg)
        elif report_type == "combined":
            report_data = self._build_combined_report(cfg)
        else:
            raise ExecutorException(f"지원하지 않는 report_type: {report_type}")

        self._update_job_status(ExecutorStatus.RUNNING, progress=70)

        # 저장
        saved_paths: dict[str, str] = {}
        base_path = f"reports/{cfg['output_id']}"

        if "json" in output_fmt:
            p = self._save_json(report_data, f"{base_path}.json")
            saved_paths["json"] = p

        if "excel" in output_fmt:
            p = self._save_excel(report_data, f"{base_path}.xlsx")
            saved_paths["excel"] = p

        self._update_job_status(ExecutorStatus.RUNNING, progress=95)

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result": {
                "output_id":    cfg["output_id"],
                "report_type":  report_type,
                "saved_paths":  saved_paths,
                "sections":     list(report_data.keys()),
            },
            "message": f"리포트 생성 완료  type={report_type}  formats={output_fmt}",
        }

    # ------------------------------------------------------------------

    def _build_model_performance_report(self, cfg) -> dict:
        import json
        meta_path = self.file_root / cfg["model_meta_path"]
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        report = {
            "title":       cfg.get("report_name", "Model Performance Report"),
            "model_info": {
                "model_id":   meta.get("model_id"),
                "model_type": meta.get("model_type"),
                "algorithm":  meta.get("model_type"),
            },
            "metrics":     meta.get("metrics", {}),
            "feature_importance": meta.get("varimp", {}),
        }

        # 예측 결과가 있으면 분포 추가
        if "prediction_path" in cfg:
            pred_df = self._load_dataframe(cfg["prediction_path"])
            score_col  = cfg.get("score_col", "score")
            target_col = cfg.get("target_col")
            if score_col in pred_df.columns:
                report["score_distribution"] = _decile_table(pred_df, score_col, target_col)

        return report

    def _build_eda_report(self, cfg) -> dict:
        import json
        eda_path = self.file_root / cfg["eda_result_path"]
        with open(eda_path, encoding="utf-8") as f:
            eda = json.load(f)

        return {
            "title":         cfg.get("report_name", "EDA Report"),
            "data_shape":    eda.get("shape"),
            "missing":       eda.get("missing", [])[:20],
            "outliers":      eda.get("outliers", [])[:20],
            "high_corr":     eda.get("correlation", {}).get("high_corr_pairs", []),
            "target_analysis": eda.get("target_analysis", [])[:20],
        }

    def _build_scorecard_report(self, cfg) -> dict:
        import json
        sc_path = self.file_root / cfg["scorecard_path"]
        with open(sc_path, encoding="utf-8") as f:
            sc = json.load(f)

        return {
            "title":          cfg.get("report_name", "Scorecard Report"),
            "selected_vars":  sc.get("selected_cols", []),
            "iv_summary":     sc.get("iv_dict", {}),
            "metrics":        sc.get("metrics", {}),
            "scorecard_table": sc.get("scorecard", []),
        }

    def _build_prediction_report(self, cfg) -> dict:
        pred_df    = self._load_dataframe(cfg["prediction_path"])
        score_col  = cfg.get("score_col", "score")
        target_col = cfg.get("target_col")

        report = {
            "title":       cfg.get("report_name", "Prediction Report"),
            "total_rows":  len(pred_df),
            "score_stats": _series_stats(pred_df[score_col]) if score_col in pred_df.columns else {},
        }
        if score_col in pred_df.columns:
            report["decile_table"] = _decile_table(pred_df, score_col, target_col)
        if "grade" in pred_df.columns:
            report["grade_dist"] = pred_df["grade"].value_counts().to_dict()
        if "decision" in pred_df.columns:
            report["decision_dist"] = pred_df["decision"].value_counts().to_dict()
        return report

    def _build_combined_report(self, cfg) -> dict:
        combined: dict = {"title": cfg.get("report_name", "Combined Report"), "sections": {}}
        if "model_meta_path" in cfg:
            combined["sections"]["model_performance"] = self._build_model_performance_report(cfg)
        if "eda_result_path" in cfg:
            combined["sections"]["eda"] = self._build_eda_report(cfg)
        if "scorecard_path" in cfg:
            combined["sections"]["scorecard"] = self._build_scorecard_report(cfg)
        if "prediction_path" in cfg:
            combined["sections"]["prediction"] = self._build_prediction_report(cfg)
        return combined

    def _save_excel(self, report_data: dict, relative_path: str) -> str:
        full_path = self.file_root / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(str(full_path), engine="openpyxl") as writer:
            for section_name, section_data in report_data.items():
                if isinstance(section_data, list) and section_data:
                    df = pd.DataFrame(section_data)
                    df.to_excel(writer, sheet_name=str(section_name)[:31], index=False)
                elif isinstance(section_data, dict):
                    rows = [{"key": k, "value": str(v)} for k, v in section_data.items()]
                    pd.DataFrame(rows).to_excel(writer, sheet_name=str(section_name)[:31], index=False)
        return str(full_path)


def _decile_table(df: pd.DataFrame, score_col: str, target_col: Optional[str]) -> list:
    df2 = df[[score_col] + ([target_col] if target_col and target_col in df.columns else [])].copy()
    df2["decile"] = pd.qcut(df2[score_col], q=10, labels=False, duplicates="drop") + 1
    rows = []
    for d, grp in df2.groupby("decile"):
        row = {"decile": int(d), "count": len(grp),
               "score_min": round(float(grp[score_col].min()), 2),
               "score_max": round(float(grp[score_col].max()), 2)}
        if target_col and target_col in grp.columns:
            row["bad_count"] = int(grp[target_col].sum())
            row["bad_rate"]  = round(float(grp[target_col].mean()), 4)
        rows.append(row)
    return rows


def _series_stats(s: pd.Series) -> dict:
    return {k: round(float(v), 4) for k, v in {
        "mean": s.mean(), "std": s.std(),
        "min": s.min(), "p25": s.quantile(0.25),
        "p50": s.quantile(0.5), "p75": s.quantile(0.75), "max": s.max(),
    }.items()}


from typing import Optional
