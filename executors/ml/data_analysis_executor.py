"""
data_analysis_executor.py
--------------------------
데이터 탐색적 분석(EDA) 실행기.

모델 학습 전 데이터 품질을 진단하고 변수 특성을 파악한다.
결과는 JSON 요약 파일과 분석 리포트용 데이터로 저장된다.

분석 항목:
  - 기초 통계 (mean, std, min, max, percentiles)
  - 결측치 현황 (건수, 비율)
  - 이상값 탐지 (IQR, Z-score)
  - 분포 요약 (skewness, kurtosis)
  - 카테고리 변수 빈도 분포
  - 타깃 대비 변수 분리도 (target_col 지정 시)
  - 변수 간 상관계수 행렬
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)


class DataAnalysisExecutor(BaseExecutor):
    """
    데이터 분석 executor.

    config 필수 키
    --------------
    source_path : str   분석 대상 데이터 상대 경로
    output_id   : str   결과 저장 식별자

    config 선택 키
    --------------
    target_col      : str   타깃 컬럼명 (이진 분류용)
    exclude_cols    : list  분석 제외 컬럼 목록
    corr_threshold  : float 상관계수 경고 임계값 (기본 0.9)
    missing_threshold : float 결측 경고 임계값 비율 (기본 0.3)
    """

    def execute(self) -> dict:
        cfg = self.config

        df = self._load_dataframe(cfg["source_path"])
        logger.info("data loaded  shape=%s", df.shape)

        exclude = cfg.get("exclude_cols", [])
        df_ana = df.drop(columns=[c for c in exclude if c in df.columns])
        target_col = cfg.get("target_col")

        self._update_job_status(ExecutorStatus.RUNNING, progress=20)

        result = {
            "shape":          list(df.shape),
            "columns":        list(df_ana.columns),
            "basic_stats":    self._basic_stats(df_ana),
            "missing":        self._missing_summary(df_ana, cfg.get("missing_threshold", 0.3)),
            "outliers":       self._outlier_summary(df_ana),
            "distribution":   self._distribution_summary(df_ana),
            "category_freq":  self._category_freq(df_ana),
        }

        self._update_job_status(ExecutorStatus.RUNNING, progress=60)

        result["correlation"] = self._correlation_matrix(df_ana, cfg.get("corr_threshold", 0.9))

        if target_col and target_col in df.columns:
            result["target_analysis"] = self._target_analysis(df_ana, df[target_col])

        self._update_job_status(ExecutorStatus.RUNNING, progress=90)

        output_path = f"analysis/{cfg['output_id']}_eda.json"
        self._save_json(result, output_path)

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  {
                "output_path":         output_path,
                "total_rows":          result["shape"][0],
                "total_cols":          result["shape"][1],
                "high_missing_cols":   [m["column"] for m in result["missing"] if m["is_warning"]],
                "high_corr_pairs":     result["correlation"]["high_corr_pairs"],
            },
            "message": f"EDA 완료: {df.shape[0]:,}행 × {df.shape[1]}열",
        }

    # ------------------------------------------------------------------

    def _basic_stats(self, df: pd.DataFrame) -> dict:
        num_df = df.select_dtypes(include=[np.number])
        stats_df = num_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
        return stats_df.round(4).to_dict(orient="index")

    def _missing_summary(self, df: pd.DataFrame, threshold: float) -> list:
        result = []
        for col in df.columns:
            missing_cnt = int(df[col].isna().sum())
            missing_rate = missing_cnt / len(df)
            result.append({
                "column":       col,
                "missing_count": missing_cnt,
                "missing_rate":  round(missing_rate, 4),
                "is_warning":   missing_rate > threshold,
            })
        return sorted(result, key=lambda x: x["missing_rate"], reverse=True)

    def _outlier_summary(self, df: pd.DataFrame) -> list:
        result = []
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_cnt = int(((series < lower) | (series > upper)).sum())
            result.append({
                "column":       col,
                "iqr_lower":    round(float(lower), 4),
                "iqr_upper":    round(float(upper), 4),
                "outlier_count": outlier_cnt,
                "outlier_rate": round(outlier_cnt / len(series), 4),
            })
        return result

    def _distribution_summary(self, df: pd.DataFrame) -> list:
        result = []
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            result.append({
                "column":   col,
                "skewness": round(float(stats.skew(series)), 4),
                "kurtosis": round(float(stats.kurtosis(series)), 4),
            })
        return result

    def _category_freq(self, df: pd.DataFrame, top_n: int = 10) -> dict:
        result = {}
        for col in df.select_dtypes(include=["object", "category"]).columns:
            freq = df[col].value_counts(normalize=True).head(top_n)
            result[col] = {str(k): round(float(v), 4) for k, v in freq.items()}
        return result

    def _correlation_matrix(self, df: pd.DataFrame, threshold: float) -> dict:
        num_df = df.select_dtypes(include=[np.number])
        corr = num_df.corr().round(4)
        high_pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = abs(corr.iloc[i, j])
                if val >= threshold:
                    high_pairs.append({
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "corr":  round(float(corr.iloc[i, j]), 4),
                    })
        return {
            "matrix":          corr.to_dict(),
            "high_corr_pairs": high_pairs,
        }

    def _target_analysis(self, df: pd.DataFrame, target: pd.Series) -> dict:
        """수치형 변수별 타깃 클래스 간 분리도(KS statistic) 산출."""
        result = []
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            aligned_target = target.loc[series.index]
            good = series[aligned_target == 0]
            bad  = series[aligned_target == 1]
            if len(good) == 0 or len(bad) == 0:
                continue
            ks_stat, ks_pval = stats.ks_2samp(good, bad)
            result.append({
                "column":  col,
                "ks_stat": round(float(ks_stat), 4),
                "ks_pval": round(float(ks_pval), 6),
                "mean_good": round(float(good.mean()), 4),
                "mean_bad":  round(float(bad.mean()), 4),
            })
        return sorted(result, key=lambda x: x["ks_stat"], reverse=True)
