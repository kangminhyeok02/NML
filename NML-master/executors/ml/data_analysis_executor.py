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


# =============================================================================
# Module-level functions
# =============================================================================


def _make_engine(db_info: dict):
    from sqlalchemy import create_engine
    url = (
        f"postgresql+psycopg2://{db_info['user']}:{db_info['password']}"
        f"@{db_info['host']}:{db_info['port']}/{db_info['db']}"
    )
    return create_engine(url, pool_pre_ping=True)


def _variable_analysis_for_each_feature(
    col: str,
    series: "pd.Series",
    target: "pd.Series | None" = None,
) -> dict:
    """단일 변수에 대한 통계 분석 (병렬화용)."""
    import numpy as np
    from scipy import stats as sp_stats

    series = series.dropna()
    info: dict = {
        "column":       col,
        "dtype":        str(series.dtype),
        "count":        int(len(series)),
        "missing_rate": round(1 - len(series) / max(len(series), 1), 4),
    }
    if pd.api.types.is_numeric_dtype(series):
        info.update({
            "mean": round(float(series.mean()), 4),
            "std":  round(float(series.std()), 4),
            "min":  round(float(series.min()), 4),
            "max":  round(float(series.max()), 4),
        })
        if target is not None:
            aligned = target.loc[series.index]
            good = series[aligned == 0]
            bad  = series[aligned == 1]
            if len(good) > 0 and len(bad) > 0:
                ks_stat, _ = sp_stats.ks_2samp(good, bad)
                info["ks_stat"] = round(float(ks_stat), 4)
    else:
        info["unique_count"] = int(series.nunique())
    return info


def variable_analysis(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """변수 분석을 병렬로 수행하고 결과를 JSON 파일로 저장한다."""
    import json as _json
    import concurrent.futures
    from pathlib import Path

    mart_path  = json_obj["mart_path"]
    root_dir   = json_obj.get("root_dir", "/data")
    target_col = json_obj.get("target_col")
    process_seq = json_obj.get("process_seq", "")
    stg_id      = json_obj.get("stg_id", "")

    df = pd.read_parquet(mart_path)
    target = df[target_col] if target_col and target_col in df.columns else None
    cols = [c for c in df.columns if c != target_col]

    with concurrent.futures.ProcessPoolExecutor() as pool:
        futures = {pool.submit(_variable_analysis_for_each_feature, c, df[c], target): c for c in cols}
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    out_path = Path(root_dir) / "processes" / process_seq / "output" / f"{stg_id}_var_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(results, f, ensure_ascii=False)

    logger.info("variable_analysis: process_seq=%s stg_id=%s cols=%d", process_seq, stg_id, len(results))
    return {"result": results, "output_file": str(out_path)}


def data_sampling(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """마트 데이터를 샘플링하고 출력 파라미터 파일에 저장한다."""
    import json as _json
    from pathlib import Path

    mart_path     = json_obj["mart_path"]
    process_seq   = json_obj["process_seq"]
    stg_id        = json_obj["stg_id"]
    sample_method = json_obj.get("sample_method", "random")
    sample_n      = int(json_obj.get("sample_n", 1000))
    target_col    = json_obj.get("target_col")
    root_dir      = json_obj.get("root_dir", "/data")

    df = pd.read_parquet(mart_path)
    if sample_method == "stratified" and target_col and target_col in df.columns:
        sampled = df.groupby(target_col, group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(1, int(sample_n * len(g) / len(df)))), random_state=42)
        )
    else:
        sampled = df.sample(min(sample_n, len(df)), random_state=42)

    out_dir  = Path(root_dir) / "processes" / process_seq / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{stg_id}_output_param.json"
    payload  = {
        "process_seq": process_seq,
        "stg_id":      stg_id,
        "params":      sampled.to_dict(orient="records"),
    }
    with open(out_file, "w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False)

    logger.info("data_sampling: process_seq=%s stg_id=%s rows=%d", process_seq, stg_id, len(sampled))
    return {"result": "ok", "row_count": len(sampled), "output_file": str(out_file)}


def data_union(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """여러 parquet 파일을 수직 결합하고 key_col 기준으로 정렬하여 저장한다.
    정렬하지 않으면 마트와 JOIN 시 잘못된 결과가 발생한다."""
    from pathlib import Path

    file_paths  = json_obj["file_paths"]
    output_path = json_obj["output_path"]
    key_col     = json_obj.get("key_col")

    dfs = [pd.read_parquet(p) for p in file_paths]
    merged = pd.concat(dfs, ignore_index=True)
    if key_col and key_col in merged.columns:
        merged = merged.sort_values(key_col).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    logger.info("data_union: files=%d rows=%d output=%s", len(file_paths), len(merged), output_path)
    return {"result": "ok", "row_count": len(merged), "output_path": output_path}


def check_output_param_prob_in_correct_range(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """출력 파라미터의 확률값이 [0, 1] 범위인지 확인한다."""
    import json as _json
    from pathlib import Path

    process_seq = json_obj["process_seq"]
    stg_id      = json_obj["stg_id"]
    prob_col    = json_obj.get("prob_col", "prob")
    root_dir    = json_obj.get("root_dir", "/data")

    out_file = Path(root_dir) / "processes" / process_seq / "output" / f"{stg_id}_output_param.json"
    with open(out_file, "r", encoding="utf-8") as f:
        data = _json.load(f)

    params = data.get("params", [])
    out_of_range = sum(1 for p in params if not (0.0 <= float(p.get(prob_col, 0.5)) <= 1.0))
    in_range = out_of_range == 0
    logger.debug("check_output_param_prob_in_correct_range: process_seq=%s out_of_range=%d", process_seq, out_of_range)
    return {"in_range": in_range, "out_of_range_count": out_of_range, "total": len(params)}


def get_distinct_values(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """마트 데이터의 특정 컬럼 고유값 목록을 반환한다."""
    mart_path = json_obj["mart_path"]
    col       = json_obj["column"]
    top_n     = int(json_obj.get("top_n", 100))

    df = pd.read_parquet(mart_path, columns=[col])
    values = [str(v) for v in df[col].dropna().unique()[:top_n]]
    logger.debug("get_distinct_values: col=%s count=%d", col, len(values))
    return {"column": col, "values": values, "total": len(values)}


def get_profiling_result(
    service_db_info: dict,
    auth_key: str,
    process_seq: str,
    stg_id: str,
    root_dir: str = "/data",
) -> dict:
    """저장된 프로파일링 결과 JSON을 읽어 반환한다."""
    import json as _json
    from pathlib import Path

    prof_file = Path(root_dir) / "processes" / process_seq / "output" / f"{stg_id}_profiling.json"
    if not prof_file.exists():
        raise FileNotFoundError(f"프로파일링 결과 파일이 없습니다: {prof_file}")
    with open(prof_file, "r", encoding="utf-8") as f:
        result = _json.load(f)
    logger.debug("get_profiling_result: process_seq=%s stg_id=%s", process_seq, stg_id)
    return {"result": result}


def delete_profiling_result(
    service_db_info: dict,
    auth_key: str,
    process_seq: str,
    stg_id: str,
    root_dir: str = "/data",
) -> dict:
    """프로파일링 결과 JSON 파일을 삭제한다."""
    from pathlib import Path

    prof_file = Path(root_dir) / "processes" / process_seq / "output" / f"{stg_id}_profiling.json"
    if prof_file.exists():
        prof_file.unlink()
        logger.info("delete_profiling_result: deleted %s", prof_file)
    else:
        logger.warning("delete_profiling_result: file not found: %s", prof_file)
    return {"result": "ok", "process_seq": process_seq, "stg_id": stg_id}


def variable_profiling_from_node(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """프로파일링 노드에서만 호출되는 변수 프로파일링.
    프로파일링 결과 JSON 업로드 + 출력 파라미터에도 추가한다."""
    import json as _json
    from pathlib import Path

    mart_path   = json_obj["mart_path"]
    process_seq = json_obj["process_seq"]
    stg_id      = json_obj["stg_id"]
    root_dir    = json_obj.get("root_dir", "/data")

    df = pd.read_parquet(mart_path)
    profiling = {}
    for col in df.columns:
        s = df[col]
        info = {
            "dtype":        str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_rate": round(float(s.isna().mean()), 4),
            "unique_count": int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s):
            info.update({
                "mean": round(float(s.mean()), 4),
                "std":  round(float(s.std()), 4),
                "min":  round(float(s.min()), 4),
                "max":  round(float(s.max()), 4),
            })
        else:
            top = s.value_counts().head(10)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
        profiling[col] = info

    out_dir = Path(root_dir) / "processes" / process_seq / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 프로파일링 결과 저장
    prof_file = out_dir / f"{stg_id}_profiling.json"
    with open(prof_file, "w", encoding="utf-8") as f:
        _json.dump(profiling, f, ensure_ascii=False)

    # 출력 파라미터에도 추가 (마트 업로드 액션 제외, 프로파일링 노드에서만)
    out_param_file = out_dir / f"{stg_id}_output_param.json"
    param_payload = {"process_seq": process_seq, "stg_id": stg_id, "profiling": profiling, "params": []}
    with open(out_param_file, "w", encoding="utf-8") as f:
        _json.dump(param_payload, f, ensure_ascii=False)

    logger.info("variable_profiling_from_node: process_seq=%s stg_id=%s cols=%d", process_seq, stg_id, len(profiling))
    return {"result": "ok", "profiling_file": str(prof_file), "column_count": len(profiling)}


def data_snapshot(
    service_db_info: dict,
    file_server_host: str,
    file_server_port: int,
    json_obj: dict,
) -> dict:
    """마트 데이터의 스냅샷(head/sample)을 저장하고 반환한다."""
    import json as _json
    from pathlib import Path

    mart_path   = json_obj["mart_path"]
    process_seq = json_obj.get("process_seq", "")
    stg_id      = json_obj.get("stg_id", "")
    n           = int(json_obj.get("n", 100))
    method      = json_obj.get("method", "head")  # "head" or "sample"
    root_dir    = json_obj.get("root_dir", "/data")

    df = pd.read_parquet(mart_path)
    snapshot = df.head(n) if method == "head" else df.sample(min(n, len(df)), random_state=42)

    if process_seq and stg_id:
        out_dir = Path(root_dir) / "processes" / process_seq / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        snap_file = out_dir / f"{stg_id}_snapshot.json"
        records = snapshot.where(snapshot.notna(), other=None).to_dict(orient="records")
        with open(snap_file, "w", encoding="utf-8") as f:
            _json.dump(records, f, ensure_ascii=False, default=str)
        logger.info("data_snapshot: process_seq=%s stg_id=%s rows=%d", process_seq, stg_id, len(snapshot))

    return {
        "result":    snapshot.where(snapshot.notna(), other=None).to_dict(orient="records"),
        "row_count": len(snapshot),
        "columns":   list(snapshot.columns),
    }
