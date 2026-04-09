# data_analysis_executor.py — 탐색적 데이터 분석 (EDA)

**파일:** `executors/ml/data_analysis_executor.py`  
**클래스:** `DataAnalysisExecutor(BaseExecutor)`

## 개요

모델 학습 전 데이터 품질을 진단하고 변수 특성을 파악하는 executor.  
분석 결과는 JSON 파일로 저장되어 리포트 및 변수 선택의 기초 자료로 활용된다.

**분석 항목:**
- 기초 통계 (mean, std, min, max, percentiles)
- 결측치 현황 (건수, 비율)
- 이상값 탐지 (IQR 1.5배)
- 분포 요약 (skewness, kurtosis)
- 카테고리 변수 빈도 분포 (상위 10개)
- 변수 간 상관계수 행렬
- 타깃 대비 변수 분리도 (target_col 지정 시, KS 통계량)

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `source_path` | ✅ | `str` | 분석 대상 데이터 상대 경로 |
| `output_id` | ✅ | `str` | 결과 저장 식별자 |
| `target_col` | ❌ | `str` | 타깃 컬럼명 (이진 분류용 KS 분리도 분석) |
| `exclude_cols` | ❌ | `list` | 분석 제외 컬럼 목록 |
| `corr_threshold` | ❌ | `float` | 고상관 경고 임계값 (기본: `0.9`) |
| `missing_threshold` | ❌ | `float` | 결측 경고 임계값 비율 (기본: `0.3`) |

---

## 내부 메서드

### `_basic_stats(df)` → `dict`

수치형 컬럼에 대해 기술통계를 산출한다.

```python
df.select_dtypes(include=[np.number]).describe(
    percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
).T.round(4).to_dict(orient="index")
```

반환: `{컬럼명: {count, mean, std, min, 1%, 5%, 25%, 50%, 75%, 95%, 99%, max}}`

---

### `_missing_summary(df, threshold)` → `list`

모든 컬럼의 결측 현황을 집계한다.

| 출력 필드 | 설명 |
|----------|------|
| `column` | 컬럼명 |
| `missing_count` | 결측 건수 |
| `missing_rate` | 결측 비율 (0~1, 소수점 4자리) |
| `is_warning` | `missing_rate > threshold` 여부 |

결측률 내림차순 정렬.

---

### `_outlier_summary(df)` → `list`

IQR(사분위 범위) 1.5배 기준으로 수치형 컬럼의 이상값을 탐지한다.

```
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
이상값 = lower 미만 또는 upper 초과
```

| 출력 필드 | 설명 |
|----------|------|
| `column` | 컬럼명 |
| `outlier_count` | 이상값 건수 |
| `outlier_rate` | 이상값 비율 |
| `lower_bound` | 하한값 |
| `upper_bound` | 상한값 |

---

### `_distribution_summary(df)` → `list`

수치형 컬럼별 분포 형태를 요약한다.

| 출력 필드 | 설명 |
|----------|------|
| `column` | 컬럼명 |
| `skewness` | 왜도 (scipy.stats.skew) |
| `kurtosis` | 첨도 (scipy.stats.kurtosis) |

---

### `_category_freq(df)` → `dict`

범주형 컬럼(object, category)의 상위 10개 빈도를 산출한다.

```python
{"gender": {"M": 0.52, "F": 0.48}, "region": {"서울": 0.32, ...}}
```

---

### `_correlation_matrix(df, threshold)` → `dict`

수치형 컬럼 간 상관계수 행렬을 산출하고, threshold 초과 쌍을 경고로 추출한다.

```python
{
    "matrix": {col: {col2: corr, ...}},
    "high_corr_pairs": [
        {"col1": "income", "col2": "salary", "corr": 0.97},
        ...
    ]
}
```

---

### `_target_analysis(df, target_series)` → `dict`

타깃(이진) 대비 각 수치형 변수의 분리도를 KS 통계량으로 측정한다.

```python
{
    "income": {"ks_stat": 0.342, "p_value": 0.0001},
    "debt":   {"ks_stat": 0.218, "p_value": 0.0023},
    ...
}
```

---

## execute() 진행률

| 단계 | progress |
|------|----------|
| 데이터 로드 완료 | 20% |
| 기초통계·결측·이상값·분포·빈도 완료 | 60% |
| 상관계수·타깃분석 완료 | 90% |

---

## 반환값

```python
{
    "status": "COMPLETED",
    "result": {
        "output_path":       "analysis/loan_eda_202312_eda.json",
        "total_rows":        100000,
        "total_cols":        45,
        "high_missing_cols": ["credit_bureau_score"],   # missing_rate > threshold
        "high_corr_pairs":   [{"col1": "income", "col2": "salary", "corr": 0.97}],
    },
    "message": "EDA 완료: 100,000행 × 45열",
    "job_id":  str,
    "elapsed_sec": float,
}
```

---

## 출력 파일

| 파일 | 경로 |
|------|------|
| EDA 결과 JSON | `analysis/{output_id}_eda.json` |

JSON 구조:
```json
{
  "shape": [100000, 45],
  "columns": [...],
  "basic_stats": {...},
  "missing": [...],
  "outliers": [...],
  "distribution": [...],
  "category_freq": {...},
  "correlation": {"matrix": {...}, "high_corr_pairs": [...]},
  "target_analysis": {...}
}
```

---

## 사용 예시

```python
config = {
    "job_id":            "eda_job_001",
    "source_path":       "mart/retail_mart_v2_train.parquet",
    "output_id":         "retail_eda_v2",
    "target_col":        "default_yn",
    "corr_threshold":    0.85,
    "missing_threshold": 0.2,
    "exclude_cols":      ["cust_id", "base_dt"],
}

from executors.ml.data_analysis_executor import DataAnalysisExecutor
result = DataAnalysisExecutor(config=config).run()
```
