# data_analysis_executor.py

데이터 탐색적 분석(EDA) 실행기.

모델 학습 전 데이터 품질을 진단하고 변수 특성을 파악한다.  
결과는 JSON 요약 파일로 저장된다.

---

## 클래스

### `DataAnalysisExecutor(BaseExecutor)`

데이터 분석 executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `source_path` | `str` | 분석 대상 데이터 상대 경로 |
| `output_id` | `str` | 결과 저장 식별자 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|---|---|---|
| `target_col` | - | 타깃 컬럼명 (이진 분류용, KS 분리도 산출 시 필요) |
| `exclude_cols` | `[]` | 분석 제외 컬럼 목록 |
| `corr_threshold` | `0.9` | 고상관 변수 쌍 경고 임계값 |
| `missing_threshold` | `0.3` | 결측 경고 비율 임계값 |

---

### `execute() → dict`

EDA 전체 분석을 수행하고 결과를 JSON 파일로 저장한다.

**분석 항목**

| 항목 | 메서드 |
|---|---|
| 기초 통계 | `_basic_stats` |
| 결측치 현황 | `_missing_summary` |
| 이상값 탐지 | `_outlier_summary` |
| 분포 요약 | `_distribution_summary` |
| 카테고리 빈도 | `_category_freq` |
| 상관계수 행렬 | `_correlation_matrix` |
| 타깃 대비 분리도 | `_target_analysis` (target_col 지정 시) |

**반환값**
```python
{
    "status": "COMPLETED",
    "result": {
        "output_path":       str,   # "analysis/{output_id}_eda.json"
        "total_rows":        int,
        "total_cols":        int,
        "high_missing_cols": list[str],
        "high_corr_pairs":   list[dict],
    },
    "message": str,
}
```

---

### `_basic_stats(df) → dict`

수치형 컬럼에 대한 기술통계를 산출한다.

- `describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])` 사용
- 컬럼명을 키로 하는 딕셔너리 반환

### `_missing_summary(df, threshold) → list`

전체 컬럼의 결측 현황을 분석한다.

**각 항목 구조**
```python
{
    "column":        str,
    "missing_count": int,
    "missing_rate":  float,
    "is_warning":    bool,   # missing_rate > threshold
}
```
결측률 내림차순으로 정렬하여 반환.

### `_outlier_summary(df) → list`

수치형 컬럼에 대해 IQR 방식으로 이상값을 탐지한다.

- 하한: `Q1 - 1.5 * IQR`, 상한: `Q3 + 1.5 * IQR`

**각 항목 구조**
```python
{
    "column":        str,
    "iqr_lower":     float,
    "iqr_upper":     float,
    "outlier_count": int,
    "outlier_rate":  float,
}
```

### `_distribution_summary(df) → list`

수치형 컬럼의 왜도(skewness)와 첨도(kurtosis)를 `scipy.stats`로 산출한다.

```python
{"column": str, "skewness": float, "kurtosis": float}
```

### `_category_freq(df, top_n=10) → dict`

object/category 타입 컬럼의 상위 `top_n`개 빈도 분포를 반환한다.

```python
{"col_name": {"value1": 0.35, "value2": 0.20, ...}}
```

### `_correlation_matrix(df, threshold) → dict`

수치형 컬럼 간 Pearson 상관계수 행렬을 산출하고 고상관 쌍을 추출한다.

```python
{
    "matrix":          dict,   # corr().to_dict()
    "high_corr_pairs": [{"col_a": str, "col_b": str, "corr": float}, ...]
}
```
`|corr| >= threshold`인 쌍만 `high_corr_pairs`에 포함.

### `_target_analysis(df, target) → list`

수치형 변수별로 타깃 클래스(0=Good, 1=Bad) 간 KS 통계량을 산출한다.

- `scipy.stats.ks_2samp(good, bad)` 사용
- KS 통계량 내림차순 정렬

```python
{
    "column":    str,
    "ks_stat":   float,
    "ks_pval":   float,
    "mean_good": float,
    "mean_bad":  float,
}
```

---

## 모듈 레벨 함수

### `_make_engine(db_info) → Engine`

PostgreSQL SQLAlchemy engine을 생성한다.

### `_variable_analysis_for_each_feature(col, series, target) → dict`

단일 변수에 대한 통계 분석. `variable_analysis()`에서 병렬 실행용으로 사용된다.

- 수치형: mean, std, min, max, ks_stat(target 있을 때) 산출
- 범주형: unique_count 산출

### `variable_analysis(service_db_info, file_server_host, file_server_port, json_obj) → dict`

변수 분석을 **ProcessPoolExecutor**로 병렬 수행하고 결과를 JSON 파일로 저장한다.

| `json_obj` 키 | 설명 |
|---|---|
| `mart_path` | 데이터 파일 경로 |
| `target_col` | 타깃 컬럼명 (선택) |
| `process_seq` | 프로세스 시퀀스 |
| `stg_id` | 스테이지 ID |
| `root_dir` | 루트 디렉터리 (기본: `/data`) |

결과 파일: `{root_dir}/processes/{process_seq}/output/{stg_id}_var_analysis.json`

### `data_sampling(service_db_info, file_server_host, file_server_port, json_obj) → dict`

마트 데이터를 샘플링하고 출력 파라미터 파일로 저장한다.

| `json_obj` 키 | 기본값 | 설명 |
|---|---|---|
| `sample_method` | `"random"` | `"random"` 또는 `"stratified"` |
| `sample_n` | `1000` | 샘플 수 |
| `target_col` | - | stratified 샘플링 시 기준 컬럼 |

### `data_union(service_db_info, file_server_host, file_server_port, json_obj) → dict`

여러 parquet 파일을 수직 결합하고 `key_col` 기준으로 정렬하여 저장한다.  
정렬하지 않으면 마트와 JOIN 시 잘못된 결과가 발생할 수 있다.

### `check_output_param_prob_in_correct_range(service_db_info, file_server_host, file_server_port, json_obj) → dict`

출력 파라미터의 확률값(`prob_col`)이 `[0, 1]` 범위인지 검증한다.

```python
{"in_range": bool, "out_of_range_count": int, "total": int}
```

### `get_distinct_values(service_db_info, file_server_host, file_server_port, json_obj) → dict`

마트 데이터의 특정 컬럼 고유값 목록을 반환한다 (최대 `top_n`개).

### `get_profiling_result(service_db_info, auth_key, process_seq, stg_id, root_dir) → dict`

저장된 프로파일링 결과 JSON 파일을 읽어 반환한다.

### `delete_profiling_result(service_db_info, auth_key, process_seq, stg_id, root_dir) → dict`

프로파일링 결과 JSON 파일을 삭제한다.

### `variable_profiling_from_node(service_db_info, file_server_host, file_server_port, json_obj) → dict`

프로파일링 노드에서 호출되는 변수 프로파일링.

각 컬럼에 대해 dtype, 결측, 고유값, 수치 통계(또는 top_values)를 수집하고  
프로파일링 JSON과 출력 파라미터 JSON 두 파일에 저장한다.

### `data_snapshot(service_db_info, file_server_host, file_server_port, json_obj) → dict`

마트 데이터의 스냅샷(head 또는 sample)을 저장하고 반환한다.

| `json_obj` 키 | 기본값 | 설명 |
|---|---|---|
| `n` | `100` | 스냅샷 행 수 |
| `method` | `"head"` | `"head"` 또는 `"sample"` |
