# data_analysis_executor.py — 탐색적 데이터 분석(EDA)

## 개요

모델 학습 전 데이터 품질을 진단하고 변수 특성을 파악하는 executor.  
분석 결과는 JSON 파일로 저장되어 리포트 및 변수 선택의 기초 자료로 활용된다.

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `source_path` | ✅ | `str` | 분석 대상 데이터 상대 경로 |
| `output_id` | ✅ | `str` | 결과 저장 식별자 |
| `target_col` | ❌ | `str` | 타깃 컬럼명 (이진 분류용 분리도 분석) |
| `exclude_cols` | ❌ | `list` | 분석 제외 컬럼 목록 |
| `corr_threshold` | ❌ | `float` | 고상관 경고 임계값 (기본 `0.9`) |
| `missing_threshold` | ❌ | `float` | 결측 경고 임계값 비율 (기본 `0.3`) |

---

## 분석 항목별 메서드

### `_basic_stats(df)` → `dict`

수치형 컬럼에 대해 기술통계를 산출한다.

- `describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])`
- 반환: `{컬럼명: {count, mean, std, min, 1%, 5%, 25%, 50%, 75%, 95%, 99%, max}}`

---

### `_missing_summary(df, threshold)` → `list`

모든 컬럼의 결측 현황을 집계하고, 임계값 초과 시 경고 플래그를 설정한다.

| 출력 필드 | 설명 |
|----------|------|
| `column` | 컬럼명 |
| `missing_count` | 결측 건수 |
| `missing_rate` | 결측 비율 (0~1) |
| `is_warning` | 결측률 > threshold 여부 |

결측률 내림차순 정렬.

---

### `_outlier_summary(df)` → `list`

IQR(사분위 범위) 1.5배 기준으로 이상값을 탐지한다.

```
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
이상값 = lower 미만 또는 upper 초과
```

| 출력 필드 | 설명 |
|----------|------|
| `iqr_lower` / `iqr_upper` | IQR 경계값 |
| `outlier_count` | 이상값 건수 |
| `outlier_rate` | 이상값 비율 |

---

### `_distribution_summary(df)` → `list`

수치형 컬럼의 분포 형태를 요약한다.

| 출력 필드 | 설명 |
|----------|------|
| `skewness` | 왜도 (scipy.stats.skew) |
| `kurtosis` | 첨도 (scipy.stats.kurtosis) |

---

### `_category_freq(df, top_n=10)` → `dict`

범주형 컬럼의 빈도 분포 상위 `top_n`개를 산출한다.

```python
{"gender": {"M": 0.55, "F": 0.44, "U": 0.01}, ...}
```

---

### `_correlation_matrix(df, threshold)` → `dict`

수치형 컬럼 간 피어슨 상관계수 행렬을 계산하고,  
절댓값이 `threshold` 이상인 쌍을 `high_corr_pairs`로 반환한다.

```python
{
    "matrix": {컬럼: {컬럼: 상관계수}},
    "high_corr_pairs": [
        {"col_a": "income", "col_b": "salary", "corr": 0.97},
        ...
    ]
}
```

---

### `_target_analysis(df, target)` → `list`

`target_col` 지정 시 수치형 변수별로 **KS 통계량**을 산출하여 분리도를 평가한다.

```
Good (target=0) vs Bad (target=1) 두 분포 간 KS 검정
```

| 출력 필드 | 설명 |
|----------|------|
| `ks_stat` | KS 통계량 (0~1, 높을수록 분리도 좋음) |
| `ks_pval` | p-value |
| `mean_good` / `mean_bad` | 클래스별 평균 |

KS 통계량 내림차순 정렬 → 상위 변수 = 예측력 높은 변수.

---

## 실행 흐름

```
1. source_path 데이터 로드
2. exclude_cols 제거
3. 기초 통계 (_basic_stats)          [progress 20%]
4. 결측 요약 (_missing_summary)
5. 이상값 탐지 (_outlier_summary)
6. 분포 요약 (_distribution_summary)
7. 카테고리 빈도 (_category_freq)
8. 상관계수 행렬 (_correlation_matrix) [progress 60%]
9. 타깃 분리도 (_target_analysis)      [target_col 있을 때만]
10. analysis/{output_id}_eda.json 저장  [progress 90%]
```

---

## 출력 결과

**저장 경로:** `analysis/{output_id}_eda.json`

```python
{
    "shape":         [행수, 열수],
    "columns":       [컬럼명 목록],
    "basic_stats":   {...},
    "missing":       [...],
    "outliers":      [...],
    "distribution":  [...],
    "category_freq": {...},
    "correlation":   {"matrix": {...}, "high_corr_pairs": [...]},
    "target_analysis": [...]   # target_col 지정 시
}
```

**반환 요약:**
```python
{
    "output_path":       "analysis/my_eda.json",
    "total_rows":        100000,
    "total_cols":        50,
    "high_missing_cols": ["col_a", "col_b"],    # missing_threshold 초과
    "high_corr_pairs":   [{"col_a": ..., "col_b": ..., "corr": 0.95}]
}
```

---

## 활용 패턴

- **모델 전 변수 선택**: `target_analysis.ks_stat` 기준으로 예측력 높은 변수 선별
- **데이터 품질 점검**: `high_missing_cols` → 결측 대체 전략 수립
- **다중공선성 진단**: `high_corr_pairs` → 상관 높은 변수 제거 또는 PCA 검토
- **분포 이상 감지**: `skewness > 2` → 로그 변환 고려
