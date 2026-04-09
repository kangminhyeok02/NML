# report_executor.py

분석/모델링 결과를 정형화된 리포트로 생성하는 실행기.

출력 형식: JSON (구조화 데이터), Excel (.xlsx)

---

## 클래스

### `ReportExecutor(BaseExecutor)`

리포트 생성 executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `report_type` | `str` | 리포트 유형 |
| `output_id` | `str` | 출력 식별자 |
| `report_name` | `str` | 리포트 제목 |

**지원 `report_type`**

| 값 | 설명 |
|---|---|
| `"model_performance"` | 모델 성능 요약 (AUC, KS, 혼동행렬, ROC 데이터) |
| `"eda_report"` | 데이터 분석 요약 |
| `"scorecard_report"` | 스코어카드 변수 요약표 및 성능 |
| `"prediction_report"` | 예측 결과 분포 요약 |
| `"combined"` | 위 항목을 통합한 종합 리포트 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|---|---|---|
| `model_meta_path` | - | 모델 메타 JSON 경로 (`model_performance`용) |
| `eda_result_path` | - | EDA 결과 JSON 경로 (`eda_report`용) |
| `scorecard_path` | - | 스코어카드 결과 JSON 경로 |
| `prediction_path` | - | 예측 결과 parquet 경로 |
| `score_col` | `"score"` | 점수 컬럼명 |
| `target_col` | - | 타깃 컬럼명 |
| `output_format` | `["json", "excel"]` | 출력 포맷 목록 |

---

### `execute() → dict`

리포트 데이터를 수집하고 지정 포맷으로 저장한다.

**실행 순서**
1. `report_type`에 따라 해당 빌더 메서드 호출
2. `output_format`에 따라 JSON 및/또는 Excel 저장
3. 저장 경로: `reports/{output_id}.json`, `reports/{output_id}.xlsx`

**반환값**
```python
{
    "status": "COMPLETED",
    "result": {
        "output_id":   str,
        "report_type": str,
        "saved_paths": {"json": str, "excel": str},
        "sections":    list[str],
    },
    "message": str,
}
```

---

### `_build_model_performance_report(cfg) → dict`

모델 성능 리포트를 생성한다.

- 모델 메타 JSON에서 `model_info`, `metrics`, `feature_importance` 추출
- `prediction_path`가 있으면 `_decile_table()`로 점수 분포 추가

**반환 구조**
```python
{
    "title":              str,
    "model_info":         {model_id, model_type, algorithm},
    "metrics":            dict,
    "feature_importance": dict,
    "score_distribution": list,  # prediction_path 있을 때
}
```

---

### `_build_eda_report(cfg) → dict`

EDA 결과 JSON에서 요약 정보를 추출한다.

- `missing` 상위 20개, `outliers` 상위 20개, `target_analysis` 상위 20개만 포함

**반환 구조**
```python
{
    "title":           str,
    "data_shape":      list,
    "missing":         list,
    "outliers":        list,
    "high_corr":       list,
    "target_analysis": list,
}
```

---

### `_build_scorecard_report(cfg) → dict`

스코어카드 결과 JSON에서 요약 정보를 추출한다.

**반환 구조**
```python
{
    "title":           str,
    "selected_vars":   list,
    "iv_summary":      dict,
    "metrics":         dict,
    "scorecard_table": list,
}
```

---

### `_build_prediction_report(cfg) → dict`

예측 결과 데이터의 분포 요약 리포트를 생성한다.

- `score_stats`: 점수 기술통계 (`_series_stats`)
- `decile_table`: 10분위 구간별 count, bad_rate (`_decile_table`)
- `grade_dist`: 등급 분포 (`grade` 컬럼 있을 때)
- `decision_dist`: 의사결정 분포 (`decision` 컬럼 있을 때)

---

### `_build_combined_report(cfg) → dict`

config에 존재하는 경로 키에 따라 각 리포트를 섹션별로 통합한다.

```python
{
    "title": str,
    "sections": {
        "model_performance": dict,  # model_meta_path 있을 때
        "eda":               dict,  # eda_result_path 있을 때
        "scorecard":         dict,  # scorecard_path 있을 때
        "prediction":        dict,  # prediction_path 있을 때
    }
}
```

---

### `_save_excel(report_data, relative_path) → str`

리포트 데이터를 Excel 파일로 저장한다.

- `openpyxl` 엔진 사용
- 섹션별로 별도 시트 생성 (시트명 최대 31자)
- `list` 타입 섹션 → DataFrame으로 변환하여 저장
- `dict` 타입 섹션 → key/value 테이블로 저장

---

## 모듈 레벨 함수

### `_decile_table(df, score_col, target_col) → list`

점수를 10분위로 분할하고 분위별 통계를 계산한다.

```python
# 각 항목 구조:
{
    "decile":    int,    # 1 ~ 10
    "count":     int,
    "score_min": float,
    "score_max": float,
    "bad_count": int,    # target_col 있을 때
    "bad_rate":  float,  # target_col 있을 때
}
```

### `_series_stats(s) → dict`

Series의 기술통계 7개 지표를 반환한다.

```python
{mean, std, min, p25, p50, p75, max}
```
