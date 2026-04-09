# report_executor.py — 분석/모델링 결과 리포트 생성

## 개요

분석·모델링 결과를 정형화된 리포트로 생성하는 executor.  
JSON 및 Excel 형태로 출력하며, 모델 성능·EDA·스코어카드·예측 분포를 단독 또는 통합으로 요약한다.

```
모델 메타 / EDA 결과 / 스코어카드 / 예측 parquet
    ↓ 리포트 유형별 데이터 수집
    ↓ JSON / Excel 저장
    ↓ reports/{output_id}.json & .xlsx
```

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `report_type` | ✅ | `str` | 리포트 유형 (아래 표 참조) |
| `output_id` | ✅ | `str` | 결과 저장 식별자 |
| `report_name` | ✅ | `str` | 리포트 제목 |
| `model_meta_path` | 조건부 | `str` | 모델 메타 JSON 경로 (`model_performance`·`combined` 필수) |
| `eda_result_path` | 조건부 | `str` | EDA 결과 JSON 경로 (`eda_report`·`combined` 필수) |
| `scorecard_path` | 조건부 | `str` | 스코어카드 JSON 경로 (`scorecard_report`·`combined` 필수) |
| `prediction_path` | 조건부 | `str` | 예측 결과 parquet 경로 (`prediction_report`·`combined` 필수) |
| `score_col` | ❌ | `str` | 점수 컬럼명 (기본: `"score"`) |
| `target_col` | ❌ | `str` | 타깃 컬럼명 (bad_rate 산출 시 필요) |
| `output_format` | ❌ | `list` | 출력 포맷 (기본: `["json", "excel"]`) |

---

## 리포트 유형

| `report_type` | 설명 | 필수 입력 |
|---------------|------|-----------|
| `model_performance` | 모델 성능 요약 (AUC, KS, 변수중요도, 점수분포) | `model_meta_path` |
| `eda_report` | EDA 요약 (결측, 이상치, 상관관계, 타깃 분석) | `eda_result_path` |
| `scorecard_report` | 스코어카드 변수 요약표 및 성능 | `scorecard_path` |
| `prediction_report` | 예측 분포 요약 (데사일, 등급·의사결정 분포) | `prediction_path` |
| `combined` | 위 항목 통합 종합 리포트 | 각 섹션 경로 중 존재하는 것만 포함 |

---

## 파일 저장 규칙

| 파일 | 경로 |
|------|------|
| JSON 리포트 | `reports/{output_id}.json` |
| Excel 리포트 | `reports/{output_id}.xlsx` |

Excel은 섹션별로 시트가 생성된다. list 타입은 DataFrame으로, dict 타입은 key-value 테이블로 변환된다.

---

## 모듈 수준 함수

### `_decile_table(df, score_col, target_col)`

점수 컬럼을 10분위로 나누어 데사일 테이블을 반환한다.

```python
[
  {"decile": 1, "count": 100, "score_min": 0.1, "score_max": 0.2,
   "bad_count": 30, "bad_rate": 0.3},
  ...
]
```

### `_series_stats(s)`

pandas Series의 기술 통계를 반환한다 (mean, std, min, p25, p50, p75, max).

---

## 사용 예시

### 모델 성능 리포트

```python
config = {
    "report_type":     "model_performance",
    "output_id":       "loan_model_report",
    "report_name":     "신용평가 모델 성능 리포트",
    "model_meta_path": "models/lgbm_loan_v1_meta.json",
    "prediction_path": "predict/loan_pred_result.parquet",
    "score_col":       "score",
    "target_col":      "default",
}
```

### 종합 리포트

```python
config = {
    "report_type":     "combined",
    "output_id":       "loan_combined_report",
    "report_name":     "신용평가 종합 리포트",
    "model_meta_path": "models/lgbm_loan_v1_meta.json",
    "eda_result_path": "analysis/loan_eda.json",
    "prediction_path": "predict/loan_pred_result.parquet",
    "score_col":       "score",
    "target_col":      "default",
    "output_format":   ["json", "excel"],
}
```

---

## 표준 출력 (result)

```json
{
  "status": "COMPLETED",
  "result": {
    "output_id":   "loan_model_report",
    "report_type": "model_performance",
    "saved_paths": {
      "json":  "/data/reports/loan_model_report.json",
      "excel": "/data/reports/loan_model_report.xlsx"
    },
    "sections": ["title", "model_info", "metrics", "feature_importance", "score_distribution"]
  },
  "message": "리포트 생성 완료  type=model_performance  formats=['json', 'excel']"
}
```

---

## 주의사항

- `combined` 타입은 경로가 지정된 섹션만 포함하며, 일부 경로가 없어도 오류가 발생하지 않는다.
- Excel 시트명은 31자를 초과할 수 없어 자동 절잘된다.
- `output_format`에서 `"json"`을 제외하면 JSON 저장을 건너뛴다.
- EDA JSON은 `DataAnalysisExecutor`가 생성한 `analysis/{id}_eda.json` 형식을 따른다.
