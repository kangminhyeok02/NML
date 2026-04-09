# report_executor.py — 분석/모델링 결과 리포트 생성

**파일:** `executors/ml/report_executor.py`  
**클래스:** `ReportExecutor(BaseExecutor)`

## 개요

분석·모델링 결과를 정형화된 리포트로 생성하는 executor.  
JSON 및 Excel 형태로 출력하며, 모델 성능·EDA·스코어카드·예측 분포를 단독 또는 통합으로 요약한다.

```
리포트 유형별 데이터 수집
  ├── model_performance  → 모델 메타 JSON 파싱
  ├── eda_report         → EDA JSON 파싱
  ├── scorecard_report   → 스코어카드 JSON 파싱
  ├── prediction_report  → 예측 parquet 분석
  └── combined           → 위 항목 통합
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
|---|---|---|
| `model_performance` | AUC, KS, 변수중요도, 점수분포 요약 | `model_meta_path` |
| `eda_report` | 결측·이상치·상관관계·타깃 분석 요약 | `eda_result_path` |
| `scorecard_report` | 스코어카드 변수 요약표(IV, WOE) 및 성능 | `scorecard_path` |
| `prediction_report` | 예측 분포 요약 (데사일, 등급·의사결정 분포) | `prediction_path` |
| `combined` | 위 항목 통합 종합 리포트 (존재하는 섹션만 포함) | 각 섹션 경로 |

---

## 내부 메서드

### `_build_model_performance_report(cfg)` → `dict`

모델 메타 JSON에서 성능 지표, 피처 중요도, 점수 분포를 추출한다.

```json
{
  "report_name":   "Retail Credit Scoring v2",
  "model_id":      "lgbm_retail_v2",
  "algorithm":     "lightgbm",
  "metrics":       {"auc": 0.8923, "accuracy": 0.8412},
  "feature_importance": {"income": 0.31, "debt_ratio": 0.22},
  "score_distribution": {...}
}
```

### `_build_eda_report(cfg)` → `dict`

EDA JSON에서 결측·이상치·상관관계 경고를 추출하여 요약한다.

### `_build_scorecard_report(cfg)` → `dict`

스코어카드 JSON에서 변수별 IV, WOE 테이블, 성능 지표를 추출한다.

### `_build_prediction_report(cfg)` → `dict`

예측 parquet에서 데사일 테이블, 등급 분포, bad_rate를 산출한다.

```python
# _decile_table(df, score_col, target_col) 모듈 수준 함수
# 점수 기준 10분위 → 각 구간의 count, bad_rate, cum_bad_rate
```

### `_build_combined_report(cfg)` → `dict`

각 섹션 빌더를 호출하여 통합 리포트를 생성한다.  
경로가 없는 섹션은 건너뛴다.

---

## Excel 저장 방식

섹션별로 시트가 생성된다.

- `list` 타입 → `pd.DataFrame`으로 변환 후 시트
- `dict` 타입 → key-value 2컬럼 테이블로 시트

---

## execute() 진행률

| 단계 | progress |
|------|----------|
| 데이터 수집 완료 | 15% |
| 리포트 데이터 생성 완료 | 70% |
| JSON·Excel 저장 완료 | 95% |

---

## 반환값

```python
{
    "status": "COMPLETED",
    "result": {
        "output_id":   "retail_report_v2",
        "report_type": "model_performance",
        "saved_paths": {
            "json":  "reports/retail_report_v2.json",
            "excel": "reports/retail_report_v2.xlsx",
        },
    },
    "message": "리포트 생성 완료  type=model_performance",
    "job_id":  str,
    "elapsed_sec": float,
}
```

---

## 출력 파일

| 파일 | 경로 |
|------|------|
| JSON 리포트 | `reports/{output_id}.json` |
| Excel 리포트 | `reports/{output_id}.xlsx` |

---

## 사용 예시

```python
config = {
    "job_id":          "report_001",
    "report_type":     "combined",
    "report_name":     "Retail Credit Scoring v2 종합 리포트",
    "output_id":       "retail_report_v2",
    "model_meta_path": "models/lgbm_retail_v2_meta.json",
    "eda_result_path": "analysis/retail_eda_v2_eda.json",
    "prediction_path": "predict/retail_pred_v2_result.parquet",
    "score_col":       "score",
    "target_col":      "default_yn",
    "output_format":   ["json", "excel"],
}

from executors.ml.report_executor import ReportExecutor
result = ReportExecutor(config=config).run()
```
