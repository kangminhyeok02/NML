# predict_executor.py — 모델 예측 실행

## 개요

저장된 모델을 로드하여 신규 데이터에 점수(score), 확률(probability), 등급(grade)을  
산출하는 executor. 운영 환경에서 가장 빈번하게 호출된다.

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `model_id` | ✅ | `str` | 사용할 모델 식별자 |
| `input_path` | ✅ | `str` | 예측 대상 데이터 경로 (.parquet) |
| `output_id` | ✅ | `str` | 결과 저장 식별자 |
| `model_type` | ❌ | `str` | `"python"` \| `"h2o"` \| `"r"` (기본: `"python"`) |
| `score_col` | ❌ | `str` | 예측 점수 컬럼명 (기본: `"score"`) |
| `grade_mapping` | ❌ | `dict` | 점수 → 등급 매핑 구간 |
| `threshold` | ❌ | `float` | 이진 분류 임계값 (기본: `0.5`) |
| `output_path` | ❌ | `str` | 결과 파일 저장 경로 (기본: 자동 생성) |

---

## 모델 유형별 로드 및 예측

### `model_type="python"` — Pickle 모델

```python
with open(models/{model_id}.pkl, "rb") as f:
    model = pickle.load(f)

# predict_proba 지원 시
proba = model.predict_proba(X)[:, 1]   # Bad 확률
result_df[score_col]    = proba
result_df["pred_class"] = (proba >= threshold).astype(int)

# 회귀 모델 등
pred = model.predict(X)
result_df[score_col] = pred
```

---

### `model_type="h2o"` — H2O MOJO

```python
model = h2o.import_mojo(meta["model_path"])
h2o_frame = h2o.H2OFrame(X)
preds = model.predict(h2o_frame).as_data_frame()
result_df[score_col] = preds.iloc[:, -1].values   # 마지막 컬럼 = Bad 확률
```

---

### `model_type="r"` — R 스크립트

```bash
Rscript r_scripts/predict.R \
    --input  /tmp/input.csv \
    --model  models/{model_id}.rds \
    --output /tmp/output.csv
```

- R 프로세스를 subprocess로 호출
- 임시 파일로 입출력 교환
- returncode != 0 이면 `ExecutorException` 발생

---

## 등급 부여 (`grade_mapping`)

```python
grade_mapping = {
    "A": [800, 1000],
    "B": [600, 800],
    "C": [400, 600],
    "D": [0,   400],
}
# score=750 → "B"
# 범위 밖은 "UNKNOWN"
```

---

## 실행 흐름

```
1. models/{model_id}_meta.json 로드               [progress 15%]
2. 모델 파일 로드 (_load_model)
3. input_path 데이터 로드
4. feature_cols 정렬 (meta 기준, 누락 컬럼 → 예외)  [progress 35%]
5. model_type별 예측 수행                          [progress 70%]
6. grade_mapping 있으면 등급 컬럼 추가
7. predict/{output_id}_result.parquet 저장         [progress 95%]
```

---

## 출력 결과

**저장 경로:** `predict/{output_id}_result.parquet`

포함 컬럼:
- 원본 입력 컬럼 전체
- `score` (또는 `score_col`)
- `pred_class` (python 분류 모델 시)
- `grade` (grade_mapping 지정 시)

**반환 요약:**
```python
{
    "output_id":   "loan_pred_20260407",
    "model_id":    "lgbm_v1",
    "total_rows":  100000,
    "output_path": "predict/loan_pred_20260407_result.parquet",
    "score_stats": {
        "mean": 0.2341,
        "std":  0.1523,
        "min":  0.0012,
        "p25":  0.1123,
        "p50":  0.2089,
        "p75":  0.3421,
        "max":  0.9871,
    },
    "grade_dist": {"A": 12000, "B": 35000, "C": 41000, "D": 12000}
}
```

---

## `_series_stats(series)` — 내부 유틸

점수 컬럼의 분포 요약 통계(mean, std, min, p25, p50, p75, max)를 딕셔너리로 반환.

---

## StrategyExecutor와의 연계

PredictExecutor 결과 parquet을 StrategyExecutor의 `input_path`로 연결하면  
점수 → 등급 → 업무 의사결정까지 파이프라인이 완성된다.

```
PredictExecutor  →  score 컬럼 포함 parquet
    ↓ input_from
StrategyExecutor →  grade, decision, limit_amt 컬럼 추가
```
