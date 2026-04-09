# predict_executor.py — 모델 예측 실행

**파일:** `executors/ml/predict_executor.py`  
**클래스:** `PredictExecutor(BaseExecutor)`

## 개요

저장된 모델을 로드하여 신규 데이터에 점수(score), 확률(probability), 등급(grade)을  
산출하는 executor. 운영 환경에서 가장 빈번하게 호출된다.

```
모델 메타 JSON 로드
    ↓ 모델 파일 로드 (pickle / H2O MOJO / R 스크립트)
    ↓ 입력 데이터 로드
    ↓ 피처 정렬 (feature_cols 기준)
    ↓ 예측 수행 (_predict_python / _predict_h2o / _predict_r)
    ↓ grade_mapping 있으면 등급 부여
    ↓ predict/{output_id}_result.parquet 저장
```

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `model_id` | ✅ | `str` | 사용할 모델 식별자 (`models/{model_id}_meta.json` 참조) |
| `input_path` | ✅ | `str` | 예측 대상 데이터 경로 (.parquet) |
| `output_id` | ✅ | `str` | 결과 저장 식별자 |
| `model_type` | ❌ | `str` | `"python"` \| `"h2o"` \| `"r"` (기본: `"python"`) |
| `score_col` | ❌ | `str` | 예측 점수 컬럼명 (기본: `"score"`) |
| `grade_mapping` | ❌ | `dict` | 점수 → 등급 매핑 구간 `{"A": [800, 1000], "B": [600, 800]}` |
| `threshold` | ❌ | `float` | 이진 분류 임계값 (기본: `0.5`) |
| `output_path` | ❌ | `str` | 결과 파일 저장 경로 (기본: 자동 생성) |

---

## 모델 유형별 로드 및 예측

### `model_type="python"` — Pickle 모델

```python
with open(f"models/{model_id}.pkl", "rb") as f:
    model = pickle.load(f)

# predict_proba 지원 시 (분류)
proba = model.predict_proba(X)[:, 1]   # Bad 확률
result_df[score_col]    = proba
result_df["pred_class"] = (proba >= threshold).astype(int)

# predict만 지원 시 (회귀)
result_df[score_col] = model.predict(X)
```

---

### `model_type="h2o"` — H2O MOJO

```python
import h2o
mojo_path = meta["mojo_path"]    # models/{model_id}/model.zip
model = h2o.import_mojo(mojo_path)
h2o_frame = h2o.H2OFrame(X)
preds = model.predict(h2o_frame).as_data_frame()
result_df[score_col] = preds.iloc[:, -1].values    # 마지막 컬럼 = Bad 확률
```

---

### `model_type="r"` — R 스크립트

```bash
Rscript {r_script} --input /tmp/input.csv --model_dir /data/models/{model_id} --output /tmp/output.csv
```

R 스크립트가 생성한 `output.csv`를 읽어 `score_col`로 매핑한다.

---

## `_assign_grade(score, grade_mapping)` → `str`

```python
grade_mapping = {"A": [800, 1000], "B": [600, 800], "C": [400, 600], "D": [0, 400]}
# score=720 → "B"
# 매핑에 해당하지 않으면 "UNKNOWN"
```

---

## execute() 진행률

| 단계 | progress |
|------|----------|
| 모델 로드 완료 | 15% |
| 입력 데이터 로드 완료 | 35% |
| 예측 완료 | 70% |
| 등급 부여·저장 완료 | 95% |

---

## 반환값

```python
{
    "status": "COMPLETED",
    "result": {
        "output_id":   "loan_pred_202312",
        "model_id":    "lgbm_loan_v1",
        "total_rows":  50000,
        "output_path": "predict/loan_pred_202312_result.parquet",
        "score_stats": {
            "mean": 0.312, "std": 0.182,
            "p25": 0.158,  "p50": 0.291, "p75": 0.463,
        },
        "grade_dist": {"A": 12000, "B": 18000, "C": 15000, "D": 5000},
    },
    "message": "예측 완료: 50,000건  model=lgbm_loan_v1",
    "job_id":  str,
    "elapsed_sec": float,
}
```

- `grade_dist`는 `grade_mapping` 지정 시에만 포함

---

## 출력 파일

| 파일 | 경로 |
|------|------|
| 예측 결과 parquet | `predict/{output_id}_result.parquet` |

결과 파일에는 원본 데이터의 모든 컬럼 + `score_col` + `grade`(있으면) + `pred_class`(분류 시) 포함.

---

## 사용 예시

```python
config = {
    "job_id":     "predict_001",
    "model_id":   "lgbm_loan_v1",
    "input_path": "mart/loan_mart_test.parquet",
    "output_id":  "loan_pred_202312",
    "model_type": "python",
    "score_col":  "ml_score",
    "threshold":  0.5,
    "grade_mapping": {
        "A": [800, 1000],
        "B": [600, 800],
        "C": [400, 600],
        "D": [0,   400],
    },
}

from executors.ml.predict_executor import PredictExecutor
result = PredictExecutor(config=config).run()
```
