# automl_executor.py — 자동 모델 탐색 (AutoML)

**파일:** `executors/ml/automl_executor.py`  
**클래스:** `AutoMLExecutor(BaseExecutor)`

## 개요

다양한 AutoML 프레임워크를 통합 지원하며, 여러 알고리즘 후보를 자동으로 탐색하고  
최적 모델을 선택하여 pickle로 저장하는 executor.

```
데이터 로드 → train/valid 분리
    ↓ 프레임워크별 AutoML 실행 (dispatch)
      ├── h2o_automl   → H2O 리더보드
      ├── autosklearn  → sklearn 앙상블 탐색
      ├── tpot         → 유전 알고리즘 파이프라인
      ├── optuna       → 베이지안 최적화 (LightGBM)
      └── pycaret      → 다중 모델 비교
    ↓ 최적 모델 → models/{model_id}_automl.pkl 저장
    ↓ 최종 AUC 산출 (predict_proba 지원 시)
    ↓ 리더보드 + meta → models/{model_id}_meta.json
```

---

## 지원 프레임워크 (`framework`)

| `framework` | 탐색 방식 | 의존 라이브러리 |
|---|---|---|
| `h2o_automl` | H2O 리더보드 (GBM/DRF/XGBoost/GLM/DL 동시 탐색) | `h2o` |
| `autosklearn` | sklearn 파이프라인 앙상블 탐색 (Linux 전용) | `auto-sklearn` |
| `tpot` | 유전 알고리즘 기반 파이프라인 구조 탐색 | `tpot` |
| `optuna` | LightGBM 고정 + 하이퍼파라미터 베이지안 탐색 | `optuna`, `lightgbm` |
| `pycaret` | 15+ 알고리즘 교차검증 비교 | `pycaret` |

지원하지 않는 프레임워크는 `ExecutorException` 발생.

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `framework` | ✅ | `str` | AutoML 프레임워크 이름 |
| `train_path` | ✅ | `str` | 학습 데이터 경로 (.parquet) |
| `target_col` | ✅ | `str` | 타깃 컬럼명 |
| `model_id` | ✅ | `str` | 결과 저장 식별자 |
| `valid_path` | ❌ | `str` | 검증 데이터 경로 (없으면 train 20% 자동 분리) |
| `feature_cols` | ❌ | `list` | 사용 피처 목록 (없으면 타깃 외 전체) |
| `max_runtime_sec` | ❌ | `int` | 최대 탐색 시간(초) (기본: `300`) |
| `n_trials` | ❌ | `int` | Optuna 시도 횟수 (기본: `50`) |
| `metric` | ❌ | `str` | 최적화 지표 (기본: `"auc"`) |
| `h2o_ip` | ❌ | `str` | H2O 서버 IP (`h2o_automl` 전용, 기본: `localhost`) |
| `h2o_port` | ❌ | `int` | H2O 서버 포트 (`h2o_automl` 전용, 기본: `54321`) |

---

## 프레임워크별 동작

### `_run_h2o_automl`

```python
aml = H2OAutoML(max_runtime_secs=cfg.get("max_runtime_sec", 300), seed=42)
aml.train(x=features, y=target, training_frame=train_h2o, leaderboard_frame=valid_h2o)
best_model = aml.leader    # 리더보드 1위 (H2OWrapper로 predict_proba 제공)
leaderboard = aml.leaderboard.as_data_frame().to_dict(orient="records")
```

### `_run_autosklearn`

```python
model = AutoSklearnClassifier(
    time_left_for_this_task=cfg.get("max_runtime_sec", 300),
    per_run_time_limit=30,
    seed=42,
)
model.fit(X_train, y_train)
leaderboard = [{"rank": i+1, "weight": w, "model": str(m)}
               for i, (w, m) in enumerate(model.get_models_with_weights())]
```

### `_run_tpot`

```python
model = TPOTClassifier(max_time_mins=cfg.get("max_runtime_sec", 300) // 60, random_state=42)
model.fit(X_train, y_train)
best = model.fitted_pipeline_    # 최적 sklearn Pipeline
```

### `_run_optuna`

LightGBM을 고정하고 하이퍼파라미터 공간을 베이지안 최적화로 탐색한다.

**탐색 공간:**
```python
n_estimators    : 50 ~ 500
max_depth       : 3 ~ 12
learning_rate   : 0.001 ~ 0.3  (log scale)
num_leaves      : 16 ~ 128
subsample       : 0.5 ~ 1.0
colsample_bytree: 0.5 ~ 1.0
```

- `direction="maximize"` (AUC 최대화)
- 리더보드: 상위 10개 trial의 AUC + 파라미터

### `_run_pycaret`

```python
setup(data=train_df, target=target_col, session_id=42)
best = compare_models(sort="AUC", n_select=1)
# 내부적으로 15+ 알고리즘을 교차검증으로 비교
```

---

## execute() 진행률

| 단계 | progress |
|------|----------|
| 데이터 로드·분리 완료 | 20% |
| AutoML 탐색 완료 | 80% |
| 모델 저장·성능 평가 완료 | 95% |

---

## 반환값

```python
{
    "status": "COMPLETED",
    "result": {
        "model_id":    "automl_credit_v1",
        "framework":   "optuna",
        "feature_cols": ["age", "income", ...],
        "target_col":  "default",
        "final_auc":   0.9234,
        "leaderboard": [
            {"rank": 1, "auc": 0.9234, "params": {"n_estimators": 412, ...}},
            {"rank": 2, "auc": 0.9198, "params": {"n_estimators": 289, ...}},
        ],
        "model_path":  "models/automl_credit_v1_automl.pkl",
        "model_type":  "python",
    },
    "message": "AutoML 완료  framework=optuna  AUC=0.9234  후보=10개",
    "job_id":  str,
    "elapsed_sec": float,
}
```

---

## 출력 파일

| 파일 | 경로 |
|------|------|
| 최적 모델 (pickle) | `models/{model_id}_automl.pkl` |
| 메타 + 리더보드 JSON | `models/{model_id}_meta.json` |

---

## 프레임워크 선택 가이드

| 상황 | 권장 |
|------|------|
| 가장 범용적, 의존성 최소 | `optuna` |
| 대용량 + 다알고리즘 앙상블 | `h2o_automl` |
| 파이프라인 구조 자체를 탐색 | `tpot` |
| 빠른 초기 베이스라인 비교 | `pycaret` |
| 재현 가능한 앙상블 (Linux) | `autosklearn` |

---

## 사용 예시

```python
config = {
    "job_id":          "automl_001",
    "framework":       "optuna",
    "train_path":      "mart/retail_mart_train.parquet",
    "valid_path":      "mart/retail_mart_valid.parquet",
    "target_col":      "default_yn",
    "model_id":        "automl_retail_v1",
    "max_runtime_sec": 600,
    "n_trials":        100,
}

from executors.ml.automl_executor import AutoMLExecutor
result = AutoMLExecutor(config=config).run()
```
