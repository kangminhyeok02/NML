# automl_executor.py — 자동 모델 탐색 (AutoML)

## 개요

다양한 AutoML 프레임워크를 통합 지원하며, 여러 알고리즘 후보를 자동으로 탐색하고  
최적 모델을 선택하여 저장하는 executor.

---

## 지원 프레임워크 (`framework`)

| `framework` | 탐색 방식 | 특징 |
|---|---|---|
| `h2o_automl` | H2O 리더보드 | 대용량 데이터, 다수 알고리즘 동시 비교 |
| `autosklearn` | 앙상블 탐색 | scikit-learn 파이프라인 자동 구성 |
| `tpot` | 유전 알고리즘 | 파이프라인 구조 자체를 진화로 탐색 |
| `optuna` | 베이지안 최적화 | LightGBM 하이퍼파라미터 탐색 |
| `pycaret` | 비교 실험 | 다수 알고리즘 자동 비교 (AUC 기준) |

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `framework` | ✅ | `str` | AutoML 프레임워크 이름 |
| `train_path` | ✅ | `str` | 학습 데이터 경로 (.parquet) |
| `target_col` | ✅ | `str` | 타깃 컬럼명 |
| `model_id` | ✅ | `str` | 결과 저장 식별자 |
| `valid_path` | ❌ | `str` | 검증 데이터 경로 (없으면 train 20% 분리) |
| `feature_cols` | ❌ | `list` | 사용 피처 목록 |
| `max_runtime_sec` | ❌ | `int` | 최대 탐색 시간(초) (기본: `300`) |
| `n_trials` | ❌ | `int` | Optuna 시도 횟수 (기본: `50`) |
| `metric` | ❌ | `str` | 최적화 지표 (기본: `"auc"`) |
| `h2o_ip` | ❌ | `str` | H2O 서버 IP (h2o_automl 전용) |
| `h2o_port` | ❌ | `int` | H2O 서버 포트 (h2o_automl 전용) |

---

## 프레임워크별 동작

### `_run_h2o_automl`

```python
aml = H2OAutoML(max_runtime_secs=300, seed=42)
aml.train(x=features, y=target, training_frame=train_h2o, leaderboard_frame=valid_h2o)
best_model = aml.leader   # 리더보드 1위
```
- H2O 내부적으로 GBM, DRF, XGBoost, GLM, DeepLearning 등을 동시 탐색
- `H2OWrapper`로 sklearn-compatible `predict_proba` 인터페이스 제공

---

### `_run_autosklearn`

```python
model = AutoSklearnClassifier(
    time_left_for_this_task=300,
    per_run_time_limit=30,
    seed=42,
)
model.fit(X_train, y_train)
```
- 리더보드: `get_models_with_weights()` → 앙상블 구성 모델과 가중치

---

### `_run_tpot`

```python
model = TPOTClassifier(max_time_mins=5, random_state=42)
model.fit(X_train, y_train)
best = model.fitted_pipeline_   # 최적 sklearn Pipeline
```
- 유전 알고리즘으로 전처리+모델 파이프라인 구조 자체를 탐색

---

### `_run_optuna`

LightGBM을 고정하고 하이퍼파라미터 공간을 베이지안 최적화로 탐색한다.

```python
# 탐색 공간
n_estimators   : 50 ~ 500
max_depth      : 3 ~ 12
learning_rate  : 0.001 ~ 0.3 (log scale)
num_leaves     : 16 ~ 128
subsample      : 0.5 ~ 1.0
colsample_bytree: 0.5 ~ 1.0
```

- `direction="maximize"` (AUC 최대화)
- 리더보드: 상위 10개 trial의 AUC + 파라미터 반환

---

### `_run_pycaret`

```python
setup(data=train_df, target=target_col, session_id=42)
best = compare_models(sort="AUC", n_select=1)
```
- 내부적으로 15+ 알고리즘을 교차검증으로 비교

---

## 실행 흐름

```
1. 데이터 로드 및 train/valid 분리                   [progress 20%]
2. 프레임워크별 AutoML 실행                          [progress 80%]
3. 최적 모델 → models/{model_id}_automl.pkl 저장
4. 최종 검증 AUC 산출 (predict_proba 지원 시)
5. 리더보드 + 메타 → models/{model_id}_meta.json
```

---

## 출력 결과

**모델 파일:** `models/{model_id}_automl.pkl`

**메타 파일:** `models/{model_id}_meta.json`
```json
{
  "model_id":    "automl_credit_v1",
  "framework":   "optuna",
  "feature_cols": ["age", "income", ...],
  "target_col":  "default",
  "final_auc":   0.9234,
  "leaderboard": [
    {"rank": 1, "auc": 0.9234, "params": {"n_estimators": 412, ...}},
    {"rank": 2, "auc": 0.9198, "params": {"n_estimators": 289, ...}},
    ...
  ],
  "model_path": "models/automl_credit_v1_automl.pkl",
  "model_type": "python"
}
```

---

## 프레임워크 선택 가이드

| 상황 | 권장 프레임워크 |
|------|--------------|
| 빠른 베이스라인 | `pycaret` |
| 대용량 + 다알고리즘 비교 | `h2o_automl` |
| LightGBM 세밀 튜닝 | `optuna` |
| 파이프라인 구조 탐색 | `tpot` |
| 재현 가능한 앙상블 | `autosklearn` |
