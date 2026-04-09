# python_model_executor.py — Python ML 모델 학습

**파일:** `executors/ml/python_model_executor.py`  
**클래스:** `PythonModelExecutor(BaseExecutor)`

## 개요

scikit-learn, XGBoost, LightGBM, CatBoost 등 순수 Python 기반 ML 모델을  
학습하고 평가한 뒤 pickle로 저장하는 executor.

```
학습/검증 데이터 로드
    ↓ 피처/타깃 분리
    ↓ _build_model(model_type, model_params)
    ↓ model.fit(X_train, y_train)
    ↓ _evaluate(model, X_valid, y_valid, task)
    ↓ pickle 저장 + meta JSON 저장
```

---

## 지원 모델 (`model_type`)

| `model_type` | 클래스 | 라이브러리 | task |
|---|---|---|---|
| `logistic_regression` | `LogisticRegression` | scikit-learn | 분류 |
| `random_forest` | `RandomForestClassifier` | scikit-learn | 분류 |
| `gradient_boosting` | `GradientBoostingClassifier` | scikit-learn | 분류 |
| `decision_tree` | `DecisionTreeClassifier` | scikit-learn | 분류 |
| `xgboost` | `XGBClassifier` | xgboost | 분류 |
| `lightgbm` | `LGBMClassifier` | lightgbm | 분류 |
| `catboost` | `CatBoostClassifier` | catboost | 분류 |
| `linear_regression` | `LinearRegression` | scikit-learn | 회귀 |
| `random_forest_regressor` | `RandomForestRegressor` | scikit-learn | 회귀 |

지원하지 않는 `model_type`은 `ExecutorException` 발생.

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `model_type` | ✅ | `str` | 모델 유형 (위 표 참조) |
| `train_path` | ✅ | `str` | 학습 데이터 경로 (.parquet) |
| `target_col` | ✅ | `str` | 타깃 컬럼명 |
| `model_id` | ✅ | `str` | 저장할 모델 식별자 |
| `valid_path` | ❌ | `str` | 검증 데이터 경로 (없으면 train 20% 자동 분리) |
| `feature_cols` | ❌ | `list` | 사용 피처 목록 (없으면 타깃 외 전체) |
| `model_params` | ❌ | `dict` | 모델 하이퍼파라미터 |
| `task` | ❌ | `str` | `"classification"` \| `"regression"` (기본: `"classification"`) |

---

## `_build_model(model_type, params)` — 모듈 수준 함수

`model_type` 문자열로 모델 인스턴스를 생성하는 팩토리 함수.  
모든 `model_params`는 생성자에 `**params`로 전달된다.

```python
model = _build_model("lightgbm", {"n_estimators": 300, "learning_rate": 0.05, "verbose": -1})
```

---

## `_evaluate(model, X, y, task)` → `dict`

### 분류 (`task="classification"`)

| 지표 | 산출 방법 |
|------|----------|
| `auc` | `roc_auc_score(y, proba[:, 1])` (predict_proba 지원 시) |
| `accuracy` | `accuracy_score(y, y_pred)` |
| `f1` | `f1_score(y, y_pred, average="binary")` |
| `precision` | `precision_score(y, y_pred, average="binary")` |
| `recall` | `recall_score(y, y_pred, average="binary")` |

### 회귀 (`task="regression"`)

| 지표 | 산출 방법 |
|------|----------|
| `rmse` | `sqrt(mean_squared_error(y, y_pred))` |
| `r2` | `r2_score(y, y_pred)` |

---

## execute() 진행률

| 단계 | progress |
|------|----------|
| 데이터 로드 완료 | 20% |
| 피처/타깃 분리 완료 | 35% |
| 모델 학습 완료 | 75% |
| 평가·저장 완료 | 95% |

---

## 반환값

```python
{
    "status": "COMPLETED",
    "result": {
        "model_id":     "lgbm_loan_v1",
        "model_type":   "lightgbm",
        "model_path":   "models/lgbm_loan_v1.pkl",
        "feature_cols": ["income", "debt_ratio", ...],
        "target_col":   "default",
        "task":         "classification",
        "metrics":      {"auc": 0.8923, "accuracy": 0.8412, "f1": 0.7231, ...},
        "model_params": {"n_estimators": 500, "learning_rate": 0.05},
    },
    "message": "모델 학습 완료  lightgbm  AUC=0.8923",
    "job_id":  str,
    "elapsed_sec": float,
}
```

---

## 출력 파일

| 파일 | 경로 |
|------|------|
| 모델 파일 (pickle) | `models/{model_id}.pkl` |
| 메타 정보 JSON | `models/{model_id}_meta.json` |

`_meta.json` 구조:
```json
{
  "model_id":     "lgbm_loan_v1",
  "model_type":   "lightgbm",
  "model_path":   "models/lgbm_loan_v1.pkl",
  "feature_cols": ["income", "debt_ratio", ...],
  "target_col":   "default",
  "task":         "classification",
  "metrics":      {"auc": 0.8923, "accuracy": 0.8412},
  "model_params": {"n_estimators": 500}
}
```

---

## 사용 예시

```python
config = {
    "job_id":      "train_job_001",
    "model_type":  "lightgbm",
    "train_path":  "mart/retail_mart_v2_train.parquet",
    "valid_path":  "mart/retail_mart_v2_valid.parquet",
    "target_col":  "default_yn",
    "model_id":    "lgbm_retail_v2",
    "model_params": {
        "n_estimators":  500,
        "learning_rate": 0.05,
        "num_leaves":    64,
        "verbose":       -1,
    },
}

from executors.ml.python_model_executor import PythonModelExecutor
result = PythonModelExecutor(config=config).run()
```
