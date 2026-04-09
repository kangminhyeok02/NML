# python_model_executor.py — Python ML 모델 학습

## 개요

scikit-learn, XGBoost, LightGBM, CatBoost 등 순수 Python 기반 ML 모델을  
학습하고 평가한 뒤 pickle로 저장하는 executor.

---

## 지원 모델 (`model_type`)

| `model_type` | 클래스 | 라이브러리 |
|---|---|---|
| `logistic_regression` | `LogisticRegression` | scikit-learn |
| `random_forest` | `RandomForestClassifier` | scikit-learn |
| `gradient_boosting` | `GradientBoostingClassifier` | scikit-learn |
| `decision_tree` | `DecisionTreeClassifier` | scikit-learn |
| `linear_regression` | `LinearRegression` | scikit-learn |
| `random_forest_regressor` | `RandomForestRegressor` | scikit-learn |
| `xgboost` | `XGBClassifier` | xgboost |
| `lightgbm` | `LGBMClassifier` | lightgbm |
| `catboost` | `CatBoostClassifier` | catboost |

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `model_type` | ✅ | `str` | 모델 유형 (위 표 참조) |
| `train_path` | ✅ | `str` | 학습 데이터 경로 (.parquet) |
| `target_col` | ✅ | `str` | 타깃 컬럼명 |
| `model_id` | ✅ | `str` | 저장할 모델 식별자 |
| `valid_path` | ❌ | `str` | 검증 데이터 경로 (없으면 train 20% 자동 분리) |
| `feature_cols` | ❌ | `list` | 사용 피처 목록 (없으면 target 외 전체) |
| `model_params` | ❌ | `dict` | 모델 하이퍼파라미터 |
| `task` | ❌ | `str` | `"classification"` \| `"regression"` (기본: classification) |

---

## 메서드 상세

### `_build_model(model_type, params)` — 모듈 수준 함수

`model_type` 문자열로 모델 인스턴스를 생성한다.  
지원하지 않는 타입은 `ExecutorException` 발생.

```python
model = _build_model("lightgbm", {"n_estimators": 300, "learning_rate": 0.05})
```

---

### `_evaluate(model, X, y, task)` → `dict`

**분류 (`task="classification"`):**

| 지표 | 설명 |
|------|------|
| `accuracy` | 정확도 |
| `precision` | 정밀도 (binary) |
| `recall` | 재현율 (binary) |
| `f1` | F1 Score (binary) |
| `auc` | ROC-AUC (predict_proba 지원 시) |

**회귀 (`task="regression"`):**

| 지표 | 설명 |
|------|------|
| `rmse` | Root Mean Squared Error |
| `r2` | R² 결정계수 |

---

## 실행 흐름

```
1. train_path 로드                                   [progress 20%]
   - valid_path 없으면 train_test_split(test_size=0.2)
2. feature_cols / target_col 분리
3. _build_model() 로 모델 인스턴스 생성               [progress 40%]
4. model.fit(X_train, y_train)                       [progress 75%]
5. _evaluate() 로 검증 성능 계산
6. models/{model_id}.pkl 저장
7. models/{model_id}_meta.json 저장                  [progress 95%]
```

---

## 출력 결과

**모델 파일:** `models/{model_id}.pkl` (pickle)

**메타 파일:** `models/{model_id}_meta.json`
```json
{
  "model_id":     "lgbm_credit_v1",
  "model_type":   "lightgbm",
  "model_params": {"n_estimators": 300},
  "feature_cols": ["age", "income", "debt_ratio"],
  "target_col":   "default",
  "task":         "classification",
  "metrics": {
    "accuracy":  0.8823,
    "precision": 0.7641,
    "recall":    0.6938,
    "f1":        0.7273,
    "auc":       0.9102
  },
  "model_path": "models/lgbm_credit_v1.pkl"
}
```

---

## 사용 예시

```python
config = {
    "model_type":  "lightgbm",
    "train_path":  "mart/loan_mart_train.parquet",
    "valid_path":  "mart/loan_mart_valid.parquet",
    "target_col":  "default",
    "model_id":    "lgbm_v1",
    "model_params": {
        "n_estimators":  500,
        "learning_rate": 0.05,
        "num_leaves":    64,
        "verbose":       -1,
    },
    "task": "classification",
}
executor = PythonModelExecutor(config=config)
result   = executor.run()
# result["result"]["metrics"]["auc"] → 0.9102
```

---

## PredictExecutor와의 연계

`model_type="python"` 설정 시 `PredictExecutor`가 이 파일을 pickle로 로드한다.  
메타 파일의 `feature_cols`로 컬럼 정렬을 자동 수행한다.
