# python_model_executor.py

순수 Python 기반 ML 모델 학습/평가/저장 실행기.

scikit-learn, XGBoost, LightGBM, CatBoost 계열 모델을 지원한다.

---

## 모듈 레벨 함수

### `_build_model(model_type, params) → estimator`

`model_type` 문자열과 `params` 딕셔너리로 모델 인스턴스를 생성한다.

| `model_type` | 클래스 |
|---|---|
| `"logistic_regression"` | `sklearn.linear_model.LogisticRegression` |
| `"random_forest"` | `sklearn.ensemble.RandomForestClassifier` |
| `"xgboost"` | `xgboost.XGBClassifier` |
| `"lightgbm"` | `lightgbm.LGBMClassifier` |
| `"catboost"` | `catboost.CatBoostClassifier` |
| `"gradient_boosting"` | `sklearn.ensemble.GradientBoostingClassifier` |
| `"decision_tree"` | `sklearn.tree.DecisionTreeClassifier` |
| `"linear_regression"` | `sklearn.linear_model.LinearRegression` |
| `"random_forest_regressor"` | `sklearn.ensemble.RandomForestRegressor` |

미지원 타입은 `ExecutorException` 발생.

---

## 클래스

### `PythonModelExecutor(BaseExecutor)`

Python ML 모델 학습 executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `model_type` | `str` | 모델 유형 문자열 |
| `train_path` | `str` | 학습 데이터 상대 경로 (`.parquet`) |
| `target_col` | `str` | 타깃 컬럼명 |
| `model_id` | `str` | 저장할 모델 식별자 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|---|---|---|
| `valid_path` | - | 검증 데이터 경로. 없으면 학습 데이터의 20% 자동 분리 |
| `feature_cols` | 타깃 외 전체 | 사용할 피처 목록 |
| `model_params` | `{}` | 모델 하이퍼파라미터 |
| `task` | `"classification"` | `"classification"` \| `"regression"` |

---

### `execute() → dict`

모델 학습 전체 파이프라인을 실행한다.

**실행 순서**
1. 학습/검증 데이터 로드 (검증 데이터 없으면 `train_test_split` 80:20)
2. 피처/타깃 분리
3. `_build_model()`로 모델 인스턴스 생성
4. `model.fit(X_train, y_train)` 학습
5. `_evaluate()`로 검증 세트 성능 평가
6. 모델을 `models/{model_id}.pkl`에 pickle 저장
7. 메타 정보를 `models/{model_id}_meta.json`에 저장

**저장되는 메타 정보**
```python
{
    "model_id":     str,
    "model_type":   str,
    "model_params": dict,
    "feature_cols": list,
    "target_col":   str,
    "task":         str,
    "metrics":      dict,
    "model_path":   str,
    "model_type":   "python",
}
```

**반환값**
```python
{
    "status":  "COMPLETED",
    "result":  meta,
    "message": str,
}
```

---

### `_evaluate(model, X_valid, y_valid, task) → dict`

검증 세트에 대한 성능 지표를 산출한다.

**분류(`classification`) 지표**
- `auc`: ROC AUC
- `accuracy`: 정확도
- `f1`: F1 Score (weighted)
- `precision`: 정밀도 (weighted)
- `recall`: 재현율 (weighted)

**회귀(`regression`) 지표**
- `rmse`: Root Mean Squared Error
- `r2`: R² Score

### `_get_feature_importance(model, feature_cols) → dict`

모델의 변수 중요도를 추출한다.

- `feature_importances_` 속성이 있는 모델(트리 계열)에서 추출
- 정규화하여 상위 20개 반환
- 속성이 없는 모델은 빈 딕셔너리 반환
