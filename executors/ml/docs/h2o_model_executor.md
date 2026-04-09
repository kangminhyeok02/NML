# h2o_model_executor.py

H2O 프레임워크 기반 모델 학습/예측 실행기.

H2O는 GBM, DRF, XGBoost, DeepLearning, GLM 등 다양한 알고리즘을 제공하며,  
H2O 서버와 연동하여 대용량 데이터 학습과 빠른 예측을 지원한다.

---

## 모듈 상수

### `H2O_ALGORITHMS`

지원 알고리즘 키 → H2O 클래스명 매핑.

| 키 | H2O 클래스 |
|---|---|
| `"gbm"` | `H2OGradientBoostingEstimator` |
| `"drf"` | `H2ORandomForestEstimator` |
| `"xgboost"` | `H2OXGBoostEstimator` |
| `"glm"` | `H2OGeneralizedLinearEstimator` |
| `"deeplearning"` | `H2ODeepLearningEstimator` |
| `"automl"` | `H2OAutoML` |

---

## 클래스

### `H2OModelExecutor(BaseExecutor)`

H2O 모델 학습/예측 executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `algorithm` | `str` | H2O 알고리즘 키 |
| `train_path` | `str` | 학습 데이터 경로 (`.parquet`) |
| `target_col` | `str` | 타깃 컬럼명 |
| `model_id` | `str` | 저장 식별자 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|---|---|---|
| `valid_path` | - | 검증 데이터 경로. 없으면 학습 데이터를 80:20으로 분할 |
| `feature_cols` | 타깃 외 전체 | 사용 피처 목록 |
| `model_params` | `{}` | H2O 알고리즘 파라미터 |
| `h2o_ip` | `"localhost"` | H2O 서버 IP |
| `h2o_port` | `54321` | H2O 서버 포트 |
| `max_runtime_sec` | `300` | AutoML 최대 실행 시간(초) |

---

### `execute() → dict`

H2O 모델 학습 전체 파이프라인을 실행한다.

**실행 순서**
1. `h2o.init(ip, port)`로 H2O 클러스터에 연결
2. 데이터를 `pd.DataFrame` → `H2OFrame`으로 변환
3. 타깃 컬럼을 factor(범주형)로 변환
4. `_train_model()`로 모델 학습
5. `model.model_performance(valid_h2o)`로 성능 평가 (AUC, LogLoss, KS)
6. `model.save_mojo(mojo_dir)`로 MOJO 파일 저장
7. `model.varimp()`로 변수 중요도 추출 (상위 20개)
8. 메타 정보를 `models/{model_id}_meta.json`에 저장

**저장되는 메타 정보**
```python
{
    "model_id":     str,
    "algorithm":    str,
    "h2o_model_id": str,
    "mojo_path":    str,
    "feature_cols": list,
    "target_col":   str,
    "metrics":      {auc, logloss, ks},
    "varimp":       {variable: percentage},
    "model_type":   "h2o",
}
```

**반환값**
```python
{
    "status":  "COMPLETED",
    "result":  meta,
    "message": str,   # AUC 포함
}
```

---

### `_train_model(h2o, algorithm, x, y, train_h2o, valid_h2o, cfg, params)`

알고리즘에 따라 H2O 모델을 학습한다.

- `"automl"`: `H2OAutoML(max_runtime_secs=...)` 실행 후 `aml.leader` 반환
- 그 외: `algo_map[algorithm](**params)` 후 `model.train()` 실행

알고리즘이 `algo_map`에 없으면 `ExecutorException` 발생.

**지원 알고리즘별 매핑**

| 알고리즘 | 클래스 |
|---|---|
| `gbm` | `H2OGradientBoostingEstimator` |
| `drf` | `H2ORandomForestEstimator` |
| `xgboost` | `H2OXGBoostEstimator` |
| `glm` | `H2OGeneralizedLinearEstimator` |
| `deeplearning` | `H2ODeepLearningEstimator` |
| `automl` | `H2OAutoML` |
