# h2o_model_executor.py — H2O 모델 학습

## 개요

H2O 클러스터와 연동하여 GBM, DRF, XGBoost, GLM, DeepLearning, AutoML 등  
H2O 알고리즘으로 모델을 학습하고 MOJO 파일로 저장하는 executor.

---

## 지원 알고리즘 (`algorithm`)

| `algorithm` | H2O 클래스 | 설명 |
|---|---|---|
| `gbm` | `H2OGradientBoostingEstimator` | Gradient Boosting Machine |
| `drf` | `H2ORandomForestEstimator` | Distributed Random Forest |
| `xgboost` | `H2OXGBoostEstimator` | XGBoost (H2O 래핑) |
| `glm` | `H2OGeneralizedLinearEstimator` | 일반화 선형모델 |
| `deeplearning` | `H2ODeepLearningEstimator` | 심층 신경망 |
| `automl` | `H2OAutoML` | 자동 모델 탐색 (리더보드) |

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `algorithm` | ✅ | `str` | H2O 알고리즘 이름 |
| `train_path` | ✅ | `str` | 학습 데이터 경로 (.parquet) |
| `target_col` | ✅ | `str` | 타깃 컬럼명 |
| `model_id` | ✅ | `str` | 저장 식별자 |
| `valid_path` | ❌ | `str` | 검증 데이터 경로 (없으면 H2O 내부에서 80:20 분리) |
| `feature_cols` | ❌ | `list` | 사용 피처 목록 |
| `model_params` | ❌ | `dict` | H2O 알고리즘 파라미터 |
| `h2o_ip` | ❌ | `str` | H2O 서버 IP (기본: `localhost`) |
| `h2o_port` | ❌ | `int` | H2O 서버 포트 (기본: `54321`) |
| `max_runtime_sec` | ❌ | `int` | AutoML 최대 실행 시간(초) |

---

## 메서드 상세

### `_train_model(h2o, algorithm, x, y, train_h2o, valid_h2o, cfg, params)`

`algorithm` 값에 따라 H2O Estimator를 선택하여 학습한다.

```python
# AutoML 특수 처리
if algorithm == "automl":
    aml = H2OAutoML(max_runtime_secs=cfg.get("max_runtime_sec", 300), seed=42)
    aml.train(x=x, y=y, training_frame=train_h2o, leaderboard_frame=valid_h2o)
    return aml.leader   # 리더보드 1위 모델 반환

# 일반 알고리즘
estimator = algo_map[algorithm](**params)
estimator.train(x=x, y=y, training_frame=train_h2o, validation_frame=valid_h2o)
return estimator
```

---

## 실행 흐름

```
1. H2O 클러스터 초기화 (h2o.init)                [progress 15%]
2. parquet → H2OFrame 변환
   - valid_path 없으면 split_frame(ratios=[0.8])
   - 타깃 컬럼 → asfactor() (분류)
3. _train_model() 로 H2O 모델 학습               [progress 70%]
4. 성능 평가 (model_performance)
   - AUC, LogLoss, KS 산출
5. MOJO 저장 (model.save_mojo)                   [progress 90%]
6. 변수 중요도 추출 (model.varimp, 상위 20개)
7. models/{model_id}_meta.json 저장
```

---

## 성능 지표

| 지표 | 설명 |
|------|------|
| `auc` | ROC-AUC |
| `logloss` | Log Loss |
| `ks` | Kolmogorov-Smirnov 통계량 (지원 시) |

---

## 출력 결과

**MOJO 파일:** `models/{model_id}/model.zip`

**메타 파일:** `models/{model_id}_meta.json`
```json
{
  "model_id":     "gbm_credit_v1",
  "algorithm":    "gbm",
  "h2o_model_id": "GBM_model_python_1234",
  "mojo_path":    "/data/models/gbm_credit_v1/model.zip",
  "feature_cols": ["age", "income"],
  "target_col":   "default",
  "metrics": {
    "auc":     0.9215,
    "logloss": 0.2341,
    "ks":      0.6782
  },
  "varimp": {
    "income":     0.3421,
    "debt_ratio": 0.2108,
    "age":        0.1543
  },
  "model_type": "h2o"
}
```

---

## PythonH2OModelExecutor / PredictExecutor와의 연계

- `model_type="h2o"` 설정 시 `PredictExecutor`가 MOJO를 `h2o.import_mojo()`로 로드
- `PythonH2OModelExecutor`는 Python 전처리 후 이 MOJO로 추론하는 혼합 파이프라인 구성 가능

```
H2OModelExecutor (학습) → MOJO 저장
    ↓
PythonH2OModelExecutor (운영 추론)
    Python 전처리 → H2O MOJO 추론 → Python 후처리
```

---

## H2O vs Python 모델 선택 기준

| 상황 | 권장 |
|------|------|
| 대용량 데이터 (수억 건) | `H2OModelExecutor` |
| AutoML 리더보드 탐색 | `H2OModelExecutor (algorithm=automl)` |
| 커스텀 피처 파이프라인 필요 | `PythonModelExecutor` |
| 배포 환경에 H2O 서버 없음 | `PythonModelExecutor` (ONNX 내보내기 고려) |
