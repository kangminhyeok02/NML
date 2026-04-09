# h2o_model_executor.py — H2O 모델 학습

**파일:** `executors/ml/h2o_model_executor.py`  
**클래스:** `H2OModelExecutor(BaseExecutor)`

## 개요

H2O 클러스터와 연동하여 GBM, DRF, XGBoost, GLM, DeepLearning, AutoML 등  
H2O 알고리즘으로 모델을 학습하고 MOJO 파일로 저장하는 executor.

```
H2O 클러스터 초기화 (h2o.init)
    ↓ parquet 로드 → H2OFrame 변환
    ↓ valid_path 없으면 split_frame(ratios=[0.8], seed=42)
    ↓ 타깃 컬럼 → asfactor() (분류)
    ↓ _train_model() 로 H2O 모델 학습
    ↓ model_performance() → AUC, LogLoss, KS 산출
    ↓ model.save_mojo() → MOJO 저장
    ↓ model.varimp() → 변수 중요도 상위 20개 추출
    ↓ meta JSON 저장
```

---

## 지원 알고리즘 (`algorithm`)

| `algorithm` | H2O 클래스 | 설명 |
|---|---|---|
| `gbm` | `H2OGradientBoostingEstimator` | Gradient Boosting Machine |
| `drf` | `H2ORandomForestEstimator` | Distributed Random Forest |
| `xgboost` | `H2OXGBoostEstimator` | XGBoost (H2O 래핑) |
| `glm` | `H2OGeneralizedLinearEstimator` | 일반화 선형모델 |
| `deeplearning` | `H2ODeepLearningEstimator` | 심층 신경망 |
| `automl` | `H2OAutoML` | 자동 모델 탐색 (리더보드 1위 반환) |

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `algorithm` | ✅ | `str` | H2O 알고리즘 이름 |
| `train_path` | ✅ | `str` | 학습 데이터 경로 (.parquet) |
| `target_col` | ✅ | `str` | 타깃 컬럼명 |
| `model_id` | ✅ | `str` | 저장 식별자 |
| `valid_path` | ❌ | `str` | 검증 데이터 경로 (없으면 80:20 자동 분리) |
| `feature_cols` | ❌ | `list` | 사용 피처 목록 (없으면 타깃 외 전체) |
| `model_params` | ❌ | `dict` | H2O 알고리즘 파라미터 |
| `h2o_ip` | ❌ | `str` | H2O 서버 IP (기본: `localhost`) |
| `h2o_port` | ❌ | `int` | H2O 서버 포트 (기본: `54321`) |
| `max_runtime_sec` | ❌ | `int` | AutoML 최대 실행 시간(초) |

---

## `_train_model(h2o, algorithm, x, y, train_h2o, valid_h2o, cfg, params)`

`algorithm` 값에 따라 H2O Estimator를 선택하여 학습한다.

```python
# AutoML 특수 처리
if algorithm == "automl":
    aml = H2OAutoML(max_runtime_secs=cfg.get("max_runtime_sec", 300), seed=42)
    aml.train(x=x, y=y, training_frame=train_h2o, leaderboard_frame=valid_h2o)
    return aml.leader   # 리더보드 1위 모델 반환

# 일반 알고리즘
estimator = H2O_ALGORITHMS[algorithm](**params)
estimator.train(x=x, y=y, training_frame=train_h2o, validation_frame=valid_h2o)
return estimator
```

---

## execute() 진행률

| 단계 | progress |
|------|----------|
| H2O 클러스터 연결 완료 | 15% |
| 데이터 로드·H2OFrame 변환 완료 | 30% |
| 모델 학습 완료 | 70% |
| MOJO 저장·변수중요도 추출 완료 | 90% |

---

## 성능 지표

| 지표 | 산출 방법 |
|------|----------|
| `auc` | `perf.auc()` |
| `logloss` | `perf.logloss()` |
| `ks` | `perf.kolmogorov_smirnov()` (지원 시) |

---

## 반환값

```python
{
    "status": "COMPLETED",
    "result": {
        "model_id":      "gbm_credit_v1",
        "algorithm":     "gbm",
        "h2o_model_id":  "GBM_model_python_1234567890",
        "mojo_path":     "/data/models/gbm_credit_v1/model.zip",
        "feature_cols":  ["age", "income", "debt_ratio"],
        "target_col":    "default",
        "metrics":       {"auc": 0.9215, "logloss": 0.2341, "ks": 0.6782},
        "varimp":        {"income": 0.3421, "debt_ratio": 0.2108, "age": 0.1543},
        "model_type":    "h2o",
    },
    "message": "H2O 모델 학습 완료  gbm  AUC=0.9215",
    "job_id":  str,
    "elapsed_sec": float,
}
```

---

## 출력 파일

| 파일 | 경로 |
|------|------|
| MOJO 파일 | `models/{model_id}/model.zip` |
| 메타 정보 JSON | `models/{model_id}_meta.json` |

---

## PredictExecutor / PythonH2OModelExecutor와의 연계

```
H2OModelExecutor (학습) → MOJO 저장
    ↓
PredictExecutor (model_type="h2o")
    → h2o.import_mojo(mojo_path)로 로드 후 예측

H2OModelExecutor (학습) → MOJO 저장
    ↓
PythonH2OModelExecutor (운영 추론)
    → Python 전처리 → H2O MOJO 추론 → Python 후처리
```

---

## H2O vs Python 모델 선택 기준

| 상황 | 권장 |
|------|------|
| 대용량 데이터 (수천만 건 이상) | `H2OModelExecutor` |
| AutoML 리더보드 탐색 | `H2OModelExecutor (algorithm=automl)` |
| H2O 없는 배포 환경 | `PythonModelExecutor` |
| H2O 학습 + Python 운영 파이프라인 | `H2OModelExecutor` → `PythonH2OModelExecutor` |

---

## 사용 예시

```python
config = {
    "job_id":    "h2o_train_001",
    "algorithm": "gbm",
    "train_path": "mart/loan_mart_train.parquet",
    "valid_path": "mart/loan_mart_valid.parquet",
    "target_col": "default",
    "model_id":   "gbm_loan_v1",
    "h2o_ip":     "localhost",
    "h2o_port":   54321,
    "model_params": {"ntrees": 200, "max_depth": 6, "learn_rate": 0.05},
}

from executors.ml.h2o_model_executor import H2OModelExecutor
result = H2OModelExecutor(config=config).run()
```
