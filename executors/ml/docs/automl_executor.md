# automl_executor.py

자동 모델 탐색(AutoML) 실행기.

다양한 AutoML 프레임워크를 통합 지원하며, 여러 알고리즘 후보를 자동으로 탐색하고 최적 모델을 선택한다.

---

## 모듈 상수

```python
SUPPORTED_FRAMEWORKS = ["h2o_automl", "autosklearn", "tpot", "optuna", "pycaret"]
```

---

## 클래스

### `AutoMLExecutor(BaseExecutor)`

AutoML 실행 executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `framework` | `str` | AutoML 프레임워크 |
| `train_path` | `str` | 학습 데이터 경로 (`.parquet`) |
| `target_col` | `str` | 타깃 컬럼명 |
| `model_id` | `str` | 결과 저장 식별자 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|---|---|---|
| `valid_path` | - | 검증 데이터 경로. 없으면 80:20 자동 분리 |
| `feature_cols` | 타깃 외 전체 | 사용 피처 목록 |
| `max_runtime_sec` | `300` | 최대 탐색 시간(초) |
| `n_trials` | `50` | Optuna 시도 횟수 |
| `metric` | `"auc"` | 최적화 지표 |

---

### `execute() → dict`

AutoML 탐색을 실행하고 최적 모델을 저장한다.

**실행 순서**
1. `framework` 검증
2. 학습/검증 데이터 로드
3. 프레임워크별 실행 메서드 호출
4. 최적 모델을 `models/{model_id}_automl.pkl`에 pickle 저장
5. 검증 AUC 산출
6. 메타 정보를 `models/{model_id}_meta.json`에 저장

**반환값**
```python
{
    "status": "COMPLETED",
    "result": {
        "model_id":    str,
        "framework":   str,
        "feature_cols": list,
        "target_col":  str,
        "final_auc":   float,
        "leaderboard": list,
        "model_path":  str,
        "model_type":  "python",
    },
    "message": str,
}
```

---

### 프레임워크별 실행 메서드

### `_run_h2o_automl(cfg, X_train, y_train, X_valid, y_valid)`

H2O AutoML을 실행한다.

- `h2o.init()`으로 H2O 클러스터에 연결
- `H2OAutoML(max_runtime_secs=...)` 실행
- 최적 모델을 sklearn 호환 `H2OWrapper`로 래핑하여 반환
- 리더보드를 DataFrame에서 dict 목록으로 변환

### `_run_autosklearn(cfg, X_train, y_train, X_valid, y_valid)`

auto-sklearn으로 앙상블 모델을 탐색한다.

- `AutoSklearnClassifier(time_left_for_this_task=..., per_run_time_limit=30)` 실행
- 리더보드: `get_models_with_weights()`로 추출

### `_run_tpot(cfg, X_train, y_train, X_valid, y_valid)`

TPOT(유전 알고리즘 기반)으로 파이프라인을 탐색한다.

- `max_time_mins = max_runtime_sec // 60`
- `fitted_pipeline_`을 최적 모델로 반환

### `_run_optuna(cfg, X_train, y_train, X_valid, y_valid)`

Optuna + LightGBM으로 베이지안 하이퍼파라미터 최적화를 수행한다.

**탐색 공간**

| 파라미터 | 범위 |
|---|---|
| `n_estimators` | 50 ~ 500 |
| `max_depth` | 3 ~ 12 |
| `learning_rate` | 1e-3 ~ 0.3 (log scale) |
| `num_leaves` | 16 ~ 128 |
| `subsample` | 0.5 ~ 1.0 |
| `colsample_bytree` | 0.5 ~ 1.0 |

- 목적함수: 검증 AUC 최대화
- 상위 10개 trial을 리더보드로 반환

### `_run_pycaret(cfg, X_train, y_train, X_valid, y_valid)`

PyCaret `compare_models(sort="AUC")`로 여러 알고리즘을 비교하고 최적 모델을 선택한다.

---

## 모듈 레벨 함수

### `fit_nice_auto_ml(..., json_obj) → dict`

NICE 시스템 통합 AutoML 학습 함수.

**주요 동작**
1. hyperopt + LightGBM으로 하이퍼파라미터 탐색
2. `search_space` 키로 커스텀 탐색 공간 지정 가능
3. n-fold CV 또는 hold-out 검증 선택 가능
4. 앙상블 가중치 자동 생성 (`n_ensemble` 설정 시)
5. 최적 모델과 앙상블 모델을 pickle로 저장
6. `result_file_path_faf`, `done_file_path_faf`에 결과 기록

**반환하는 메타 정보 키**
`model_id`, `framework`, `feature_cols`, `target_col`, `ks_dev`, `ks_val`, `n_trials`, `hidden_layers`, `ensemble`, `score_dist_train`, `score_dist_val`, `history`, `model_path`, `model_type`

---

### `convert_hidden_layer_to_dict(hidden_layer_str) → list`

문자열 형태의 DNN hidden layer 설정을 dict 목록으로 변환한다.

```python
# 입력: "64,32,16"
# 출력: [{"size": 64}, {"size": 32}, {"size": 16}]
```

---

### `convert_list_to_hp(d) → dict`

파라미터 딕셔너리를 hyperopt `hp` 탐색 공간으로 변환한다.

| 입력 형태 | 변환 결과 |
|---|---|
| `[v1, v2, ...]` | `hp.choice` |
| `[min, max]` (숫자 2개) | `hp.uniform` |
| `{"min": x, "max": y}` | `hp.uniform` |
| 단일 값 | `hp.choice([val])` |

---

### `get_automl_objective_loss(ks_dev, ks_val, best_model_multiplier=5, val_perf_rate=100) → float`

AutoML 목적함수 손실값을 계산한다.

- 검증 성능이 높을수록 손실 감소
- 개발/검증 AUC 격차가 클수록 과적합 페널티 부여
- 반환값: `max(0, min(1 - val_reward + gap_penalty, 10))`

---

### `generate_ensemble_weights(n_models, inc=0.1) → list`

합이 1이 되는 앙상블 가중치 조합 목록을 생성한다.

```python
generate_ensemble_weights(2, inc=0.5)
# → [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
```

---

### `predict_nice_auto_ml(..., json_obj) → dict`

NICE AutoML 모델로 예측을 수행한다.

- 메타에서 `feature_cols`, `ensemble.weights` 로드
- 앙상블 모델의 가중 평균으로 최종 점수 산출
- 결과를 `predict/{output_id}_result.parquet`에 저장

---

### `apply_nice_auto_ml(..., json_obj) → dict`

배포 환경에서 NICE AutoML 모델을 적용한다.

- `predict_nice_auto_ml`과 유사하나 단일 best_model만 사용
- 결과를 `apply/{output_id}_apply.parquet`에 저장
- 입력 데이터에 필요한 피처가 없으면 `ValueError` 발생
