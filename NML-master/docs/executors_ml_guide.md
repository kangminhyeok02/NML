# executors/ml 패키지 상세 가이드

> **위치**: `executors/ml/`  
> **목적**: REST API 요청을 받아 ML 작업(마트 생성 → 학습 → 예측 → 전략 적용 → 리포트 → 내보내기)을 수행하는 executor 모음

---

## 목차

1. [패키지 구조 한눈에 보기](#1-패키지-구조-한눈에-보기)
2. [상속 관계 및 실행 흐름](#2-상속-관계-및-실행-흐름)
3. [파일별 상세 설명](#3-파일별-상세-설명)
   - [__init__.py](#__initpy)
   - [base_executor.py](#base_executorpy)
   - [process_executor.py](#process_executorpy)
   - [mart_executor.py](#mart_executorpy)
   - [data_analysis_executor.py](#data_analysis_executorpy)
   - [python_model_executor.py](#python_model_executorpy)
   - [h2o_model_executor.py](#h2o_model_executorpy)
   - [python_h2o_model_executor.py](#python_h2o_model_executorpy)
   - [r_model_executor.py](#r_model_executorpy)
   - [automl_executor.py](#automl_executorpy)
   - [scorecard_executor.py](#scorecard_executorpy)
   - [predict_executor.py](#predict_executorpy)
   - [pretrained_executor.py](#pretrained_executorpy)
   - [stg_executor.py](#stg_executorpy)
   - [rulesearch_executor.py](#rulesearch_executorpy)
   - [rl_executor.py](#rl_executorpy)
   - [report_executor.py](#report_executorpy)
   - [export_executor.py](#export_executorpy)
4. [전형적인 파이프라인 예시](#4-전형적인-파이프라인-예시)
5. [config 공통 키 레퍼런스](#5-config-공통-키-레퍼런스)

---

## 1. 패키지 구조 한눈에 보기

```
executors/ml/
├── __init__.py                  # 패키지 진입점 (BaseExecutor 재노출)
├── base_executor.py             # 공통 추상 기반 클래스
├── process_executor.py          # 파이프라인 오케스트레이터
│
├── [데이터 준비]
│   ├── mart_executor.py         # 모델링용 데이터 마트 생성
│   └── data_analysis_executor.py# 탐색적 데이터 분석 (EDA)
│
├── [모델 학습]
│   ├── python_model_executor.py # sklearn / XGBoost / LightGBM / CatBoost
│   ├── h2o_model_executor.py    # H2O 프레임워크 학습
│   ├── python_h2o_model_executor.py # Python 전후처리 + H2O 추론 통합
│   ├── r_model_executor.py      # R 스크립트 기반 모델
│   ├── automl_executor.py       # 자동 하이퍼파라미터 탐색
│   └── scorecard_executor.py    # WOE/IV 기반 신용평가 스코어카드
│
├── [예측 / 추론]
│   ├── predict_executor.py      # 저장된 모델 로드 후 예측
│   └── pretrained_executor.py   # 사전학습 모델 추론 (ONNX / HuggingFace 등)
│
├── [의사결정 / 분석]
│   ├── stg_executor.py          # 예측 점수 → 업무 전략 변환
│   ├── rulesearch_executor.py   # if-then 규칙 탐색
│   └── rl_executor.py           # 강화학습 정책 학습
│
└── [출력]
    ├── report_executor.py       # 분석 리포트 생성
    └── export_executor.py       # 결과 파일/DB/API 내보내기
```

---

## 2. 상속 관계 및 실행 흐름

```
BaseExecutor (ABC)
│
├── run()          ← 외부에서 호출하는 진입점
│   ├── started_at 기록
│   ├── execute()  ← 각 서브클래스가 구현
│   └── finished_at / 상태 기록
│
└── 공통 헬퍼
    ├── _load_dataframe(path)   parquet / csv 로드
    ├── _save_dataframe(df, path)
    ├── _save_json(data, path)
    └── _update_job_status(status, progress, message)
```

**모든 executor는 반드시 `execute() → dict` 를 구현해야 한다.**  
반환 dict의 필수 키: `status`, `result`, `message`

---

## 3. 파일별 상세 설명

---

### `__init__.py`

패키지 진입점. `BaseExecutor`, `ExecutorException`, `ExecutorStatus`를 패키지 레벨로 재노출한다.

```python
from executors.ml import BaseExecutor, ExecutorException, ExecutorStatus
```

---

### `base_executor.py`

**역할**: 모든 executor의 공통 기반 클래스. 직접 인스턴스화하지 않는다.

#### 주요 클래스

| 이름 | 종류 | 설명 |
|------|------|------|
| `ExecutorStatus` | 상수 클래스 | `PENDING` / `RUNNING` / `COMPLETED` / `FAILED` |
| `ExecutorException` | Exception | executor 내부 에러 (예측 가능한 실패) |
| `BaseExecutor` | ABC | 모든 executor의 부모 |

#### `BaseExecutor.__init__` 파라미터

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `config` | dict | 실행 파라미터. 필수 키: `job_id`, `service_id`, `project_id` |
| `db_session` | optional | SQLAlchemy 세션 |
| `file_root_dir` | str | 파일 루트 경로. 없으면 환경변수 `FILE_ROOT_DIR` 사용 (기본: `/data`) |

#### 공통 헬퍼 메서드

| 메서드 | 반환 | 설명 |
|--------|------|------|
| `run()` | dict | 실행 래퍼. 직접 호출하는 진입점 |
| `_load_dataframe(path)` | DataFrame | `.parquet` / `.csv` / `.txt` 로드 |
| `_save_dataframe(df, path)` | str | parquet으로 저장, 절대경로 반환 |
| `_save_json(data, path)` | str | JSON 저장, 절대경로 반환 |
| `_update_job_status(status, progress, message, result)` | None | `jobs/{job_id}.json` 갱신 |

#### 잡 상태 파일 형식

```json
{
  "job_id": "job_001",
  "status": "RUNNING",
  "updated_at": "2026-04-09T10:00:00",
  "progress": 50.0,
  "message": "학습 중..."
}
```

---

### `process_executor.py`

**역할**: 여러 executor를 순서대로 실행하는 **파이프라인 오케스트레이터**.  
`EXECUTOR_REGISTRY`에 등록된 executor 유형을 이름으로 동적 로드(`importlib`)하여 실행한다.

#### EXECUTOR_REGISTRY (등록된 executor 유형)

| 키 | executor 클래스 |
|----|-----------------|
| `mart` | MartExecutor |
| `data_analysis` | DataAnalysisExecutor |
| `python_model` | PythonModelExecutor |
| `h2o_model` | H2OModelExecutor |
| `r_model` | RModelExecutor |
| `automl` | AutoMLExecutor |
| `scorecard` | ScorecardExecutor |
| `predict` | PredictExecutor |
| `pretrained` | PretrainedExecutor |
| `report` | ReportExecutor |
| `export` | ExportExecutor |
| `rulesearch` | RuleSearchExecutor |
| `stg` | StrategyExecutor |
| `rl` | RLExecutor |

#### config 구조

```python
config = {
    "stop_on_first_failure": True,   # 첫 실패 시 중단 여부
    "pipeline": [
        {
            "name": "make_mart",        # 단계 이름 (고유)
            "executor": "mart",         # EXECUTOR_REGISTRY 키
            "config": { ... },          # 해당 executor의 config
            "on_error": "stop",         # "stop" | "skip" | "continue"
            "input_from": "prev_step",  # 이전 단계 결과를 config에 병합 (선택)
        },
        ...
    ]
}
```

#### 실행 흐름

```
pipeline[0] → pipeline[1] → ... → pipeline[n]
      ↓               ↓
  context 공유  (이전 결과를 다음 단계 config에 주입 가능)
```

---

### `mart_executor.py`

**역할**: 원천 DB 또는 파일에서 데이터를 읽어 **모델링용 데이터 마트**를 생성한다.

#### 처리 단계

1. 원천 데이터 로드 (SQL 쿼리 또는 parquet/csv 파일)
2. 타입 변환 / 결측 처리 / 이상값 클리핑
3. 파생 변수 생성 (`feature_rules` 기반)
4. 학습/검증/테스트 분리 (선택)
5. parquet으로 저장 및 메타 정보 기록

#### config 필수 키

| 키 | 설명 |
|----|------|
| `source_query` | 원천 SQL (`db_session` 필요) — `source_path`와 택1 |
| `source_path` | 원천 파일 경로 |
| `target_id` | 마트 식별자 |
| `target_path` | 저장 경로 (예: `"mart/{target_id}.parquet"`) |

#### config 선택 키

| 키 | 기본값 | 설명 |
|----|--------|------|
| `feature_rules` | `[]` | 파생 변수 생성 규칙 목록 |
| `split` | None | `{"train": 0.7, "valid": 0.15, "test": 0.15}` |
| `target_col` | None | 타깃 컬럼 (분할 시 층화 추출에 사용) |

---

### `data_analysis_executor.py`

**역할**: 모델 학습 전 **탐색적 데이터 분석(EDA)**을 수행하고 결과를 JSON으로 저장한다.

#### 분석 항목

| 항목 | 메서드 | 설명 |
|------|--------|------|
| 기초 통계 | `_basic_stats` | mean, std, min, max, percentile(1/5/25/50/75/95/99) |
| 결측치 | `_missing_summary` | 건수, 비율, 임계값 초과 여부 |
| 이상값 | `_outlier_summary` | IQR 방식 (1.5×IQR 기준) |
| 분포 요약 | `_distribution_summary` | skewness, kurtosis |
| 카테고리 빈도 | `_category_freq` | 상위 10개 값 비율 |
| 상관관계 | `_correlation_matrix` | 상관계수 행렬 + 고상관 변수 쌍 목록 |
| 타깃 분리도 | `_target_analysis` | KS statistic (target_col 지정 시) |

#### 모듈 수준 함수 (module-level)

| 함수 | 설명 |
|------|------|
| `variable_analysis(...)` | 변수별 통계를 ProcessPoolExecutor로 병렬 산출 |
| `data_sampling(...)` | random / stratified 샘플링 |
| `data_union(...)` | 여러 parquet 수직 결합 + key_col 정렬 |
| `variable_profiling_from_node(...)` | 프로파일링 결과 JSON 저장 |
| `data_snapshot(...)` | head/sample 방식으로 데이터 스냅샷 저장 |
| `check_output_param_prob_in_correct_range(...)` | 확률값 [0,1] 범위 검증 |
| `get_distinct_values(...)` | 특정 컬럼의 고유값 목록 반환 |

#### config 필수 키

| 키 | 설명 |
|----|------|
| `source_path` | 분석 대상 데이터 경로 |
| `output_id` | 결과 식별자 → `analysis/{output_id}_eda.json` 저장 |

---

### `python_model_executor.py`

**역할**: **scikit-learn / XGBoost / LightGBM / CatBoost** 계열 모델을 학습하고 평가/저장한다.

#### 지원 model_type

| model_type | 라이브러리 | 과제 |
|------------|-----------|------|
| `logistic_regression` | sklearn | 분류 |
| `random_forest` | sklearn | 분류 |
| `gradient_boosting` | sklearn | 분류 |
| `decision_tree` | sklearn | 분류 |
| `xgboost` | xgboost | 분류 |
| `lightgbm` | lightgbm | 분류 |
| `catboost` | catboost | 분류 |
| `linear_regression` | sklearn | 회귀 |

#### 출력

- `models/{model_id}.pkl` — pickle 모델 파일
- `models/{model_id}_meta.json` — 피처 목록, 성능 지표, 모델 유형 등

#### config 필수 키

| 키 | 설명 |
|----|------|
| `train_path` | 학습 데이터 경로 |
| `target_col` | 타깃 컬럼명 |
| `model_id` | 모델 저장 식별자 |
| `model_type` | 위 표의 유형 중 하나 |

---

### `h2o_model_executor.py`

**역할**: **H2O 서버**와 연동하여 GBM / DRF / XGBoost / DeepLearning / GLM 등 H2O 알고리즘으로 모델을 학습한다.

#### 지원 알고리즘

| algorithm | H2O 클래스 |
|-----------|-----------|
| `gbm` | H2OGradientBoostingEstimator |
| `drf` | H2ORandomForestEstimator |
| `xgboost` | H2OXGBoostEstimator |
| `glm` | H2OGeneralizedLinearEstimator |
| `deeplearning` | H2ODeepLearningEstimator |
| `automl` | H2OAutoML |

#### 실행 순서

1. H2O 클러스터 초기화 (`h2o.init`)
2. DataFrame → H2OFrame 변환
3. 학습/검증 분리
4. 알고리즘별 모델 학습
5. 성능 평가 (AUC, RMSE 등)
6. MOJO 파일 저장 → `models/{model_id}.mojo`
7. H2O 세션 종료 (선택)

#### config 필수 키

| 키 | 설명 |
|----|------|
| `algorithm` | 위 표의 알고리즘 키 |
| `train_path` | 학습 데이터 경로 |
| `target_col` | 타깃 컬럼명 |
| `model_id` | 모델 저장 식별자 |

---

### `python_h2o_model_executor.py`

**역할**: **Python 전처리/후처리 + H2O MOJO 추론**을 하나의 파이프라인으로 통합한다.  
H2O로 학습했지만 추론 파이프라인을 Python으로 제어해야 할 때 사용한다.

#### 처리 순서

```
입력 데이터
  ↓ Python 전처리 (preprocess_steps)
  ↓ H2O MOJO 추론
  ↓ Python 후처리 (postprocess_steps)
  ↓ 결과 저장
```

#### config 필수 키

| 키 | 설명 |
|----|------|
| `model_id` | H2O 모델 식별자 (MOJO 경로 포함된 메타) |
| `input_path` | 입력 데이터 경로 |
| `output_id` | 결과 저장 식별자 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|----|--------|------|
| `preprocess_steps` | `[]` | Python 전처리 스텝 목록 |
| `postprocess_steps` | `[]` | Python 후처리 스텝 목록 |
| `use_mojo` | `True` | MOJO 사용 여부 |
| `h2o_ip` / `h2o_port` | `localhost:54321` | H2O 서버 주소 |

---

### `r_model_executor.py`

**역할**: **R 스크립트**를 subprocess로 실행하여 학습/예측을 수행한다.  
Python ↔ R 간 데이터 교환은 CSV 또는 RDS 파일을 경유한다.

#### 사용 목적

- 기존 R 자산(통계모형, 스코어카드) 재사용
- R 전용 라이브러리 활용 (`glm`, `caret`, `survival`, `creditR`)
- 레거시 R 코드와의 통합 운영

#### 실행 흐름

```
Python: 데이터 → 임시 CSV 저장
  ↓
Rscript {r_script} --args {csv_path} {meta_path}
  ↓
Python: R 결과(메타 JSON, 예측 CSV) 읽기 → 저장
```

#### config 필수 키

| 키 | 설명 |
|----|------|
| `r_script` | R 스크립트 경로 |
| `mode` | `"train"` \| `"predict"` |
| `model_id` | 모델 저장 식별자 |
| `train_path` | (train 모드) 학습 데이터 경로 |
| `target_col` | (train 모드) 타깃 컬럼명 |
| `input_path` | (predict 모드) 예측 데이터 경로 |

---

### `automl_executor.py`

**역할**: 다양한 **AutoML 프레임워크**를 통합 지원하여 자동으로 최적 모델을 탐색한다.

#### 지원 프레임워크 (AutoMLExecutor)

| framework | 탐색 방식 |
|-----------|-----------|
| `h2o_automl` | H2O 리더보드 기반 |
| `autosklearn` | 앙상블 탐색 (auto-sklearn) |
| `tpot` | 유전 알고리즘 기반 파이프라인 탐색 |
| `optuna` | 베이지안 최적화 + LightGBM |
| `pycaret` | 비교 실험 자동화 |

#### 모듈 수준 함수 (NICE AutoML 통합)

| 함수 | 설명 |
|------|------|
| `fit_nice_auto_ml(...)` | hyperopt(TPE) + LightGBM으로 학습, 앙상블 지원 |
| `predict_nice_auto_ml(...)` | 저장된 앙상블 모델로 배치 예측 |
| `apply_nice_auto_ml(...)` | 배포 환경에서 단일 모델 예측 수행 |
| `get_automl_objective_loss(...)` | KS 기반 목적함수 손실 계산 (과적합 페널티 포함) |
| `generate_ensemble_weights(...)` | 합이 1이 되는 앙상블 가중치 조합 열거 |
| `convert_list_to_hp(...)` | dict 파라미터를 hyperopt 탐색공간으로 변환 |
| `convert_hidden_layer_to_dict(...)` | `"64,32,16"` 문자열 → 레이어 설정 dict 변환 |

**`fit_nice_auto_ml` 내부 흐름:**
```
데이터 로드 → 학습/검증 분리
  ↓
hyperopt + TPE 알고리즘으로 n_trials회 탐색
  (각 trial: LGBMClassifier 학습 → AUC 평가 → 과적합 페널티 손실 계산)
  ↓
최적 파라미터로 LGBMClassifier 재학습
  ↓
상위 N개 모델로 앙상블 구성
  ↓
모델(.pkl) + 메타(.json) 저장
```

#### config 필수 키 (AutoMLExecutor)

| 키 | 설명 |
|----|------|
| `framework` | 위 표의 프레임워크 키 |
| `train_path` | 학습 데이터 경로 |
| `target_col` | 타깃 컬럼명 |
| `model_id` | 결과 저장 식별자 |

---

### `scorecard_executor.py`

**역할**: 금융/리스크 도메인에서 사용하는 **신용평가 스코어카드**를 생성한다.  
WOE 변환 → 로지스틱 회귀 → 점수 스케일링 순으로 처리한다.

#### 처리 단계

```
학습 데이터
  ↓ Binning (구간화, n_bins 기준)
  ↓ WOE(Weight of Evidence) 계산
  ↓ IV(Information Value) 산출 → iv_threshold 미만 변수 제거
  ↓ WOE 변환된 데이터로 LogisticRegression 학습
  ↓ 점수 스케일링 (base_score, PDO 기준)
  ↓ 스코어카드 포인트 테이블 생성
  ↓ 결과 저장 (KS, AUC, Gini 포함)
```

#### 출력물

- `models/{model_id}_scorecard.json` — 변수별 binning/WOE/IV/포인트 테이블
- `models/{model_id}_meta.json` — 성능 지표(KS, AUC, Gini) 및 모델 정보

#### config 필수 키

| 키 | 설명 |
|----|------|
| `train_path` | 학습 데이터 경로 |
| `target_col` | 타깃 컬럼 (1=Bad, 0=Good) |
| `model_id` | 모델 저장 식별자 |
| `feature_cols` | 스코어카드에 사용할 변수 목록 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|----|--------|------|
| `n_bins` | 10 | binning 구간 수 |
| `min_bin_rate` | 0.05 | 최소 bin 비율 |
| `iv_threshold` | 0.02 | IV 필터 기준 |
| `base_score` | 600 | 기준 점수 |
| `pdo` | 20 | PDO (Odds 2배당 점수 감소 폭) |

---

### `predict_executor.py`

**역할**: 저장된 모델을 로드하여 신규 데이터에 **예측(점수/확률/등급)**을 수행한다.  
운영 환경에서 가장 빈번하게 호출되는 executor.

#### 지원 모델 형식

| model_type | 파일 형식 | 비고 |
|------------|-----------|------|
| `python` | `.pkl` (pickle) | sklearn / XGBoost / LightGBM 등 |
| `h2o` | `.mojo` | H2O MOJO |
| `r` | `.rds` | R 모델 |

#### 실행 순서

1. 모델 메타 JSON 로드 (`models/{model_id}_meta.json`)
2. 모델 파일 로드
3. 예측 대상 데이터 로드
4. 피처 정렬 및 누락 컬럼 검증
5. 예측 수행 → 점수 / 확률 / 클래스
6. 등급 부여 (`grade_mapping` 설정 시)
7. 결과 parquet 저장

#### config 필수 키

| 키 | 설명 |
|----|------|
| `model_id` | 사용할 모델 식별자 |
| `input_path` | 예측 대상 데이터 경로 |
| `output_id` | 결과 저장 식별자 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|----|--------|------|
| `model_type` | `"python"` | `"python"` \| `"h2o"` \| `"r"` |
| `score_col` | `"score"` | 예측 점수 컬럼명 |
| `threshold` | `0.5` | 이진 분류 임계값 |
| `grade_mapping` | None | `{"A": [800,1000], "B": [600,800], ...}` |

---

### `pretrained_executor.py`

**역할**: 재학습 없이 기존 **사전 학습 모델**을 로드하여 추론(임베딩 추출 / 점수 산출)을 수행한다.

#### 지원 모델 형식

| model_format | 설명 |
|--------------|------|
| `pickle` | scikit-learn / XGBoost / LightGBM (.pkl) |
| `onnx` | ONNX Runtime 기반 추론 (.onnx) |
| `h2o` | H2O MOJO |
| `hugging` | HuggingFace Transformers |

#### 사용 사례

- 내부 리스크 팀이 학습/검증한 모델을 운영 배포
- 외부 공개 모델(HuggingFace, ONNX) 활용
- Transfer Learning의 feature extractor 단계
- A/B 테스트를 위한 챔피언/챌린저 모델 동시 배포

#### config 필수 키

| 키 | 설명 |
|----|------|
| `model_id` | 모델 식별자 (메타 JSON 기준) |
| `input_path` | 입력 데이터 경로 |
| `output_id` | 결과 저장 식별자 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|----|--------|------|
| `model_format` | 메타에서 자동 감지 | `"pickle"` \| `"onnx"` \| `"h2o"` \| `"hugging"` |
| `output_mode` | `"score"` | `"score"` \| `"embedding"` \| `"both"` |
| `batch_size` | 10000 | 배치 추론 크기 |

---

### `stg_executor.py`

**역할**: 모델이 산출한 점수/확률값을 실제 **업무 의사결정(전략)**으로 변환한다.  
금융/리스크 도메인에서 승인/거절/한도/등급 결정에 사용한다.

#### 지원 전략 유형

| strategy_type | 설명 |
|---------------|------|
| `grade` | 점수 → A/B/C/D/E 등급 매핑 |
| `threshold` | 임계값 기반 이진 승인/거절 |
| `tiered` | 다단계 정책 (등급별 한도/조건) |
| `matrix` | 2차원 매트릭스 (score × 기존등급) |

#### 오버라이드 룰

전략 적용 후 `override_rules` 목록의 조건을 순서대로 평가하여 강제 적용한다.  
예: "연체 90일 이상이면 등급 무관하게 거절"

#### config 필수 키

| 키 | 설명 |
|----|------|
| `input_path` | 예측 점수가 포함된 데이터 경로 |
| `score_col` | 점수 컬럼명 |
| `strategy_type` | 위 표의 전략 유형 |
| `output_id` | 결과 저장 식별자 |

---

### `rulesearch_executor.py`

**역할**: 데이터에서 **설명 가능한 if-then 규칙**을 자동으로 탐색한다.  
신용/리스크 정책 룰 설계나 해석 가능한 분류 기준을 만들 때 활용한다.

#### 탐색 방식

| method | 알고리즘 | 출력 |
|--------|---------|------|
| `decision_tree` | 의사결정트리 경로 추출 | 조건 + 지지도 + bad_rate |
| `association` | Apriori / FP-Growth | 지지도 + 신뢰도 + lift |
| `woe_rule` | WOE 기반 구간 규칙 | IV + bad_rate |

#### config 필수 키

| 키 | 설명 |
|----|------|
| `source_path` | 입력 데이터 경로 |
| `target_col` | 타깃 컬럼 (1=Bad/Event) |
| `output_id` | 결과 저장 식별자 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|----|--------|------|
| `method` | `"decision_tree"` | 탐색 방식 |
| `max_depth` | 4 | 트리 최대 깊이 |
| `min_support` | 0.05 | 연관규칙 최소 지지도 |
| `min_confidence` | 0.3 | 연관규칙 최소 신뢰도 |
| `top_n` | 50 | 상위 N개 규칙 반환 |

---

### `rl_executor.py`

**역할**: **강화학습(RL)** 기반 의사결정 정책을 학습하고 배포한다.  
금융/리스크에서 한도/금리 최적화, 컬렉션 전략 학습, 상품 추천에 활용한다.

#### 지원 알고리즘

| algorithm | 설명 | 사용 조건 |
|-----------|------|----------|
| `q_learning` | 테이블 기반 Q-Learning | 소규모 이산 상태공간 |
| `dqn` | Deep Q-Network (stable-baselines3) | 연속/고차원 상태공간 |
| `ppo` | Proximal Policy Optimization (stable-baselines3) | 연속 행동공간 |
| `contextual_bandit` | 컨텍스추얼 밴딧 | 단순 단기 의사결정 |

#### 실행 모드

| mode | 설명 |
|------|------|
| `train` | 환경 설정 기반 정책 학습 |
| `evaluate` | 저장된 정책을 데이터로 평가 |
| `deploy` | 배포 환경에서 정책 적용 (실시간 추론) |

#### config 필수 키

| 키 | 설명 |
|----|------|
| `algorithm` | 위 표의 알고리즘 |
| `mode` | `"train"` \| `"evaluate"` \| `"deploy"` |
| `model_id` | 모델 저장 식별자 |

---

### `report_executor.py`

**역할**: 분석/모델링 결과를 **정형화된 리포트**로 생성한다.  
출력 형식: JSON (구조화 데이터), Excel (.xlsx)

#### 지원 리포트 유형

| report_type | 설명 |
|-------------|------|
| `model_performance` | 모델 성능 요약 (AUC, KS, 혼동행렬, ROC 데이터) |
| `eda_report` | 데이터 분석 요약 |
| `scorecard_report` | 스코어카드 변수 요약표 및 성능 |
| `prediction_report` | 예측 결과 분포 요약 (점수 구간별 분포, bad_rate) |
| `combined` | 위 항목을 통합한 종합 리포트 |

#### config 필수 키

| 키 | 설명 |
|----|------|
| `report_type` | 위 표의 리포트 유형 |
| `output_id` | 출력 식별자 |
| `report_name` | 리포트 제목 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|----|--------|------|
| `output_format` | `["json", "excel"]` | 출력 포맷 목록 |
| `model_meta_path` | - | 모델 메타 JSON 경로 |
| `eda_result_path` | - | EDA 결과 JSON 경로 |
| `scorecard_path` | - | 스코어카드 결과 JSON 경로 |
| `prediction_path` | - | 예측 결과 parquet 경로 |

---

### `export_executor.py`

**역할**: 분석/예측 결과를 **파일 / DB / REST API** 세 가지 대상으로 내보낸다.

#### 지원 export_type

| export_type | 설명 |
|-------------|------|
| `file` | CSV / Excel / JSON / Parquet 파일 저장 |
| `db` | DB 테이블에 적재 (append / replace / upsert) |
| `api` | REST endpoint로 POST 전송 (배치) |

#### config 필수 키

| 키 | 설명 |
|----|------|
| `source_path` | 내보낼 원본 데이터 경로 (.parquet) |
| `export_type` | `"file"` \| `"db"` \| `"api"` |
| `output_id` | 출력 식별자 |

#### file 모드 추가 키

| 키 | 설명 |
|----|------|
| `file_format` | `"csv"` \| `"excel"` \| `"json"` \| `"parquet"` |
| `output_path` | 저장 경로 |

#### db 모드 추가 키

| 키 | 설명 |
|----|------|
| `table_name` | 대상 테이블명 |
| `write_mode` | `"append"` \| `"replace"` \| `"upsert"` |
| `key_cols` | upsert 키 컬럼 목록 |

#### api 모드 추가 키

| 키 | 기본값 | 설명 |
|----|--------|------|
| `endpoint` | - | POST 대상 URL |
| `batch_size` | 1000 | 배치 크기 |
| `headers` | `{}` | 요청 헤더 |

---

## 4. 전형적인 파이프라인 예시

### 신용평가 모델 학습 파이프라인

```python
config = {
    "job_id": "job_credit_001",
    "pipeline": [
        {
            "name": "build_mart",
            "executor": "mart",
            "config": {
                "source_query": "SELECT * FROM raw.loan_data WHERE dt = '2026-03-31'",
                "target_id":    "loan_mart_v1",
                "target_path":  "mart/loan_mart_v1.parquet",
                "split": {"train": 0.7, "valid": 0.15, "test": 0.15},
                "target_col": "is_bad",
            }
        },
        {
            "name": "eda",
            "executor": "data_analysis",
            "config": {
                "source_path": "mart/loan_mart_v1_train.parquet",
                "output_id":   "loan_v1",
                "target_col":  "is_bad",
            }
        },
        {
            "name": "train_scorecard",
            "executor": "scorecard",
            "config": {
                "train_path":   "mart/loan_mart_v1_train.parquet",
                "valid_path":   "mart/loan_mart_v1_valid.parquet",
                "target_col":   "is_bad",
                "model_id":     "loan_sc_v1",
                "feature_cols": ["age", "income", "debt_ratio", "delinq_cnt"],
                "base_score":   600,
                "pdo":          20,
            }
        },
        {
            "name": "predict",
            "executor": "predict",
            "config": {
                "model_id":   "loan_sc_v1",
                "model_type": "python",
                "input_path": "mart/loan_mart_v1_test.parquet",
                "output_id":  "loan_pred_v1",
                "score_col":  "score",
            }
        },
        {
            "name": "strategy",
            "executor": "stg",
            "config": {
                "input_path":    "predict/loan_pred_v1_result.parquet",
                "score_col":     "score",
                "strategy_type": "grade",
                "output_id":     "loan_stg_v1",
                "grade_map": {
                    "A": [750, 1000],
                    "B": [600, 750],
                    "C": [450, 600],
                    "D": [300, 450],
                    "E": [0, 300],
                }
            }
        },
        {
            "name": "report",
            "executor": "report",
            "config": {
                "report_type":   "combined",
                "output_id":     "loan_report_v1",
                "report_name":   "신용평가 모델 성능 리포트",
                "output_format": ["json", "excel"],
            }
        },
        {
            "name": "export_to_db",
            "executor": "export",
            "config": {
                "source_path":  "strategy/loan_stg_v1_result.parquet",
                "export_type":  "db",
                "output_id":    "loan_export_v1",
                "table_name":   "result.loan_grade",
                "write_mode":   "upsert",
                "key_cols":     ["customer_id"],
            }
        },
    ]
}

from executors.ml.process_executor import ProcessExecutor
result = ProcessExecutor(config).run()
```

---

## 5. config 공통 키 레퍼런스

모든 executor의 config에 공통으로 적용되는 키 (BaseExecutor에서 처리):

| 키 | 필수 | 설명 |
|----|------|------|
| `job_id` | 권장 | 잡 상태 파일명 기준. 없으면 `"unknown"` |
| `service_id` | 선택 | 서비스 식별자 |
| `project_id` | 선택 | 프로젝트 식별자 |

잡 상태는 `{FILE_ROOT_DIR}/jobs/{job_id}.json`에 자동 기록된다.
