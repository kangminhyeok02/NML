# executors/ml — ML Executor 가이드

## 개요

`executors/ml` 폴더는 N-Strategy ML 플랫폼의 **핵심 실행 레이어**다.  
API 서버에서 받은 분석/모델링/예측 요청을 실제 작업으로 변환하여 수행하는 **실행 단위(executor)** 모음이다.

모든 executor는 `BaseExecutor`를 상속하며 동일한 인터페이스로 실행된다.

---

## 아키텍처

```
API Server  ──►  ProcessExecutor (오케스트레이터)
                    │
                    ├──► MartExecutor          (데이터 마트 생성)
                    ├──► DataAnalysisExecutor  (EDA)
                    ├──► PythonModelExecutor   (Python 모델 학습)
                    ├──► H2OModelExecutor      (H2O 모델 학습)
                    ├──► RModelExecutor        (R 모델 학습)
                    ├──► AutoMLExecutor        (자동 모델 탐색)
                    ├──► ScorecardExecutor     (스코어카드 생성)
                    ├──► PredictExecutor       (예측 실행)
                    ├──► PretrainedExecutor    (사전학습 모델 추론)
                    ├──► RuleSearchExecutor    (규칙 탐색)
                    ├──► StrategyExecutor      (업무 전략 적용)
                    ├──► RLExecutor            (강화학습 정책)
                    ├──► ReportExecutor        (리포트 생성)
                    └──► ExportExecutor        (결과 내보내기)
```

### 데이터 흐름

```
[DB Server]    ──► MartExecutor ──► [File Server: .parquet]
                                         │
                                 [Model Executor] ──► [File Server: .pkl / MOJO / RDS]
                                         │
                                 PredictExecutor  ──► [File Server: predict result]
                                         │
                                 StrategyExecutor ──► [File Server: strategy result]
                                         │
                                 ReportExecutor   ──► [File Server: .xlsx / .json]
                                         │
                                 ExportExecutor   ──► [CSV / DB table / API]
```

---

## 공통 인터페이스

### BaseExecutor

모든 executor의 기반 클래스. `executors/ml/base_executor.py`에 정의.

```python
executor = SomExecutor(
    config = { ... },          # 실행 파라미터
    db_session = session,      # DB 연결 (선택)
    file_root_dir = "/data",   # 파일 서버 루트 (기본: FILE_ROOT_DIR 환경변수)
)
result = executor.run()        # 실행 (공통 래퍼)
```

`run()`은 내부적으로 `execute()`를 호출하고 다음을 자동 처리한다:
- 시작/종료 시각 기록
- 잡 상태 파일 업데이트 (`/data/jobs/{job_id}.json`)
- 예외 처리 및 에러 반환

### 반환 형식

```python
{
    "status":      "COMPLETED" | "FAILED",
    "job_id":      "job_001",
    "result":      { ... },    # executor별 결과
    "message":     "완료 요약",
    "elapsed_sec": 12.4,
}
```

### 잡 상태 폴링

비동기 실행 시 클라이언트는 아래 파일로 진행 상황을 확인할 수 있다:

```
/data/jobs/{job_id}.json
{
  "job_id": "job_001",
  "status": "RUNNING",
  "progress": 45.0,
  "updated_at": "2024-01-15T10:30:22"
}
```

---

## 파일별 상세 설명

### base_executor.py — 공통 기반 클래스

| 항목 | 내용 |
|------|------|
| **역할** | 모든 executor의 추상 기반 클래스 |
| **핵심 메서드** | `run()` (공통 래퍼), `execute()` (추상, 서브클래스 구현) |
| **제공 기능** | 잡 상태 관리, 데이터 로드/저장, 로깅, 예외 처리 |
| **상속 방법** | `class MyExecutor(BaseExecutor)` + `execute()` 구현 |

---

### mart_executor.py — 데이터 마트 생성

| 항목 | 내용 |
|------|------|
| **역할** | 원천 DB/파일 → 모델 입력용 마트 데이터셋 생성 |
| **입력** | SQL 쿼리 또는 원천 파일 |
| **출력** | `.parquet` 마트 파일 + 메타 JSON |
| **주요 기능** | 결측 처리, 이상값 클리핑, 파생 변수 생성, train/valid/test 분리 |
| **호출 시점** | 모델 학습/분석 전 가장 먼저 실행 |

**config 예시:**
```python
{
    "source_path":  "raw/customer_data.parquet",
    "target_id":    "mart_v1",
    "target_path":  "mart/mart_v1_{split}.parquet",
    "target_col":   "default_yn",
    "split":        {"train": 0.7, "valid": 0.15, "test": 0.15},
    "feature_rules": [
        {"name": "debt_ratio", "expr": "total_debt / (income + 1)"}
    ]
}
```

---

### data_analysis_executor.py — 데이터 탐색 분석 (EDA)

| 항목 | 내용 |
|------|------|
| **역할** | 모델 학습 전 데이터 품질 진단 및 변수 특성 파악 |
| **입력** | 분석 대상 `.parquet` 파일 |
| **출력** | EDA 결과 JSON |
| **분석 항목** | 기초 통계, 결측률, 이상값(IQR), 분포(skewness/kurtosis), 상관계수, 타깃 대비 KS 통계량 |
| **호출 시점** | MartExecutor 이후, 모델 학습 전 |

**주요 출력 항목:**
- `high_missing_cols`: 결측률 30% 초과 컬럼 목록
- `high_corr_pairs`: 상관계수 0.9 초과 변수 쌍
- `target_analysis`: 타깃별 KS 분리도 (내림차순 정렬)

---

### python_model_executor.py — Python ML 모델 학습

| 항목 | 내용 |
|------|------|
| **역할** | scikit-learn / XGBoost / LightGBM / CatBoost 계열 모델 학습 |
| **입력** | 학습 `.parquet` 데이터 |
| **출력** | `.pkl` 모델 파일 + 메타 JSON |
| **지원 모델** | logistic_regression, random_forest, xgboost, lightgbm, catboost, gradient_boosting, decision_tree |
| **평가 지표** | 분류: AUC, Accuracy, F1, Precision, Recall / 회귀: RMSE, R² |

**config 예시:**
```python
{
    "model_type":   "lightgbm",
    "train_path":   "mart/mart_v1_train.parquet",
    "valid_path":   "mart/mart_v1_valid.parquet",
    "target_col":   "default_yn",
    "model_id":     "lgbm_v1",
    "model_params": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 64}
}
```

---

### h2o_model_executor.py — H2O 모델 학습

| 항목 | 내용 |
|------|------|
| **역할** | H2O 서버와 연동하여 GBM/DRF/XGBoost/GLM/DeepLearning 학습 |
| **입력** | 학습 `.parquet` 데이터 |
| **출력** | H2O MOJO 파일 + 메타 JSON |
| **지원 알고리즘** | gbm, drf, xgboost, glm, deeplearning, automl |
| **평가 지표** | AUC, LogLoss, KS |
| **특이사항** | H2O 서버 실행 중이어야 함 (기본: localhost:54321) |

---

### python_h2o_model_executor.py — Python + H2O 통합

| 항목 | 내용 |
|------|------|
| **역할** | Python 전/후처리 + H2O MOJO 추론을 하나의 파이프라인으로 통합 |
| **입력** | 입력 `.parquet` + H2O MOJO 모델 |
| **출력** | 점수 포함 결과 `.parquet` |
| **h2o_model_executor와 차이** | 학습(H2O) + 추론 전처리/후처리(Python)를 결합한 운영 배포용 executor |

**preprocess_steps 예시:**
```python
"preprocess_steps": [
    {"type": "fillna",  "columns": ["income"], "value": 0},
    {"type": "log1p",   "columns": ["total_debt"]},
    {"type": "eval",    "name": "debt_ratio", "expr": "total_debt / (income + 1)"}
]
```

---

### r_model_executor.py — R 모델 학습/예측

| 항목 | 내용 |
|------|------|
| **역할** | R 스크립트를 subprocess로 호출하여 학습/예측 수행 |
| **입력** | 학습/예측 `.parquet` 데이터 |
| **출력** | RDS 모델 파일 + 예측 결과 `.parquet` |
| **데이터 교환 방식** | Python → CSV 임시파일 → R 스크립트 → 결과 CSV → Python |
| **사용 사례** | glm, caret, survival, creditR 등 R 전용 통계모형 |

**R 스크립트 인터페이스:**
```r
# R 스크립트는 --args_file 인자로 JSON 파일 경로를 받음
args_file <- commandArgs(trailingOnly=TRUE)[2]
args <- jsonlite::fromJSON(args_file)

# args$train_path, args$target_col, args$model_dir, args$meta_path 사용
```

---

### automl_executor.py — 자동 모델 탐색

| 항목 | 내용 |
|------|------|
| **역할** | 다중 AutoML 프레임워크를 통합 지원하는 자동 모델 탐색 executor |
| **지원 프레임워크** | h2o_automl, autosklearn, tpot, optuna, pycaret |
| **입력** | 학습 `.parquet` 데이터 |
| **출력** | 최적 모델 `.pkl` + 리더보드 + 메타 JSON |
| **optuna 상세** | LightGBM 하이퍼파라미터 베이지안 최적화 (기본 50 trials) |

**framework 선택 가이드:**
- `optuna`: 가장 범용적, 설치 의존성 낮음 (권장)
- `h2o_automl`: H2O 서버 필요, 앙상블 성능 우수
- `autosklearn`: Linux 전용, sklearn 기반 강력한 앙상블
- `tpot`: 파이프라인 구조 탐색, 시간 소요 많음
- `pycaret`: 빠른 비교 실험, 코드 간결

---

### scorecard_executor.py — 스코어카드 모델 생성

| 항목 | 내용 |
|------|------|
| **역할** | 금융/신용평가용 WOE 기반 스코어카드 모델 생성 |
| **입력** | 학습 `.parquet` 데이터 |
| **출력** | 스코어카드 JSON (binning/WOE/IV 테이블 + 점수표) + 학습 점수 |
| **처리 단계** | binning → WOE/IV 계산 → IV 필터 → 로지스틱 회귀 → PDO 점수 변환 |
| **성능 지표** | KS, AUC, Gini |
| **도메인** | 신용평가, 부도 예측, 한도 심사 |

**스코어카드 출력 예시:**
```json
{"variable": "debt_ratio", "bin": "(0.5, 1.0]", "WOE": -0.42, "points": -15}
```

---

### predict_executor.py — 예측 실행

| 항목 | 내용 |
|------|------|
| **역할** | 저장된 모델을 로드하여 신규 데이터 예측 수행 |
| **지원 모델** | Python(.pkl), H2O(MOJO), R(subprocess) |
| **입력** | 예측 대상 `.parquet` + 모델 메타 JSON |
| **출력** | 점수/확률/등급 포함 결과 `.parquet` |
| **호출 시점** | 운영 배치 또는 실시간 API 요청 시 가장 빈번하게 실행 |

**등급 매핑 예시:**
```python
"grade_mapping": {
    "A": [800, 1000],
    "B": [650, 800],
    "C": [500, 650],
    "D": [0, 500]
}
```

---

### pretrained_executor.py — 사전 학습 모델 추론

| 항목 | 내용 |
|------|------|
| **역할** | 재학습 없이 기존 완성된 모델로 inference만 수행 |
| **지원 형식** | pickle, ONNX, H2O MOJO, HuggingFace |
| **predict_executor와 차이** | predict는 일반 운영 예측, pretrained는 외부 모델/임베딩 추출 등 특수 활용 |
| **사용 사례** | NLP 임베딩 추출, 챔피언 모델 재배포, 외부 기관 제공 모델 |

---

### rulesearch_executor.py — 규칙 탐색

| 항목 | 내용 |
|------|------|
| **역할** | 설명 가능한 if-then 규칙 후보 발굴 |
| **탐색 방법** | decision_tree (트리 경로 추출), association (FP-Growth), woe_rule (WOE 구간) |
| **출력** | 규칙 목록 JSON (조건, 지지도, 신뢰도, bad_rate, lift) |
| **사용 사례** | 정책 룰 설계, 설명 가능한 분류 기준, 고위험 구간 식별 |

**출력 규칙 예시:**
```json
{
  "condition_str": "debt_ratio > 0.85 AND overdue_cnt > 2",
  "bad_rate": 0.72,
  "support": 0.03,
  "lift": 3.6
}
```

---

### stg_executor.py — 업무 전략 적용 (Strategy)

| 항목 | 내용 |
|------|------|
| **역할** | 모델 점수를 실제 업무 의사결정(승인/거절/등급/한도)으로 변환 |
| **전략 유형** | grade (등급화), threshold (임계값), tiered (다단계), matrix (2차원) |
| **추가 기능** | 오버라이드 룰 적용 (사기이력, 고DSR 등 정책 강제 반영) |
| **출력** | decision/grade/override 포함 결과 `.parquet` + 요약 JSON |
| **호출 시점** | PredictExecutor 이후 마지막 업무 처리 단계 |

**tiered_rules 예시:**
```python
[
    {"score_min": 800, "grade": "A", "limit_pct": 1.0, "rate": 3.5},
    {"score_min": 600, "grade": "B", "limit_pct": 0.7, "rate": 5.0},
    {"score_min": 0,   "grade": "C", "reject": True}
]
```

---

### rl_executor.py — 강화학습 정책

| 항목 | 내용 |
|------|------|
| **역할** | 강화학습 기반 동적 의사결정 정책 학습 및 배포 |
| **지원 알고리즘** | q_learning, dqn (stable-baselines3), ppo (stable-baselines3), contextual_bandit (LinUCB) |
| **모드** | train (학습), evaluate (오프라인 평가), deploy (단건 추론) |
| **사용 사례** | 한도/금리 동적 최적화, 컬렉션 전략 시퀀스, 추천 채널 최적화 |

---

### report_executor.py — 리포트 생성

| 항목 | 내용 |
|------|------|
| **역할** | 분석/모델 결과를 정형화된 리포트로 생성 |
| **리포트 유형** | model_performance, eda_report, scorecard_report, prediction_report, combined |
| **출력 포맷** | JSON, Excel (.xlsx) |
| **주요 내용** | 성능 지표, 디사일 테이블, 변수 요약, WOE 테이블 |

---

### export_executor.py — 결과 내보내기

| 항목 | 내용 |
|------|------|
| **역할** | 분석/예측 결과를 외부 시스템으로 내보내는 최종 단계 |
| **export 대상** | 파일 (CSV/Excel/JSON/Parquet), DB 테이블 (append/replace/upsert), REST API |
| **호출 시점** | 파이프라인의 마지막 단계 |

---

### process_executor.py — 파이프라인 오케스트레이터

| 항목 | 내용 |
|------|------|
| **역할** | 여러 executor를 순차적으로 실행하는 파이프라인 제어기 |
| **기능** | 단계 간 결과 전달, 실패 정책(stop/skip/continue), 진행 상태 기록 |
| **등록 executor** | EXECUTOR_REGISTRY에 모든 executor가 key-value로 등록됨 |

**파이프라인 config 예시:**
```python
{
    "pipeline_name": "credit_scoring_v1",
    "pipeline": [
        {
            "name":     "make_mart",
            "executor": "mart",
            "config":   {"source_path": "raw/data.parquet", "target_id": "mart_v1"}
        },
        {
            "name":      "train_model",
            "executor":  "python_model",
            "input_from": "make_mart",
            "config":    {"model_type": "lightgbm", "target_col": "default_yn", "model_id": "lgbm_v1"}
        },
        {
            "name":     "predict",
            "executor": "predict",
            "config":   {"model_id": "lgbm_v1", "input_path": "mart/test.parquet", "output_id": "pred_v1"}
        },
        {
            "name":     "apply_strategy",
            "executor": "stg",
            "config":   {"strategy_type": "grade", "score_col": "score", "output_id": "stg_v1"}
        },
        {
            "name":     "export_result",
            "executor": "export",
            "config":   {"export_type": "file", "file_format": "csv", "output_id": "final_v1"}
        }
    ]
}
```

---

## 파일 저장 구조

```
{FILE_ROOT_DIR}/
├── jobs/
│   └── {job_id}.json          ← 잡 상태 파일
├── mart/
│   └── {target_id}.parquet    ← 마트 데이터
├── models/
│   ├── {model_id}.pkl         ← Python 모델
│   ├── {model_id}/            ← H2O MOJO / R RDS
│   └── {model_id}_meta.json   ← 모델 메타정보
├── predict/
│   └── {output_id}_result.parquet
├── analysis/
│   └── {output_id}_eda.json
├── strategy/
│   └── {output_id}_result.parquet
├── reports/
│   ├── {output_id}.json
│   └── {output_id}.xlsx
└── exports/
    └── {output_id}.csv
```

---

## 새 Executor 추가 방법

1. `executors/ml/my_executor.py` 생성
2. `BaseExecutor` 상속 + `execute()` 구현
3. `process_executor.py`의 `EXECUTOR_REGISTRY`에 등록

```python
# my_executor.py
from executors.ml.base_executor import BaseExecutor, ExecutorStatus

class MyExecutor(BaseExecutor):
    def execute(self) -> dict:
        # 작업 수행
        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  {},
            "message": "완료",
        }

# process_executor.py EXECUTOR_REGISTRY에 추가
"my_executor": ("executors.ml.my_executor", "MyExecutor"),
```

---

## 의존성 요약

| 라이브러리 | 용도 | 필수 여부 |
|-----------|------|----------|
| pandas, numpy | 데이터 처리 | 필수 |
| scikit-learn | Python 모델 | 필수 |
| lightgbm | LightGBM | 권장 |
| xgboost | XGBoost | 선택 |
| catboost | CatBoost | 선택 |
| h2o | H2O 연동 | H2O 사용 시 |
| optuna | 하이퍼파라미터 최적화 | AutoML 사용 시 |
| stable-baselines3 | DQN/PPO | RL 사용 시 |
| scipy | 통계 분석 | EDA 사용 시 |
| mlxtend | 연관규칙 탐색 | RuleSearch 사용 시 |
| openpyxl | Excel 출력 | Report/Export 사용 시 |
| onnxruntime | ONNX 추론 | Pretrained ONNX 사용 시 |
| transformers | HuggingFace | NLP 모델 사용 시 |
