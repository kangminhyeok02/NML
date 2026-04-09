# N-Strategy ML Executor Framework — 프로젝트 개요

> **참고:** 이 저장소는 실제 상용 ML 플랫폼의 **추상화(abstracted) 버전**입니다.
> 핵심 설계 패턴과 실행 흐름은 실제 시스템과 동일하나, 비즈니스 로직·데이터·인프라 세부 사항은 일반화되어 있습니다.

---

## 1. 프로젝트 목적

N-Strategy ML Executor Framework는 **금융/리스크 도메인**에서 ML 파이프라인 전 과정을 실행하기 위한 **실행 레이어(execution layer)**다.

- API 서버로부터 받은 분석·모델링·예측 요청을 실제 작업으로 변환해 수행한다.
- 데이터 마트 생성 → EDA → 모델 학습 → 예측 → 전략 적용 → 리포트 → 내보내기까지 전체 ML 워크플로를 하나의 통일된 인터페이스로 처리한다.
- 각 단계는 독립적인 **Executor** 단위로 캡슐화되어 있어 단독 실행 또는 파이프라인 조합이 모두 가능하다.

---

## 2. 디렉터리 구조

```
NML_clone/
├── executors/
│   └── ml/
│       ├── base_executor.py          # 추상 기반 클래스 (공통 인터페이스)
│       ├── process_executor.py       # 파이프라인 오케스트레이터
│       ├── mart_executor.py          # 데이터 마트 생성
│       ├── data_analysis_executor.py # 탐색적 데이터 분석 (EDA)
│       ├── python_model_executor.py  # Python ML 모델 학습
│       ├── h2o_model_executor.py     # H2O 모델 학습
│       ├── python_h2o_model_executor.py # Python 전처리 + H2O MOJO 통합
│       ├── r_model_executor.py       # R 모델 학습/예측
│       ├── automl_executor.py        # AutoML (자동 모델 탐색)
│       ├── scorecard_executor.py     # WOE 스코어카드 생성
│       ├── predict_executor.py       # 모델 예측 실행
│       ├── pretrained_executor.py    # 사전학습 모델 추론
│       ├── rulesearch_executor.py    # if-then 규칙 탐색
│       ├── stg_executor.py           # 업무 전략 적용
│       ├── rl_executor.py            # 강화학습 정책 학습/배포
│       ├── report_executor.py        # 리포트 생성
│       └── export_executor.py        # 결과 내보내기
└── docs/
    ├── executors_ml_guide.md         # Executor 상세 가이드
    └── project_overview.md           # 이 파일
```

---

## 3. 전체 아키텍처

### 3-1. 레이어 구조

```
┌─────────────────────────────────────────┐
│            API Server / Scheduler        │  요청 진입점
└──────────────────┬──────────────────────┘
                   │ config dict
                   ▼
┌─────────────────────────────────────────┐
│           ProcessExecutor               │  파이프라인 오케스트레이터
│   (EXECUTOR_REGISTRY로 동적 로딩)       │
└──┬────────┬────────┬────────┬───────────┘
   │        │        │        │
   ▼        ▼        ▼        ▼
Mart     Model    Predict   Export   ...    개별 Executor들
   │        │        │        │
   ▼        ▼        ▼        ▼
┌─────────────────────────────────────────┐
│           File Server  /  DB            │  영속성 레이어
│  (.parquet, .pkl, .json, .xlsx, DB)     │
└─────────────────────────────────────────┘
```

### 3-2. 전형적인 데이터 흐름

```
[원천 DB/파일]
      │
      ▼
MartExecutor          → mart/{id}.parquet
      │
      ▼
DataAnalysisExecutor  → analysis/{id}_eda.json
      │
      ▼
ModelExecutor(*)      → models/{id}.pkl / MOJO / RDS
      │
      ▼
PredictExecutor       → predict/{id}_result.parquet
      │
      ▼
StrategyExecutor      → strategy/{id}_result.parquet
      │
      ▼
ReportExecutor        → reports/{id}.json / .xlsx
      │
      ▼
ExportExecutor        → CSV / DB table / REST API
```

(*) ModelExecutor = PythonModelExecutor, H2OModelExecutor, RModelExecutor, AutoMLExecutor, ScorecardExecutor 중 하나

---

## 4. 핵심 설계 패턴

### 4-1. BaseExecutor — Template Method Pattern

모든 Executor는 `BaseExecutor`를 상속하며 `execute()` 메서드 하나만 구현하면 된다.
공통 관심사(잡 상태 기록, 시간 측정, 예외 처리, 파일 I/O)는 기반 클래스가 담당한다.

```python
class BaseExecutor(ABC):
    def run(self) -> dict:          # 공통 래퍼: 상태 기록 + 예외 처리
        ...
        result = self.execute()     # ← 서브클래스 구현부
        ...

    @abstractmethod
    def execute(self) -> dict:      # 서브클래스가 반드시 구현
        ...

    # 공통 유틸 메서드 (상속하여 바로 사용)
    def _load_dataframe(self, path) -> pd.DataFrame
    def _save_dataframe(self, df, path) -> str
    def _save_json(self, data, path) -> str
    def _update_job_status(self, status, progress, message)
```

### 4-2. ProcessExecutor — Composite + Registry Pattern

`ProcessExecutor`는 파이프라인의 각 단계를 `EXECUTOR_REGISTRY`에서 동적으로 로딩하여 순차 실행한다.

```python
EXECUTOR_REGISTRY = {
    "mart":          ("executors.ml.mart_executor",    "MartExecutor"),
    "python_model":  ("executors.ml.python_model_executor", "PythonModelExecutor"),
    ...
}
```

- 단계 간 결과 전달: `input_from` 키로 이전 단계 결과를 다음 단계 config에 자동 주입
- 실패 정책: 단계별 `on_error: "stop" | "skip" | "continue"` 설정

### 4-3. 잡 상태 관리 — Polling 기반

비동기 실행 시 클라이언트는 파일 폴링으로 진행 상황을 확인한다.

```
{FILE_ROOT_DIR}/jobs/{job_id}.json
{
  "job_id": "job_001",
  "status": "RUNNING",      // PENDING → RUNNING → COMPLETED | FAILED
  "progress": 45.0,         // 0~100
  "updated_at": "2024-01-15T10:30:22"
}
```

### 4-4. 통일된 반환 형식

모든 Executor의 `run()`은 동일한 구조의 dict를 반환한다.

```python
{
    "status":      "COMPLETED" | "FAILED",
    "job_id":      "job_001",
    "result":      { ... },    # executor별 결과
    "message":     "완료 요약",
    "elapsed_sec": 12.4,
}
```

---

## 5. Executor 목록 및 역할

| Executor | 키 (`EXECUTOR_REGISTRY`) | 주요 역할 | 입력 | 출력 |
|----------|--------------------------|-----------|------|------|
| `MartExecutor` | `mart` | 원천 데이터 → 마트 데이터셋 생성 | SQL / 파일 | `.parquet` |
| `DataAnalysisExecutor` | `data_analysis` | EDA (결측, 이상값, 상관계수, KS) | `.parquet` | EDA JSON |
| `PythonModelExecutor` | `python_model` | sklearn/XGBoost/LightGBM/CatBoost 학습 | `.parquet` | `.pkl` + 메타 |
| `H2OModelExecutor` | `h2o_model` | H2O GBM/DRF/XGBoost/GLM 학습 | `.parquet` | MOJO + 메타 |
| `PythonH2OModelExecutor` | *(직접 호출)* | Python 전처리 + H2O MOJO 통합 파이프라인 | `.parquet` + MOJO | 점수 `.parquet` |
| `RModelExecutor` | `r_model` | R 스크립트 subprocess 학습/예측 | `.parquet` | RDS + 메타 |
| `AutoMLExecutor` | `automl` | H2O AutoML / autosklearn / TPOT / Optuna / PyCaret | `.parquet` | 최적 모델 + 리더보드 |
| `ScorecardExecutor` | `scorecard` | WOE 기반 스코어카드 (신용평가) | `.parquet` | 스코어카드 JSON |
| `PredictExecutor` | `predict` | 저장된 모델로 예측 (Python/H2O/R) | `.parquet` + 모델 | 점수 `.parquet` |
| `PretrainedExecutor` | `pretrained` | 외부 완성 모델 추론 (ONNX/HuggingFace 등) | `.parquet` + 모델 | 결과 `.parquet` |
| `RuleSearchExecutor` | `rulesearch` | if-then 규칙 탐색 (트리/연관규칙/WOE) | `.parquet` | 규칙 JSON |
| `StrategyExecutor` | `stg` | 점수 → 업무 의사결정 (등급/한도/승인) | `.parquet` | decision `.parquet` |
| `RLExecutor` | `rl` | 강화학습 정책 학습/평가/배포 | 환경설정 / `.parquet` | `.pkl` 정책 모델 |
| `ReportExecutor` | `report` | 성능/EDA/스코어카드/예측 결과 리포트 | JSON / `.parquet` | `.json` / `.xlsx` |
| `ExportExecutor` | `export` | 결과 외부 내보내기 (파일/DB/API) | `.parquet` | CSV/Excel/DB/API |
| `ProcessExecutor` | *(최상위)* | 파이프라인 오케스트레이션 | 파이프라인 config | 단계별 결과 집합 |

---

## 6. Executor별 심층 설명

### 6-1. MartExecutor

데이터 파이프라인의 **첫 번째 관문**. 원천 데이터를 ML에 적합한 형태로 변환한다.

**처리 단계:**
1. 원천 로드: DB SQL(`source_query`) 또는 파일(`source_path`)
2. 기본 전처리: 문자열→카테고리 자동 변환, 수치형 결측→중앙값, 이상값 IQR 클리핑(1~99 퍼센타일)
3. 파생 변수: `feature_rules`의 pandas eval 표현식으로 새 컬럼 생성
4. 분할: train/valid/test 비율 지정 시 랜덤 셔플 후 분할
5. 메타 저장: shape, 컬럼 목록, dtype, 저장 경로 → `{target_id}_meta.json`

**도메인 맥락:** 신용평가 프로젝트에서 고객 거래이력 DB를 조회해 `debt_ratio`, `overdue_cnt` 같은 파생 변수를 만들고 train/test 분리까지 자동 처리하는 역할.

---

### 6-2. DataAnalysisExecutor

모델 학습 전 **데이터 품질 진단** 역할.

**분석 항목:**
- `basic_stats`: 수치형 변수 기술통계 (mean, std, 백분위)
- `missing`: 컬럼별 결측률, 30% 초과 시 경고 플래그
- `outliers`: IQR 1.5배 기준 이상값 건수/비율
- `distribution`: skewness, kurtosis (분포 형태 판단)
- `category_freq`: 범주형 변수 상위 10개 빈도
- `correlation`: 수치형 변수 간 상관계수 행렬, 0.9 초과 쌍 경고
- `target_analysis`: 타깃 클래스 간 KS 분리도 (변수 선별 힌트)

**출력 경고 활용:** `high_missing_cols`, `high_corr_pairs`를 보고 전처리 전략을 결정한다.

---

### 6-3. PythonModelExecutor

scikit-learn 생태계 모델의 **표준 학습 executor**.

**지원 모델:**
- 분류: `logistic_regression`, `random_forest`, `xgboost`, `lightgbm`, `catboost`, `gradient_boosting`, `decision_tree`
- 회귀: `linear_regression`, `random_forest_regressor`

**평가 지표:**
- 분류: AUC, Accuracy, F1, Precision, Recall
- 회귀: RMSE, R²

검증 데이터(`valid_path`)가 없으면 학습 데이터의 20%를 자동 분리한다.
모델은 pickle로 저장되며 메타 JSON에 피처 목록, 하이퍼파라미터, 성능 지표가 함께 기록된다.

---

### 6-4. AutoMLExecutor

여러 AutoML 프레임워크를 **단일 인터페이스**로 통합한 executor.

| 프레임워크 | 특징 | 권장 상황 |
|-----------|------|----------|
| `optuna` | LightGBM 베이지안 최적화 (기본 50 trials) | 가장 범용, 의존성 최소 |
| `h2o_automl` | 앙상블 리더보드, H2O 서버 필요 | 앙상블 성능 극대화 |
| `autosklearn` | sklearn 기반 강력한 앙상블, Linux 전용 | 메타러닝 활용 |
| `tpot` | 유전 알고리즘 파이프라인 탐색 | 파이프라인 구조 자동화 |
| `pycaret` | 빠른 다중 모델 비교 | 초기 탐색 단계 |

모든 프레임워크의 결과는 동일한 리더보드 + pickle 모델 형태로 저장된다.

---

### 6-5. ScorecardExecutor

**금융 신용평가** 전용 스코어카드 모델을 생성한다.

**처리 순서:**
1. 수치형 변수 분위수 binning
2. 구간별 WOE(Weight of Evidence) 및 IV(Information Value) 계산
3. IV 임계값 미만 변수 제거
4. 선택된 변수의 WOE 값으로 로지스틱 회귀 학습
5. PDO(Points to Double Odds) 기반 점수 변환 (예: 기준점 600, PDO 50)
6. 스코어카드 테이블 생성 (변수 × 구간 × WOE × 점수)

**성능 지표:** KS, AUC, Gini

---

### 6-6. PredictExecutor

운영 환경에서 **가장 빈번하게 호출**되는 executor.

- Python 모델(`.pkl`): `predict_proba` 또는 `predict` 자동 판별
- H2O 모델(MOJO): H2O 서버 연동
- R 모델(RDS): subprocess로 Rscript 호출, CSV 파일로 입출력 교환

예측 후 `grade_mapping`이 있으면 점수 범위에 따라 A/B/C/D 등급을 자동 부여한다.
결과에는 score 통계(mean, std, p25/p50/p75)와 등급 분포가 포함된다.

---

### 6-7. StrategyExecutor

모델 점수를 **실제 업무 의사결정**으로 변환하는 마지막 ML 단계.

| 전략 유형 | 설명 |
|-----------|------|
| `grade` | 점수 구간 → 등급 (A/B/C/D) |
| `threshold` | 임계값 기준 승인/거절 |
| `tiered` | 다단계 등급별 한도·금리 적용 |
| `matrix` | 2개 변수(예: 점수 × 직종) 교차 매트릭스 |

오버라이드 룰: 사기이력, 고DSR 등 정책 강제 반영 기능으로 모델 결과를 덮어쓴다.

---

### 6-8. RLExecutor

**강화학습**을 활용한 동적 의사결정 executor. 금융 도메인에서 아래 문제에 적용된다:
- 고객별 대출 한도/금리 동적 최적화
- 연체 고객 컬렉션 전략 최적 시퀀스
- 상품/채널 추천 최적화

| 알고리즘 | 적용 | 비고 |
|----------|------|------|
| `q_learning` | 소규모 이산 상태공간 | 배치 데이터 기반 오프라인 학습 |
| `dqn` | 연속 상태공간 | stable-baselines3 |
| `ppo` | 연속 상태공간, 안정적 정책 | stable-baselines3 |
| `contextual_bandit` | 단순 의사결정, LinUCB | 실시간 추론에 유리 |

모드: `train`(학습) → `evaluate`(오프라인 평가) → `deploy`(단건 실시간 추론)

---

### 6-9. ExportExecutor

파이프라인의 **최종 출구**. 결과를 외부 시스템으로 내보낸다.

- **파일**: CSV, Excel, JSON, Parquet
- **DB**: append / replace / upsert (MariaDB/MySQL 호환 DELETE+INSERT 방식)
- **API**: REST POST, 배치 전송 (기본 1000건씩)

컬럼 선택(`columns`)과 이름 변경(`rename_map`)을 지원해 외부 시스템 스키마에 맞춰 출력할 수 있다.

---

## 7. 파일 저장 구조

```
{FILE_ROOT_DIR}/                        # 환경변수 FILE_ROOT_DIR (기본: /data)
├── jobs/
│   └── {job_id}.json                   # 잡 상태 파일 (폴링용)
├── mart/
│   ├── {target_id}_train.parquet
│   ├── {target_id}_valid.parquet
│   ├── {target_id}_test.parquet
│   └── {target_id}_meta.json
├── models/
│   ├── {model_id}.pkl                  # Python 모델
│   ├── {model_id}/                     # H2O MOJO 디렉터리
│   ├── {model_id}.rds                  # R 모델
│   └── {model_id}_meta.json            # 모델 메타 (피처, 하이퍼파라미터, 성능)
├── predict/
│   └── {output_id}_result.parquet
├── analysis/
│   ├── {output_id}_eda.json
│   └── {output_id}_rules.json
├── strategy/
│   └── {output_id}_result.parquet
├── reports/
│   ├── {output_id}.json
│   └── {output_id}.xlsx
└── exports/
    ├── {output_id}.csv
    └── {output_id}_summary.json
```

---

## 8. 공통 config 키

모든 Executor에 공통으로 적용되는 config 키:

| 키 | 타입 | 설명 |
|----|------|------|
| `job_id` | str | 잡 식별자 (상태 파일명) |
| `service_id` | str | 서비스/프로젝트 식별자 |
| `project_id` | str | 프로젝트 식별자 |

ProcessExecutor가 하위 Executor를 생성할 때 이 세 키를 자동으로 전파한다.

---

## 9. 새 Executor 추가 방법

```python
# 1. 파일 생성: executors/ml/my_executor.py
from executors.ml.base_executor import BaseExecutor, ExecutorStatus

class MyExecutor(BaseExecutor):
    def execute(self) -> dict:
        # config에서 파라미터 읽기
        cfg = self.config
        
        # 공통 유틸 활용
        df = self._load_dataframe(cfg["input_path"])
        self._update_job_status(ExecutorStatus.RUNNING, progress=50)
        
        # ... 실제 작업 ...
        
        saved = self._save_dataframe(result_df, f"output/{cfg['output_id']}.parquet")
        
        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  {"output_path": saved, "rows": len(result_df)},
            "message": "완료",
        }

# 2. process_executor.py의 EXECUTOR_REGISTRY에 등록
"my_executor": ("executors.ml.my_executor", "MyExecutor"),
```

---

## 10. 의존성

| 라이브러리 | 용도 | 필수 여부 |
|-----------|------|----------|
| `pandas`, `numpy` | 데이터 처리 | **필수** |
| `scikit-learn` | Python 모델, 전처리 | **필수** |
| `scipy` | 통계 분석 (KS, skewness) | EDA 사용 시 |
| `lightgbm` | LightGBM 모델, Optuna AutoML | 권장 |
| `xgboost` | XGBoost 모델 | 선택 |
| `catboost` | CatBoost 모델 | 선택 |
| `h2o` | H2O 모델, H2O AutoML | H2O 사용 시 |
| `optuna` | 베이지안 하이퍼파라미터 최적화 | AutoML 사용 시 |
| `stable-baselines3`, `gymnasium` | DQN/PPO 강화학습 | RL 사용 시 |
| `mlxtend` | FP-Growth 연관규칙 | RuleSearch 사용 시 |
| `openpyxl` | Excel 출력 | Report/Export 사용 시 |
| `onnxruntime` | ONNX 모델 추론 | Pretrained ONNX 사용 시 |
| `transformers` | HuggingFace NLP 모델 | Pretrained NLP 사용 시 |
| `requests` | API export | ExportExecutor API 모드 |

---

## 11. 실사용 패턴 — 신용평가 파이프라인 예시

```python
config = {
    "job_id":        "credit_score_job_001",
    "service_id":    "credit_scoring",
    "project_id":    "retail_loan_v2",
    "pipeline_name": "retail_credit_scoring_v2",
    "pipeline": [
        {
            "name": "make_mart",
            "executor": "mart",
            "config": {
                "source_query":  "SELECT * FROM customer_txn WHERE yymm = '202312'",
                "target_id":     "retail_mart_v2",
                "target_col":    "default_yn",
                "split":         {"train": 0.7, "valid": 0.15, "test": 0.15},
                "feature_rules": [
                    {"name": "debt_ratio",   "expr": "total_debt / (income + 1)"},
                    {"name": "util_rate",    "expr": "used_limit / (credit_limit + 1)"}
                ]
            }
        },
        {
            "name":     "eda",
            "executor": "data_analysis",
            "config": {
                "source_path": "mart/retail_mart_v2_train.parquet",
                "output_id":   "retail_eda_v2",
                "target_col":  "default_yn"
            }
        },
        {
            "name":     "train_lgbm",
            "executor": "python_model",
            "config": {
                "model_type":  "lightgbm",
                "train_path":  "mart/retail_mart_v2_train.parquet",
                "valid_path":  "mart/retail_mart_v2_valid.parquet",
                "target_col":  "default_yn",
                "model_id":    "lgbm_retail_v2",
                "model_params": {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 64}
            }
        },
        {
            "name":     "predict_test",
            "executor": "predict",
            "config": {
                "model_id":   "lgbm_retail_v2",
                "input_path": "mart/retail_mart_v2_test.parquet",
                "output_id":  "retail_pred_v2",
                "grade_mapping": {"A": [800, 1000], "B": [650, 800], "C": [500, 650], "D": [0, 500]}
            }
        },
        {
            "name":     "apply_strategy",
            "executor": "stg",
            "config": {
                "strategy_type": "tiered",
                "score_col":     "score",
                "output_id":     "retail_stg_v2",
                "tiered_rules": [
                    {"score_min": 800, "grade": "A", "limit_pct": 1.0, "rate": 3.5},
                    {"score_min": 600, "grade": "B", "limit_pct": 0.7, "rate": 5.0},
                    {"score_min": 0,   "grade": "C", "reject": True}
                ]
            }
        },
        {
            "name":     "generate_report",
            "executor": "report",
            "config": {
                "report_type":     "model_performance",
                "report_name":     "Retail Credit Scoring v2",
                "output_id":       "retail_report_v2",
                "model_meta_path": "models/lgbm_retail_v2_meta.json",
                "prediction_path": "predict/retail_pred_v2_result.parquet",
                "score_col":       "score",
                "target_col":      "default_yn"
            }
        },
        {
            "name":     "export_result",
            "executor": "export",
            "config": {
                "source_path": "strategy/retail_stg_v2_result.parquet",
                "export_type": "db",
                "output_id":   "retail_export_v2",
                "table_name":  "ml_scoring_result",
                "write_mode":  "upsert",
                "key_cols":    ["customer_id", "yymm"]
            }
        }
    ]
}

from executors.ml.process_executor import ProcessExecutor
result = ProcessExecutor(config=config, db_session=session).run()
```

---

## 12. 설계 원칙 및 추상화 배경

이 프레임워크는 실제 금융 ML 플랫폼의 실행 레이어를 추상화한 것으로, 다음 원칙 아래 설계되었다.

1. **단일 책임**: 각 Executor는 하나의 ML 단계만 담당한다.
2. **공통 인터페이스**: `run()` → `execute()` 패턴으로 모든 Executor가 동일하게 호출된다.
3. **설정 주도(Config-driven)**: 코드 변경 없이 config dict만으로 동작을 제어한다.
4. **파일 기반 통신**: 단계 간 데이터는 Parquet 파일로 교환해 메모리 부담을 최소화하고 재시작을 용이하게 한다.
5. **관측 가능성**: 잡 상태 파일과 로깅으로 모든 단계의 진행 상황을 추적할 수 있다.
6. **확장성**: `EXECUTOR_REGISTRY`에 새 항목을 추가하는 것만으로 파이프라인에 새 단계를 삽입할 수 있다.

실제 상용 버전에서는 이 레이어 위에 API 서버, 스케줄러, 웹 UI, 모델 레지스트리, 알림 시스템이 연동되어 있다.
