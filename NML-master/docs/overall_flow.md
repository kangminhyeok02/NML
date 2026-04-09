# NML Executor Framework — 전체 흐름 분석

## 시스템 개요

금융/리스크 도메인 특화 ML 파이프라인 프레임워크.  
신용평가, 이상탐지, 강화학습 기반 전략 최적화 등 다양한 ML 워크플로우를  
모듈화된 executor 단위로 조합하여 실행한다.

---

## 아키텍처 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                      ProcessExecutor                         │
│              (파이프라인 오케스트레이터)                       │
│                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  MartEx  │→ │DataAnalysisEx│→ │    Model Executor      │  │
│  │ 마트생성  │  │  EDA 분석    │  │ (아래 5종 중 선택)     │  │
│  └──────────┘  └──────────────┘  └───────────────────────┘  │
│                                          ↓                   │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ ExportEx │← │  StrategyEx  │← │     PredictExecutor    │  │
│  │  결과배포 │  │  전략적용    │  │       예측수행         │  │
│  └──────────┘  └──────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

Model Executor 선택지:
  ├── PythonModelExecutor    (scikit-learn / XGBoost / LightGBM / CatBoost)
  ├── H2OModelExecutor       (H2O GBM / DRF / GLM / DeepLearning / AutoML)
  ├── AutoMLExecutor         (H2O / Optuna / TPOT / AutoSklearn / PyCaret)
  ├── ScorecardExecutor      (WOE / Binning / LogisticRegression)
  └── RLExecutor             (Q-Learning / DQN / PPO / LinUCB Bandit)

보조 Executor:
  ├── PythonH2OModelExecutor (Python 전처리 + H2O MOJO 추론 혼합)
  ├── PretrainedExecutor     (Pickle / ONNX / H2O MOJO / HuggingFace)
  └── RuleSearchExecutor     (Decision Tree / Association / WOE Rule)
```

---

## BaseExecutor 공통 인터페이스

모든 executor가 공유하는 생명주기:

```
BaseExecutor.run()
    │
    ├── started_at 기록
    ├── 잡 상태 → RUNNING  (jobs/{job_id}.json)
    │
    ├── execute()  ← 서브클래스 구현
    │       │
    │       ├── _load_dataframe()    parquet / csv 로드
    │       ├── _save_dataframe()    parquet 저장
    │       ├── _save_json()         메타/요약 JSON 저장
    │       └── _update_job_status() 진행률(0~100) 기록
    │
    ├── 성공: 잡 상태 → COMPLETED
    └── 실패: 잡 상태 → FAILED  (에러 메시지 포함)
```

**표준 반환 딕셔너리:**
```python
{
    "status":      "COMPLETED" | "FAILED",
    "job_id":      "job_abc123",
    "result":      { ... },   # executor별 결과
    "message":     "요약 메시지",
    "elapsed_sec": 12.3,
}
```

---

## 파일 저장 규칙

모든 executor는 `FILE_ROOT_DIR` (환경변수) 하위의 규약된 경로에 저장한다.

```
FILE_ROOT_DIR/
├── jobs/
│   └── {job_id}.json              ← 잡 상태 (폴링용)
├── mart/
│   ├── {target_id}.parquet        ← MartExecutor 전체 마트
│   ├── {target_id}_train.parquet  ← 분할 시
│   ├── {target_id}_valid.parquet
│   ├── {target_id}_test.parquet
│   └── {target_id}_meta.json
├── analysis/
│   ├── {output_id}_eda.json       ← DataAnalysisExecutor
│   └── {output_id}_rules.json     ← RuleSearchExecutor
├── models/
│   ├── {model_id}.pkl             ← Python 모델 (pickle)
│   ├── {model_id}_automl.pkl      ← AutoML 최적 모델
│   ├── {model_id}/model.zip       ← H2O MOJO
│   ├── {model_id}_meta.json       ← 모델 메타 (공통)
│   ├── {model_id}_scorecard.json  ← 스코어카드 테이블
│   └── {model_id}_rl.pkl          ← RL 정책 모델
├── predict/
│   ├── {output_id}_result.parquet          ← PredictExecutor
│   ├── {output_id}_py_h2o.parquet          ← PythonH2OModelExecutor
│   └── {output_id}_pretrained.parquet      ← PretrainedExecutor
├── strategy/
│   ├── {output_id}_result.parquet  ← StrategyExecutor 결과
│   └── {output_id}_summary.json    ← 전략 요약
└── exports/
    └── {output_id}_summary.json    ← ExportExecutor 요약
```

---

## 표준 파이프라인 플로우 (신용 스코어링)

```
[1] MartExecutor
    source_query → SQL 조회
    기본 전처리 (타입/결측/클리핑)
    파생변수 생성
    train/valid/test 분리
    → mart/{id}_train.parquet

[2] DataAnalysisExecutor
    기초 통계 / 결측 / 이상값 / 분포
    상관계수 행렬 (다중공선성 진단)
    KS 통계량 (변수 예측력 순위)
    → analysis/{id}_eda.json

[3-A] PythonModelExecutor (또는 H2OModelExecutor, AutoMLExecutor)
    피처/타깃 분리
    모델 학습 (LightGBM / GBM / AutoML)
    검증 성능 평가 (AUC, KS)
    → models/{id}.pkl + _meta.json

[3-B] ScorecardExecutor (전통 금융 스코어링)
    WOE/IV 산출 → IV 필터
    LogisticRegression → PDO 스케일링
    → models/{id}_scorecard.json

[4] PredictExecutor
    모델 로드 (pickle / MOJO / R)
    신규 데이터 예측
    grade_mapping으로 등급 부여
    → predict/{id}_result.parquet

[5] StrategyExecutor
    점수 → 등급/의사결정 변환 (grade/threshold/tiered/matrix)
    오버라이드 룰 적용 (사기, 고DSR 등)
    → strategy/{id}_result.parquet

[6] ExportExecutor
    컬럼 선택 / 이름 변경
    CSV / Excel / DB 적재 / API 전송
    → 운영계 DB 또는 외부 시스템
```

---

## 대표 파이프라인 config 예시

```python
config = {
    "pipeline_name": "credit_scoring_v2",
    "stop_on_first_failure": True,
    "pipeline": [
        {
            "name":     "mart",
            "executor": "mart",
            "config": {
                "source_query": "SELECT * FROM RAW_LOAN WHERE base_dt='2026-04-07'",
                "target_id":    "loan_20260407",
                "feature_rules": [
                    {"name": "dti", "expr": "total_debt / (income + 1)"}
                ],
                "split": {"train": 0.7, "valid": 0.15, "test": 0.15},
            }
        },
        {
            "name":       "eda",
            "executor":   "data_analysis",
            "input_from": "mart",
            "config": {
                "source_path":    "mart/loan_20260407_train.parquet",
                "output_id":      "loan_20260407",
                "target_col":     "default",
                "corr_threshold": 0.85,
            }
        },
        {
            "name":     "train",
            "executor": "python_model",
            "config": {
                "model_type":  "lightgbm",
                "train_path":  "mart/loan_20260407_train.parquet",
                "valid_path":  "mart/loan_20260407_valid.parquet",
                "target_col":  "default",
                "model_id":    "lgbm_loan_v1",
                "model_params": {"n_estimators": 500, "learning_rate": 0.05, "verbose": -1},
            }
        },
        {
            "name":     "predict",
            "executor": "predict",
            "config": {
                "model_id":   "lgbm_loan_v1",
                "input_path": "mart/loan_20260407_test.parquet",
                "output_id":  "loan_pred_20260407",
                "model_type": "python",
            }
        },
        {
            "name":     "strategy",
            "executor": "stg",
            "config": {
                "input_path":    "predict/loan_pred_20260407_result.parquet",
                "score_col":     "score",
                "strategy_type": "tiered",
                "output_id":     "loan_stg_20260407",
                "tiered_rules": [
                    {"score_min": 0.7, "grade": "A", "limit_pct": 1.0},
                    {"score_min": 0.4, "grade": "B", "limit_pct": 0.6},
                    {"score_min": 0.0, "grade": "C", "limit_pct": 0.0, "reject": True},
                ],
                "override_rules": [
                    {"condition": "fraud_flag == 1", "decision": "REJECT", "reason": "사기이력"},
                ],
                "key_cols": ["cust_id"],
            }
        },
        {
            "name":     "export",
            "executor": "export",
            "config": {
                "source_path": "strategy/loan_stg_20260407_result.parquet",
                "export_type": "db",
                "output_id":   "loan_stg_20260407",
                "table_name":  "ML_LOAN_DECISION",
                "write_mode":  "upsert",
                "key_cols":    ["cust_id"],
                "rename_map":  {"score": "ML_SCORE", "grade": "ML_GRADE"},
            }
        },
    ]
}
```

---

## 비표준 파이프라인 시나리오

### AutoML 탐색 → 최적 모델 배포

```
AutoMLExecutor (optuna/h2o_automl)
    → 리더보드 + best model pkl
PredictExecutor (model_type="python")
    → 예측 결과
```

### H2O 학습 → Python 운영 추론

```
H2OModelExecutor
    → MOJO 파일 저장
PythonH2OModelExecutor
    → Python 전처리 + MOJO 추론 + Python 후처리
```

### 사전 학습 모델 챔피언/챌린저

```
PretrainedExecutor (model_format="pickle", model_id="champion")
PretrainedExecutor (model_format="h2o",    model_id="challenger")
    → 두 점수 비교 후 우수 모델 승격
```

### 강화학습 전략 최적화

```
RLExecutor (mode="train", algorithm="contextual_bandit")
    → 정책 학습
RLExecutor (mode="evaluate", input_path=...)
    → 오프라인 평가 / recommended_action 컬럼 생성
StrategyExecutor
    → RL 추천 행동 기반 한도/금리 결정
```

### 규칙 탐색 → 정책 룰 설계

```
RuleSearchExecutor (method="decision_tree")
    → if-then 규칙 목록 (bad_rate, lift 기준)
StrategyExecutor (override_rules에 상위 규칙 적용)
    → 고위험 구간 자동 거절
```

---

## Executor 선택 가이드

| 상황 | 권장 Executor |
|------|-------------|
| 대용량 DB → 마트 구축 | `MartExecutor` |
| 모델 전 변수 탐색/품질 진단 | `DataAnalysisExecutor` |
| 빠른 Python 모델 학습 | `PythonModelExecutor` |
| 대용량 + H2O 클러스터 활용 | `H2OModelExecutor` |
| 최적 알고리즘 자동 탐색 | `AutoMLExecutor` |
| 금융 규제 준수 필요 (해석 가능) | `ScorecardExecutor` |
| 운영 배포 추론 | `PredictExecutor` |
| H2O 학습 + Python 운영 파이프라인 | `PythonH2OModelExecutor` |
| 기존 모델 재활용 / A-B 테스트 | `PretrainedExecutor` |
| 동적 정책 최적화 | `RLExecutor` |
| 설명 가능한 비즈니스 룰 발굴 | `RuleSearchExecutor` |
| 점수 → 업무 의사결정 변환 | `StrategyExecutor` |
| 결과 DB/API 배포 | `ExportExecutor` |
| 위 모든 단계 조합 | `ProcessExecutor` |

---

## 데이터 흐름 요약 다이어그램

```
원천 DB / 파일
      │
      ▼
[MartExecutor]────────────────────────────────── mart/*.parquet
      │
      ▼
[DataAnalysisExecutor]────────────────────────── analysis/*_eda.json
      │
      ▼
┌─────────────────────────────┐
│     Model Executor 선택     │
│                             │
│  PythonModelExecutor        │──── models/*.pkl
│  H2OModelExecutor           │──── models/*/model.zip (MOJO)
│  AutoMLExecutor             │──── models/*_automl.pkl
│  ScorecardExecutor          │──── models/*_scorecard.json
│  RLExecutor                 │──── models/*_rl.pkl
└─────────────────────────────┘
      │
      ▼
[PredictExecutor]─────────────────────────────── predict/*_result.parquet
  또는
[PythonH2OModelExecutor]──────────────────────── predict/*_py_h2o.parquet
  또는
[PretrainedExecutor]──────────────────────────── predict/*_pretrained.parquet
      │
      ▼
[StrategyExecutor]────────────────────────────── strategy/*_result.parquet
  (grade / threshold / tiered / matrix + override)
      │
      ▼
[ExportExecutor]──────────────────────────────── CSV / DB / API
      │
      ▼
  운영계 시스템
```

---

## 핵심 설계 원칙

1. **단일 책임**: 각 executor는 하나의 역할만 담당 — 학습은 학습, 예측은 예측
2. **결합 가능성**: `ProcessExecutor`의 `input_from`으로 단계 간 데이터 자동 전달
3. **실패 격리**: 단계별 `on_error` 정책으로 부분 실패 시 전체 중단 방지
4. **관찰 가능성**: 모든 단계가 `jobs/{job_id}.json`에 진행률 기록 → 폴링 지원
5. **확장성**: 새 executor는 `BaseExecutor` 상속 + `EXECUTOR_REGISTRY` 등록만으로 추가 가능
6. **재현 가능성**: 모든 출력물이 파일로 저장 → 단계별 재실행 및 결과 비교 가능
