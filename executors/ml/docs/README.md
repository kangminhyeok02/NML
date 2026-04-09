# executors/ml 문서 인덱스

`executors/ml/` 하위 각 Python 파일에 대한 함수/클래스 상세 문서 목록.

---

## 파일별 문서

| 파일 | 문서 | 역할 요약 |
|---|---|---|
| `base_executor.py` | [base_executor.md](base_executor.md) | 모든 executor의 추상 기반 클래스. 공통 실행 래퍼, 상태 관리, 파일 I/O 제공 |
| `process_executor.py` | [process_executor.md](process_executor.md) | ML 파이프라인 오케스트레이션. 단계별 executor를 순차 실행 |
| `mart_executor.py` | [mart_executor.md](mart_executor.md) | 데이터 마트 생성. 원천 데이터 로드 → 전처리 → 파생변수 → 분할 → 저장 |
| `data_analysis_executor.py` | [data_analysis_executor.md](data_analysis_executor.md) | 탐색적 데이터 분석(EDA). 결측/이상값/분포/상관/KS 분리도 분석 |
| `python_model_executor.py` | [python_model_executor.md](python_model_executor.md) | Python ML 모델 학습 (sklearn, XGBoost, LightGBM, CatBoost 등) |
| `h2o_model_executor.py` | [h2o_model_executor.md](h2o_model_executor.md) | H2O 프레임워크 기반 모델 학습 (GBM, DRF, XGBoost, DeepLearning, AutoML) |
| `python_h2o_model_executor.py` | [python_h2o_model_executor.md](python_h2o_model_executor.md) | Python 전처리/후처리 + H2O MOJO 추론 통합 파이프라인 |
| `automl_executor.py` | [automl_executor.md](automl_executor.md) | AutoML 탐색 (H2O AutoML, Optuna, auto-sklearn, TPOT, PyCaret) |
| `scorecard_executor.py` | [scorecard_executor.md](scorecard_executor.md) | 신용 스코어카드 생성 (Binning → WOE → IV → LogisticRegression → PDO 스케일링) |
| `predict_executor.py` | [predict_executor.md](predict_executor.md) | 저장된 모델로 신규 데이터 예측 (Python / H2O / R 모델 지원) |
| `pretrained_executor.py` | [pretrained_executor.md](pretrained_executor.md) | 사전 학습 모델 추론 (pickle / ONNX / H2O MOJO / HuggingFace) |
| `stg_executor.py` | [stg_executor.md](stg_executor.md) | 업무 전략 적용 (등급화 / 임계값 / 다단계 / 매트릭스 / 오버라이드) |
| `r_model_executor.py` | [r_model_executor.md](r_model_executor.md) | R 스크립트 기반 모델 학습/예측 (subprocess + CSV 파일 교환) |
| `rulesearch_executor.py` | [rulesearch_executor.md](rulesearch_executor.md) | if-then 규칙 탐색 (DecisionTree / FP-Growth / WOE 구간) + GA/Greedy 최적화 |
| `rl_executor.py` | [rl_executor.md](rl_executor.md) | 강화학습 정책 학습 (Q-Learning / DQN / PPO / LinUCB Bandit) |
| `report_executor.py` | [report_executor.md](report_executor.md) | 분석/모델 결과 리포트 생성 (JSON / Excel 출력) |
| `export_executor.py` | [export_executor.md](export_executor.md) | 결과 데이터 내보내기 (파일 / DB 적재 / REST API 전송) |

---

## 실행 흐름 요약

```
ProcessExecutor (오케스트레이션)
├── MartExecutor          → 데이터 마트 생성
├── DataAnalysisExecutor  → EDA 분석
├── PythonModelExecutor   → Python 모델 학습
│   H2OModelExecutor      → H2O 모델 학습
│   AutoMLExecutor        → AutoML 탐색
│   RModelExecutor        → R 모델 학습
│   ScorecardExecutor     → 스코어카드 생성
│   RuleSearchExecutor    → 규칙 탐색
│   RLExecutor            → 강화학습
├── PredictExecutor       → 예측 수행
│   PretrainedExecutor    → 사전학습 모델 추론
│   PythonH2OModelExecutor→ Python+H2O 통합 추론
├── StrategyExecutor      → 업무 전략 적용
├── ReportExecutor        → 리포트 생성
└── ExportExecutor        → 결과 내보내기
```

---

## 공통 규칙

- 모든 executor는 `BaseExecutor`를 상속하며 `execute() → dict`를 구현
- 직접 호출하는 대신 `run()`을 통해 실행 → 상태 관리, 예외 처리, 경과 시간 자동 기록
- 데이터 파일은 `FILE_ROOT_DIR` 환경변수 기준 상대 경로로 참조
- 잡 상태는 `{FILE_ROOT_DIR}/jobs/{job_id}.json`에 실시간 기록
