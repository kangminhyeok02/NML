# process_executor.py — ML 파이프라인 오케스트레이터

## 개요

전체 ML 파이프라인을 순차적으로 제어하는 오케스트레이션 executor.  
개별 executor를 단계별로 호출하고, 각 단계의 결과를 다음 단계로 전달한다.

```
ProcessExecutor
    step 1 → MartExecutor
    step 2 → DataAnalysisExecutor
    step 3 → PythonModelExecutor | H2OModelExecutor | AutoMLExecutor
    step 4 → ScorecardExecutor (선택)
    step 5 → PredictExecutor
    step 6 → StrategyExecutor
    step 7 → ExportExecutor
```

---

## EXECUTOR_REGISTRY

등록된 executor 유형 목록. `config.executor` 값으로 참조한다.

| 키 | 클래스 |
|----|--------|
| `mart` | `MartExecutor` |
| `data_analysis` | `DataAnalysisExecutor` |
| `python_model` | `PythonModelExecutor` |
| `h2o_model` | `H2OModelExecutor` |
| `r_model` | `RModelExecutor` |
| `automl` | `AutoMLExecutor` |
| `scorecard` | `ScorecardExecutor` |
| `predict` | `PredictExecutor` |
| `pretrained` | `PretrainedExecutor` |
| `report` | `ReportExecutor` |
| `export` | `ExportExecutor` |
| `rulesearch` | `RuleSearchExecutor` |
| `stg` | `StrategyExecutor` |
| `rl` | `RLExecutor` |

---

## config 구조

```python
{
    "pipeline_name": "credit_scoring_v2",       # 파이프라인 이름 (선택)
    "stop_on_first_failure": True,               # 첫 실패 시 중단 여부 (기본 True)
    "pipeline": [
        {
            "name":       "mart_build",          # 단계 이름 (고유해야 함)
            "executor":   "mart",                # EXECUTOR_REGISTRY 키
            "config":     { ... },               # 해당 executor config
            "on_error":   "stop",                # "stop" | "skip" | "continue"
            "input_from": "prev_step_name",      # 이전 단계 결과 주입 (선택)
        },
        ...
    ]
}
```

---

## 메서드 상세

### `execute()` → `dict`

파이프라인을 순차 실행한다.

**실행 흐름:**
```
pipeline 순회
    ├── input_from 지정 시 이전 단계 결과를 현재 step_config에 병합
    ├── job_id = "{parent_job_id}__{step_name}" 자동 생성
    ├── 진행률 = step_index / total_steps * 90 (%)
    ├── executor 인스턴스 생성 (_build_executor)
    ├── executor.run() 호출
    └── 실패 시
        ├── on_error="stop"  → 파이프라인 중단
        ├── on_error="skip"  → 해당 단계 건너뜀
        └── on_error="continue" → 실패 기록 후 계속
```

**반환 예시:**
```python
{
    "status": "COMPLETED",
    "result": {
        "pipeline_name": "credit_scoring_v2",
        "total_steps":   5,
        "executed":      5,
        "failed_steps":  [],
        "step_results": [
            {"step": "mart_build",   "status": "COMPLETED", ...},
            {"step": "eda",          "status": "COMPLETED", ...},
            ...
        ]
    },
    "message": "파이프라인 완료  5/5단계"
}
```

---

### `_build_executor(executor_type, config)` → `BaseExecutor`

`EXECUTOR_REGISTRY`에서 모듈 경로와 클래스명을 조회하여 `importlib`로 동적 로딩한다.

```python
module = importlib.import_module("executors.ml.mart_executor")
cls    = getattr(module, "MartExecutor")
return cls(config=config, db_session=self.db_session, file_root_dir=...)
```

---

## 단계 간 결과 전달 (`input_from`)

`context` 딕셔너리로 단계별 결과를 관리한다.

```python
# step A 완료 후
context["mart_build"] = step_result["result"]

# step B에서 input_from="mart_build" 설정 시
step_config.update(context["mart_build"])
# → mart_build 결과(output_path 등)가 B의 config에 자동 주입
```

---

## 진행률 계산

```
각 단계 시작 시 progress = (step_idx / total_steps) * 90
마지막 10%는 결과 요약을 위해 예약
```

---

## 오류 처리 정책

| `on_error` | `stop_on_first_failure` | 동작 |
|------------|------------------------|------|
| `"stop"` | `True` | 즉시 파이프라인 중단 |
| `"stop"` | `False` | 실패 기록 후 계속 |
| `"skip"` | - | 해당 단계 건너뜀 |
| `"continue"` | - | 실패 기록 후 계속 |

---

## 사용 예시

```python
config = {
    "pipeline_name": "loan_model",
    "pipeline": [
        {
            "name": "mart",
            "executor": "mart",
            "config": {"source_path": "raw/loan.parquet", "target_id": "loan_mart"},
        },
        {
            "name": "train",
            "executor": "python_model",
            "input_from": "mart",               # mart 결과의 saved_paths 등이 주입됨
            "config": {"model_type": "lightgbm", "target_col": "default", "model_id": "lgbm_v1"},
            "on_error": "stop",
        },
        {
            "name": "predict",
            "executor": "predict",
            "input_from": "train",
            "config": {"input_path": "mart/loan_mart.parquet", "output_id": "loan_pred"},
        },
    ]
}

executor = ProcessExecutor(config=config, db_session=db, file_root_dir="/data")
result   = executor.run()
```
