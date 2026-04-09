# process_executor.py — ML 파이프라인 오케스트레이터

**파일:** `executors/ml/process_executor.py`  
**클래스:** `ProcessExecutor(BaseExecutor)`

## 개요

개별 executor를 단계별로 호출하고 각 단계의 결과를 다음 단계로 전달하는  
파이프라인 오케스트레이션 executor.

```
ProcessExecutor
    step 1 → MartExecutor
    step 2 → DataAnalysisExecutor
    step 3 → PythonModelExecutor | H2OModelExecutor | AutoMLExecutor | ScorecardExecutor | RLExecutor
    step 4 → PredictExecutor | PythonH2OModelExecutor | PretrainedExecutor
    step 5 → StrategyExecutor
    step 6 → ReportExecutor
    step 7 → ExportExecutor
```

---

## EXECUTOR_REGISTRY

`EXECUTOR_REGISTRY`는 `(module_path, class_name)` 튜플을 값으로 하는 딕셔너리다.  
`_build_executor()`가 `importlib`로 동적 로딩하여 인스턴스를 생성한다.

| 키 | 모듈 | 클래스 |
|----|------|--------|
| `mart` | `executors.ml.mart_executor` | `MartExecutor` |
| `data_analysis` | `executors.ml.data_analysis_executor` | `DataAnalysisExecutor` |
| `python_model` | `executors.ml.python_model_executor` | `PythonModelExecutor` |
| `h2o_model` | `executors.ml.h2o_model_executor` | `H2OModelExecutor` |
| `r_model` | `executors.ml.r_model_executor` | `RModelExecutor` |
| `automl` | `executors.ml.automl_executor` | `AutoMLExecutor` |
| `scorecard` | `executors.ml.scorecard_executor` | `ScorecardExecutor` |
| `predict` | `executors.ml.predict_executor` | `PredictExecutor` |
| `pretrained` | `executors.ml.pretrained_executor` | `PretrainedExecutor` |
| `report` | `executors.ml.report_executor` | `ReportExecutor` |
| `export` | `executors.ml.export_executor` | `ExportExecutor` |
| `rulesearch` | `executors.ml.rulesearch_executor` | `RuleSearchExecutor` |
| `stg` | `executors.ml.stg_executor` | `StrategyExecutor` |
| `rl` | `executors.ml.rl_executor` | `RLExecutor` |

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `pipeline` | ✅ | `list` | 실행할 단계 목록 (아래 단계 구조 참조) |
| `pipeline_name` | ❌ | `str` | 파이프라인 이름 (결과 메타에 기록) |
| `stop_on_first_failure` | ❌ | `bool` | 첫 실패 시 전체 중단 여부 (기본: `True`) |

### 단계(step) 구조

```python
{
    "name":       str,    # 단계 이름 (고유해야 함, context 키로 사용)
    "executor":   str,    # EXECUTOR_REGISTRY 키
    "config":     dict,   # 해당 executor의 config
    "on_error":   str,    # "stop" | "skip" | "continue" (기본: "stop")
    "input_from": str,    # 이전 단계 이름 → 결과를 이 단계 config에 병합 (선택)
}
```

---

## execute() 실행 흐름

```python
pipeline 순회 (step_idx, step)
    ├── input_from 지정 시:
    │       step_config.update(context[input_from])   # 이전 단계 result 병합
    ├── job_id  자동 생성: "{parent_job_id}__{step_name}"
    ├── service_id, project_id 자동 전파
    ├── progress = int(step_idx / total_steps * 90)  # 0~90%
    ├── _build_executor(executor_type, step_config)
    ├── executor.run() 호출
    ├── 결과를 context[step_name]에 저장
    └── 실패 시:
        ├── on_error="stop" + stop_on_first_failure=True → break
        ├── on_error="skip"     → continue
        └── on_error="continue" → 실패 기록 후 계속
```

---

## 반환값

```python
{
    "status": "COMPLETED" | "FAILED",
    "result": {
        "pipeline_name": str,
        "total_steps":   int,
        "executed":      int,         # 실제 실행된 단계 수
        "failed_steps":  list[str],   # 실패한 단계 이름 목록
        "step_results":  [
            {"step": "mart_build", "status": "COMPLETED", "result": {...}, ...},
            ...
        ],
    },
    "message": "파이프라인 완료  5/5단계",
    "job_id":  str,
    "elapsed_sec": float,
}
```

- `failed_steps`가 비어 있으면 전체 `COMPLETED`, 하나라도 있으면 `FAILED`

---

## `_build_executor(executor_type, config)` → `BaseExecutor`

```python
if executor_type not in EXECUTOR_REGISTRY:
    raise ExecutorException(f"등록되지 않은 executor: {executor_type}")

module_path, class_name = EXECUTOR_REGISTRY[executor_type]
module = importlib.import_module(module_path)
cls    = getattr(module, class_name)
return cls(config=config, db_session=self.db_session, file_root_dir=str(self.file_root))
```

---

## 단계 간 결과 전달 (`input_from`)

`context` 딕셔너리로 단계별 결과(`result` 딕셔너리)를 관리한다.

```python
# step A 완료 후
context["mart_build"] = step_result["result"]
# → {"saved_paths": {...}, "shape": [100000, 45], ...}

# step B에서 input_from="mart_build" 설정 시
step_config.update(context["mart_build"])
# → mart_build의 saved_paths, shape 등이 B의 config에 자동 주입
```

---

## 오류 처리 정책

| `on_error` | `stop_on_first_failure` | 동작 |
|------------|------------------------|------|
| `"stop"` | `True` (기본) | 즉시 파이프라인 중단 |
| `"stop"` | `False` | 실패 기록 후 계속 |
| `"skip"` | 무관 | 해당 단계 건너뜀 |
| `"continue"` | 무관 | 실패 기록 후 계속 |

---

## 사용 예시

```python
from executors.ml.process_executor import ProcessExecutor

config = {
    "pipeline_name": "loan_credit_v1",
    "stop_on_first_failure": True,
    "pipeline": [
        {
            "name":     "make_mart",
            "executor": "mart",
            "config": {
                "source_query": "SELECT * FROM loan_raw WHERE yymm='202312'",
                "target_id":    "loan_mart_202312",
                "split":        {"train": 0.7, "valid": 0.15, "test": 0.15},
            },
        },
        {
            "name":       "eda",
            "executor":   "data_analysis",
            "input_from": "make_mart",
            "config": {
                "source_path": "mart/loan_mart_202312_train.parquet",
                "output_id":   "loan_eda_202312",
                "target_col":  "default",
            },
        },
        {
            "name":     "train",
            "executor": "python_model",
            "config": {
                "model_type":  "lightgbm",
                "train_path":  "mart/loan_mart_202312_train.parquet",
                "valid_path":  "mart/loan_mart_202312_valid.parquet",
                "target_col":  "default",
                "model_id":    "lgbm_loan_v1",
                "model_params": {"n_estimators": 500, "learning_rate": 0.05},
            },
        },
        {
            "name":     "predict",
            "executor": "predict",
            "config": {
                "model_id":   "lgbm_loan_v1",
                "input_path": "mart/loan_mart_202312_test.parquet",
                "output_id":  "loan_pred_202312",
            },
        },
        {
            "name":     "export",
            "executor": "export",
            "config": {
                "source_path": "predict/loan_pred_202312_result.parquet",
                "export_type": "db",
                "output_id":   "loan_pred_202312",
                "table_name":  "ML_LOAN_SCORE",
                "write_mode":  "upsert",
                "key_cols":    ["cust_id"],
            },
        },
    ],
}

result = ProcessExecutor(
    config=config,
    db_session=session,
    file_root_dir="/data"
).run()
```
