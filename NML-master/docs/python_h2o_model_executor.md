# python_h2o_model_executor.py — Python + H2O 통합 추론

**파일:** `executors/ml/python_h2o_model_executor.py`  
**클래스:** `PythonH2OModelExecutor(BaseExecutor)`

## 개요

Python 기반 전처리/후처리와 H2O MOJO 모델 추론을 결합하는 executor.  
H2O로 학습된 모델을 운영 파이프라인에서 Python 코드로 제어해야 할 때 사용한다.

```
입력 데이터 (.parquet)
    ↓ Python 전처리 (preprocess_steps)
    ↓ H2O MOJO 추론 (use_mojo=True) 또는 H2O 서버 Live 추론 (use_mojo=False)
    ↓ Python 후처리 (postprocess_steps)
    ↓ 결과 저장 → predict/{output_id}_py_h2o.parquet
```

**대표 시나리오:**
1. H2OModelExecutor로 GBM 학습 → MOJO 저장
2. 운영계에서 이 executor로 Python 전처리 + MOJO 추론 + 등급화 후처리

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `model_id` | ✅ | `str` | H2O 모델 식별자 (`models/{model_id}_meta.json` 참조) |
| `input_path` | ✅ | `str` | 입력 데이터 경로 (.parquet) |
| `output_id` | ✅ | `str` | 결과 저장 식별자 |
| `preprocess_steps` | ❌ | `list` | Python 전처리 스텝 목록 |
| `postprocess_steps` | ❌ | `list` | Python 후처리 스텝 목록 |
| `score_col` | ❌ | `str` | 점수 컬럼명 (기본: `"score"`) |
| `h2o_ip` | ❌ | `str` | H2O 서버 IP (기본: `localhost`) |
| `h2o_port` | ❌ | `int` | H2O 서버 포트 (기본: `54321`) |
| `use_mojo` | ❌ | `bool` | MOJO 사용 여부 (기본: `True`) |

---

## 전처리 스텝 (`preprocess_steps`)

각 스텝은 `{"type": ..., ...}` 형식의 딕셔너리.

| `type` | 파라미터 | 설명 |
|--------|---------|------|
| `fillna` | `columns`, `value` | 결측값을 지정값으로 대체 |
| `clip` | `columns`, `lower`, `upper` | 값 범위 클리핑 |
| `log1p` | `columns` | `log(x+1)` 변환 (음수는 0으로 클리핑) |
| `eval` | `name`, `expr` | `df.eval(expr)`로 파생변수 생성 |
| `drop` | `columns` | 컬럼 제거 |

```python
preprocess_steps = [
    {"type": "fillna", "columns": ["income"], "value": 0},
    {"type": "clip",   "columns": ["age"],    "lower": 18, "upper": 80},
    {"type": "log1p",  "columns": ["loan_amt"]},
    {"type": "eval",   "name": "dti", "expr": "debt / (income + 1)"},
]
```

---

## 후처리 스텝 (`postprocess_steps`)

| `type` | 파라미터 | 설명 |
|--------|---------|------|
| `scale` | `factor` | `score * factor` |
| `round` | `decimals` | `round(score, decimals)` |
| `grade` | `grade_map` | 점수 → 등급 컬럼 추가 |
| `clip_score` | `lower`, `upper` | 점수 범위 클리핑 |

---

## H2O 추론 방식

### `use_mojo=True` (기본) — MOJO 파일 추론

H2O 서버 없이 MOJO 파일만으로 추론 (운영 환경 권장).

```python
import h2o
mojo_path = meta["mojo_path"]    # models/{model_id}/model.zip
model = h2o.import_mojo(mojo_path)
h2o_frame = h2o.H2OFrame(X)
preds = model.predict(h2o_frame).as_data_frame()
scores = preds.iloc[:, -1].values    # Bad 확률 컬럼
```

### `use_mojo=False` — H2O 서버 Live 추론

h2o_ip/h2o_port의 H2O 클러스터에 직접 연결하여 학습된 모델로 추론.

---

## execute() 진행률

| 단계 | progress |
|------|----------|
| 메타·데이터 로드 완료 | 15% |
| Python 전처리 완료 | 35% |
| H2O 추론 완료 | 70% |
| Python 후처리·저장 완료 | 90% |

---

## 반환값

```python
{
    "status": "COMPLETED",
    "result": {
        "output_id":   "loan_pred_202312",
        "model_id":    "gbm_loan_v1",
        "total_rows":  50000,
        "output_path": "predict/loan_pred_202312_py_h2o.parquet",
        "score_stats": {"mean": 0.312, "std": 0.18, "p25": 0.15, "p50": 0.29, "p75": 0.46},
    },
    "message": "Python+H2O 추론 완료  50,000건",
    "job_id":  str,
    "elapsed_sec": float,
}
```

---

## 출력 파일

| 파일 | 경로 |
|------|------|
| 추론 결과 | `predict/{output_id}_py_h2o.parquet` |

---

## 사용 예시

```python
config = {
    "job_id":     "py_h2o_pred_001",
    "model_id":   "gbm_loan_v1",
    "input_path": "mart/new_applicants.parquet",
    "output_id":  "new_applicants_score",
    "score_col":  "ml_score",
    "use_mojo":   True,
    "preprocess_steps": [
        {"type": "fillna", "columns": ["income", "debt"], "value": 0},
        {"type": "eval", "name": "dti", "expr": "debt / (income + 1)"},
    ],
    "postprocess_steps": [
        {"type": "scale", "factor": 1000},
        {"type": "round", "decimals": 0},
    ],
}

from executors.ml.python_h2o_model_executor import PythonH2OModelExecutor
result = PythonH2OModelExecutor(config=config).run()
```
