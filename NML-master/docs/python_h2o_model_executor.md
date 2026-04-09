# python_h2o_model_executor.py — Python + H2O 통합 추론

## 개요

Python 기반 전처리/후처리와 H2O MOJO 모델 추론을 결합하는 executor.  
H2O로 학습된 모델을 운영 파이프라인에서 Python 코드로 제어해야 할 때 사용한다.

```
입력 데이터
    ↓ Python 전처리 (fillna, clip, log1p, eval 등)
    ↓ H2O MOJO 추론 (서버 불필요) 또는 H2O 서버 Live 추론
    ↓ Python 후처리 (scale, grade, round 등)
    ↓ 결과 저장
```

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `model_id` | ✅ | `str` | H2O 모델 식별자 (메타 JSON 참조) |
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

| `type` | 필수 파라미터 | 설명 |
|--------|-------------|------|
| `fillna` | `columns`, `value` | 결측값을 지정값으로 대체 |
| `clip` | `columns`, `lower`, `upper` | 값 범위 클리핑 |
| `log1p` | `columns` | `log(x+1)` 변환 (음수는 0으로 클리핑) |
| `eval` | `name`, `expr` | `df.eval()`로 파생변수 생성 |
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
| `scale` | `col`, `method` (`minmax`\|`standard`) | 점수 정규화 |
| `grade` | `col`, `map` | 점수 구간 → 등급 문자열 매핑 |
| `round` | `col`, `decimals` | 소수점 반올림 |

```python
postprocess_steps = [
    {"type": "scale", "method": "minmax", "col": "score"},
    {"type": "grade", "col": "score",
     "map": {"A": [0.8, 1.0], "B": [0.6, 0.8], "C": [0.0, 0.6]}},
    {"type": "round", "col": "score", "decimals": 4},
]
```

---

## H2O 추론 방식

### `_predict_mojo(meta, X, cfg)` — MOJO 추론 (권장)

H2O 서버 없이 MOJO 파일만으로 추론. `h2o.import_mojo()`를 사용한다.

```python
model = h2o.import_mojo(mojo_path)         # MOJO 로드
h2oframe = h2o.H2OFrame(X)
preds = model.predict(h2oframe).as_data_frame()
scores = preds.iloc[:, -1].values          # 마지막 컬럼 = Bad 확률
```

### `_predict_h2o_live(meta, X, cfg)` — Live 서버 추론

H2O 클러스터에 올라간 모델을 실시간으로 호출한다.

```python
model = h2o.get_model(meta["h2o_model_id"])
```

---

## 실행 흐름

```
1. models/{model_id}_meta.json 로드
2. input_path 데이터 로드                            [progress 15%]
3. _apply_preprocess() — Python 전처리               [progress 35%]
4. H2O 추론 (_predict_mojo 또는 _predict_h2o_live)  [progress 70%]
5. _apply_postprocess() — Python 후처리              [progress 90%]
6. predict/{output_id}_py_h2o.parquet 저장
```

---

## 출력 결과

**저장 경로:** `predict/{output_id}_py_h2o.parquet`

**반환 요약:**
```python
{
    "output_id":   "credit_run_001",
    "model_id":    "gbm_credit_v1",
    "total_rows":  50000,
    "output_path": "predict/credit_run_001_py_h2o.parquet",
    "score_stats": {
        "mean": 0.2341,
        "min":  0.0012,
        "max":  0.9871,
        "std":  0.1823
    }
}
```

---

## H2OModelExecutor와의 차이

| 항목 | `H2OModelExecutor` | `PythonH2OModelExecutor` |
|------|-------------------|--------------------------|
| 목적 | 모델 **학습** | 모델 **추론** |
| H2O 서버 | 학습 중 필수 | MOJO 사용 시 불필요 |
| 전처리/후처리 | 없음 | Python으로 완전 제어 |
| 주 사용 시점 | 개발 단계 | 운영 단계 |
