# python_h2o_model_executor.md

Python 서비스 파이프라인과 H2O 모델을 통합하는 실행기.

`h2o_model_executor.py`가 순수 H2O 로직에 집중한다면,  
이 executor는 **Python 기반 전처리/후처리와 H2O 모델 추론을 결합**한다.

**대표 시나리오**
1. Python으로 feature engineering 수행
2. H2O MOJO 모델로 점수 산출
3. Python으로 결과 후처리 (스케일링, 등급화, 마스킹)

---

## 클래스

### `PythonH2OModelExecutor(BaseExecutor)`

Python + H2O 통합 executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `model_id` | `str` | H2O 모델 식별자 (MOJO 경로 포함된 메타 기준) |
| `input_path` | `str` | 입력 데이터 경로 (`.parquet`) |
| `output_id` | `str` | 결과 저장 식별자 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|---|---|---|
| `preprocess_steps` | `[]` | Python 전처리 스텝 목록 |
| `postprocess_steps` | `[]` | Python 후처리 스텝 목록 |
| `score_col` | `"score"` | 점수 컬럼명 |
| `h2o_ip` | `"localhost"` | H2O 서버 IP |
| `h2o_port` | `54321` | H2O 서버 포트 |
| `use_mojo` | `True` | MOJO 사용 여부. `False`이면 live H2O 서버 사용 |

---

### `execute() → dict`

Python + H2O 통합 파이프라인을 실행한다.

**실행 순서**
1. `models/{model_id}_meta.json` 로드
2. 입력 데이터 로드
3. Python 전처리 (`_apply_preprocess`)
4. H2O 추론 (`_predict_mojo` 또는 `_predict_h2o_live`)
5. Python 후처리 (`_apply_postprocess`)
6. 결과를 `predict/{output_id}_py_h2o.parquet`에 저장

**반환값**
```python
{
    "status": "COMPLETED",
    "result": {
        "output_id":   str,
        "model_id":    str,
        "total_rows":  int,
        "output_path": str,
        "score_stats": {mean, std, min, p25, p50, p75, max},
    },
    "message": str,
}
```

---

### `_apply_preprocess(df, steps) → pd.DataFrame`

전처리 스텝 목록을 순서대로 적용한다.

**지원 step type**

| `type` | 필요 키 | 동작 |
|---|---|---|
| `"fillna"` | `columns`, `value` | 결측값을 `value`로 대체 |
| `"clip"` | `columns`, `lower`, `upper` | 범위 클리핑 |
| `"log1p"` | `columns` | `log(1 + max(x, 0))` 변환 |
| `"eval"` | `name`, `expr` | `df.eval(expr)`로 파생 컬럼 생성 |
| `"drop"` | `columns` | 컬럼 삭제 |

스텝 실패 시 경고 로그 출력 후 건너뜀.

**스텝 예시**
```python
[
    {"type": "fillna", "columns": ["col_a"], "value": 0},
    {"type": "clip",   "columns": ["col_b"], "lower": 0, "upper": 100},
    {"type": "log1p",  "columns": ["col_c"]},
    {"type": "eval",   "name": "ratio", "expr": "col_a / (col_b + 1)"},
    {"type": "drop",   "columns": ["tmp_col"]},
]
```

---

### `_apply_postprocess(df, steps, score_col) → pd.DataFrame`

후처리 스텝 목록을 순서대로 적용한다.

**지원 step type**

| `type` | 필요 키 | 동작 |
|---|---|---|
| `"scale"` | `col`, `method` | 점수 정규화. `"minmax"` 또는 `"standard"` |
| `"grade"` | `col`, `map` | 점수 → 등급 변환. `grade` 컬럼 추가 |
| `"round"` | `col`, `decimals` | 소수점 반올림 |

**스텝 예시**
```python
[
    {"type": "scale",  "method": "minmax",               "col": "score"},
    {"type": "grade",  "col": "score", "map": {"A": [800, 1000], "B": [600, 800]}},
    {"type": "round",  "col": "score", "decimals": 0},
]
```

**`"scale"` 동작 상세**
- `"minmax"`: `(x - min) / (max - min + 1e-9)`
- `"standard"`: `(x - mean) / (std + 1e-9)`

**`"grade"` 동작 상세**
- `grade_map`의 `[lo, hi)` 구간에 해당하는 등급 부여
- 해당 구간 없으면 `"UNKNOWN"`

---

### `_predict_mojo(meta, X, cfg) → np.ndarray`

H2O MOJO 파일을 로드하여 점수를 산출한다.

1. `h2o.init(ip, port)` 연결
2. `h2o.import_mojo(mojo_path)` 로드
3. `h2o.H2OFrame(X)`으로 변환 후 `model.predict()` 호출
4. 예측 결과의 마지막 컬럼(양성 확률)을 numpy 배열로 반환

`mojo_path` 기본값: `models/{model_id}/model.zip`

---

### `_predict_h2o_live(meta, X, cfg) → np.ndarray`

실행 중인 H2O 서버의 모델로 점수를 산출한다.

- `h2o.get_model(meta["h2o_model_id"])`로 이미 학습된 모델을 직접 참조
- `H2OFrame` 변환 후 예측, 마지막 컬럼 반환

---

## 모듈 레벨 함수

### `_stats(s) → dict`

Series의 기술통계 7개 지표를 반환한다.

```python
{mean, std, min, p25, p50, p75, max}
```
