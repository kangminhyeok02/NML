# mart_executor.py

모델링용 데이터 마트(mart) 생성 및 적재 실행기.

원천 DB 테이블 또는 파일에서 데이터를 읽어 feature engineering을 수행하고,  
분석/모델 입력용 마트 데이터셋을 생성하여 파일 서버 또는 DB에 저장한다.

---

## 클래스

### `MartExecutor(BaseExecutor)`

데이터 마트 생성 executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `source_query` | `str` | 원천 데이터 SQL (`db_session` 필요). `source_path`와 택일 |
| `source_path` | `str` | 원천 파일 상대 경로. `source_query` 없을 때 사용 |
| `target_id` | `str` | 생성할 마트 식별자 (파일명 / DB 키) |
| `target_path` | `str` | 저장 경로 (예: `"mart/{target_id}.parquet"`) |

#### config 선택 키

| 키 | 타입 | 설명 |
|---|---|---|
| `feature_rules` | `list` | 파생 변수 생성 규칙 목록 |
| `split` | `dict` | `{"train": 0.7, "valid": 0.15, "test": 0.15}` |
| `target_col` | `str` | 타깃 컬럼명 |
| `clip_columns` | `list` | 이상값 클리핑 대상 컬럼 목록 (기본: 전체 수치형) |

---

### `execute() → dict`

마트 생성 전체 파이프라인을 실행한다.

**실행 순서**
1. 원천 데이터 로드 (`_load_source`)
2. 기본 전처리 (`_basic_preprocess`)
3. 파생 변수 생성 (`_create_derived_features`, `feature_rules` 있을 때)
4. 학습/검증/예측 분할 (`_split_dataframe`, `split` 있을 때)
5. 마트 파일 저장 및 메타 JSON 저장

**반환값**
```python
{
    "status": "COMPLETED",
    "result": {
        "target_id":   str,
        "shape":       [rows, cols],
        "columns":     list[str],
        "saved_paths": {"full": str} | {"train": str, "valid": str, "test": str},
        "dtypes":      {col: dtype},
    },
    "message": str,
}
```

---

### `_load_source(cfg) → pd.DataFrame`

원천 데이터를 로드한다.

- `source_query`가 있으면 `pd.read_sql()`로 DB 조회
- `source_path`가 있으면 `_load_dataframe()`으로 파일 로드
- 둘 다 없으면 `ExecutorException` 발생

### `_basic_preprocess(df, cfg) → pd.DataFrame`

기본 전처리 3단계를 수행한다.

1. **카테고리 변환**: object 타입 컬럼 중 고유값 비율 < 50%이면 `category` 타입으로 변환
2. **결측 처리**: 수치형 컬럼의 결측값을 **중앙값**으로 대체
3. **이상값 클리핑**: `clip_columns` 지정 컬럼에 대해 1% ~ 99% 분위수로 클리핑

### `_create_derived_features(df, rules) → pd.DataFrame`

`rules` 목록에 따라 파생 변수를 생성한다.

**rules 예시**
```python
[
    {"name": "ratio_a_b", "expr": "col_a / (col_b + 1)"},
    {"name": "log_c",     "expr": "log(col_c + 1)"}
]
```

- `df.eval(rule["expr"])`로 수식을 평가
- 생성 실패 시 경고 로그 출력 후 건너뜀

### `_split_dataframe(df, split_cfg, target_col) → dict[str, pd.DataFrame]`

비율에 따라 train / valid / test 데이터셋을 분할한다.

- 먼저 전체 데이터를 셔플 (`random_state=42`)
- `split_cfg["train"]`, `split_cfg["valid"]` 비율로 인덱스를 계산
- 나머지가 test 세트

---

## 모듈 레벨 함수

### `_make_engine(db_info) → Engine`

PostgreSQL SQLAlchemy engine을 생성한다 (`postgresql+psycopg2`).

### `get_common_code_index(service_db_info, auth_key, code, code_role) → dict`

`common_code` 테이블에서 `auth_key`, `code`, `code_role`로 단일 코드 인덱스를 조회한다.  
결과가 없으면 `-1` 반환.

```python
{"result": int, "code": str, "code_role": str}
```

### `get_common_code_indexes(service_db_info, auth_key, code) → dict`

`code`에 해당하는 모든 `code_role`과 인덱스 목록을 조회한다.

```python
{"result": [{"code_role": str, "code_index": int}, ...]}
```
