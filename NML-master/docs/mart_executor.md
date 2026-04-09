# mart_executor.py — 데이터 마트 생성

**파일:** `executors/ml/mart_executor.py`  
**클래스:** `MartExecutor(BaseExecutor)`

## 개요

원천 DB 테이블 또는 파일에서 데이터를 읽어 feature engineering을 수행하고,  
분석·모델 입력용 마트 데이터셋을 생성하여 파일 서버 또는 DB에 저장한다.

```
원천 데이터 (DB SQL 또는 파일)
    ↓ _load_source()
    ↓ _basic_preprocess()  ← 타입변환 / 결측처리 / 이상값클리핑
    ↓ _create_derived_features()  ← feature_rules 적용
    ↓ _split_dataframe()  ← train/valid/test 분할 (선택)
    ↓ parquet 저장 + meta JSON 저장
```

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `source_query` | ✅* | `str` | 원천 SQL (db_session 필요) |
| `source_path` | ✅* | `str` | 원천 파일 경로 (source_query 없을 때) |
| `target_id` | ✅ | `str` | 생성할 마트 식별자 (파일명·DB 키) |
| `target_path` | ❌ | `str` | 저장 경로 템플릿 (기본: `mart/{target_id}.parquet`) |
| `feature_rules` | ❌ | `list` | 파생 변수 생성 규칙 목록 |
| `split` | ❌ | `dict` | train/valid/test 분할 비율 예: `{"train": 0.7, "valid": 0.15, "test": 0.15}` |
| `target_col` | ❌ | `str` | 타깃 컬럼명 (층화 분할 시 활용) |
| `clip_columns` | ❌ | `list` | 이상값 클리핑 대상 컬럼 (기본: 전체 수치형) |

\* `source_query` 또는 `source_path` 중 하나 필수

---

## 내부 메서드

### `_load_source(cfg)` → `pd.DataFrame`

```python
# DB 조회 (SQLAlchemy)
if "source_query" in cfg and self.db_session is not None:
    return pd.read_sql(cfg["source_query"], self.db_session.bind)

# 파일 로드 (parquet / csv)
elif "source_path" in cfg:
    return self._load_dataframe(cfg["source_path"])

# 둘 다 없으면 ExecutorException
```

---

### `_basic_preprocess(df, cfg)` → `pd.DataFrame`

3단계 기본 전처리를 순서대로 적용한다.

**① 타입 변환**
```
object 컬럼 중 (nunique / len) < 0.5 → category 타입 자동 변환
```

**② 결측 처리**
```
수치형 컬럼 결측 → 해당 컬럼의 중앙값(median)으로 대체
```

**③ 이상값 클리핑 (IQR 3배)**
```python
clip_cols = cfg.get("clip_columns", list(num_cols))
# 기본: 전체 수치형 컬럼
# 1퍼센타일 미만 → 1퍼센타일 클리핑
# 99퍼센타일 초과 → 99퍼센타일 클리핑
```

---

### `_create_derived_features(df, feature_rules)` → `pd.DataFrame`

`feature_rules` 리스트의 각 규칙을 `df.eval()`로 적용하여 새 컬럼을 생성한다.

```python
feature_rules = [
    {"name": "debt_ratio", "expr": "total_debt / (income + 1)"},
    {"name": "util_rate",  "expr": "used_limit / (credit_limit + 1)"},
]
# df.eval("total_debt / (income + 1)")로 실행 후 df["debt_ratio"]에 저장
```

---

### `_split_dataframe(df, split_cfg, target_col)` → `dict[str, pd.DataFrame]`

랜덤 셔플 후 비율대로 분할한다.

```python
split = {"train": 0.7, "valid": 0.15, "test": 0.15}
# → {"train": df_train, "valid": df_valid, "test": df_test}
```

---

## execute() 진행률

| 단계 | progress |
|------|----------|
| 원천 로드 완료 | 20% |
| 기본 전처리 완료 | 50% |
| 파생 변수 생성 완료 | 70% |
| 파일 저장 완료 | 90% |

---

## 반환값

```python
{
    "status": "COMPLETED",
    "result": {
        "target_id":   "loan_mart_202312",
        "shape":       [100000, 45],
        "columns":     ["cust_id", "income", ...],
        "saved_paths": {
            "train": "/data/mart/loan_mart_202312_train.parquet",
            "valid": "/data/mart/loan_mart_202312_valid.parquet",
            "test":  "/data/mart/loan_mart_202312_test.parquet",
        },
        "dtypes": {"cust_id": "int64", "income": "float64", ...},
    },
    "message": "마트 생성 완료: loan_mart_202312  shape=(100000, 45)",
    "job_id":  str,
    "elapsed_sec": float,
}
```

- `split` 미지정 시 `saved_paths = {"full": "..."}` 형태

---

## 출력 파일

| 파일 | 경로 |
|------|------|
| 전체 마트 | `mart/{target_id}.parquet` |
| train 분할 | `mart/{target_id}_train.parquet` |
| valid 분할 | `mart/{target_id}_valid.parquet` |
| test 분할 | `mart/{target_id}_test.parquet` |
| 메타 정보 | `mart/{target_id}_meta.json` |

---

## 사용 예시

```python
config = {
    "job_id":       "mart_job_001",
    "source_query": "SELECT * FROM customer_txn WHERE yymm='202312'",
    "target_id":    "retail_mart_v2",
    "target_col":   "default_yn",
    "feature_rules": [
        {"name": "debt_ratio", "expr": "total_debt / (income + 1)"},
        {"name": "util_rate",  "expr": "used_limit / (credit_limit + 1)"},
    ],
    "split": {"train": 0.7, "valid": 0.15, "test": 0.15},
}

from executors.ml.mart_executor import MartExecutor
result = MartExecutor(config=config, db_session=session).run()
```
