# mart_executor.py — 데이터 마트 생성

## 개요

원천 DB 테이블 또는 파일에서 데이터를 읽어 feature engineering을 수행하고,  
모델 입력용 마트 데이터셋을 생성하여 저장하는 executor.

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `source_query` | ✅* | `str` | 원천 SQL (db_session 필요) |
| `source_path` | ✅* | `str` | 원천 파일 경로 (source_query 없을 때) |
| `target_id` | ✅ | `str` | 생성할 마트 식별자 |
| `target_path` | ❌ | `str` | 저장 경로 템플릿 (기본: `mart/{target_id}.parquet`) |
| `feature_rules` | ❌ | `list` | 파생 변수 생성 규칙 |
| `split` | ❌ | `dict` | train/valid/test 분할 비율 |
| `target_col` | ❌ | `str` | 타깃 컬럼명 |
| `clip_columns` | ❌ | `list` | 이상값 클리핑 대상 컬럼 (기본: 전체 수치형) |

\* `source_query` 또는 `source_path` 중 하나 필수

---

## 메서드 상세

### `_load_source(cfg)` → `pd.DataFrame`

원천 데이터를 로드한다.

```python
# DB 조회 (SQLAlchemy)
if "source_query" in cfg:
    pd.read_sql(cfg["source_query"], db_session.bind)

# 파일 로드 (parquet / csv)
elif "source_path" in cfg:
    self._load_dataframe(cfg["source_path"])
```

---

### `_basic_preprocess(df, cfg)` → `pd.DataFrame`

3단계 기본 전처리를 순서대로 적용한다.

**① 타입 변환**
```
object 컬럼 중 카디널리티 비율 < 50% → category 타입 변환
```

**② 결측 처리**
```
수치형 컬럼 결측 → 해당 컬럼의 중앙값(median)으로 대체
```

**③ 이상값 클리핑**
```
1% ~ 99% 분위수 범위로 클리핑 (극단값 제거)
```

---

### `_create_derived_features(df, rules)` → `pd.DataFrame`

`df.eval()`을 사용해 파생 변수를 생성한다.

```python
# rules 예시
feature_rules = [
    {"name": "debt_ratio",  "expr": "total_debt / (income + 1)"},
    {"name": "log_income",  "expr": "log(income + 1)"},
    {"name": "age_sq",      "expr": "age ** 2"},
]
```

- 개별 rule 실패 시 경고 로그만 남기고 계속 진행 (전체 실패 없음)

---

### `_split_dataframe(df, split_cfg, target_col)` → `dict`

데이터를 랜덤 셔플(`random_state=42`) 후 비율에 따라 분할한다.

```python
split_cfg = {"train": 0.7, "valid": 0.15, "test": 0.15}
# → train: 0~70%, valid: 70~85%, test: 85~100%
```

반환: `{"train": df_train, "valid": df_valid, "test": df_test}`

---

## 실행 흐름

```
1. 원천 데이터 로드 (_load_source)             [progress 20%]
2. 기본 전처리 (_basic_preprocess)             [progress 50%]
   - 타입 변환 → 결측 처리 → 클리핑
3. 파생 변수 생성 (_create_derived_features)   [progress 70%]
4. 분할 저장 또는 전체 저장                    [progress 90%]
   - split 설정 시: train/valid/test 분리 저장
   - 미설정 시:     mart/{target_id}.parquet 전체 저장
5. 메타 JSON 저장 (mart/{target_id}_meta.json)
```

---

## 출력 결과

**데이터 파일:**
```
mart/{target_id}.parquet              # split 미설정 시
mart/{target_id}_train.parquet        # split 설정 시
mart/{target_id}_valid.parquet
mart/{target_id}_test.parquet
```

**메타 파일:** `mart/{target_id}_meta.json`
```json
{
  "target_id":   "loan_mart",
  "shape":       [100000, 45],
  "columns":     ["age", "income", ...],
  "saved_paths": {"train": "/data/mart/loan_mart_train.parquet", ...},
  "dtypes":      {"age": "int64", "income": "float64", ...}
}
```

---

## ProcessExecutor와의 연계

`input_from`으로 다음 단계에 메타 정보를 자동 전달한다.

```python
# mart 단계 결과 result
{
    "target_id":   "loan_mart",
    "saved_paths": {"train": "...", "valid": "..."},
    ...
}

# 다음 단계(python_model)의 config에 자동 병합됨
# → train_path 등을 명시적으로 지정하지 않아도 됨
```
