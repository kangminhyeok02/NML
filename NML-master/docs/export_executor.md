# export_executor.py — 결과 외부 내보내기

**파일:** `executors/ml/export_executor.py`  
**클래스:** `ExportExecutor(BaseExecutor)`

## 개요

분석/예측 결과를 외부로 내보내는 파이프라인 최종 executor.  
파일 저장, DB 적재, API 전송 세 가지 대상을 지원한다.

```
원본 데이터 로드 (.parquet)
    ↓ 컬럼 선택 (columns) + 이름 변경 (rename_map)
    ↓ export_type에 따라 실행
      ├── file → CSV / Excel / JSON / Parquet 저장
      ├── db   → INSERT / UPSERT / TRUNCATE-INSERT
      └── api  → REST POST (배치 전송)
    ↓ exports/{output_id}_summary.json 저장
```

---

## config 파라미터

### 공통

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `source_path` | ✅ | `str` | 내보낼 원본 데이터 경로 (.parquet) |
| `export_type` | ✅ | `str` | `"file"` \| `"db"` \| `"api"` |
| `output_id` | ✅ | `str` | 출력 식별자 |
| `columns` | ❌ | `list` | 내보낼 컬럼 선택 (없으면 전체) |
| `rename_map` | ❌ | `dict` | 컬럼 이름 변경 `{"original": "new_name"}` |
| `date_format` | ❌ | `str` | 날짜 포맷 (기본: `"%Y-%m-%d"`) |

### `file` 모드

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `file_format` | ❌ | `str` | `"csv"` \| `"excel"` \| `"json"` \| `"parquet"` (기본: `"csv"`) |
| `output_path` | ❌ | `str` | 저장 경로 (기본: `exports/{output_id}.{format}`) |

### `db` 모드

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `table_name` | ✅ | `str` | 대상 테이블명 |
| `write_mode` | ✅ | `str` | `"append"` \| `"replace"` \| `"upsert"` |
| `key_cols` | upsert 필수 | `list` | upsert 키 컬럼 목록 |

### `api` 모드

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `endpoint` | ✅ | `str` | POST 대상 URL |
| `batch_size` | ❌ | `int` | 배치 크기 (기본: `1000`) |
| `headers` | ❌ | `dict` | 요청 헤더 |

---

## 내부 메서드

### `_export_to_file(df, cfg)` → `dict`

```python
if file_format == "csv":
    df.to_csv(full_path, index=False, date_format=date_fmt)
elif file_format == "excel":
    df.to_excel(full_path, index=False, engine="openpyxl")
elif file_format == "json":
    df.to_json(full_path, orient="records", force_ascii=False, indent=2)
elif file_format == "parquet":
    df.to_parquet(full_path, index=False)
```

---

### `_export_to_db(df, cfg)` → `dict`

```python
# append: DataFrame.to_sql(if_exists="append")
# replace: DataFrame.to_sql(if_exists="replace")
# upsert: DELETE WHERE key IN (...) → INSERT (MariaDB/MySQL 호환)
```

upsert는 `key_cols` 기준으로 기존 행 삭제 후 신규 INSERT한다.

---

### `_export_to_api(df, cfg)` → `dict`

```python
for batch in chunks(df, batch_size):
    payload = batch.to_dict(orient="records")
    resp = requests.post(endpoint, json=payload, headers=headers)
    # 실패 시 ExecutorException
```

기본 `batch_size=1000`건씩 분할 POST.

---

## execute() 진행률

| 단계 | progress |
|------|----------|
| 데이터 로드·컬럼처리 완료 | 40% |
| export 실행 완료 | 90% |
| 요약 저장 완료 | 95% |

---

## 반환값

```python
{
    "status": "COMPLETED",
    "result": {
        "output_id":     "loan_export_202312",
        "export_type":   "db",
        "exported_rows": 50000,
        "exported_cols": 8,
        "table_name":    "ML_LOAN_DECISION",
        "write_mode":    "upsert",
    },
    "message": "Export 완료  type=db  50,000행",
    "job_id":  str,
    "elapsed_sec": float,
}
```

---

## 출력 파일

| 파일 | 경로 |
|------|------|
| 내보낸 파일 (file 모드) | `exports/{output_id}.{format}` |
| 요약 JSON | `exports/{output_id}_summary.json` |

---

## 사용 예시

```python
# DB upsert 예시
config = {
    "job_id":      "export_001",
    "source_path": "strategy/loan_stg_202312_result.parquet",
    "export_type": "db",
    "output_id":   "loan_export_202312",
    "table_name":  "ML_LOAN_DECISION",
    "write_mode":  "upsert",
    "key_cols":    ["cust_id", "base_dt"],
    "columns":     ["cust_id", "base_dt", "ml_score", "grade", "decision"],
    "rename_map":  {"ml_score": "ML_SCORE", "grade": "ML_GRADE"},
}

from executors.ml.export_executor import ExportExecutor
result = ExportExecutor(config=config, db_session=session).run()

# CSV 파일 저장 예시
config_csv = {
    "job_id":      "export_csv_001",
    "source_path": "predict/loan_pred_202312_result.parquet",
    "export_type": "file",
    "output_id":   "loan_pred_csv",
    "file_format": "csv",
    "output_path": "exports/loan_pred_202312.csv",
}
```
