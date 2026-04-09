# export_executor.py

분석/예측 결과를 외부로 내보내는 실행기.

파일 export, DB 적재, API 전송의 세 가지 출력 대상을 지원한다.

---

## 클래스

### `ExportExecutor(BaseExecutor)`

결과 export executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `source_path` | `str` | 내보낼 원본 데이터 경로 (`.parquet`) |
| `export_type` | `str` | `"file"` \| `"db"` \| `"api"` |
| `output_id` | `str` | 출력 식별자 |

**file 모드 추가 키**

| 키 | 기본값 | 설명 |
|---|---|---|
| `file_format` | `"csv"` | `"csv"` \| `"excel"` \| `"json"` \| `"parquet"` |
| `output_path` | 자동 생성 | 저장 경로 (FILE_ROOT_DIR 기준) |

**db 모드 추가 키**

| 키 | 기본값 | 설명 |
|---|---|---|
| `table_name` | - | 대상 테이블명 |
| `write_mode` | `"append"` | `"append"` \| `"replace"` \| `"upsert"` |
| `key_cols` | - | upsert 키 컬럼 목록 (upsert 모드 필수) |

**api 모드 추가 키**

| 키 | 기본값 | 설명 |
|---|---|---|
| `endpoint` | - | POST 대상 URL |
| `batch_size` | `1000` | 배치 크기 |
| `headers` | `{"Content-Type": "application/json"}` | 요청 헤더 |

#### config 선택 키

| 키 | 타입 | 설명 |
|---|---|---|
| `columns` | `list` | 내보낼 컬럼 선택. 없으면 전체 컬럼 |
| `rename_map` | `dict` | 컬럼 이름 변경 `{"original": "new_name"}` |
| `date_format` | `str` | 날짜 포맷 (기본: `"%Y-%m-%d"`) |

---

### `execute() → dict`

데이터를 로드하고 지정 대상으로 내보낸다.

**실행 순서**
1. 원본 데이터 로드
2. `columns` 선택 및 `rename_map` 적용
3. `export_type`에 따라 export 실행
4. 결과 요약을 `exports/{output_id}_summary.json`에 저장

**반환값**
```python
{
    "status": "COMPLETED",
    "result": {
        "output_id":     str,
        "export_type":   str,
        "exported_rows": int,
        "exported_cols": int,
        ...              # export 대상별 추가 정보
    },
    "message": str,
}
```

---

### `_export_to_file(df, cfg) → dict`

데이터프레임을 파일로 저장한다.

| `file_format` | 저장 방식 |
|---|---|
| `"csv"` | `df.to_csv(index=False, date_format=...)` |
| `"excel"` | `df.to_excel(index=False, engine="openpyxl")` |
| `"json"` | `df.to_json(orient="records", force_ascii=False)` |
| `"parquet"` | `df.to_parquet(index=False)` |

**반환값**: `{file_path, file_format}`

---

### `_export_to_db(df, cfg) → dict`

데이터프레임을 DB 테이블에 적재한다.

| `write_mode` | 동작 |
|---|---|
| `"append"` | `df.to_sql(if_exists="append")` |
| `"replace"` | `df.to_sql(if_exists="replace")` |
| `"upsert"` | `_upsert_to_db()` 호출 |

`db_session`이 `None`이면 `ExecutorException` 발생.  
청크 크기: `10,000행`

**반환값**: `{table_name, write_mode}`

---

### `_upsert_to_db(df, table_name, key_cols)`

단순 upsert를 구현한다 (MariaDB/MySQL 호환).

1. `key_cols` 고유 조합에 대해 `DELETE FROM {table} WHERE ...` 실행
2. 이후 `df.to_sql(if_exists="append")`로 INSERT

---

### `_export_to_api(df, cfg) → dict`

데이터를 REST API로 배치 전송한다.

1. `df.to_dict(orient="records")`로 변환
2. `batch_size` 단위로 분할하여 `requests.post(endpoint, json=batch)` 호출
3. HTTP 상태코드가 200/201/202가 아니면 `ExecutorException` 발생

**반환값**: `{endpoint, total_sent, n_batches}`
