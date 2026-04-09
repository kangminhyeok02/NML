# export_executor.py — 결과 외부 내보내기

## 개요

분석/예측 결과를 파일, DB, REST API 등 외부로 내보내는 executor.  
파이프라인 마지막 단계에서 최종 결과를 운영계·데이터웨어하우스·외부 시스템으로 전달한다.

---

## config 파라미터 (공통)

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `source_path` | ✅ | `str` | 내보낼 원본 데이터 경로 (.parquet) |
| `export_type` | ✅ | `str` | `"file"` \| `"db"` \| `"api"` |
| `output_id` | ✅ | `str` | 출력 식별자 |
| `columns` | ❌ | `list` | 내보낼 컬럼 선택 (없으면 전체) |
| `rename_map` | ❌ | `dict` | 컬럼 이름 변경 `{"original": "new_name"}` |
| `date_format` | ❌ | `str` | 날짜 포맷 (기본: `"%Y-%m-%d"`) |

---

## export_type별 추가 파라미터

### `"file"` 모드

| 키 | 필수 | 설명 |
|----|------|------|
| `file_format` | ❌ | `"csv"` \| `"excel"` \| `"json"` \| `"parquet"` (기본: `"csv"`) |
| `output_path` | ❌ | 저장 경로 (기본: `exports/{output_id}.{format}`) |

| 포맷 | 저장 메서드 | 특이사항 |
|------|-----------|---------|
| `csv` | `df.to_csv()` | date_format 적용 |
| `excel` | `df.to_excel()` | openpyxl 엔진 |
| `json` | `df.to_json(orient="records")` | ensure_ascii=False, indent=2 |
| `parquet` | `df.to_parquet()` | 가장 효율적, 대용량 권장 |

---

### `"db"` 모드

| 키 | 필수 | 설명 |
|----|------|------|
| `table_name` | ✅ | 대상 테이블명 |
| `write_mode` | ❌ | `"append"` \| `"replace"` \| `"upsert"` (기본: `"append"`) |
| `key_cols` | ❌* | upsert 키 컬럼 목록 (\* upsert 시 필수) |

**write_mode 동작:**

| 모드 | 동작 |
|------|------|
| `append` | 기존 테이블에 추가 (`if_exists="append"`) |
| `replace` | 테이블 삭제 후 재생성 (`if_exists="replace"`) |
| `upsert` | 키 기준 DELETE → INSERT (MariaDB/MySQL 호환) |

**`_upsert_to_db` 구현:**
```python
# 키 값별 기존 데이터 삭제
DELETE FROM {table} WHERE key1=:k1 AND key2=:k2

# 전체 재삽입
df.to_sql(table, if_exists="append", chunksize=10_000)
```

---

### `"api"` 모드

| 키 | 필수 | 설명 |
|----|------|------|
| `endpoint` | ✅ | POST 대상 URL |
| `batch_size` | ❌ | 배치 크기 (기본: `1000`) |
| `headers` | ❌ | 요청 헤더 (기본: `{"Content-Type": "application/json"}`) |

```python
# 배치 전송 루프
records = df.to_dict(orient="records")
for batch in chunks(records, batch_size):
    resp = requests.post(endpoint, json=batch, headers=headers, timeout=30)
    if resp.status_code not in (200, 201, 202):
        raise ExecutorException(...)
```

---

## 실행 흐름

```
1. source_path 데이터 로드                           [progress 20%]
2. columns 선택 + rename_map 적용                   [progress 40%]
3. export_type별 내보내기                            [progress 90%]
   - file: 파일 저장
   - db:   DB 적재 (chunksize=10,000)
   - api:  배치 POST 전송
4. exports/{output_id}_summary.json 저장
```

---

## 출력 결과

**요약 파일:** `exports/{output_id}_summary.json`

```json
{
  "output_id":     "credit_result_20260407",
  "export_type":   "file",
  "exported_rows": 100000,
  "exported_cols": 8,
  "file_path":     "/data/exports/credit_result_20260407.csv",
  "file_format":   "csv"
}
```

---

## 파이프라인 위치

```
PredictExecutor / ScorecardExecutor
    ↓ 점수·등급 포함 parquet
StrategyExecutor (선택)
    ↓ 의사결정 포함 parquet
ExportExecutor
    ↓ CSV / DB / API로 최종 전달
```

---

## 사용 예시

```python
# DB upsert 예시
config = {
    "source_path": "strategy/loan_result.parquet",
    "export_type": "db",
    "output_id":   "loan_20260407",
    "table_name":  "ML_SCORE_RESULT",
    "write_mode":  "upsert",
    "key_cols":    ["CUST_ID", "BASE_DT"],
    "columns":     ["CUST_ID", "BASE_DT", "score", "grade", "decision"],
    "rename_map":  {"score": "ML_SCORE", "grade": "ML_GRADE"},
}
```
