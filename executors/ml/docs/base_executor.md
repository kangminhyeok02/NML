# base_executor.py

모든 ML Executor의 공통 인터페이스와 유틸리티를 정의하는 추상 기반 클래스 모듈.

---

## 클래스

### `ExecutorStatus`
잡 상태 상수를 정의하는 네임스페이스 클래스.

| 상수 | 값 |
|---|---|
| `PENDING` | `"PENDING"` |
| `RUNNING` | `"RUNNING"` |
| `COMPLETED` | `"COMPLETED"` |
| `FAILED` | `"FAILED"` |

---

### `ExecutorException`
Executor 실행 중 발생하는 내부 예외 클래스. `Exception`을 상속.

---

### `BaseExecutor` (추상 클래스)

모든 ML Executor의 추상 기반 클래스. 서브클래스는 반드시 `execute()`를 구현해야 한다.

#### 생성자 `__init__(config, db_session, file_root_dir)`

| 파라미터 | 타입 | 설명 |
|---|---|---|
| `config` | `dict` | 실행에 필요한 파라미터. 필수 키: `job_id`, `service_id`, `project_id` |
| `db_session` | optional | SQLAlchemy 또는 커스텀 DB 세션 객체 |
| `file_root_dir` | `str`, optional | 파일 서버 루트 경로. 기본값은 `FILE_ROOT_DIR` 환경변수 (`/data`) |

초기화 시 `_setup_logger()`를 자동 호출한다.

---

## 추상 메서드

### `execute() → dict` *(추상)*

실제 ML 작업을 수행하고 결과 딕셔너리를 반환한다. 서브클래스에서 반드시 구현해야 한다.

**반환값 필수 키**

| 키 | 타입 | 설명 |
|---|---|---|
| `status` | `str` | `"COMPLETED"` 또는 `"FAILED"` |
| `job_id` | `str` | 잡 식별자 |
| `result` | `dict` | executor별 결과 데이터 |
| `message` | `str` | 요약 메시지 |

---

## 공통 실행 메서드

### `run() → dict`

`execute()`를 감싸는 공통 실행 래퍼.

- 시작/종료 시각(`started_at`, `finished_at`) 기록
- 실행 전 상태를 `RUNNING`으로 업데이트
- `execute()` 호출 후 `COMPLETED`로 업데이트
- `ExecutorException` 및 일반 예외 모두 `_handle_failure()`로 처리
- 반환 딕셔너리에 `job_id`, `elapsed_sec`을 자동 추가

---

## 잡 상태 관리

### `_update_job_status(status, progress, message, result)`

잡 상태를 JSON 파일(`{FILE_ROOT_DIR}/jobs/{job_id}.json`)에 기록한다.

| 파라미터 | 타입 | 설명 |
|---|---|---|
| `status` | `str` | `ExecutorStatus` 상수 |
| `progress` | `float`, optional | 진행률 (0~100) |
| `message` | `str`, optional | 상태 메시지 |
| `result` | `dict`, optional | 결과 데이터 |

---

## 데이터 로드 / 저장 헬퍼

### `_load_dataframe(relative_path) → pd.DataFrame`

`FILE_ROOT_DIR` 기준 상대 경로의 parquet 또는 CSV 파일을 로드한다.

- `.parquet` → `pd.read_parquet()`
- `.csv`, `.txt` → `pd.read_csv()`
- 파일 미존재 또는 미지원 형식 시 `ExecutorException` 발생

### `_save_dataframe(df, relative_path) → str`

DataFrame을 `FILE_ROOT_DIR` 기준 상대 경로에 **parquet** 형식으로 저장한다.  
저장된 절대 경로를 반환한다. 부모 디렉터리가 없으면 자동 생성.

### `_save_json(data, relative_path) → str`

`dict`를 JSON 파일로 저장하고 절대 경로를 반환한다.  
`ensure_ascii=False`, `indent=2` 옵션으로 저장.

---

## 내부 유틸

### `_elapsed() → float`

`started_at` 부터 `finished_at`(없으면 현재 시각)까지의 경과 초를 반환한다.

### `_handle_failure(error_msg) → dict`

실패 처리 공통 로직.

1. `finished_at` 기록
2. 에러 로그 출력
3. 상태를 `FAILED`로 업데이트
4. 표준 실패 결과 딕셔너리 반환

**반환값 구조**
```python
{
    "status":      "FAILED",
    "job_id":      str,
    "result":      {},
    "message":     str,   # 에러 메시지 또는 traceback
    "elapsed_sec": float,
}
```

### `_setup_logger()`

`logging.basicConfig()`으로 기본 로거를 설정한다.  
포맷: `YYYY-MM-DD HH:MM:SS  LEVEL  name  message`
