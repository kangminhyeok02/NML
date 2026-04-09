# base_executor.py — 공통 추상 기반 클래스

## 개요

모든 ML Executor의 공통 인터페이스와 유틸리티를 정의하는 추상 기반 클래스.  
각 executor는 이 클래스를 상속받아 `execute()` 메서드 하나만 구현하면 된다.

```
BaseExecutor (ABC)
    ├── ProcessExecutor
    ├── MartExecutor
    ├── DataAnalysisExecutor
    ├── PythonModelExecutor
    ├── H2OModelExecutor
    ├── PythonH2OModelExecutor
    ├── AutoMLExecutor
    ├── ScorecardExecutor
    ├── PredictExecutor
    ├── PretrainedExecutor
    ├── ExportExecutor
    ├── RuleSearchExecutor
    ├── StrategyExecutor
    └── RLExecutor
```

---

## 핵심 클래스

### `ExecutorStatus`

| 상수 | 값 | 의미 |
|------|-----|------|
| `PENDING` | `"PENDING"` | 실행 대기 중 |
| `RUNNING` | `"RUNNING"` | 실행 중 |
| `COMPLETED` | `"COMPLETED"` | 정상 완료 |
| `FAILED` | `"FAILED"` | 실패 |

### `ExecutorException`

Executor 내부에서 예상된 실패를 표현하는 커스텀 예외.  
`run()` 내부에서 catch되어 FAILED 상태로 변환된다.

---

## `BaseExecutor` 생성자 파라미터

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `config` | `dict` | executor 실행 파라미터. `job_id`, `service_id`, `project_id` 포함 |
| `db_session` | optional | SQLAlchemy DB 세션 |
| `file_root_dir` | `str` | 파일 서버 루트 경로. 미지정 시 `FILE_ROOT_DIR` 환경변수 사용 (기본 `/data`) |

---

## 메서드 상세

### `execute()` — 추상 메서드 (서브클래스 구현 필수)

실제 ML 작업을 수행하고 결과 딕셔너리를 반환한다.

**반환 형식:**
```python
{
    "status":  "COMPLETED" | "FAILED",
    "result":  dict,      # executor별 결과 데이터
    "message": str,       # 요약 메시지
}
```

---

### `run()` — 공통 실행 래퍼

`execute()`를 호출하기 전후로 공통 처리를 수행한다.

**처리 순서:**
1. `started_at` 기록
2. 잡 상태를 `RUNNING`으로 갱신
3. `execute()` 호출
4. `elapsed_sec`, `job_id` 자동 주입
5. 잡 상태를 `COMPLETED`로 갱신
6. 예외 발생 시 `FAILED` 처리 후 에러 딕셔너리 반환

**반환 보장 키:**
```python
{
    "status":      str,
    "job_id":      str,
    "result":      dict,
    "message":     str,
    "elapsed_sec": float,
}
```

---

### `_update_job_status(status, progress, message, result)`

잡 상태를 `FILE_ROOT_DIR/jobs/{job_id}.json` 파일에 기록한다.  
폴링 또는 UI에서 진행률을 표시할 때 이 파일을 읽는다.

**저장 예시:**
```json
{
  "job_id": "job_abc123",
  "status": "RUNNING",
  "progress": 45.0,
  "message": "모델 학습 중...",
  "updated_at": "2026-04-07T10:30:00"
}
```

---

### `_load_dataframe(relative_path)` → `pd.DataFrame`

`FILE_ROOT_DIR` 기준 상대 경로의 데이터 파일을 로드한다.

| 확장자 | 처리 방식 |
|--------|----------|
| `.parquet` | `pd.read_parquet()` |
| `.csv`, `.txt` | `pd.read_csv()` |
| 기타 | `ExecutorException` 발생 |

---

### `_save_dataframe(df, relative_path)` → `str`

DataFrame을 parquet 형식으로 저장한다. 중간 디렉토리는 자동 생성.  
저장된 절대 경로를 문자열로 반환한다.

---

### `_save_json(data, relative_path)` → `str`

딕셔너리를 UTF-8 JSON 파일로 저장한다. `ensure_ascii=False`, `indent=2`.

---

## 파일 저장 구조

```
FILE_ROOT_DIR/
├── jobs/
│   └── {job_id}.json          ← 잡 상태 파일
├── mart/                       ← MartExecutor 출력
├── analysis/                   ← DataAnalysisExecutor, RuleSearchExecutor 출력
├── models/                     ← 모델 파일 및 메타
├── predict/                    ← PredictExecutor, PretrainedExecutor 출력
├── strategy/                   ← StrategyExecutor 출력
└── exports/                    ← ExportExecutor 출력
```

---

## 설계 원칙

- **단일 책임**: `execute()` 하나만 구현하면 상태관리·로깅·저장은 자동 처리
- **실패 격리**: `ExecutorException`은 예측된 실패, 일반 `Exception`은 예측 못한 실패 — 둘 다 `FAILED`로 수렴
- **진행률 표준화**: `_update_job_status(progress=0~100)`으로 단계별 진행률 기록
- **파일 기반 상태**: DB 없이도 폴링 가능한 잡 상태 파일 제공
