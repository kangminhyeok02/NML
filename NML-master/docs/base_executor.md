# base_executor.py — 공통 추상 기반 클래스

**파일:** `executors/ml/base_executor.py`  
**클래스:** `BaseExecutor` (ABC)

## 개요

모든 ML Executor의 공통 인터페이스와 유틸리티를 정의하는 추상 기반 클래스.  
서브클래스는 `execute()` 메서드 하나만 구현하면 되고,  
공통 관심사(잡 상태 기록, 시간 측정, 파일 I/O, 예외 처리)는 이 클래스가 담당한다.

```
BaseExecutor (ABC)
    ├── ProcessExecutor
    ├── MartExecutor
    ├── DataAnalysisExecutor
    ├── PythonModelExecutor
    ├── H2OModelExecutor
    ├── PythonH2OModelExecutor
    ├── RModelExecutor
    ├── AutoMLExecutor
    ├── ScorecardExecutor
    ├── PredictExecutor
    ├── PretrainedExecutor
    ├── RuleSearchExecutor
    ├── StrategyExecutor
    ├── RLExecutor
    ├── ReportExecutor
    └── ExportExecutor
```

---

## 보조 클래스

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

## `BaseExecutor.__init__` 파라미터

```python
def __init__(
    self,
    config: dict,
    db_session=None,
    file_root_dir: Optional[str] = None,
)
```

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `config` | `dict` | executor 실행 파라미터. `job_id`, `service_id`, `project_id` 포함 |
| `db_session` | optional | SQLAlchemy DB 세션 (MartExecutor의 SQL 조회에 사용) |
| `file_root_dir` | `str` | 파일 서버 루트 경로. 미지정 시 `FILE_ROOT_DIR` 환경변수, 기본값 `/data` |

**초기화 시 설정되는 인스턴스 변수:**

```python
self.config      = config
self.db_session  = db_session
self.file_root   = Path(file_root_dir or os.getenv("FILE_ROOT_DIR", "/data"))
self.job_id      = config.get("job_id", "unknown")
self.service_id  = config.get("service_id", "unknown")
self.project_id  = config.get("project_id", "unknown")
self.started_at  = None
self.finished_at = None
self._status     = ExecutorStatus.PENDING
```

---

## 메서드 상세

### `execute()` — 추상 메서드 (서브클래스 구현 필수)

실제 ML 작업을 수행하고 결과 딕셔너리를 반환한다.

**반환 형식 (최소 보장):**
```python
{
    "status":  "COMPLETED" | "FAILED",
    "result":  dict,    # executor별 결과 데이터
    "message": str,     # 요약 메시지
}
```

---

### `run()` — 공통 실행 래퍼

`execute()`를 감싸는 공통 실행 메서드.  
직접 호출하는 것은 `run()`이며, `execute()`는 서브클래스에서 구현한다.

**처리 순서:**
1. `started_at` 기록
2. 잡 상태를 `RUNNING`으로 갱신 → `jobs/{job_id}.json` 기록
3. `execute()` 호출
4. 완료 시: `elapsed_sec`, `job_id` 자동 주입 후 `COMPLETED` 갱신
5. `ExecutorException` 발생 시: `FAILED` 처리
6. 일반 `Exception` 발생 시: traceback 포함 `FAILED` 처리

**반환 보장 키:**
```python
{
    "status":      "COMPLETED" | "FAILED",
    "job_id":      str,
    "result":      dict,
    "message":     str,
    "elapsed_sec": float,   # run() 종료 시 자동 주입
}
```

---

### `_update_job_status(status, progress=None, message=None, result=None)`

잡 상태를 `FILE_ROOT_DIR/jobs/{job_id}.json` 파일에 기록한다.  
UI나 폴링 클라이언트가 이 파일을 읽어 진행률을 표시한다.

**저장 예시:**
```json
{
  "job_id": "job_abc123",
  "status": "RUNNING",
  "progress": 45.0,
  "message": "모델 학습 중...",
  "updated_at": "2026-04-09T10:30:00"
}
```

---

### `_load_dataframe(relative_path)` → `pd.DataFrame`

`file_root` 기준 상대 경로의 데이터 파일을 로드한다.

| 확장자 | 처리 방식 |
|--------|----------|
| `.parquet` | `pd.read_parquet()` |
| `.csv`, `.txt` | `pd.read_csv()` |
| 기타 | `ExecutorException` 발생 |

---

### `_save_dataframe(df, relative_path)` → `str`

DataFrame을 parquet 형식으로 저장한다.  
중간 디렉토리를 자동 생성하고, 저장된 절대 경로 문자열을 반환한다.

---

### `_save_json(data, relative_path)` → `str`

딕셔너리를 UTF-8 JSON 파일로 저장한다.

- `ensure_ascii=False`
- `indent=2`

---

## 파일 저장 구조

```
FILE_ROOT_DIR/                          # 환경변수 (기본: /data)
├── jobs/
│   └── {job_id}.json                   ← 잡 상태 파일 (폴링용)
├── mart/                               ← MartExecutor 출력
│   ├── {target_id}.parquet
│   ├── {target_id}_train.parquet
│   ├── {target_id}_valid.parquet
│   ├── {target_id}_test.parquet
│   └── {target_id}_meta.json
├── analysis/                           ← DataAnalysisExecutor, RuleSearchExecutor 출력
│   ├── {output_id}_eda.json
│   └── {output_id}_rules.json
├── models/                             ← 모델 파일 및 메타
│   ├── {model_id}.pkl
│   ├── {model_id}_automl.pkl
│   ├── {model_id}/model.zip            ← H2O MOJO
│   ├── {model_id}.rds                  ← R 모델
│   ├── {model_id}_rl.pkl               ← RL 정책
│   └── {model_id}_meta.json
├── predict/                            ← PredictExecutor, PretrainedExecutor, PythonH2OModelExecutor
│   ├── {output_id}_result.parquet
│   ├── {output_id}_py_h2o.parquet
│   └── {output_id}_pretrained.parquet
├── strategy/                           ← StrategyExecutor
│   ├── {output_id}_result.parquet
│   └── {output_id}_summary.json
├── reports/                            ← ReportExecutor
│   ├── {output_id}.json
│   └── {output_id}.xlsx
└── exports/                            ← ExportExecutor
    ├── {output_id}.csv / .xlsx / .json / .parquet
    └── {output_id}_summary.json
```

---

## 설계 원칙

| 원칙 | 내용 |
|------|------|
| **단일 책임** | `execute()` 하나만 구현하면 상태관리·로깅·저장은 자동 처리 |
| **실패 격리** | `ExecutorException`은 예측된 실패, 일반 `Exception`은 예측 못한 실패 — 둘 다 FAILED로 수렴 |
| **진행률 표준화** | `_update_job_status(progress=0~100)`으로 단계별 진행률 기록 |
| **파일 기반 상태** | DB 없이도 폴링 가능한 잡 상태 파일 제공 |
| **공통 config 키** | 모든 executor에 `job_id`, `service_id`, `project_id` 공통 전파 |

---

## 새 Executor 추가 방법

```python
# 1. executors/ml/my_executor.py 생성
from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

class MyExecutor(BaseExecutor):
    def execute(self) -> dict:
        cfg = self.config
        df = self._load_dataframe(cfg["input_path"])
        self._update_job_status(ExecutorStatus.RUNNING, progress=50)
        # ... 실제 작업 ...
        saved = self._save_dataframe(result_df, f"output/{cfg['output_id']}.parquet")
        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  {"output_path": saved, "rows": len(result_df)},
            "message": "완료",
        }

# 2. process_executor.py의 EXECUTOR_REGISTRY에 등록
"my_executor": ("executors.ml.my_executor", "MyExecutor"),
```
