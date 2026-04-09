# process_executor.py

전체 ML 파이프라인을 순차적으로 제어하는 오케스트레이션 실행기.

개별 executor를 단계별로 호출하고 각 단계의 결과를 다음 단계로 전달한다.  
파이프라인 진행 상황을 잡 상태 파일에 기록하며, 특정 단계 실패 시 `on_error` 정책에 따라 동작한다.

---

## 모듈 상수

### `EXECUTOR_REGISTRY`

executor 유형 문자열 → (모듈 경로, 클래스명) 매핑 딕셔너리.

| 키 | 클래스 |
|---|---|
| `"mart"` | `MartExecutor` |
| `"data_analysis"` | `DataAnalysisExecutor` |
| `"python_model"` | `PythonModelExecutor` |
| `"h2o_model"` | `H2OModelExecutor` |
| `"r_model"` | `RModelExecutor` |
| `"automl"` | `AutoMLExecutor` |
| `"scorecard"` | `ScorecardExecutor` |
| `"predict"` | `PredictExecutor` |
| `"pretrained"` | `PretrainedExecutor` |
| `"report"` | `ReportExecutor` |
| `"export"` | `ExportExecutor` |
| `"rulesearch"` | `RuleSearchExecutor` |
| `"stg"` | `StrategyExecutor` |
| `"rl"` | `RLExecutor` |

---

## 클래스

### `ProcessExecutor(BaseExecutor)`

ML 파이프라인 오케스트레이션 executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `pipeline` | `list` | 실행할 단계 목록 |

**파이프라인 단계 항목 구조**

| 키 | 설명 |
|---|---|
| `name` | 단계 이름 (고유) |
| `executor` | executor 유형 (`EXECUTOR_REGISTRY` 키) |
| `config` | 해당 executor의 config 딕셔너리 |
| `on_error` | `"stop"` \| `"skip"` \| `"continue"` (기본: `"stop"`) |
| `input_from` | 이전 단계 결과를 이 단계 config에 병합할 단계명 (선택) |

#### config 선택 키

| 키 | 기본값 | 설명 |
|---|---|---|
| `stop_on_first_failure` | `True` | 첫 번째 실패 시 파이프라인 중단 여부 |

---

### `execute() → dict`

파이프라인을 순차 실행하고 전체 결과를 반환한다.

**동작 순서**
1. `pipeline` 목록을 순회하며 각 단계의 executor를 생성
2. `input_from`이 지정된 경우 이전 단계 결과를 현재 단계 config에 병합
3. `job_id`, `service_id`, `project_id`를 하위 executor에 전파
4. 진행률을 `(step_idx / total_steps) * 90` 으로 계산하여 상태 업데이트
5. 단계 실패 시 `on_error` 정책 적용:
   - `"stop"`: 즉시 중단
   - `"skip"`: 해당 단계 건너뜀, 계속 진행
   - `"continue"`: 결과 기록 후 계속 진행
6. 단계 간 결과는 `context[step_name]`에 저장되어 이후 단계에서 참조 가능

**반환값 구조**
```python
{
    "status": "COMPLETED" | "FAILED",
    "result": {
        "pipeline_name": str,
        "total_steps":   int,
        "executed":      int,
        "failed_steps":  list[str],
        "step_results":  list[dict],
    },
    "message": str,
}
```

---

### `_build_executor(executor_type, config) → BaseExecutor`

`EXECUTOR_REGISTRY`에서 executor 클래스를 동적으로 로드하여 인스턴스를 생성한다.

- `importlib.import_module()`로 모듈을 동적 로드
- `db_session`과 `file_root_dir`을 상속하여 전달
- 등록되지 않은 `executor_type`이면 `ExecutorException` 발생

---

## 모듈 레벨 함수

### `_make_engine(db_info) → Engine`

`db_info` 딕셔너리로 PostgreSQL SQLAlchemy engine을 생성한다.

| 파라미터 키 | 설명 |
|---|---|
| `host` | DB 호스트 |
| `port` | DB 포트 |
| `db` | DB 명 |
| `user` | 사용자명 |
| `password` | 비밀번호 |

드라이버: `postgresql+psycopg2`, `pool_pre_ping=True` 설정.

---

### `_check_process_name_already_exists(company_db_info, prcs_name) → bool`

프로세스 이름(`prcs_name`) 중복 여부를 `tb_process` 테이블에서 조회한다.  
이미 존재하면 `True`, 없으면 `False` 반환.

---

### `get_system_auth_id(service_db_info, auth_key, id_gb_cd) → dict`

`auth_key`로 사용자를 검증하고 `id_gb_cd`로 시스템 인증 ID를 조회한다.

1. `tb_user_auth` 테이블에서 `auth_key`로 `user_id` 조회 (없으면 `ValueError`)
2. `id_gb_cd` 기준으로 인증 ID 필터링 후 반환
