# Executor 코드 심층 분석

각 Executor의 소스코드를 메서드 단위로 분석한 문서.
설계 의도, 핵심 로직, 주의사항, 실제 동작 흐름을 다룬다.

---

## 목차

1. [BaseExecutor](#1-baseexecutor)
2. [ProcessExecutor](#2-processexecutor)
3. [MartExecutor](#3-martexecutor)
4. [DataAnalysisExecutor](#4-dataanalysisexecutor)
5. [PythonModelExecutor](#5-pythonmodelexecutor)
6. [H2OModelExecutor](#6-h2omodelexecutor)
7. [PythonH2OModelExecutor](#7-pythonh2omodelexecutor)
8. [RModelExecutor](#8-rmodelexecutor)
9. [AutoMLExecutor](#9-automlexecutor)
10. [ScorecardExecutor](#10-scorecardexecutor)
11. [PredictExecutor](#11-predictexecutor)
12. [PretrainedExecutor](#12-pretrainedexecutor)
13. [RuleSearchExecutor](#13-rulesearchexecutor)
14. [StrategyExecutor](#14-strategyexecutor)
15. [RLExecutor](#15-rlexecutor)
16. [ReportExecutor](#16-reportexecutor)
17. [ExportExecutor](#17-exportexecutor)

---

## 1. BaseExecutor

**파일:** `executors/ml/base_executor.py`

모든 Executor의 추상 기반 클래스. Template Method 패턴으로 공통 관심사를 캡슐화한다.

### 클래스 계층

```
BaseExecutor (ABC)
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
  ├── ExportExecutor
  └── ProcessExecutor
```

### `__init__`

```python
def __init__(self, config, db_session=None, file_root_dir=None):
    self.config      = config
    self.db_session  = db_session
    self.file_root   = Path(file_root_dir or os.getenv("FILE_ROOT_DIR", "/data"))
    self.job_id      = config.get("job_id", "unknown")
    self.service_id  = config.get("service_id", "unknown")
    self.project_id  = config.get("project_id", "unknown")
    self.started_at  = None
    self.finished_at = None
    self._status     = ExecutorStatus.PENDING
    self._setup_logger()
```

- `file_root`: 파일 서버 루트. `file_root_dir` 인자 → `FILE_ROOT_DIR` 환경변수 → `/data` 순서로 결정.
- `job_id` / `service_id` / `project_id`: config에서 추출. 잡 상태 파일명과 로그 추적에 사용.

### `run()` — 공통 실행 래퍼

```python
def run(self) -> dict:
    self.started_at = datetime.now()
    self._update_job_status(ExecutorStatus.RUNNING)

    try:
        result = self.execute()          # ← 서브클래스 구현
        self.finished_at = datetime.now()
        result.setdefault("job_id", self.job_id)
        result.setdefault("elapsed_sec", self._elapsed())
        self._update_job_status(ExecutorStatus.COMPLETED, result=result)
        return result

    except ExecutorException as exc:
        return self._handle_failure(str(exc))
    except Exception:
        return self._handle_failure(traceback.format_exc())
```

**핵심 포인트:**
- `execute()`를 `try/except`로 감싸 모든 예외를 `FAILED` 반환으로 변환.
- `ExecutorException`: 의도된 실패 (입력 오류 등) — 스택 트레이스 없이 메시지만.
- `Exception`: 예상치 못한 실패 — 전체 스택 트레이스 포함.
- 결과 dict에 `job_id`, `elapsed_sec`을 자동 주입 (`setdefault`이므로 서브클래스가 이미 설정했다면 덮어쓰지 않음).

### `_update_job_status()`

```python
def _update_job_status(self, status, progress=None, message=None, result=None):
    payload = {
        "job_id":     self.job_id,
        "status":     status,
        "updated_at": datetime.now().isoformat(),
    }
    # progress, message, result는 있을 때만 포함
    with open(job_dir / f"{self.job_id}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
```

잡 상태 파일을 갱신한다. `progress`는 0~100 float. 각 `execute()` 구현 내부에서
`self._update_job_status(ExecutorStatus.RUNNING, progress=50)` 형태로 중간 진행률을 기록한다.

### 데이터 I/O 헬퍼

| 메서드 | 동작 |
|--------|------|
| `_load_dataframe(relative_path)` | `.parquet` 또는 `.csv/.txt`를 `pd.DataFrame`으로 로드. 파일 미존재 시 `ExecutorException` |
| `_save_dataframe(df, relative_path)` | `df.to_parquet()`으로 저장. 부모 디렉터리 자동 생성. 절대경로 반환 |
| `_save_json(data, relative_path)` | dict를 UTF-8 JSON으로 저장. 절대경로 반환 |

모든 경로는 `self.file_root` 기준 상대 경로로 입력한다.

### `ExecutorStatus` 상수

```python
class ExecutorStatus:
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"
```

---

## 2. ProcessExecutor

**파일:** `executors/ml/process_executor.py`

파이프라인을 순차적으로 실행하는 오케스트레이터. Registry 패턴으로 executor를 동적 로딩한다.

### EXECUTOR_REGISTRY

```python
EXECUTOR_REGISTRY: dict[str, tuple[str, str]] = {
    "mart":          ("executors.ml.mart_executor",           "MartExecutor"),
    "data_analysis": ("executors.ml.data_analysis_executor",  "DataAnalysisExecutor"),
    "python_model":  ("executors.ml.python_model_executor",   "PythonModelExecutor"),
    ...
}
```

키: 파이프라인 config의 `executor` 필드값.  
값: `(module_path, class_name)` — `importlib.import_module` + `getattr`로 동적 로딩.

### `execute()` 핵심 흐름

```python
def execute(self) -> dict:
    pipeline = self.config.get("pipeline", [])
    context: dict[str, Any] = {}   # 단계 간 결과 공유

    for step_idx, step in enumerate(pipeline):
        step_config = dict(step.get("config", {}))

        # 이전 단계 결과 주입
        input_from = step.get("input_from")
        if input_from and input_from in context:
            step_config.update(context[input_from])

        # job_id 자동 생성: "{parent_job_id}__{step_name}"
        step_config.setdefault("job_id", f"{self.job_id}__{step_name}")

        executor = self._build_executor(executor_type, step_config)
        step_result = executor.run()

        context[step_name] = step_result.get("result", {})

        # 실패 정책
        if step_result["status"] == ExecutorStatus.FAILED:
            on_error = step.get("on_error", "stop")
            if on_error == "stop" and stop_on_fail:
                break   # 파이프라인 중단
            elif on_error == "skip":
                continue  # 실패 무시하고 다음 단계 진행
            # "continue"는 실패 기록 후 계속
```

**`input_from` 동작:**
- `step_config.update(context[input_from])`는 이전 단계의 `result` dict를 현재 단계 config에 병합.
- 예: mart 단계 결과의 `train_path`, `valid_path`를 model 단계에서 자동으로 받아 쓸 수 있다.

**진행률 계산:**
```python
progress_start = int(step_idx / total_steps * 90)  # 최대 90%, 나머지는 마무리용
```

### `_build_executor()`

```python
def _build_executor(self, executor_type: str, config: dict) -> BaseExecutor:
    module_path, class_name = EXECUTOR_REGISTRY[executor_type]
    module = importlib.import_module(module_path)
    cls    = getattr(module, class_name)
    return cls(config=config, db_session=self.db_session, file_root_dir=str(self.file_root))
```

`db_session`과 `file_root`는 부모 ProcessExecutor에서 그대로 전파된다.

---

## 3. MartExecutor

**파일:** `executors/ml/mart_executor.py`

원천 데이터를 ML용 마트로 변환한다.

### `execute()` 흐름

```
_load_source()          → 원천 로드
_basic_preprocess()     → 타입 변환 + 결측 + 이상값
_create_derived_features() → 파생 변수
_split_dataframe()      → train/valid/test 분할 (선택)
_save_dataframe()       → parquet 저장
_save_json()            → 메타 저장
```

### `_load_source()`

```python
def _load_source(self, cfg):
    if "source_query" in cfg and self.db_session is not None:
        return pd.read_sql(cfg["source_query"], self.db_session.bind)
    elif "source_path" in cfg:
        return self._load_dataframe(cfg["source_path"])
    else:
        raise ExecutorException("source_query 또는 source_path 중 하나가 필요합니다.")
```

DB 쿼리와 파일 로드를 투명하게 처리. `db_session`이 없으면 파일 모드로 자동 전환된다.

### `_basic_preprocess()`

```python
# 1) 문자열 → 카테고리 (cardinality가 50% 미만인 경우)
for col in df.select_dtypes(include="object").columns:
    if df[col].nunique() / len(df) < 0.5:
        df[col] = df[col].astype("category")

# 2) 수치형 결측 → 중앙값 대체
for col in num_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# 3) 이상값 클리핑: 1~99 퍼센타일로 제한
for col in clip_cols:
    q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lower=q1, upper=q3)
```

**주의:** 클리핑은 IQR이 아닌 1~99 퍼센타일 기준. 극단값 제거가 목적이므로 IQR보다 넓게 잡는다.

### `_create_derived_features()`

```python
for rule in rules:
    df[rule["name"]] = df.eval(rule["expr"])
```

`pandas.DataFrame.eval()`을 사용해 문자열 수식으로 파생 변수를 만든다.
실패한 rule은 경고 로그를 남기고 건너뜀 (파이프라인을 중단하지 않음).

### `_split_dataframe()`

```python
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 셔플
n = len(df)
train_end = int(n * split_cfg.get("train", 0.7))
valid_end  = train_end + int(n * split_cfg.get("valid", 0.15))

return {
    "train": df.iloc[:train_end],
    "valid": df.iloc[train_end:valid_end],
    "test":  df.iloc[valid_end:],
}
```

`random_state=42` 고정으로 재현성 보장. 층화 추출(stratify)은 미구현 — 불균형 타깃이면 주의.

---

## 4. DataAnalysisExecutor

**파일:** `executors/ml/data_analysis_executor.py`

EDA 결과를 JSON으로 저장한다. 6개 분석 항목 + 타깃 분리도.

### 분석 항목별 메서드

#### `_basic_stats()`

```python
stats_df = num_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
return stats_df.round(4).to_dict(orient="index")
```

수치형 컬럼만 대상. 9개 백분위 포함 기술통계.

#### `_missing_summary()`

```python
{
    "column":        col,
    "missing_count": int,
    "missing_rate":  float,        # 0~1
    "is_warning":    bool,         # missing_rate > threshold (기본 0.3)
}
```

`is_warning=True` 컬럼 목록이 `high_missing_cols`로 요약에 포함된다.

#### `_outlier_summary()`

```python
q1, q3 = series.quantile(0.25), series.quantile(0.75)
iqr    = q3 - q1
lower  = q1 - 1.5 * iqr   # Tukey Fence
upper  = q3 + 1.5 * iqr
outlier_cnt = ((series < lower) | (series > upper)).sum()
```

Tukey Fence(IQR × 1.5) 방식. MartExecutor의 클리핑 기준(1~99 퍼센타일)과 다르게, 분포 이상값 '진단'이 목적이므로 표준 IQR 방식 사용.

#### `_distribution_summary()`

```python
{
    "column":   col,
    "skewness": scipy.stats.skew(series),    # 0이면 대칭
    "kurtosis": scipy.stats.kurtosis(series), # 0이면 정규분포
}
```

`scipy.stats` 사용. skewness > 1 이면 오른쪽 꼬리, < -1 이면 왼쪽 꼬리.

#### `_correlation_matrix()`

```python
corr = num_df.corr().round(4)
# 상삼각 행렬 순회하여 |corr| >= threshold 인 쌍 추출
high_pairs = [
    {"col_a": cols[i], "col_b": cols[j], "corr": corr.iloc[i,j]}
    for i in range(len(cols))
    for j in range(i+1, len(cols))
    if abs(corr.iloc[i, j]) >= threshold
]
```

Pearson 상관계수. 다중공선성 의심 쌍을 `high_corr_pairs`로 반환.

#### `_target_analysis()`

```python
ks_stat, ks_pval = scipy.stats.ks_2samp(good, bad)
```

2-sample KS test: Good(y=0) vs Bad(y=1) 분포 비교. KS 값이 클수록 해당 변수가 타깃을 잘 분리한다.
결과는 KS 값 내림차순 정렬 → 변수 선별 우선순위로 활용.

---

## 5. PythonModelExecutor

**파일:** `executors/ml/python_model_executor.py`

scikit-learn 생태계 모델 학습. 모델 레지스트리 함수와 평가 로직이 핵심.

### `_build_model()` — 모듈 수준 함수

```python
def _build_model(model_type: str, params: dict):
    model_type = model_type.lower()
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    ...
```

지연 임포트(lazy import): `_build_model()` 호출 시점에만 해당 라이브러리를 로드.
설치되지 않은 라이브러리는 사용하지 않으면 ImportError가 발생하지 않는다.

### `execute()` 핵심 흐름

```python
# 검증 데이터 없으면 자동 분리
if "valid_path" in cfg:
    valid_df = self._load_dataframe(cfg["valid_path"])
else:
    from sklearn.model_selection import train_test_split
    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)

# 피처 목록: 명시하지 않으면 타깃 외 전체
feature_cols = cfg.get("feature_cols") or [c for c in train_df.columns if c != target_col]

model = _build_model(model_type, model_params)
model.fit(X_train, y_train)

# pickle로 저장
with open(full_path, "wb") as f:
    pickle.dump(model, f)
```

### `_evaluate()`

```python
def _evaluate(self, model, X, y, task):
    y_pred = model.predict(X)
    if task == "classification":
        metrics["accuracy"]  = accuracy_score(y, y_pred)
        metrics["precision"] = precision_score(y, y_pred, average="binary", zero_division=0)
        metrics["recall"]    = recall_score(y, y_pred, average="binary", zero_division=0)
        metrics["f1"]        = f1_score(y, y_pred, average="binary", zero_division=0)
        if hasattr(model, "predict_proba"):
            metrics["auc"] = roc_auc_score(y, model.predict_proba(X)[:, 1])
    else:  # regression
        metrics["rmse"] = np.sqrt(mean_squared_error(y, y_pred))
        metrics["r2"]   = r2_score(y, y_pred)
```

`zero_division=0`: 양성 샘플 없는 엣지 케이스에서 0 반환 (경고 없음).
AUC는 `predict_proba`가 있는 모델(softmax 출력)에서만 계산.

### 저장되는 메타 JSON

```json
{
  "model_id":     "lgbm_v1",
  "model_type":   "lightgbm",
  "model_params": {"n_estimators": 300, "learning_rate": 0.05},
  "feature_cols": ["age", "income", "debt_ratio"],
  "target_col":   "default_yn",
  "task":         "classification",
  "metrics":      {"auc": 0.82, "f1": 0.61, "accuracy": 0.89},
  "model_path":   "models/lgbm_v1.pkl"
}
```

`feature_cols`가 특히 중요 — PredictExecutor가 이 목록으로 피처를 정렬한다.

---

## 6. H2OModelExecutor

**파일:** `executors/ml/h2o_model_executor.py`

H2O 클러스터와 연동하여 학습하고 MOJO 포맷으로 저장한다.

### H2O 초기화

```python
h2o.init(ip=h2o_ip, port=h2o_port, nthreads=-1)
```

`nthreads=-1`: 가용한 모든 CPU 코어 사용. 이미 실행 중인 클러스터에 연결할 때도 같은 코드를 쓴다.

### 타깃을 factor로 변환

```python
train_h2o[target] = train_h2o[target].asfactor()
valid_h2o[target] = valid_h2o[target].asfactor()
```

H2O에서 분류 문제는 타깃을 `factor` 타입으로 설정해야 한다. 하지 않으면 회귀로 처리된다.

### `_train_model()`

```python
algo_map = {
    "gbm":          H2OGradientBoostingEstimator,
    "drf":          H2ORandomForestEstimator,
    "xgboost":      H2OXGBoostEstimator,
    "glm":          H2OGeneralizedLinearEstimator,
    "deeplearning": H2ODeepLearningEstimator,
}

if algorithm == "automl":
    aml = H2OAutoML(max_runtime_secs=max_rt, seed=42)
    aml.train(...)
    return aml.leader   # 리더보드 1위 모델 반환
else:
    model = estimator_cls(**params)
    model.train(x=x, y=y, training_frame=train_h2o, validation_frame=valid_h2o)
    return model
```

`automl` 선택 시 자동 앙상블 탐색 후 리더 모델만 반환한다.

### MOJO 저장

```python
mojo_path = model.save_mojo(mojo_dir)
```

MOJO(Model Object, Optimized)는 H2O 서버 없이 Java/Python으로 추론할 수 있는 이식성 높은 포맷이다.

### 변수 중요도

```python
varimp = model.varimp(use_pandas=True)[["variable", "percentage"]].head(20)
```

상위 20개 변수만 메타에 저장. H2O는 트리 기반 모델에서 자동으로 feature importance를 제공한다.

---

## 7. PythonH2OModelExecutor

**파일:** `executors/ml/python_h2o_model_executor.py`

Python 전/후처리 + H2O 추론을 하나의 파이프라인으로 통합. 운영 배포용.

### 파이프라인 흐름

```
Python 전처리 (_apply_preprocess)
      ↓
H2O 추론 (_predict_mojo 또는 _predict_h2o_live)
      ↓
Python 후처리 (_apply_postprocess)
      ↓
저장
```

### `_apply_preprocess()` — step 타입

| `type` | 동작 | 필수 파라미터 |
|--------|------|--------------|
| `fillna` | 결측 대체 | `columns`, `value` |
| `clip` | 범위 클리핑 | `columns`, `lower`, `upper` |
| `log1p` | log(1+x) 변환 (음수 클리핑 후) | `columns` |
| `eval` | pandas eval 수식으로 신규 컬럼 생성 | `name`, `expr` |
| `drop` | 컬럼 제거 | `columns` |

각 step은 실패해도 warning 로그만 남기고 계속 진행한다 (파이프라인 중단 없음).

### `_predict_mojo()` vs `_predict_h2o_live()`

| 메서드 | H2O 서버 필요 | 사용 시점 |
|--------|--------------|----------|
| `_predict_mojo()` | 필요 (MOJO 로드에 h2o.init 사용) | `use_mojo: true` (기본) |
| `_predict_h2o_live()` | 필요 | 학습한 모델이 서버 메모리에 상주 중일 때 |

```python
# MOJO 방식
model = h2o.import_mojo(mojo_path)
preds = model.predict(h2o_frame).as_data_frame()
return preds.iloc[:, -1].values   # 마지막 컬럼 = p1 (양성 클래스 확률)
```

### `_apply_postprocess()` — step 타입

| `type` | 동작 |
|--------|------|
| `scale` | `minmax` 또는 `standard` 정규화 |
| `grade` | 점수 구간 → 등급 매핑 |
| `round` | 소수점 반올림 |

---

## 8. RModelExecutor

**파일:** `executors/ml/r_model_executor.py`

Python에서 Rscript를 subprocess로 호출. Python ↔ R 데이터 교환은 CSV/JSON 파일 기반.

### 통신 프로토콜

```
Python                         R 스크립트
  │                                │
  ├─ train.csv (임시파일) ──────►  │  args$train_path
  ├─ args.json (임시파일) ──────►  │  fromJSON(args_file)
  │                                │  ... 학습 ...
  │◄─ meta.json ────────────────── │  저장
  │◄─ model.rds ─────────────────  │  saveRDS()
```

### `_run_rscript()`

```python
# args dict를 JSON 파일로 직렬화
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(args, f)
    args_file = f.name

cmd = [rscript_cmd, str(script_path), "--args_file", args_file]
proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
os.unlink(args_file)  # 임시파일 삭제

if proc.returncode != 0:
    raise ExecutorException(f"R 스크립트 실행 실패\n{proc.stderr[-500:]}")
```

- `timeout=3600`: 최대 1시간 대기.
- `stderr`는 최대 500자만 예외 메시지에 포함 (너무 길면 로그 폭주 방지).
- `proc.stdout`의 각 줄은 `logger.info("[R] ...")` 로 전달 → Python 로그와 통합.

### R 스크립트 인터페이스 규약

```r
# R 스크립트가 받아야 할 인터페이스
args_file <- commandArgs(trailingOnly=TRUE)[2]  # --args_file 다음 값
args      <- jsonlite::fromJSON(args_file)

# train 모드에서 Python이 기대하는 출력:
# 1) args$model_dir 에 RDS 파일 저장
# 2) args$meta_path 에 JSON 메타 저장 (선택, 없으면 기본 메타 사용)
saveRDS(model, file.path(args$model_dir, "model.rds"))
jsonlite::write_json(meta, args$meta_path)
```

### 임시 디렉터리 관리

```python
with tempfile.TemporaryDirectory() as tmpdir:
    train_csv = os.path.join(tmpdir, "train.csv")
    train_df.to_csv(train_csv, index=False)
    ...
    self._run_rscript(...)  # R이 tmpdir 내 파일 읽음
    meta = json.load(open(meta_path))
# with 블록 종료 시 tmpdir 자동 삭제
```

`TemporaryDirectory` 컨텍스트 매니저로 임시파일 누수 방지.

---

## 9. AutoMLExecutor

**파일:** `executors/ml/automl_executor.py`

5개 AutoML 프레임워크를 단일 인터페이스로 통합.

### dispatch 패턴

```python
dispatch = {
    "h2o_automl":  self._run_h2o_automl,
    "autosklearn": self._run_autosklearn,
    "tpot":        self._run_tpot,
    "optuna":      self._run_optuna,
    "pycaret":     self._run_pycaret,
}
best_model, leaderboard = dispatch[framework](cfg, X_train, y_train, X_valid, y_valid)
```

모든 메서드는 `(best_model, leaderboard)` 튜플을 반환하는 동일한 시그니처를 가진다.
`best_model`은 sklearn-compatible(`predict_proba` 메서드 존재)이어야 한다.

### `_run_optuna()` — 가장 범용적

```python
def objective(trial):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 50, 500),
        "max_depth":       trial.suggest_int("max_depth", 3, 12),
        "learning_rate":   trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves":      trial.suggest_int("num_leaves", 16, 128),
        "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    return roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=cfg.get("n_trials", 50))
```

- 탐색 공간: LightGBM 6개 주요 하이퍼파라미터.
- `log=True` for `learning_rate`: 로그 스케일 탐색 (0.001~0.3을 선형이 아닌 로그 간격으로).
- 리더보드: 상위 10개 trial의 AUC와 파라미터 조합.

### `_run_h2o_automl()` — H2OWrapper

```python
class H2OWrapper:
    def __init__(self, m): self.m = m
    def predict_proba(self, X):
        frame = h2o.H2OFrame(X)
        preds = self.m.predict(frame).as_data_frame()
        return np.column_stack([1 - preds.iloc[:, -1].values,
                                    preds.iloc[:, -1].values])
```

H2O 모델을 sklearn-compatible로 래핑. `predict_proba`가 `(n, 2)` numpy array를 반환하도록 맞춤.

---

## 10. ScorecardExecutor

**파일:** `executors/ml/scorecard_executor.py`

금융 신용평가 전용 스코어카드. WOE → 로지스틱 회귀 → PDO 점수화.

### WOE/IV 계산 (`_calc_woe()`)

```python
for bin_label, group in df.groupby("bin"):
    good = (group["y"] == 0).sum()
    bad  = (group["y"] == 1).sum()
    dist_good = good / total_good   # Good 중 이 구간 비율
    dist_bad  = bad  / total_bad    # Bad  중 이 구간 비율
    woe = log(dist_good / dist_bad) # WOE 공식
    iv  = (dist_good - dist_bad) * woe  # IV 기여도
```

**WOE 해석:**
- WOE > 0: 해당 구간에 Good(정상) 고객이 상대적으로 많음
- WOE < 0: Bad(불량) 고객이 상대적으로 많음
- WOE = 0: Good/Bad 비율이 전체 평균과 동일

**IV 기준:**
- IV < 0.02: 예측력 없음 → `iv_threshold` 필터로 제거
- 0.02 ~ 0.1: 약한 예측력
- 0.1 ~ 0.3: 보통 예측력
- > 0.3: 강한 예측력

### PDO 점수 변환 (`_build_scorecard()`)

```python
factor  = pdo / log(2)       # PDO = Points to Double Odds
offset  = base_score - factor * log(1)

# 변수별 구간 점수
point = -(coef / scale) * woe * factor
```

**PDO 공식 배경:** 오즈(Odds = P(Good)/P(Bad))가 2배가 될 때 점수가 `pdo`만큼 증가.
기준점 600에서 오즈 1:1이 되도록 설정하면 600점 이상 = Good 고객 구간.

**최종 점수 산출 (`_score_data()`):**
```python
# 각 변수의 구간 포인트를 합산
for col in selected_cols:
    score_map = scorecard[scorecard["variable"] == col].set_index("bin")["points"]
    scores += bins.map(score_map).fillna(0)
```

### 성능 지표 (`_evaluate_scorecard()`)

```python
# KS 계산: 누적 Good률 - 누적 Bad률의 최대값
score_df = score_df.sort_values("score", ascending=False)
score_df["cum_good"] = (score_df["y"] == 0).cumsum() / n_good
score_df["cum_bad"]  = (score_df["y"] == 1).cumsum() / n_bad
ks = (score_df["cum_bad"] - score_df["cum_good"]).abs().max()
```

KS는 금융권 스코어카드 주요 성능 지표. 통상 0.3 이상이면 사용 가능 수준.

---

## 11. PredictExecutor

**파일:** `executors/ml/predict_executor.py`

운영 환경에서 가장 빈번히 호출. Python/H2O/R 3가지 모델 타입을 통합 지원.

### 피처 정렬

```python
feature_cols = meta.get("feature_cols", [...])
missing_cols = set(feature_cols) - set(df.columns)
if missing_cols:
    raise ExecutorException(f"예측 데이터에 필요한 컬럼이 없습니다: {missing_cols}")
X = df[feature_cols]  # 순서 일치시켜 선택
```

`feature_cols`는 학습 시 저장된 메타에서 가져옴. 순서까지 정확히 맞춰야 tree 모델이 올바르게 작동.

### `_predict_python()`

```python
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)[:, 1]
    result_df[score_col]    = np.round(proba, 6)
    result_df["pred_class"] = (proba >= threshold).astype(int)
else:
    result_df[score_col] = np.round(model.predict(X), 6)
```

`predict_proba` 유무로 분류/회귀를 자동 판별. 점수는 소수점 6자리로 반올림.

### `_predict_r()`

```python
# 임시파일로 입출력 교환
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_in:
    X.to_csv(tmp_in.name, index=False)
cmd = ["Rscript", r_script, "--input", tmp_input, "--model", model_path, "--output", tmp_output]
proc = subprocess.run(cmd, capture_output=True, text=True)
preds = pd.read_csv(tmp_output)
os.unlink(tmp_input); os.unlink(tmp_output)
```

RModelExecutor의 `_run_rscript()`와 달리 PredictExecutor는 `--input/--model/--output` 인자 방식을 사용.
R 스크립트는 이 인자를 직접 파싱해야 한다.

### `_assign_grade()`

```python
def _assign_grade(self, score, grade_mapping):
    for grade, (low, high) in grade_mapping.items():
        if low <= score < high:
            return grade
    return "UNKNOWN"
```

범위 밖이면 "UNKNOWN" 반환. 운영 시 "UNKNOWN" 건수를 모니터링할 필요가 있다.

### `_series_stats()` — 모듈 수준 유틸

```python
def _series_stats(series):
    return {
        "mean": round(float(series.mean()), 4),
        "std":  round(float(series.std()), 4),
        "min":  ..., "p25": ..., "p50": ..., "p75": ..., "max": ...
    }
```

예측 점수 분포 요약. 운영 모니터링에서 점수 분포 이동(drift)을 감지하는 데 사용된다.

---

## 12. PretrainedExecutor

**파일:** `executors/ml/pretrained_executor.py`

재학습 없이 기존 완성 모델로 추론. 4가지 모델 포맷 지원.

### `output_mode` 옵션

| 모드 | 동작 |
|------|------|
| `"score"` | `predict_proba` 또는 `predict`로 점수 컬럼 추가 |
| `"embedding"` | `model.transform(X)`로 임베딩 컬럼 추가 (`emb_0`, `emb_1`, ...) |
| `"both"` | 점수 + 임베딩 모두 추가 |

### `_infer_onnx()` — 배치 추론

```python
for i in range(0, len(X), batch_size):
    batch = X.iloc[i:i+batch_size].values.astype(np.float32)
    preds = sess.run(None, {input_name: batch})
    if len(preds) > 1 and preds[1] is not None:
        scores.extend(preds[1][:, 1].tolist())  # 분류: 2번째 출력의 p1
    else:
        scores.extend(preds[0].flatten().tolist())  # 회귀: 첫 번째 출력
```

ONNX 모델의 출력 구조 (분류 모델은 보통 `[labels, probabilities]` 두 출력).
`float32` 강제 변환 필수 — ONNX Runtime은 `float64` 미지원.

### `_infer_hugging()` — HuggingFace

```python
pipe = hf_pipeline("text-classification", model=model_path)
for i in range(0, len(texts), batch_size):
    batch_res = pipe(texts[i:i+batch_size], truncation=True)
    for r in batch_res:
        labels.append(r["label"])
        scores.append(round(r["score"], 6))
```

텍스트 분류만 지원 (현재 구현). NLP 피처 추출이나 다른 태스크는 확장 필요.
`truncation=True`: 최대 토큰 길이 초과 시 자동 잘라냄.

---

## 13. RuleSearchExecutor

**파일:** `executors/ml/rulesearch_executor.py`

설명 가능한 if-then 규칙 발굴. 3가지 탐색 방법.

### `_search_tree_rules()` — 의사결정트리 경로 추출

```python
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)
tree.fit(X, y)

def _recurse(node_id, conditions):
    if children_l[node_id] == children_r[node_id]:  # 리프 노드
        # 조건 목록, bad_rate, lift 계산 후 rules에 추가
        return
    feat_name = feature_cols[feature[node_id]]
    thresh    = threshold[node_id]
    _recurse(children_l[node_id], conditions + [f"{feat_name} <= {thresh}"])
    _recurse(children_r[node_id], conditions + [f"{feat_name} > {thresh}"])

_recurse(0, [])  # 루트에서 시작
```

트리의 각 리프 경로를 AND 조건 목록으로 변환.
`min_samples_leaf=50`: 너무 작은 집단의 규칙 방지.

**lift 계산:**
```python
lift = bad_rate / overall_bad_rate  # 전체 bad_rate 대비 이 구간의 bad_rate 배율
```

### `_search_association_rules()` — FP-Growth

```python
# 수치형 → 5분위 구간화 → 이진 인코딩
binary_df = pd.get_dummies(pd.qcut(df[col], q=5).astype(str), prefix=col)
binary_df[f"{target}=1"] = (df[target] == 1).astype(int)

freq = fpgrowth(binary_df.astype(bool), min_support=0.05)
ar   = association_rules(freq, metric="confidence", min_threshold=0.3)

# consequent가 "{target}=1"인 규칙만 필터
rules = [r for r in ar if f"{target}=1" in r["consequents"]]
```

`mlxtend` 라이브러리 사용. support는 해당 조건 조합이 전체에서 차지하는 비율, confidence는 조건 충족 시 Bad일 확률.

### `_search_woe_rules()` — WOE 구간 탐색

```python
for col in feature_cols:
    bins = pd.qcut(df[col], q=10, duplicates="drop")
    for bin_label, group in df.groupby(bins):
        bad_rate = group[target].sum() / len(group)
        lift     = bad_rate / overall_bad
        # min_bad_rate 초과 구간만 규칙으로 등록
```

변수 × 구간을 단순 규칙으로 만든다. 조합은 아니지만 가장 빠르고 직관적.

### 공통 필터 및 정렬

```python
rules = [r for r in rules if r.get("bad_rate", 0) >= min_bad_rate]
rules = sorted(rules, key=lambda x: x.get("lift", x.get("bad_rate", 0)), reverse=True)
rules = rules[:top_n]
```

lift 내림차순 정렬 후 상위 `top_n`개만 반환.

---

## 14. StrategyExecutor

**파일:** `executors/ml/stg_executor.py`

모델 점수를 업무 의사결정으로 변환. 4가지 전략 + 오버라이드.

### `_apply_tiered_strategy()` — 가장 복잡한 전략

```python
rules_sorted = sorted(tiered_rules, key=lambda r: r["score_min"], reverse=True)  # 높은 점수부터

for score in df[score_col]:
    for rule in rules_sorted:
        if score >= rule["score_min"]:
            grades.append(rule.get("grade", "UNKNOWN"))
            decisions.append("REJECT" if rule.get("reject") else "APPROVE")
            limit_pcts.append(rule.get("limit_pct", 0.0))
            break  # 첫 번째 매칭 규칙 적용
```

점수 내림차순으로 정렬된 규칙을 순서대로 비교하여 첫 번째 매칭 규칙 적용 (waterfall 방식).

### `_apply_matrix_strategy()` — 2차원 교차

```python
def _get_score_band(score):
    if score >= 700: return "700+"
    elif score >= 500: return "500-700"
    else: return "500-"

decision = matrix.get(ex_grade, {}).get(score_band, "REVIEW")
```

기존 등급(외부 신용등급 등) × 모델 점수 구간 → 의사결정. 정의되지 않은 조합은 "REVIEW".

### `_apply_overrides()` — 정책 강제 오버라이드

```python
for rule in override_rules:
    mask = df.eval(rule["condition"])    # pandas eval로 조건 평가
    df.loc[mask, "decision"]       = rule["decision"]
    df.loc[mask, "override_flag"]  = 1
    df.loc[mask, "override_reason"] = rule.get("reason", rule["condition"])
```

`df.eval()`로 임의 조건식 평가. 오버라이드된 건에는 `override_flag=1`, `override_reason`이 기록된다.

**오버라이드 예시:**
```python
[
    {"condition": "fraud_flag == 1",  "decision": "REJECT", "reason": "사기이력"},
    {"condition": "dsr > 0.7",        "decision": "REJECT", "reason": "고DSR"},
    {"condition": "overdue_cnt >= 3", "decision": "REJECT", "reason": "연체3회이상"},
]
```

### `_build_summary()`

```python
summary = {
    "strategy_type": ...,
    "total":         len(df),
    "decision_dist": df["decision"].value_counts().to_dict(),
    "approve_rate":  approve_cnt / len(df),
    "grade_dist":    df["grade"].value_counts().to_dict(),
    "override_count": int(df["override_flag"].sum()),
}
```

운영 모니터링용 요약. 승인율 급변, 오버라이드 건수 이상은 알람 조건으로 활용.

---

## 15. RLExecutor

**파일:** `executors/ml/rl_executor.py`

강화학습 정책 학습/평가/배포. 3가지 모드 × 4가지 알고리즘.

### 모드별 흐름

```python
def execute(self):
    mode = cfg["mode"]
    if mode == "train":    return self._run_train(cfg, algorithm)
    elif mode == "evaluate": return self._run_evaluate(cfg, algorithm)
    elif mode == "deploy":   return self._run_deploy(cfg, algorithm)
```

`deploy` 모드는 단건 실시간 추론 — 배치가 아닌 단일 state dict를 받아 action을 반환.

### `_train_q_learning()` — 테이블 기반

```python
from sklearn.preprocessing import KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
states = discretizer.fit_transform(df[state_cols].fillna(0)).astype(int)

Q = np.zeros((n_states, n_actions))

for i in range(len(df) - 1):
    s  = ravel_multi_index(states[i], ...)      # 다차원 상태 → 1D 인덱스
    s2 = ravel_multi_index(states[i+1], ...)
    a  = action_map[df.iloc[i][action_col]]
    r  = float(df.iloc[i][reward_col])
    Q[s, a] += lr * (r + gamma * Q[s2].max() - Q[s, a])  # Bellman update
```

오프라인 배치 데이터로 Q-테이블을 업데이트. `ravel_multi_index`로 다차원 이산 상태를 1D 인덱스로 변환.

**스케일 주의:** 상태 변수가 2개이고 각 10개 구간이면 Q 테이블 크기 = 10² × n_actions = 100 × n_actions.
변수가 늘어날수록 지수적으로 증가 (차원의 저주).

### `_train_bandit()` — LinUCB 컨텍스추얼 밴딧

```python
A = {a: np.eye(n_features) for a in actions}  # 피처 × 피처 행렬
b = {a: np.zeros(n_features) for a in actions} # 보상 누적 벡터

for _, row in df.iterrows():
    x, a, r = ...
    A[a] += np.outer(x, x)   # 리지 업데이트
    b[a] += r * x

# 추론 시: UCB 스코어 계산
theta = np.linalg.solve(A[a], b[a])  # 최적 파라미터 추정
p     = theta.dot(x) + alpha * sqrt(x.dot(solve(A[a], x)))  # UCB 상한
```

LinUCB: 선형 보상 모델 + 불확실성 보너스(UCB). `alpha`가 클수록 탐색(exploration) 중시.

### `_run_deploy()` — 실시간 단건 추론

```python
state = cfg.get("state")  # {"income": 5000, "debt": 3000, "age": 35}
state_array = np.array(list(state.values()), dtype=np.float32).reshape(1, -1)
actions, values = self._predict_actions(model, algorithm, state_array)

return {
    "recommended_action": int(actions[0]),
    "action_value":       float(values[0]),
}
```

state dict의 values()를 순서대로 배열로 변환. 키 순서가 학습 시와 일치해야 한다.

---

## 16. ReportExecutor

**파일:** `executors/ml/report_executor.py`

분석 결과를 JSON과 Excel로 출력.

### `_build_model_performance_report()`

```python
report = {
    "title":       ...,
    "model_info":  {"model_id": ..., "model_type": ...},
    "metrics":     meta.get("metrics", {}),         # AUC, F1 등
    "feature_importance": meta.get("varimp", {}),
}
if "prediction_path" in cfg:
    report["score_distribution"] = _decile_table(pred_df, score_col, target_col)
```

예측 결과가 있으면 데사일 테이블(10분위 분포표)을 추가한다.

### `_decile_table()` — 모듈 수준 유틸

```python
df2["decile"] = pd.qcut(df2[score_col], q=10, labels=False, duplicates="drop") + 1
for d, grp in df2.groupby("decile"):
    row = {
        "decile":    int(d),
        "count":     len(grp),
        "score_min": grp[score_col].min(),
        "score_max": grp[score_col].max(),
        "bad_count": grp[target_col].sum(),    # target_col 있을 때만
        "bad_rate":  grp[target_col].mean(),
    }
```

10개 구간별 점수 범위와 bad rate를 보여주는 표. 신용평가 성능 검증의 표준 출력.

### `_save_excel()`

```python
with pd.ExcelWriter(full_path, engine="openpyxl") as writer:
    for section_name, section_data in report_data.items():
        if isinstance(section_data, list):
            pd.DataFrame(section_data).to_excel(writer, sheet_name=section_name[:31])
        elif isinstance(section_data, dict):
            rows = [{"key": k, "value": str(v)} for k, v in section_data.items()]
            pd.DataFrame(rows).to_excel(writer, sheet_name=section_name[:31])
```

sheet_name은 Excel 제한(31자)으로 잘라냄. list → 테이블 시트, dict → key/value 시트.

---

## 17. ExportExecutor

**파일:** `executors/ml/export_executor.py`

파이프라인 최종 출구. 파일/DB/API 3가지 대상 지원.

### `_export_to_file()`

```python
if file_format == "csv":
    df.to_csv(full_path, index=False, date_format=date_fmt)
elif file_format == "excel":
    df.to_excel(full_path, index=False, engine="openpyxl")
elif file_format == "json":
    df.to_json(full_path, orient="records", force_ascii=False, indent=2, date_format="iso")
elif file_format == "parquet":
    df.to_parquet(full_path, index=False)
```

`force_ascii=False`: JSON에 한글 등 비ASCII 문자 그대로 저장.
`date_format="iso"`: JSON 날짜를 ISO 8601 형식(`2024-01-15T10:30:00`)으로 출력.

### `_export_to_db()` — upsert 구현

```python
def _upsert_to_db(self, df, table_name, key_cols):
    conn = self.db_session.bind
    where_clause = " AND ".join([f"{c} = :{c}" for c in key_cols])

    for _, row in keys.iterrows():
        params = {c: row[c] for c in key_cols}
        conn.execute(f"DELETE FROM {table_name} WHERE {where_clause}", params)

    df.to_sql(table_name, con=conn, if_exists="append", index=False, chunksize=10_000)
```

표준 SQL UPSERT(`INSERT ON DUPLICATE KEY UPDATE`) 대신 DELETE + INSERT 방식.
MariaDB/MySQL 모두 호환. 단, 트랜잭션 격리 수준에 따라 동시성 이슈 가능 — 단일 배치 작업에 적합.

### `_export_to_api()`

```python
for i in range(0, len(records), batch_size):
    batch = records[i: i + batch_size]
    resp  = requests.post(endpoint, json=batch, headers=headers, timeout=30)
    if resp.status_code not in (200, 201, 202):
        raise ExecutorException(f"API 전송 실패  status={resp.status_code}")
    total_sent += len(batch)
```

배치 크기(기본 1000건)로 나눠 전송. 실패 시 즉시 예외 — 재시도 로직은 미구현.
운영에서 재시도가 필요하다면 `tenacity` 같은 라이브러리 추가가 필요하다.

---

## 공통 패턴 정리

### 진행률 업데이트 컨벤션

각 `execute()` 구현에서의 progress 패턴:

| 단계 | progress 범위 |
|------|--------------|
| 데이터 로드 완료 | 15~20 |
| 주요 연산 진행 중 | 40~75 |
| 연산 완료, 저장 전 | 85~90 |
| 저장 완료 | 95 |
| `run()` 완료 처리 | 100 (자동) |

### 에러 처리 계층

```
ExecutorException       ← 예상된 비즈니스 오류 (입력 오류, 파일 없음 등)
      ↓ run()에서 잡아서 FAILED 반환
Exception (일반)        ← 예상치 못한 오류 (traceback 포함)
      ↓ run()에서 잡아서 FAILED 반환
```

서브클래스는 `ExecutorException`을 raise하고, 일반 예외는 자연스럽게 버블업시킨다.

### 메타 JSON 저장 규약

학습 executor들이 저장하는 `{model_id}_meta.json`의 공통 키:

```json
{
  "model_id":     "...",
  "model_type":   "python" | "h2o" | "r",
  "model_path":   "models/...",
  "feature_cols": [...],
  "target_col":   "...",
  "metrics":      {...}
}
```

PredictExecutor와 PretrainedExecutor가 이 구조를 읽어 모델을 로드하므로 반드시 이 키들이 있어야 한다.
