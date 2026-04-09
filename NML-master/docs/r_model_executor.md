# r_model_executor.py — R 기반 모델 학습/예측

**파일:** `executors/ml/r_model_executor.py`  
**클래스:** `RModelExecutor(BaseExecutor)`

## 개요

R 스크립트를 subprocess로 호출하여 학습·예측을 수행하는 executor.  
Python ↔ R 간 데이터 교환은 **CSV 임시파일 + JSON 인자파일**을 통해 이루어진다.

```
Python 데이터 (.parquet)
    ↓ CSV 임시파일로 변환 (tempfile)
    ↓ args JSON 파일 생성
    ↓ subprocess: Rscript {r_script} --args_file {path}
    ↓ R 결과 파일 읽기 (meta JSON, 예측 CSV)
    ↓ 결과 저장 및 반환
```

**주요 활용:**
- 기존 R 자산(통계모형, 스코어카드) 재사용
- R 전용 라이브러리 활용 (glm, caret, survival, creditR 등)
- 레거시 R 코드와의 통합 운영

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `mode` | ✅ | `str` | `"train"` \| `"predict"` |
| `r_script` | ✅ | `str` | R 스크립트 경로 (file_root 기준) |
| `model_id` | ✅ | `str` | 모델 저장 식별자 |
| `train_path` | train 필수 | `str` | 학습 데이터 경로 (.parquet) |
| `target_col` | train 필수 | `str` | 타깃 컬럼명 |
| `input_path` | predict 필수 | `str` | 예측 대상 데이터 경로 (.parquet) |
| `valid_path` | ❌ | `str` | 검증 데이터 경로 (train 모드) |
| `r_args` | ❌ | `dict` | R 스크립트에 전달할 추가 인자 |
| `rscript_cmd` | ❌ | `str` | Rscript 실행 명령 (기본: `"Rscript"`) |
| `output_id` | ❌ | `str` | 예측 결과 파일 식별자 (predict 모드) |

---

## R 스크립트 인터페이스

R 스크립트는 `--args_file <path>` 형태로 실행된다.  
`args_file`은 JSON 파일이며 다음 키를 포함한다.

### train 모드 args JSON

```json
{
  "mode":       "train",
  "train_path": "/tmp/xxx/train.csv",
  "valid_path": "/tmp/xxx/valid.csv",
  "model_dir":  "/data/models/my_model_id",
  "target_col": "default",
  "meta_path":  "/tmp/xxx/meta.json"
}
```

R 스크립트는 학습 후 `meta_path`에 JSON을 생성해야 한다.  
최종적으로 `models/{model_id}_meta.json`으로 저장된다.

### predict 모드 args JSON

```json
{
  "mode":       "predict",
  "input_path": "/tmp/xxx/input.csv",
  "model_dir":  "/data/models/my_model_id",
  "output_path": "/tmp/xxx/predictions.csv"
}
```

R 스크립트는 `output_path`에 예측 결과 CSV를 생성해야 한다.

---

## R 스크립트 실행 방식

```bash
Rscript r_scripts/train_glm.R --args_file /tmp/args_xxxx.json
```

subprocess 타임아웃 없이 실행된다.  
returncode가 0이 아니면 `ExecutorException` 발생 (stderr 포함).

---

## train 모드 실행 흐름

1. train/valid parquet → CSV 임시파일
2. args JSON 파일 생성 (model_dir, target_col, meta_path 포함)
3. Rscript 실행
4. R이 생성한 `meta.json` 읽기
5. `models/{model_id}_meta.json`으로 복사 저장
6. 임시 디렉토리 정리

## predict 모드 실행 흐름

1. input parquet → CSV 임시파일
2. args JSON 파일 생성 (model_dir, output_path 포함)
3. Rscript 실행
4. R이 생성한 `predictions.csv` 읽기
5. `predict/{output_id}_r_result.parquet`로 저장
6. 임시 디렉토리 정리

---

## 반환값 (train 모드)

```python
{
    "status": "COMPLETED",
    "result": {
        "model_id":    "glm_credit_v1",
        "model_dir":   "/data/models/glm_credit_v1",
        "meta_path":   "models/glm_credit_v1_meta.json",
        "r_meta":      {...},    # R 스크립트가 생성한 메타 JSON 내용
    },
    "message": "R 모델 학습 완료  glm_credit_v1",
    "job_id":  str,
    "elapsed_sec": float,
}
```

## 반환값 (predict 모드)

```python
{
    "status": "COMPLETED",
    "result": {
        "output_id":   "credit_pred_v1",
        "output_path": "predict/credit_pred_v1_r_result.parquet",
        "total_rows":  50000,
    },
    "message": "R 예측 완료  50,000건",
    "job_id":  str,
    "elapsed_sec": float,
}
```

---

## R 스크립트 작성 가이드

```r
# R 스크립트 내부에서 args 읽기
library(jsonlite)
args_file <- commandArgs(trailingOnly=TRUE)[which(commandArgs(trailingOnly=TRUE) == "--args_file") + 1]
args <- fromJSON(args_file)

train_df  <- read.csv(args$train_path)
target    <- args$target_col
model_dir <- args$model_dir

# 학습
model <- glm(as.formula(paste(target, "~ .")), data=train_df, family=binomial)

# 메타 저장 (Python이 읽을 JSON)
meta <- list(model_id=basename(model_dir), auc=auc_val, ...)
write_json(meta, args$meta_path)
saveRDS(model, file.path(model_dir, "model.rds"))
```

---

## 사용 예시

```python
config = {
    "job_id":     "r_train_001",
    "mode":       "train",
    "r_script":   "r_scripts/train_glm.R",
    "model_id":   "glm_credit_v1",
    "train_path": "mart/loan_mart_train.parquet",
    "valid_path": "mart/loan_mart_valid.parquet",
    "target_col": "default",
    "r_args":     {"max_iter": 1000},
}

from executors.ml.r_model_executor import RModelExecutor
result = RModelExecutor(config=config).run()
```
