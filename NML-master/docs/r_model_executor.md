# r_model_executor.py — R 기반 모델 학습/예측

## 개요

R 스크립트를 subprocess로 호출하여 학습·예측을 수행하는 executor.  
Python ↔ R 간 데이터 교환은 CSV 임시파일 + JSON 인자파일을 통해 이루어진다.

```
Python 데이터 (parquet)
    ↓ CSV 임시파일로 저장
    ↓ Rscript 실행 (--args_file)
    ↓ R 스크립트 결과(모델 메타 JSON, 예측 CSV) 읽기
    ↓ 결과 저장 및 반환
```

주요 활용:
- 기존 R 자산(통계모형, 스코어카드) 재사용
- R 전용 라이브러리 활용 (glm, caret, survival, creditR 등)
- 레거시 R 코드와의 통합 운영

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `mode` | ✅ | `str` | `"train"` \| `"predict"` |
| `r_script` | ✅ | `str` | R 스크립트 경로 (FILE_ROOT_DIR 기준) |
| `model_id` | ✅ | `str` | 모델 저장 식별자 |
| `train_path` | train필수 | `str` | 학습 데이터 경로 (.parquet) |
| `target_col` | train필수 | `str` | 타깃 컬럼명 |
| `input_path` | predict필수 | `str` | 예측 대상 데이터 경로 (.parquet) |
| `valid_path` | ❌ | `str` | 검증 데이터 경로 (train 모드) |
| `r_args` | ❌ | `dict` | R 스크립트에 전달할 추가 인자 |
| `rscript_cmd` | ❌ | `str` | Rscript 실행 명령 (기본: `"Rscript"`) |
| `output_id` | ❌ | `str` | 예측 결과 파일 식별자 (predict 모드) |

---

## R 스크립트 인터페이스

R 스크립트는 `--args_file <path>` 형태로 실행된다.  
args_file은 JSON 파일이며 다음 키를 포함한다.

### train 모드

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

R 스크립트는 학습 후 `meta_path`에 JSON 파일을 생성해야 한다.  
이 파일은 최종적으로 `models/{model_id}_meta.json`으로 저장된다.

### predict 모드

```json
{
  "mode":        "predict",
  "input_path":  "/tmp/xxx/input.csv",
  "output_path": "/tmp/xxx/output.csv",
  "model_dir":   "/data/models/my_model_id"
}
```

R 스크립트는 예측 후 `output_path`에 CSV 파일을 생성해야 한다.  
예측 CSV는 원본 데이터에 컬럼으로 병합되어 parquet으로 저장된다.

---

## 파일 저장 규칙

| 파일 | 경로 |
|------|------|
| 모델 파일 (R 스크립트 생성) | `models/{model_id}/` |
| 모델 메타 | `models/{model_id}_meta.json` |
| 예측 결과 | `predict/{output_id}_r_result.parquet` |

---

## 사용 예시

### train

```python
config = {
    "mode":       "train",
    "r_script":   "scripts/logit_train.R",
    "model_id":   "r_logit_v1",
    "train_path": "mart/loan_train.parquet",
    "valid_path": "mart/loan_valid.parquet",
    "target_col": "default",
    "r_args":     {"max_iter": 200},
}
```

### predict

```python
config = {
    "mode":       "predict",
    "r_script":   "scripts/logit_train.R",
    "model_id":   "r_logit_v1",
    "input_path": "mart/loan_test.parquet",
    "output_id":  "loan_pred_r_v1",
}
```

---

## 표준 출력 (result)

### train

```json
{
  "status":  "COMPLETED",
  "result":  {
    "model_id":   "r_logit_v1",
    "model_type": "r",
    "r_script":   "scripts/logit_train.R",
    "target_col": "default",
    "model_path": "models/r_logit_v1"
  },
  "message": "R 모델 학습 완료  model_id=r_logit_v1"
}
```

### predict

```json
{
  "status":  "COMPLETED",
  "result":  {
    "output_path": "predict/loan_pred_r_v1_r_result.parquet",
    "total_rows":  10000,
    "pred_cols":   ["score", "prob"]
  },
  "message": "R 예측 완료  10,000건"
}
```

---

## 주의사항

- R 스크립트가 `meta_path` JSON을 생성하지 않으면 최소 메타(`model_id`, `model_dir`)만 저장된다.
- R 스크립트가 예측 CSV를 생성하지 않으면 `ExecutorException`이 발생한다.
- `rscript_cmd`는 시스템에 따라 `"Rscript"`, `"/usr/bin/Rscript"` 등으로 지정한다.
- R 스크립트 실행 timeout은 3600초(1시간)이다.
- stderr 출력은 WARNING 레벨로 로깅되며 returncode != 0이면 예외가 발생한다.
