# r_model_executor.py

R 기반 모델 학습 및 예측 실행기.

R 스크립트를 subprocess로 호출하여 학습/예측을 수행한다.  
Python ↔ R 간 데이터 교환은 CSV 파일(입력)과 JSON/CSV 파일(출력)을 통해 이루어진다.

**사용 목적**
- 기존 R 자산(통계모형, 스코어카드) 재사용
- R 전용 라이브러리 활용 (glm, caret, survival, creditR 등)
- 레거시 R 코드와의 통합 운영

---

## 클래스

### `RModelExecutor(BaseExecutor)`

R 모델 실행 executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `r_script` | `str` | 실행할 R 스크립트 경로 (FILE_ROOT_DIR 기준) |
| `mode` | `str` | `"train"` \| `"predict"` |
| `model_id` | `str` | 모델 저장 식별자 |

**train 모드 추가 필수 키**

| 키 | 타입 | 설명 |
|---|---|---|
| `train_path` | `str` | 학습 데이터 경로 (`.parquet`) |
| `target_col` | `str` | 타깃 컬럼명 |

**predict 모드 추가 필수 키**

| 키 | 타입 | 설명 |
|---|---|---|
| `input_path` | `str` | 예측 대상 데이터 경로 (`.parquet`) |

#### config 선택 키

| 키 | 타입 | 설명 |
|---|---|---|
| `r_args` | `dict` | R 스크립트에 전달할 추가 인자 |
| `rscript_cmd` | `str` | Rscript 실행 명령 (기본: `"Rscript"`) |
| `valid_path` | `str` | 검증 데이터 경로 (train 모드, 선택) |
| `output_id` | `str` | 예측 결과 저장 식별자 (predict 모드, 기본: model_id) |

---

### `execute() → dict`

`mode`에 따라 `_run_train()` 또는 `_run_predict()`를 호출한다.  
미지원 mode는 `ExecutorException` 발생.

---

### `_run_train(cfg) → dict`

R 스크립트를 통해 모델을 학습하고 메타 정보를 저장한다.

**실행 순서**
1. 학습/검증 데이터를 임시 CSV 파일로 저장
2. `model_dir` 생성 (`models/{model_id}/`)
3. R 스크립트에 전달할 인자 딕셔너리를 JSON으로 준비:
   - `mode`, `train_path`, `valid_path`, `model_dir`, `target_col`, `meta_path`, `r_args`
4. `_run_rscript()` 호출
5. R 스크립트가 생성한 `meta.json` 읽기 (없으면 기본 메타 사용)
6. 메타 정보를 `models/{model_id}_meta.json`에 저장

**반환값**
```python
{
    "status": "COMPLETED",
    "result": {   # 메타 정보
        "model_id":   str,
        "model_type": "r",
        "r_script":   str,
        "target_col": str,
        "model_path": str,
        ...        # R 스크립트 생성 메타 포함
    },
    "message": str,
}
```

---

### `_run_predict(cfg) → dict`

학습된 R 모델로 예측을 수행한다.

**실행 순서**
1. 예측 대상 데이터 로드
2. `models/{model_id}_meta.json`에서 모델 경로 및 R 스크립트 경로 로드
3. 입력 데이터를 임시 CSV로 저장
4. R 스크립트에 `mode=predict`, `input_path`, `output_path`, `model_dir` 전달
5. `_run_rscript()` 호출
6. R이 생성한 `output.csv`를 읽어 원본 데이터와 수평 병합
7. 결과를 `predict/{output_id}_r_result.parquet`에 저장

**반환값**
```python
{
    "status": "COMPLETED",
    "result": {
        "output_path": str,
        "total_rows":  int,
        "pred_cols":   list[str],
    },
    "message": str,
}
```

---

### `_run_rscript(script_rel_path, args, rscript_cmd="Rscript")`

R 스크립트를 subprocess로 실행한다.

**실행 방식**
1. `args` 딕셔너리를 임시 JSON 파일로 저장
2. `{rscript_cmd} {script_path} --args_file {json_file}` 명령 실행
3. stdout은 INFO 로그, stderr는 WARNING 로그로 출력
4. 임시 JSON 파일 삭제
5. returncode가 0이 아니면 `ExecutorException` 발생 (stderr 마지막 500자 포함)

| 파라미터 | 설명 |
|---|---|
| `script_rel_path` | R 스크립트의 FILE_ROOT_DIR 기준 상대 경로 |
| `args` | R 스크립트에 전달할 인자 딕셔너리 |
| `rscript_cmd` | Rscript 실행 명령어 (기본: `"Rscript"`) |

timeout: 3600초 (1시간)
