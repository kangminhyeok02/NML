# predict_executor.py

저장된 모델을 로드하여 신규 데이터에 대한 예측을 수행하는 실행기.

운영 환경에서 가장 빈번하게 호출되는 executor로,  
점수(score), 확률(probability), 등급(grade) 형태의 예측 결과를 생성한다.

---

## 클래스

### `PredictExecutor(BaseExecutor)`

예측 실행 executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `model_id` | `str` | 사용할 모델 식별자 |
| `input_path` | `str` | 예측 대상 데이터 상대 경로 |
| `output_id` | `str` | 결과 저장 식별자 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|---|---|---|
| `score_col` | `"score"` | 예측 점수 컬럼명 |
| `grade_mapping` | - | 점수 → 등급 매핑 구간 (예: `{"A": [800, 1000]}`) |
| `threshold` | `0.5` | 이진 분류 임계값 |
| `output_path` | 자동 생성 | 결과 파일 저장 경로 |
| `model_type` | `"python"` | `"python"` \| `"h2o"` \| `"r"` |

---

### `execute() → dict`

예측 전체 파이프라인을 실행한다.

**실행 순서**
1. 모델 메타 정보 로드 (`_load_model_meta`)
2. 모델 파일 로드 (`_load_model`)
3. 예측 대상 데이터 로드
4. 피처 컬럼 정렬 및 누락 컬럼 검증
5. 모델 타입별 예측 수행
6. `grade_mapping`이 있으면 점수 → 등급 변환
7. 결과를 parquet으로 저장

**반환값**
```python
{
    "status": "COMPLETED",
    "result": {
        "output_id":   str,
        "model_id":    str,
        "total_rows":  int,
        "output_path": str,
        "score_stats": {mean, std, min, p25, p50, p75, max},
        "grade_dist":  dict,   # grade_mapping 지정 시
    },
    "message": str,
}
```

---

### `_load_model_meta(model_id) → dict`

`models/{model_id}_meta.json` 파일을 읽어 반환한다.  
파일이 없으면 `ExecutorException` 발생.

---

### `_load_model(meta, model_type)`

모델 타입에 따라 모델 객체를 로드한다.

| `model_type` | 동작 |
|---|---|
| `"python"` | `pickle.load()`로 모델 파일 로드 |
| `"h2o"` | `h2o.init()` 후 `h2o.import_mojo()` 로드 |
| `"r"` | R 모델은 메타 정보만 반환 (실제 예측은 subprocess) |

---

### `_predict_python(model, X, result_df, score_col, cfg) → pd.DataFrame`

Python 모델로 예측을 수행한다.

- `predict_proba()` 지원 모델: 확률값(양성 클래스)을 `score_col`에 저장, `pred_class` 컬럼도 추가
- `predict()` 전용 모델: 예측값을 `score_col`에 저장
- `threshold`로 이진 분류 임계값 적용

---

### `_predict_h2o(model, X, result_df, score_col) → pd.DataFrame`

H2O MOJO 모델로 예측을 수행한다.

- `h2o.H2OFrame(X)`으로 변환 후 `model.predict()` 호출
- 예측 결과의 마지막 컬럼(양성 클래스 확률)을 `score_col`에 저장

---

### `_predict_r(meta, X, result_df, score_col) → pd.DataFrame`

R 스크립트를 subprocess로 호출하여 예측을 수행한다.

1. 입력 데이터를 임시 CSV 파일로 저장
2. `Rscript {r_script} --input {tmp} --model {model_path} --output {output}` 실행
3. 결과 CSV를 읽어 `score_col`에 저장

---

### `_assign_grade(score, grade_mapping) → str`

점수를 `grade_mapping` 구간에 따라 등급 문자열로 변환한다.

```python
# grade_mapping 예시:
# {"A": [800, 1000], "B": [600, 800], "C": [0, 600]}
```

매핑 구간에 없으면 `"UNKNOWN"` 반환.

---

## 모듈 레벨 함수

### `_series_stats(s) → dict`

Series의 기술통계를 반환한다.

```python
{mean, std, min, p25, p50, p75, max}
```
