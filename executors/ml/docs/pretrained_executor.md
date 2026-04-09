# pretrained_executor.py

사전 학습된(pretrained) 모델을 활용한 추론(inference) 실행기.

재학습 없이 기존 모델을 로드하여 임베딩 추출, 피처 생성, 또는 최종 예측을 수행한다.

---

## 클래스

### `PretrainedExecutor(BaseExecutor)`

사전 학습 모델 추론 executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `model_id` | `str` | 모델 식별자 (메타 JSON 기준) |
| `input_path` | `str` | 입력 데이터 경로 (`.parquet`) |
| `output_id` | `str` | 결과 저장 식별자 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|---|---|---|
| `model_format` | 메타에서 자동 감지 | `"pickle"` \| `"onnx"` \| `"h2o"` \| `"hugging"` |
| `score_col` | `"score"` | 예측 점수 컬럼명 |
| `output_mode` | `"score"` | `"score"` \| `"embedding"` \| `"both"` |
| `batch_size` | `10000` | 배치 추론 크기 (ONNX/HuggingFace 사용 시) |

---

### `execute() → dict`

사전 학습 모델 추론 전체 파이프라인을 실행한다.

**실행 순서**
1. `_load_meta()`로 모델 메타 로드
2. `model_format` 결정 (config 우선, 없으면 메타에서)
3. 입력 데이터 로드, feature_cols로 피처 선택
4. `model_format`에 따라 추론 메서드 호출
5. 결과를 `predict/{output_id}_pretrained.parquet`에 저장

**반환값**
```python
{
    "status": "COMPLETED",
    "result": {
        "output_id":    str,
        "model_id":     str,
        "model_format": str,
        "output_path":  str,
        "total_rows":   int,
    },
    "message": str,
}
```

---

### `_load_meta(model_id) → dict`

`models/{model_id}_meta.json` 파일을 읽어 반환한다.  
파일이 없으면 `ExecutorException` 발생.

---

### `_infer_pickle(meta, X, result_df, score_col, output_mode) → pd.DataFrame`

pickle 형식(.pkl) 모델로 추론한다. scikit-learn, XGBoost, LightGBM 등에 사용.

| `output_mode` | 동작 |
|---|---|
| `"score"` | `predict_proba()[:, 1]` 또는 `predict()` 결과를 `score_col`에 저장 |
| `"embedding"` | `transform()`으로 임베딩 추출, `emb_0`, `emb_1`, ... 컬럼 추가 |
| `"both"` | score와 embedding 모두 추출 |

---

### `_infer_onnx(meta, X, result_df, score_col, batch_size) → pd.DataFrame`

ONNX Runtime으로 배치 추론을 수행한다.

1. `onnxruntime.InferenceSession(model_path)` 초기화
2. 입력을 `float32`로 변환 후 `batch_size` 단위로 배치 처리
3. 출력이 2개 이상이면 확률 행렬의 양성 클래스 컬럼, 아니면 출력값 flatten

---

### `_infer_h2o(meta, X, result_df, score_col) → pd.DataFrame`

H2O MOJO 모델로 추론한다.

1. `h2o.init()` 후 `h2o.import_mojo(model_path)` 로드
2. `h2o.H2OFrame(X)`으로 변환 후 `model.predict()` 호출
3. 예측 결과의 마지막 컬럼(양성 클래스 확률)을 `score_col`에 저장

---

### `_infer_hugging(meta, X, result_df, batch_size) → pd.DataFrame`

HuggingFace transformers 텍스트 분류 파이프라인으로 추론한다.

1. `pipeline("text-classification", model=model_path)` 초기화
2. `meta["text_col"]` 컬럼의 텍스트를 `batch_size` 단위로 배치 처리
3. `pred_label`, `pred_score` 컬럼 추가 (truncation=True 적용)

---

## 모듈 레벨 함수

### `preload_mdl(json_obj) → dict`

Base64로 인코딩된 MDL 파일 바이트를 디코딩하고 모델 타입 및 VarLayout을 반환한다.

- `json_obj["MdlFileBytes"]` Base64 디코딩
- `json_obj["VarLayout"]` JSON → DataFrame 파싱
- pickle 언피클 시도 (실패하면 바이트 원본 유지)

**반환값**: `{ModelType, VarLayout, ModelBytes, ModelObj}`

---

### `load_model_file(model_type, model_file_path, h2o_host, ...) → object`

모델 파일 경로에서 모델 객체를 로드한다.

- H2O 타입 목록에 해당하면 `h2o.import_mojo()` 시도 (실패 시 fallback)
- 그 외는 `pickle.load()`로 로드

**H2O 타입 목록**: `RANDOMFOREST`, `DEEPLEARNING`, `GRADIENTBOOSTING`, `GLM`, `XGBOOST`, `AUTOML`, `AUTOMLGLM`, `AUTOMLDL`, `KMEANS`, `ANOMALY` 등

---

### `load_pretrained_model(service_db_info, ..., json_obj) → dict`

파일 서버에서 사전 학습된 모델을 로드하고 메타 정보를 반환한다.

1. `models/{model_key}_meta.json` 파일 조회
2. 없으면 파일 서버 HTTP API(`/model/{model_key}/meta`)로 조회 시도
3. `load_model_file()`로 모델 객체 로드

**반환값**: `{model, meta, model_type, model_id}`

---

### `predict_pretrained_model(..., json_obj) → dict`

사전 학습 모델로 예측을 수행한다.

- H2O 타입이면 `h2o.H2OFrame`으로 예측, 아니면 `predict_proba()` 또는 `predict()` 사용
- `reverse_prob=True`이면 `proba = 1 - proba` 적용
- `target_col`이 있으면 AUC 계산
- 결과를 `output/{model_id}_pretrained_predict.parquet`에 저장

---

### `convert_ensemble_to_rclips_model(..., json_obj) → dict`

앙상블 모델을 rclips 호환 포맷으로 변환한다.

- `ensemble_config["member_ids"]` 각 멤버의 메타를 로드
- rclips 포맷 딕셔너리를 `models/{model_id}_rclips.pkl`에 저장
