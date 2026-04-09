# pretrained_executor.py — 사전 학습 모델 추론

## 개요

재학습 없이 기존에 학습된 모델을 로드하여 임베딩 추출, 피처 생성, 최종 예측을 수행하는 executor.  
내부 검증 완료 모델의 운영 배포, 외부 공개 모델 활용, A/B 테스트에 활용된다.

---

## 지원 모델 포맷 (`model_format`)

| `model_format` | 확장자 | 로드 방식 | 주 활용 모델 |
|---|---|---|---|
| `pickle` | `.pkl` | `pickle.load()` | scikit-learn, XGBoost, LightGBM |
| `onnx` | `.onnx` | `onnxruntime.InferenceSession` | PyTorch, TensorFlow 변환 모델 |
| `h2o` | `.zip` | `h2o.import_mojo()` | H2O GBM, DRF, XGBoost |
| `hugging` | 디렉토리 | `transformers.pipeline` | HuggingFace 텍스트 분류 모델 |

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `model_id` | ✅ | `str` | 모델 식별자 (메타 JSON 기준) |
| `input_path` | ✅ | `str` | 입력 데이터 경로 (.parquet) |
| `output_id` | ✅ | `str` | 결과 저장 식별자 |
| `model_format` | ❌ | `str` | 모델 포맷 (미지정 시 메타에서 자동 감지) |
| `score_col` | ❌ | `str` | 예측 점수 컬럼명 (기본: `"score"`) |
| `output_mode` | ❌ | `str` | `"score"` \| `"embedding"` \| `"both"` (기본: `"score"`) |
| `batch_size` | ❌ | `int` | 배치 추론 크기 (기본: `10000`, ONNX/HuggingFace) |

---

## 포맷별 추론 메서드

### `_infer_pickle(meta, X, result_df, score_col, output_mode)`

```python
model = pickle.load(open(model_path, "rb"))

# score 모드: predict_proba 지원 시
result_df[score_col] = model.predict_proba(X)[:, 1]

# embedding 모드: transform 지원 시 (PCA, Autoencoder 등)
embedding = model.transform(X)
# → emb_0, emb_1, ..., emb_N 컬럼으로 추가
```

---

### `_infer_onnx(meta, X, result_df, score_col, batch_size)`

```python
sess = ort.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name

for batch in chunks(X, batch_size):
    preds = sess.run(None, {input_name: batch.astype(np.float32)})
    # 분류: preds[1][:, 1] (확률)
    # 회귀: preds[0].flatten()
```

- 배치 단위 추론으로 메모리 효율 확보
- float32 강제 변환 필수 (ONNX Runtime 요구)

---

### `_infer_h2o(meta, X, result_df, score_col)`

```python
h2o.init()
model = h2o.import_mojo(meta["model_path"])
preds = model.predict(h2o.H2OFrame(X)).as_data_frame()
result_df[score_col] = preds.iloc[:, -1].values
```

---

### `_infer_hugging(meta, X, result_df, batch_size)`

HuggingFace `text-classification` 파이프라인으로 텍스트 분류.

```python
pipe = pipeline("text-classification", model=model_path)
for batch in chunks(texts, batch_size):
    results = pipe(batch, truncation=True)
    # → pred_label, pred_score 컬럼 추가
```

메타의 `text_col` 키로 텍스트 컬럼을 지정한다.

---

## 실행 흐름

```
1. models/{model_id}_meta.json 로드                  [progress 15%]
2. input_path 데이터 로드
3. feature_cols 정렬 (메타 기준)                     [progress 30%]
4. 포맷별 추론                                       [progress 85%]
   - pickle  → _infer_pickle
   - onnx    → _infer_onnx (배치)
   - h2o     → _infer_h2o
   - hugging → _infer_hugging (배치)
5. predict/{output_id}_pretrained.parquet 저장
```

---

## 출력 결과

**저장 경로:** `predict/{output_id}_pretrained.parquet`

추가 컬럼:
- `score` (또는 `score_col`) — score/both 모드
- `emb_0`, `emb_1`, ... — embedding/both 모드 (pickle transform)
- `pred_label`, `pred_score` — HuggingFace 모드

**반환 요약:**
```python
{
    "output_id":    "deploy_run_001",
    "model_id":     "champion_gbm",
    "model_format": "h2o",
    "output_path":  "predict/deploy_run_001_pretrained.parquet",
    "total_rows":   200000,
}
```

---

## 사용 시나리오

### 챔피언/챌린저 A/B 테스트

```python
# Champion 모델 (기존 운영 모델)
champion_cfg = {"model_id": "gbm_v3", "output_id": "champion_pred", ...}

# Challenger 모델 (신규 후보)
challenger_cfg = {"model_id": "lgbm_v5", "output_id": "challenger_pred", ...}

# 두 결과 비교 후 성능 우수 모델을 승격
```

### Transfer Learning 피처 추출

```python
config = {
    "model_id":    "autoencoder_v1",
    "output_mode": "embedding",   # transform() 결과를 컬럼으로 추출
    "model_format": "pickle",
}
# → emb_0 ~ emb_N 컬럼을 downstream 모델의 입력 피처로 활용
```
