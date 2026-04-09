# pretrained_executor.py — 사전 학습 모델 추론

**파일:** `executors/ml/pretrained_executor.py`  
**클래스:** `PretrainedExecutor(BaseExecutor)`

## 개요

재학습 없이 기존 완성 모델을 로드하여 임베딩 추출, 피처 생성,  
또는 최종 예측을 수행하는 executor.

**사용 사례:**
- 내부 리스크 팀이 이미 학습/검증한 모델을 운영에 배포
- 외부 공개 pretrained 모델(ONNX, HuggingFace) 활용
- Transfer Learning의 feature extractor 단계로 활용
- A/B 테스트를 위한 챔피언/챌린저 모델 동시 배포

```
모델 메타 JSON 로드 (_load_meta)
    ↓ 입력 데이터 로드
    ↓ model_format에 따라 추론
      ├── pickle   → _infer_pickle()
      ├── onnx     → _infer_onnx()
      ├── h2o      → _infer_h2o()
      └── hugging  → _infer_hugging()
    ↓ predict/{output_id}_pretrained.parquet 저장
```

---

## 지원 모델 포맷 (`model_format`)

| `model_format` | 파일 형식 | 추론 방식 |
|---|---|---|
| `pickle` | `.pkl` | scikit-learn / XGBoost / LightGBM 등 |
| `onnx` | `.onnx` | ONNX Runtime (`onnxruntime`) |
| `h2o` | MOJO `.zip` | H2O MOJO (`h2o.import_mojo`) |
| `hugging` | HuggingFace 모델 | `transformers` AutoModel |

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `model_id` | ✅ | `str` | 모델 식별자 (`models/{model_id}_meta.json` 참조) |
| `input_path` | ✅ | `str` | 입력 데이터 경로 (.parquet) |
| `output_id` | ✅ | `str` | 결과 저장 식별자 |
| `model_format` | ❌ | `str` | 포맷 지정 (미지정 시 메타 JSON의 `model_format` 자동 감지) |
| `score_col` | ❌ | `str` | 예측 점수 컬럼명 (기본: `"score"`) |
| `output_mode` | ❌ | `str` | `"score"` \| `"embedding"` \| `"both"` (기본: `"score"`) |
| `batch_size` | ❌ | `int` | 배치 추론 크기 (기본: `10000`, ONNX/HuggingFace 사용 시) |

---

## 내부 메서드

### `_load_meta(model_id)` → `dict`

`models/{model_id}_meta.json`을 읽어 반환한다.  
파일이 없으면 `ExecutorException` 발생.

### `_infer_pickle(meta, X, result_df, score_col, output_mode)` → `pd.DataFrame`

```python
model_path = self.file_root / meta["model_path"]
model = pickle.load(open(model_path, "rb"))

# output_mode in ("score", "both") + predict_proba 지원 시
result_df[score_col] = model.predict_proba(X)[:, 1].round(6)

# output_mode in ("embedding", "both") + transform 지원 시
emb = model.transform(X)   # embedding 추출
```

### `_infer_onnx(meta, X, result_df, score_col, batch_size)` → `pd.DataFrame`

```python
import onnxruntime as rt
sess = rt.InferenceSession(meta["model_path"])
# batch_size 단위로 분할하여 추론
```

### `_infer_h2o(meta, X, result_df, score_col)` → `pd.DataFrame`

```python
import h2o
model = h2o.import_mojo(meta["mojo_path"])
preds = model.predict(h2o.H2OFrame(X)).as_data_frame()
result_df[score_col] = preds.iloc[:, -1].values
```

### `_infer_hugging(meta, X, result_df, batch_size)` → `pd.DataFrame`

HuggingFace `transformers`를 사용한 배치 추론.  
텍스트 컬럼을 embedding 벡터로 변환하거나 분류 점수 산출.

---

## execute() 진행률

| 단계 | progress |
|------|----------|
| 메타 로드 완료 | 15% |
| 데이터 로드 완료 | 30% |
| 추론 완료 | 85% |
| 저장 완료 | 95% |

---

## 반환값

```python
{
    "status": "COMPLETED",
    "result": {
        "output_id":    "champion_pred_202312",
        "model_id":     "lgbm_champion_v3",
        "model_format": "pickle",
        "output_path":  "predict/champion_pred_202312_pretrained.parquet",
        "total_rows":   50000,
    },
    "message": "Pretrained 추론 완료  50,000건  format=pickle",
    "job_id":  str,
    "elapsed_sec": float,
}
```

---

## 출력 파일

| 파일 | 경로 |
|------|------|
| 추론 결과 | `predict/{output_id}_pretrained.parquet` |

---

## 챔피언/챌린저 A/B 테스트 패턴

```python
# 챔피언 모델 추론
config_champion = {
    "model_id": "lgbm_champion_v3", "input_path": "mart/new_data.parquet",
    "output_id": "champion_pred", "model_format": "pickle",
}

# 챌린저 모델 추론
config_challenger = {
    "model_id": "gbm_challenger_v1", "input_path": "mart/new_data.parquet",
    "output_id": "challenger_pred", "model_format": "h2o",
}

# 두 결과 비교 후 우수 모델 승격
```

---

## 사용 예시

```python
config = {
    "job_id":       "pretrained_001",
    "model_id":     "lgbm_champion_v3",
    "input_path":   "mart/new_applicants.parquet",
    "output_id":    "new_score_202312",
    "model_format": "pickle",
    "score_col":    "ml_score",
    "output_mode":  "score",
}

from executors.ml.pretrained_executor import PretrainedExecutor
result = PretrainedExecutor(config=config).run()
```
