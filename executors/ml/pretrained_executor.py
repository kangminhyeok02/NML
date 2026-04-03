"""
pretrained_executor.py
----------------------
사전 학습된(pretrained) 모델을 활용한 추론(inference) 실행기.

재학습 없이 기존 모델을 로드하여 임베딩 추출, 피처 생성,
또는 최종 예측을 수행한다.

사용 사례:
  - 내부 리스크 팀이 이미 학습/검증한 모델을 운영에 배포
  - 외부 공개 pretrained 모델(HuggingFace, ONNX 등) 활용
  - Transfer Learning의 feature extractor 단계로 활용
  - A/B 테스트를 위한 챔피언/챌린저 모델 동시 배포

모델 형식 지원:
  - pickle  (.pkl): scikit-learn / XGBoost / LightGBM
  - onnx    (.onnx): ONNX Runtime 기반 추론
  - h2o     (MOJO): H2O MOJO
  - hugging : HuggingFace transformers
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)


class PretrainedExecutor(BaseExecutor):
    """
    사전 학습 모델 추론 executor.

    config 필수 키
    --------------
    model_id    : str   모델 식별자 (메타 JSON 기준)
    input_path  : str   입력 데이터 경로 (.parquet)
    output_id   : str   결과 저장 식별자

    config 선택 키
    --------------
    model_format : str   "pickle" | "onnx" | "h2o" | "hugging" (기본: 메타에서 자동 감지)
    score_col    : str   예측 점수 컬럼명 (기본: "score")
    output_mode  : str   "score" | "embedding" | "both" (기본: "score")
    batch_size   : int   배치 추론 크기 (기본: 10000, ONNX/HuggingFace 사용 시)
    """

    def execute(self) -> dict:
        cfg = self.config

        # 모델 메타 로드
        meta = self._load_meta(cfg["model_id"])
        model_format = cfg.get("model_format") or meta.get("model_format", "pickle")
        output_mode  = cfg.get("output_mode", "score")
        score_col    = cfg.get("score_col", "score")

        self._update_job_status(ExecutorStatus.RUNNING, progress=15)

        # 데이터 로드
        df = self._load_dataframe(cfg["input_path"])
        feature_cols = meta.get("feature_cols") or list(df.columns)
        X = df[[c for c in feature_cols if c in df.columns]]
        logger.info("pretrained inference  format=%s  rows=%d  features=%d", model_format, len(X), len(X.columns))
        self._update_job_status(ExecutorStatus.RUNNING, progress=30)

        # 추론
        result_df = df.copy()
        if model_format == "pickle":
            result_df = self._infer_pickle(meta, X, result_df, score_col, output_mode)
        elif model_format == "onnx":
            result_df = self._infer_onnx(meta, X, result_df, score_col, cfg.get("batch_size", 10000))
        elif model_format == "h2o":
            result_df = self._infer_h2o(meta, X, result_df, score_col)
        elif model_format == "hugging":
            result_df = self._infer_hugging(meta, X, result_df, cfg.get("batch_size", 512))
        else:
            raise ExecutorException(f"지원하지 않는 model_format: {model_format}")

        self._update_job_status(ExecutorStatus.RUNNING, progress=85)

        # 저장
        output_path = f"predict/{cfg['output_id']}_pretrained.parquet"
        self._save_dataframe(result_df, output_path)

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result": {
                "output_id":    cfg["output_id"],
                "model_id":     cfg["model_id"],
                "model_format": model_format,
                "output_path":  output_path,
                "total_rows":   len(result_df),
            },
            "message": f"Pretrained 추론 완료  {len(result_df):,}건  format={model_format}",
        }

    # ------------------------------------------------------------------

    def _load_meta(self, model_id: str) -> dict:
        meta_path = self.file_root / f"models/{model_id}_meta.json"
        if not meta_path.exists():
            raise ExecutorException(f"모델 메타 없음: {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    def _infer_pickle(self, meta, X, result_df, score_col, output_mode) -> pd.DataFrame:
        model_path = self.file_root / meta["model_path"]
        if not model_path.exists():
            raise ExecutorException(f"모델 파일 없음: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        if output_mode in ("score", "both") and hasattr(model, "predict_proba"):
            result_df[score_col] = model.predict_proba(X)[:, 1].round(6)
        elif output_mode in ("score", "both"):
            result_df[score_col] = model.predict(X).round(6)

        if output_mode in ("embedding", "both") and hasattr(model, "transform"):
            embedding = model.transform(X)
            emb_cols  = [f"emb_{i}" for i in range(embedding.shape[1])]
            for i, col in enumerate(emb_cols):
                result_df[col] = embedding[:, i]

        return result_df

    def _infer_onnx(self, meta, X, result_df, score_col, batch_size) -> pd.DataFrame:
        import onnxruntime as ort

        model_path = str(self.file_root / meta["model_path"])
        sess = ort.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name

        scores = []
        for i in range(0, len(X), batch_size):
            batch = X.iloc[i: i + batch_size].values.astype(np.float32)
            preds = sess.run(None, {input_name: batch})
            # 분류 확률 또는 회귀값
            if len(preds) > 1 and preds[1] is not None:
                scores.extend(preds[1][:, 1].tolist())
            else:
                scores.extend(preds[0].flatten().tolist())

        result_df[score_col] = np.round(scores, 6)
        return result_df

    def _infer_h2o(self, meta, X, result_df, score_col) -> pd.DataFrame:
        import h2o
        h2o.init()
        model = h2o.import_mojo(str(self.file_root / meta["model_path"]))
        h2o_frame = h2o.H2OFrame(X)
        preds = model.predict(h2o_frame).as_data_frame()
        result_df[score_col] = preds.iloc[:, -1].values.round(6)
        return result_df

    def _infer_hugging(self, meta, X, result_df, batch_size) -> pd.DataFrame:
        from transformers import pipeline as hf_pipeline

        model_path = str(self.file_root / meta["model_path"])
        text_col   = meta.get("text_col", X.columns[0])
        pipe       = hf_pipeline("text-classification", model=model_path)

        texts  = X[text_col].fillna("").tolist()
        labels, scores = [], []
        for i in range(0, len(texts), batch_size):
            batch_res = pipe(texts[i: i + batch_size], truncation=True)
            for r in batch_res:
                labels.append(r["label"])
                scores.append(round(r["score"], 6))

        result_df["pred_label"] = labels
        result_df["pred_score"] = scores
        return result_df
