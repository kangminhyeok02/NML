"""
mart_executor.py
----------------
모델링용 데이터 마트(mart) 생성 및 적재 실행기.

원천 DB 테이블 또는 파일에서 데이터를 읽어 feature engineering을 수행하고,
분석/모델 입력용 마트 데이터셋을 생성하여 File Server 또는 DB에 저장한다.

실행 순서:
  1. 원천 데이터 조회 (DB SQL 또는 파일)
  2. 타입 변환 / 결측 처리 / 이상값 클리핑
  3. 파생 변수 생성
  4. 학습/검증/예측 분리 (선택적)
  5. 마트 파일 저장 및 DB 메타 등록
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)


class MartExecutor(BaseExecutor):
    """
    데이터 마트 생성 executor.

    config 필수 키
    --------------
    source_query  : str   원천 데이터 SQL (db_session 필요) 또는 source_path 사용
    source_path   : str   원천 파일 상대 경로 (source_query 없을 때)
    target_id     : str   생성할 마트 식별자 (파일명 / DB 키)
    target_path   : str   저장 경로 (예: "mart/{target_id}.parquet")
    feature_rules : list  파생 변수 생성 규칙 목록 (선택)
    split         : dict  {"train": 0.7, "valid": 0.15, "test": 0.15} (선택)
    target_col    : str   타깃 컬럼명 (선택)
    """

    def execute(self) -> dict:
        cfg = self.config

        # 1. 원천 데이터 로드
        df = self._load_source(cfg)
        logger.info("source loaded  shape=%s", df.shape)
        self._update_job_status(ExecutorStatus.RUNNING, progress=20)

        # 2. 기본 전처리
        df = self._basic_preprocess(df, cfg)
        self._update_job_status(ExecutorStatus.RUNNING, progress=50)

        # 3. 파생 변수 생성
        feature_rules = cfg.get("feature_rules", [])
        if feature_rules:
            df = self._create_derived_features(df, feature_rules)
        self._update_job_status(ExecutorStatus.RUNNING, progress=70)

        # 4. 분할 (선택)
        split_cfg = cfg.get("split")
        saved_paths: dict[str, str] = {}
        if split_cfg:
            splits = self._split_dataframe(df, split_cfg, cfg.get("target_col"))
            for split_name, split_df in splits.items():
                path = cfg.get("target_path", "mart/{target_id}_{split}.parquet").format(
                    target_id=cfg["target_id"], split=split_name
                )
                saved_paths[split_name] = self._save_dataframe(split_df, path)
        else:
            path = cfg.get("target_path", f"mart/{cfg['target_id']}.parquet")
            saved_paths["full"] = self._save_dataframe(df, path)

        self._update_job_status(ExecutorStatus.RUNNING, progress=90)

        # 5. 메타 정보 저장
        meta = {
            "target_id":   cfg["target_id"],
            "shape":       list(df.shape),
            "columns":     list(df.columns),
            "saved_paths": saved_paths,
            "dtypes":      {c: str(t) for c, t in df.dtypes.items()},
        }
        meta_path = f"mart/{cfg['target_id']}_meta.json"
        self._save_json(meta, meta_path)

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  meta,
            "message": f"마트 생성 완료: {cfg['target_id']}  shape={df.shape}",
        }

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _load_source(self, cfg: dict) -> pd.DataFrame:
        if "source_query" in cfg and self.db_session is not None:
            return pd.read_sql(cfg["source_query"], self.db_session.bind)
        elif "source_path" in cfg:
            return self._load_dataframe(cfg["source_path"])
        else:
            raise ExecutorException("source_query 또는 source_path 중 하나가 필요합니다.")

    def _basic_preprocess(self, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        """타입 캐스팅, 결측 처리, 이상값 클리핑."""
        # 문자열 → 카테고리 자동 변환
        for col in df.select_dtypes(include="object").columns:
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype("category")

        # 수치형 결측 → 중앙값 대체
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        # 이상값 클리핑 (IQR 3배)
        clip_cols = cfg.get("clip_columns", list(num_cols))
        for col in clip_cols:
            q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q1, upper=q3)

        return df

    def _create_derived_features(self, df: pd.DataFrame, rules: list) -> pd.DataFrame:
        """
        rules 예시:
            [{"name": "ratio_a_b", "expr": "col_a / (col_b + 1)"},
             {"name": "log_c",     "expr": "log(col_c + 1)"}]
        """
        for rule in rules:
            try:
                df[rule["name"]] = df.eval(rule["expr"])
            except Exception as exc:
                logger.warning("파생변수 생성 실패: %s  reason=%s", rule["name"], exc)
        return df

    def _split_dataframe(
        self,
        df: pd.DataFrame,
        split_cfg: dict,
        target_col: Optional[str],
    ) -> dict[str, pd.DataFrame]:
        """비율에 따라 train/valid/test 분할."""
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df)
        train_end = int(n * split_cfg.get("train", 0.7))
        valid_end = train_end + int(n * split_cfg.get("valid", 0.15))

        return {
            "train": df.iloc[:train_end],
            "valid": df.iloc[train_end:valid_end],
            "test":  df.iloc[valid_end:],
        }
