"""
scorecard_executor.py
---------------------
신용평가식 스코어카드(Credit Scorecard) 모델 생성 실행기.

금융/리스크 도메인에서 가장 전통적인 모델링 방식.
변수 구간화(binning) → WOE 변환 → IV 산출 → 로지스틱 회귀 →
점수 스케일링 순서로 스코어카드를 생성한다.

출력물:
  - 변수별 binning / WOE / IV 테이블
  - 스코어카드 포인트 테이블
  - 고객별 스코어
  - 모델 성능 지표 (KS, AUC, Gini)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)


class ScorecardExecutor(BaseExecutor):
    """
    스코어카드 모델 executor.

    config 필수 키
    --------------
    train_path  : str   학습 데이터 경로 (.parquet)
    target_col  : str   타깃 컬럼 (1=Bad, 0=Good)
    model_id    : str   모델 저장 식별자
    feature_cols: list  스코어카드에 사용할 변수 목록

    config 선택 키
    --------------
    valid_path    : str   검증 데이터 경로
    n_bins        : int   binning 구간 수 (기본 10)
    min_bin_rate  : float 최소 bin 비율 (기본 0.05)
    iv_threshold  : float IV 필터 기준 (기본 0.02)
    base_score    : int   기준 점수 (기본 600)
    pdo           : int   PDO - odds 2배당 점수 (기본 20)
    """

    def execute(self) -> dict:
        cfg = self.config
        target_col   = cfg["target_col"]
        feature_cols = cfg["feature_cols"]
        n_bins       = cfg.get("n_bins", 10)
        iv_threshold = cfg.get("iv_threshold", 0.02)

        # 1. 데이터 로드
        train_df = self._load_dataframe(cfg["train_path"])
        valid_df = self._load_dataframe(cfg["valid_path"]) if "valid_path" in cfg else None
        self._update_job_status(ExecutorStatus.RUNNING, progress=15)

        # 2. Binning & WOE 계산
        woe_tables: dict[str, pd.DataFrame] = {}
        iv_dict:    dict[str, float]        = {}
        for col in feature_cols:
            if col not in train_df.columns:
                logger.warning("컬럼 없음, 건너뜀: %s", col)
                continue
            woe_table = self._calc_woe(train_df[col], train_df[target_col], n_bins)
            iv = float(woe_table["IV"].sum())
            woe_tables[col] = woe_table
            iv_dict[col]    = round(iv, 4)

        self._update_job_status(ExecutorStatus.RUNNING, progress=40)

        # 3. IV 필터링
        selected_cols = [c for c, iv in iv_dict.items() if iv >= iv_threshold]
        logger.info("IV 필터 후 선택 변수: %d/%d", len(selected_cols), len(feature_cols))
        if len(selected_cols) == 0:
            raise ExecutorException(f"IV >= {iv_threshold} 조건을 만족하는 변수가 없습니다.")

        # 4. WOE 변환 적용
        X_woe_train = self._apply_woe(train_df, selected_cols, woe_tables, target_col)
        y_train     = train_df[target_col]

        # 5. 로지스틱 회귀 학습
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_woe_train)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_scaled, y_train)
        self._update_job_status(ExecutorStatus.RUNNING, progress=65)

        # 6. 스코어 스케일링 (PDO 방식)
        base_score = cfg.get("base_score", 600)
        pdo        = cfg.get("pdo", 20)
        scorecard  = self._build_scorecard(lr, scaler, woe_tables, selected_cols, base_score, pdo)

        # 7. 성능 평가
        train_scores = self._score_data(train_df, scorecard, selected_cols, woe_tables)
        metrics = self._evaluate_scorecard(train_scores, y_train)

        if valid_df is not None:
            valid_scores = self._score_data(valid_df, scorecard, selected_cols, woe_tables)
            y_valid = valid_df[target_col]
            metrics["valid"] = self._evaluate_scorecard(valid_scores, y_valid)

        self._update_job_status(ExecutorStatus.RUNNING, progress=85)

        # 8. 저장
        model_data = {
            "scorecard":    scorecard.to_dict(orient="records"),
            "woe_tables":   {k: v.to_dict(orient="records") for k, v in woe_tables.items()},
            "iv_dict":      iv_dict,
            "selected_cols": selected_cols,
            "metrics":      metrics,
            "base_score":   base_score,
            "pdo":          pdo,
        }
        self._save_json(model_data, f"models/{cfg['model_id']}_scorecard.json")

        # 학습 데이터 점수 저장
        train_df["score"] = train_scores
        self._save_dataframe(train_df[[target_col, "score"]], f"models/{cfg['model_id']}_train_scores.parquet")

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result": {
                "model_id":      cfg["model_id"],
                "selected_vars": selected_cols,
                "iv_dict":       iv_dict,
                "metrics":       metrics,
            },
            "message": f"스코어카드 생성 완료  변수={len(selected_cols)}개  KS={metrics.get('ks', 'N/A')}",
        }

    # ------------------------------------------------------------------
    # WOE / IV
    # ------------------------------------------------------------------

    def _calc_woe(self, x: pd.Series, y: pd.Series, n_bins: int) -> pd.DataFrame:
        """수치형 변수에 대한 WOE/IV 테이블 생성."""
        df = pd.DataFrame({"x": x, "y": y}).dropna()

        try:
            df["bin"] = pd.qcut(df["x"], q=n_bins, duplicates="drop")
        except ValueError:
            df["bin"] = pd.cut(df["x"], bins=n_bins, duplicates="drop")

        total_good = (df["y"] == 0).sum()
        total_bad  = (df["y"] == 1).sum()

        rows = []
        for bin_label, group in df.groupby("bin"):
            good = (group["y"] == 0).sum()
            bad  = (group["y"] == 1).sum()
            dist_good = good / total_good if total_good > 0 else 1e-9
            dist_bad  = bad  / total_bad  if total_bad  > 0 else 1e-9
            woe = np.log(dist_good / dist_bad) if dist_bad > 0 and dist_good > 0 else 0
            iv  = (dist_good - dist_bad) * woe
            rows.append({
                "bin":       str(bin_label),
                "count":     len(group),
                "good":      int(good),
                "bad":       int(bad),
                "bad_rate":  round(bad / len(group), 4) if len(group) > 0 else 0,
                "WOE":       round(woe, 4),
                "IV":        round(iv, 4),
            })
        return pd.DataFrame(rows)

    def _apply_woe(self, df, selected_cols, woe_tables, target_col) -> pd.DataFrame:
        """선택 변수에 WOE 값을 적용한 데이터프레임 반환."""
        result = {}
        for col in selected_cols:
            woe_map = {}
            for _, row in woe_tables[col].iterrows():
                woe_map[row["bin"]] = row["WOE"]
            try:
                bins = pd.qcut(df[col], q=10, duplicates="drop")
            except ValueError:
                bins = pd.cut(df[col], bins=10, duplicates="drop")
            result[col] = bins.astype(str).map(woe_map).fillna(0)
        return pd.DataFrame(result)

    def _build_scorecard(self, lr, scaler, woe_tables, selected_cols, base_score, pdo) -> pd.DataFrame:
        """로지스틱 계수를 점수로 변환한 스코어카드 테이블 생성."""
        factor  = pdo / np.log(2)
        offset  = base_score - factor * np.log(1)
        rows = []
        for i, col in enumerate(selected_cols):
            coef = lr.coef_[0][i]
            scale = scaler.scale_[i]
            for _, bin_row in woe_tables[col].iterrows():
                point = round(-(coef / scale) * bin_row["WOE"] * factor)
                rows.append({
                    "variable": col,
                    "bin":      bin_row["bin"],
                    "WOE":      bin_row["WOE"],
                    "points":   int(point),
                })
        return pd.DataFrame(rows)

    def _score_data(self, df, scorecard, selected_cols, woe_tables) -> pd.Series:
        """스코어카드 포인트 합산으로 점수 산출."""
        scores = pd.Series(np.zeros(len(df)), index=df.index)
        for col in selected_cols:
            try:
                bins = pd.qcut(df[col], q=10, duplicates="drop").astype(str)
            except ValueError:
                bins = pd.cut(df[col], bins=10, duplicates="drop").astype(str)
            score_map = scorecard[scorecard["variable"] == col].set_index("bin")["points"].to_dict()
            scores += bins.map(score_map).fillna(0)
        return scores.round(0).astype(int)

    def _evaluate_scorecard(self, scores: pd.Series, y: pd.Series) -> dict:
        """KS, AUC, Gini 산출."""
        auc  = float(roc_auc_score(y, scores))
        gini = 2 * auc - 1
        # KS 계산
        score_df = pd.DataFrame({"score": scores, "y": y})
        score_df = score_df.sort_values("score", ascending=False)
        score_df["cum_good"] = (score_df["y"] == 0).cumsum() / (y == 0).sum()
        score_df["cum_bad"]  = (score_df["y"] == 1).cumsum() / (y == 1).sum()
        ks = float((score_df["cum_bad"] - score_df["cum_good"]).abs().max())
        return {
            "auc":  round(auc, 4),
            "gini": round(gini, 4),
            "ks":   round(ks, 4),
        }
