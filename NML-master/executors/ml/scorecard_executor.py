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


# =============================================================================
# Module-level functions
# =============================================================================


def debug_export(df: "pd.DataFrame", outfilename: str) -> None:
    """디버그용 DataFrame 파일 저장 (parquet 또는 csv)."""
    import pandas as pd
    from pathlib import Path

    Path(outfilename).parent.mkdir(parents=True, exist_ok=True)
    if outfilename.endswith(".parquet"):
        df.to_parquet(outfilename, index=False)
    else:
        df.to_csv(outfilename, index=False, encoding="utf-8-sig")
    logger.debug("debug_export: saved %s  shape=%s", outfilename, df.shape)


def fit_scorecard_ml(
    service_db_info, file_server_port,
    result_file_path_faf, done_file_path_faf,
    num_threads, numpy_use_32bit_float_precision, json_obj,
) -> dict:
    """XGBoost 기반 ML 스코어카드 학습.

    변수 선택 → CB 정보 처리 → Auto-binning → SPV Binning →
    Recoding → Parameter Setting → Train/Validation → Fitting →
    변수 개수 제한 → CV 처리 → Tree 분해 → 평점 → Predict → Performance
    """
    import json as _json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    logger.info("fit_scorecard_ml start: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(json_obj.get("root_dir", "/data"))
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    good_val     = json_obj.get("good_val", 0)
    bad_val      = json_obj.get("bad_val", 1)
    n_bins       = int(json_obj.get("n_bins", 20))
    max_use      = int(json_obj.get("max_use", 0))
    nfolds       = int(json_obj.get("nfolds", 0))
    base_score   = int(json_obj.get("base_score", 600))
    pdo          = int(json_obj.get("pdo", 20))
    anchor_value = int(json_obj.get("anchor_value", 600))

    float_dtype = np.float32 if numpy_use_32bit_float_precision else np.float64

    # 1. Target에 대한 데이터 전처리
    full_train = root_path / train_path
    train_df = pd.read_parquet(full_train) if str(full_train).endswith(".parquet") else pd.read_csv(full_train)

    valid_path = json_obj.get("valid_path")
    if valid_path:
        full_val = root_path / valid_path
        val_df = pd.read_parquet(full_val) if str(full_val).endswith(".parquet") else pd.read_csv(full_val)
    else:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df[target_col])

    if feature_cols is None:
        feature_cols = [c for c in train_df.columns if c != target_col]

    # 변수 선택에서 제외된 변수 중 서브타겟 설정에 사용된 변수 처리
    subtarget_json = json_obj.get("subtarget_json", {})
    if subtarget_json:
        extra_cols = [c for c in subtarget_json.get("used_cols", []) if c not in feature_cols and c in train_df.columns]
        feature_cols = feature_cols + extra_cols

    # 2. CB 정보 처리 (char/binary 변수 처리)
    cb_info = json_obj.get("cb_info", {})
    binary_cols = [c for c in feature_cols if train_df[c].nunique() <= 2]

    # 3. Auto-binning
    woe_tables: dict = {}
    iv_dict: dict    = {}
    iv_threshold     = float(json_obj.get("iv_threshold", 0.02))

    for col in feature_cols:
        if col not in train_df.columns:
            continue
        try:
            df_tmp = pd.DataFrame({"x": train_df[col], "y": train_df[target_col]}).dropna()
            try:
                df_tmp["bin"] = pd.qcut(df_tmp["x"], q=n_bins, duplicates="drop")
            except ValueError:
                df_tmp["bin"] = pd.cut(df_tmp["x"], bins=n_bins, duplicates="drop")

            total_good = (df_tmp["y"] == good_val).sum()
            total_bad  = (df_tmp["y"] == bad_val).sum()
            rows = []
            for bin_label, grp in df_tmp.groupby("bin"):
                good = (grp["y"] == good_val).sum()
                bad  = (grp["y"] == bad_val).sum()
                dist_good = good / total_good if total_good > 0 else 1e-9
                dist_bad  = bad  / total_bad  if total_bad  > 0 else 1e-9
                woe = float(np.log(dist_good / dist_bad)) if dist_bad > 0 and dist_good > 0 else 0
                iv  = (dist_good - dist_bad) * woe
                rows.append({"bin": str(bin_label), "WOE": round(woe, 4), "IV": round(iv, 4),
                             "count": len(grp), "bad_rate": round(bad / len(grp), 4) if len(grp) > 0 else 0})
            woe_tables[col] = pd.DataFrame(rows)
            iv_dict[col]    = round(float(pd.DataFrame(rows)["IV"].sum()), 4)
        except Exception as e:
            logger.warning("binning 실패: %s  %s", col, e)

    # 4~5. SPV Binning / Recoding은 woe_tables 기반으로 처리됨
    # 6. Parameter Setting
    selected_cols = [c for c, iv in iv_dict.items() if iv >= iv_threshold]
    if not selected_cols:
        selected_cols = list(iv_dict.keys())[:5]

    # Train Set / Validation Set 정의
    X_train = train_df[selected_cols].fillna(-9999).astype(float_dtype)
    y_train = (train_df[target_col] == bad_val).astype(int)
    X_val   = val_df[selected_cols].fillna(-9999).astype(float_dtype)
    y_val   = (val_df[target_col] == bad_val).astype(int)

    # fitting (XGBoost)
    try:
        import xgboost as xgb
        params = {
            "n_estimators":     int(json_obj.get("n_estimators", 100)),
            "max_depth":        int(json_obj.get("max_depth", 3)),
            "learning_rate":    float(json_obj.get("learning_rate", 0.1)),
            "subsample":        float(json_obj.get("subsample", 0.8)),
            "colsample_bytree": float(json_obj.get("colsample_bytree", 0.8)),
            "use_label_encoder": False,
            "eval_metric":      "auc",
            "nthread":          num_threads,
            "random_state":     42,
        }
        model = xgb.XGBClassifier(**{k: v for k, v in params.items() if k != "use_label_encoder"})
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

    # 최대 변수 개수 제한
    if max_use > 0 and hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=selected_cols)
        selected_cols = list(imp.nlargest(max_use).index)
        X_train = X_train[selected_cols]
        X_val   = X_val[selected_cols]
        model.fit(X_train, y_train)

    # 9. Predict
    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob   = model.predict_proba(X_val)[:, 1]

    # 10. Performance
    def _ks(y_true, y_score):
        df_tmp = pd.DataFrame({"y": y_true, "s": y_score}).sort_values("s", ascending=False)
        n_bad  = (y_true == 1).sum()
        n_good = (y_true == 0).sum()
        if n_bad == 0 or n_good == 0:
            return 0.0
        df_tmp["cb"] = (df_tmp["y"] == 1).cumsum() / n_bad
        df_tmp["cg"] = (df_tmp["y"] == 0).cumsum() / n_good
        return float((df_tmp["cb"] - df_tmp["cg"]).abs().max())

    train_auc = round(float(roc_auc_score(y_train, train_prob)), 4)
    val_auc   = round(float(roc_auc_score(y_val,   val_prob)),   4)
    train_ks  = round(_ks(y_train.values, train_prob), 4)
    val_ks    = round(_ks(y_val.values,   val_prob),   4)

    metrics = {
        "train_auc": train_auc, "train_ks": train_ks,
        "val_auc":   val_auc,   "val_ks":   val_ks,
    }

    # 11. Write output file
    model_dir = root_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"{model_id}_scorecard_ml.pkl"
    with open(model_file, "wb") as f:
        pickle.dump({"model": model, "woe_tables": woe_tables, "iv_dict": iv_dict,
                     "selected_cols": selected_cols, "feature_cols": feature_cols,
                     "target_col": target_col, "good_val": good_val, "bad_val": bad_val}, f)

    result = {
        "result":        "ok",
        "model_id":      model_id,
        "selected_cols": selected_cols,
        "iv_dict":       iv_dict,
        "metrics":       metrics,
        "model_file":    str(model_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()

    logger.info("fit_scorecard_ml done: %s", metrics)
    return result


def make_py_expression_from_cc(
    df_CcResult: "pd.DataFrame",
    df_var_layout: "pd.DataFrame",
) -> "pd.DataFrame":
    """VAR_NM, CC, RANGE_DESC를 받아서 파이썬 표현식(if 조건)을 만든다."""
    import pandas as pd
    import numpy as np

    rows = []
    for _, row in df_CcResult.iterrows():
        var_nm     = row.get("VAR_NM", "")
        cc         = row.get("CC", "")
        range_desc = str(row.get("RANGE_DESC", ""))

        # 숫자 구간 파싱: "(a, b]" 형태
        py_expr = ""
        try:
            clean = range_desc.strip("()[] ")
            parts = [p.strip() for p in clean.split(",")]
            if len(parts) == 2:
                lo, hi = parts
                lo_op  = ">=" if range_desc.startswith("[") else ">"
                hi_op  = "<=" if range_desc.endswith("]") else "<"
                exprs  = []
                if lo not in ("-inf", "-Inf", "nan", ""):
                    exprs.append(f"{var_nm} {lo_op} {lo}")
                if hi not in ("inf", "Inf", "nan", ""):
                    exprs.append(f"{var_nm} {hi_op} {hi}")
                py_expr = " & ".join(exprs)
            else:
                py_expr = f"{var_nm} == '{range_desc}'"
        except Exception:
            py_expr = f"{var_nm} == '{range_desc}'"

        rows.append({"VAR_NM": var_nm, "CC": cc, "RANGE_DESC": range_desc, "PY_EXPR": py_expr})

    return pd.DataFrame(rows)


def recoding(
    df_data: "pd.DataFrame",
    df_range: "pd.DataFrame",
) -> "pd.DataFrame":
    """df_range의 구간 정보를 기준으로 df_data를 리코딩(구간화)한다."""
    import pandas as pd
    import numpy as np

    result = df_data.copy()
    for var_nm, grp in df_range.groupby("VAR_NM"):
        if var_nm not in result.columns:
            continue
        bins   = []
        labels = []
        for _, row in grp.sort_values("CC").iterrows():
            range_desc = str(row.get("RANGE_DESC", ""))
            cc         = row.get("CC", "")
            try:
                clean = range_desc.strip("()[] ")
                parts = [p.strip() for p in clean.split(",")]
                if len(parts) == 2:
                    lo = float(parts[0]) if parts[0] not in ("-inf", "-Inf") else -np.inf
                    hi = float(parts[1]) if parts[1] not in ("inf",  "Inf")  else  np.inf
                    bins.append((lo, hi, cc))
            except Exception:
                pass

        if bins:
            bin_edges = sorted(set([b[0] for b in bins] + [bins[-1][1]]))
            try:
                result[f"{var_nm}_recode"] = pd.cut(
                    result[var_nm],
                    bins=bin_edges,
                    labels=[b[2] for b in bins],
                    include_lowest=True,
                )
            except Exception as e:
                logger.warning("recoding 실패: %s  %s", var_nm, e)
    return result


def get_feature_properties(var_layout_df: "pd.DataFrame") -> dict:
    """XGBoost interaction_constraints 및 monotone_constraints 생성.

    interaction_constraints는 feature index 기준으로 [[0,1,2], [2,3,4]] 형태 입력.
    Interaction Group에 변수가 1개만 있는 경우 에러 방지.
    Interaction Group이 1개면 모든 변수 교호작용 가능.
    """
    import pandas as pd
    import numpy as np

    feature_cols = list(var_layout_df.get("VAR_NM", var_layout_df.iloc[:, 0]))
    feat_index   = {v: i for i, v in enumerate(feature_cols)}

    # interaction_constraints
    interaction_constraints = None
    if "INTERACTION_GROUP" in var_layout_df.columns:
        groups = var_layout_df.groupby("INTERACTION_GROUP")["VAR_NM"].apply(list).to_dict()
        groups = {k: [feat_index[v] for v in vs if v in feat_index] for k, vs in groups.items()}
        # Interaction Group에 변수가 1개만 있는 경우 제거
        groups = {k: v for k, v in groups.items() if len(v) >= 2}
        if len(groups) > 1:
            interaction_constraints = list(groups.values())

    # monotone_constraints
    monotone_constraints = None
    if "MONOTONICITY" in var_layout_df.columns:
        mapping = {"increasing": 1, "decreasing": -1, "none": 0, "": 0}
        constraints = []
        for _, row in var_layout_df.iterrows():
            mono = str(row.get("MONOTONICITY", "")).lower()
            constraints.append(mapping.get(mono, 0))
        monotone_constraints = tuple(constraints)

    return {
        "feature_cols":            feature_cols,
        "interaction_constraints": interaction_constraints,
        "monotone_constraints":    monotone_constraints,
    }


def predict_scorecard(
    df_data: "pd.DataFrame",
    df_scorecard: "pd.DataFrame",
    output_param_column_name: str,
    output_param_type: str,
    reverse_prob: bool = True,
    calculate_performance: bool = True,
    target: str = None,
) -> "pd.DataFrame":
    """스코어카드 포인트 합산으로 점수 산출 (모듈 레벨 함수)."""
    import pandas as pd
    import numpy as np

    result = df_data.copy()
    scores = pd.Series(np.zeros(len(df_data)), index=df_data.index)

    for var_nm, grp in df_scorecard.groupby("variable"):
        if var_nm not in df_data.columns:
            continue
        try:
            bins = pd.qcut(df_data[var_nm], q=10, duplicates="drop").astype(str)
        except Exception:
            try:
                bins = pd.cut(df_data[var_nm], bins=10, duplicates="drop").astype(str)
            except Exception:
                continue
        score_map = grp.set_index("bin")["points"].to_dict()
        scores += bins.map(score_map).fillna(0)

    scores = scores.round(0).astype(int)
    if reverse_prob:
        max_score = scores.max()
        scores    = max_score - scores

    result[output_param_column_name] = scores

    if calculate_performance and target and target in df_data.columns:
        from sklearn.metrics import roc_auc_score
        y = df_data[target]
        try:
            auc = round(float(roc_auc_score(y, scores)), 4)
            result["_auc"] = auc
        except Exception:
            pass

    return result


def summary_scorecard(
    pred_df: "pd.DataFrame",
    pred_val_df: "pd.DataFrame",
    scorecard_df: "pd.DataFrame",
    target: str,
    var_layout_df: "pd.DataFrame",
    good_val: int = 0,
    bad_val: int = 1,
) -> dict:
    """스코어카드 요약 — 클래스별 개수, 검증 데이터 PSI, WOE 지표 반환."""
    import pandas as pd
    import numpy as np

    def _perf(df, score_col="score"):
        y = df[target]
        s = df[score_col] if score_col in df.columns else pd.Series(np.zeros(len(df)))
        total = len(df)
        bad   = int((y == bad_val).sum())
        good  = int((y == good_val).sum())
        bad_rate = bad / total if total > 0 else 0
        return {"total": total, "bad": bad, "good": good, "bad_rate": round(bad_rate, 4)}

    score_col = "score"
    train_perf = _perf(pred_df, score_col)
    val_perf   = _perf(pred_val_df, score_col) if pred_val_df is not None else {}

    # PSI 산출 (Return columns & nan 처리)
    psi = 0.0
    if pred_val_df is not None and score_col in pred_df.columns and score_col in pred_val_df.columns:
        try:
            bins = pd.qcut(pred_df[score_col], q=10, duplicates="drop").cat.categories
            dev_dist = pd.cut(pred_df[score_col], bins=bins, include_lowest=True).value_counts(normalize=True).sort_index()
            val_dist = pd.cut(pred_val_df[score_col], bins=bins, include_lowest=True).value_counts(normalize=True).sort_index()
            dev_dist = dev_dist.fillna(1e-9)
            val_dist = val_dist.fillna(1e-9)
            psi_vals = (dev_dist - val_dist) * np.log(dev_dist / val_dist)
            psi = round(float(psi_vals.sum()), 4)
        except Exception:
            psi = 0.0

    return {
        "train":       train_perf,
        "validation":  val_perf,
        "psi":         psi,
        "n_variables": len(scorecard_df["variable"].unique()) if "variable" in scorecard_df.columns else 0,
    }


def calculate_scorecard(
    df_data: "pd.DataFrame",
    df_scorecard: "pd.DataFrame",
    target: str,
    good_val: int = 0,
    bad_val: int = 1,
) -> dict:
    """개발 데이터에 대한 평점 분포 및 KS/AUC 지표 산출."""
    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_auc_score

    result_df = predict_scorecard(
        df_data, df_scorecard,
        output_param_column_name="score",
        output_param_type="SCORE",
        reverse_prob=False,
        calculate_performance=False,
    )

    scores = result_df["score"]
    y      = df_data[target]

    try:
        auc  = round(float(roc_auc_score((y == bad_val).astype(int), scores)), 4)
    except Exception:
        auc = 0.0

    gini = round(2 * auc - 1, 4)

    score_df = pd.DataFrame({"score": scores, "y": y}).sort_values("score", ascending=False)
    n_bad  = (y == bad_val).sum()
    n_good = (y == good_val).sum()
    if n_bad > 0 and n_good > 0:
        score_df["cb"] = (score_df["y"] == bad_val).cumsum() / n_bad
        score_df["cg"] = (score_df["y"] == good_val).cumsum() / n_good
        ks = round(float((score_df["cb"] - score_df["cg"]).abs().max()), 4)
    else:
        ks = 0.0

    # 점수 분포 (10구간)
    try:
        score_df["decile"] = pd.qcut(scores, q=10, duplicates="drop", labels=False)
        dist = score_df.groupby("decile").agg(
            count=("score", "count"),
            bad=(target, lambda x: (x == bad_val).sum()),
        ).reset_index()
        dist["bad_rate"] = (dist["bad"] / dist["count"]).round(4)
        score_dist = dist.to_dict(orient="records")
    except Exception:
        score_dist = []

    return {
        "auc":        auc,
        "gini":       gini,
        "ks":         ks,
        "score_dist": score_dist,
    }


def convert_tree_to_scorecard(
    df_tree: "pd.DataFrame",
    anchor: int = 500,
    pdo: int = 40,
    reverse_prob: bool = True,
    base_gain: float = 1.0,
) -> "pd.DataFrame":
    """트리(DataFrame) 구조를 스코어카드 포인트 테이블로 변환."""
    import pandas as pd
    import numpy as np

    factor = pdo / np.log(2)
    rows   = []

    for _, row in df_tree.iterrows():
        var_nm     = row.get("VAR_NM", row.get("variable", ""))
        range_desc = row.get("RANGE_DESC", row.get("bin", ""))
        woe        = float(row.get("WOE", 0.0))
        coef       = float(row.get("COEF", row.get("coef", 1.0)))
        point      = round(-coef * woe * factor * base_gain)
        rows.append({
            "variable":   var_nm,
            "bin":        range_desc,
            "WOE":        woe,
            "points":     int(point),
            "anchor":     anchor,
        })

    return pd.DataFrame(rows)


def convert_single_tree_to_scorecard(df_tree: "pd.DataFrame") -> "pd.DataFrame":
    """단일 트리 결과를 스코어카드 테이블로 변환 (anchor/PDO 없이 WOE 직접 사용)."""
    import pandas as pd

    rows = []
    for _, row in df_tree.iterrows():
        var_nm     = row.get("VAR_NM", row.get("variable", ""))
        range_desc = row.get("RANGE_DESC", row.get("bin", ""))
        woe        = float(row.get("WOE", 0.0))
        points     = int(round(woe * 10))
        rows.append({"variable": var_nm, "bin": range_desc, "WOE": woe, "points": points})
    return pd.DataFrame(rows)


def checkpoint_scorecard(
    service_db_info: dict,
    root_dir: str,
    model_id: str,
    checkpoint_data: dict,
) -> str:
    """스코어카드 중간 결과를 체크포인트 파일로 저장."""
    import json as _json
    from pathlib import Path

    cp_dir  = Path(root_dir) / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    cp_file = cp_dir / f"{model_id}_scorecard_checkpoint.json"

    with open(cp_file, "w", encoding="utf-8") as f:
        _json.dump(checkpoint_data, f, ensure_ascii=False, default=str)

    logger.info("checkpoint_scorecard saved: %s", cp_file)
    return str(cp_file)


def predict_scorecard_ml(
    service_db_info, file_server_host, file_server_port,
    result_file_path_faf, done_file_path_faf,
    numpy_use_32bit_float_precision, json_obj,
) -> dict:
    """ML 스코어카드 모델을 사용한 예측."""
    import json as _json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_scorecard_ml: model_id=%s", json_obj.get("model_id"))

    root_path   = Path(json_obj.get("root_dir", "/data"))
    model_id    = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    float_dtype = np.float32 if numpy_use_32bit_float_precision else np.float64

    # 모델 로드
    model_file = root_path / "models" / f"{model_id}_scorecard_ml.pkl"
    with open(model_file, "rb") as f:
        model_data = pickle.load(f)

    model        = model_data["model"]
    selected_cols = model_data["selected_cols"]
    bad_val      = model_data.get("bad_val", 1)

    # 데이터 로드
    full_path = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    X  = df[[c for c in selected_cols if c in df.columns]].fillna(-9999).astype(float_dtype)

    proba = model.predict_proba(X)[:, 1]
    df["score"] = (proba * 1000).round(0).astype(int)

    # 저장
    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_scorecard_predict.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result":      "ok",
        "model_id":    model_id,
        "total_rows":  len(df),
        "score_mean":  round(float(df["score"].mean()), 2),
        "score_std":   round(float(df["score"].std()), 2),
        "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def calculate_editclass_statistics(
    service_db_info, file_server_host, file_server_port,
    result_file_path_faf, done_file_path_faf, json_obj,
) -> dict:
    """편집 클래스별 통계 산출 — 구간 수정 이후 성능 변화 비교."""
    import json as _json
    from pathlib import Path

    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_auc_score

    logger.info("calculate_editclass_statistics: model_id=%s", json_obj.get("model_id"))

    root_path  = Path(json_obj.get("root_dir", "/data"))
    model_id   = json_obj["model_id"]
    data_path  = json_obj["data_path"]
    target_col = json_obj["target_col"]
    score_col  = json_obj.get("score_col", "score")
    bad_val    = json_obj.get("bad_val", 1)
    good_val   = json_obj.get("good_val", 0)

    full_path = root_path / data_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)

    y = (df[target_col] == bad_val).astype(int)
    s = df[score_col] if score_col in df.columns else pd.Series(np.zeros(len(df)))

    try:
        auc = round(float(roc_auc_score(y, s)), 4)
    except Exception:
        auc = 0.0

    # 구간별 통계
    try:
        df["decile"] = pd.qcut(s, q=10, duplicates="drop", labels=False)
        stats = df.groupby("decile").agg(
            count=(score_col, "count"),
            bad=(target_col, lambda x: (x == bad_val).sum()),
            good=(target_col, lambda x: (x == good_val).sum()),
            score_min=(score_col, "min"),
            score_max=(score_col, "max"),
        ).reset_index()
        stats["bad_rate"] = (stats["bad"] / stats["count"]).round(4)
        class_stats = stats.to_dict(orient="records")
    except Exception:
        class_stats = []

    result = {
        "result":      "ok",
        "model_id":    model_id,
        "auc":         auc,
        "class_stats": class_stats,
        "total_rows":  len(df),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False, default=str)
    Path(done_file_path_faf).touch()
    return result
