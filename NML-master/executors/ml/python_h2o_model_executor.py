"""
python_h2o_model_executor.py
----------------------------
Python 서비스 파이프라인과 H2O 모델을 통합하는 실행기.

h2o_model_executor.py가 순수 H2O 로직에 집중한다면,
이 executor는 Python 기반 전처리/후처리와 H2O 모델 추론을 결합한다.

대표 시나리오:
  1. Python으로 feature engineering 수행
  2. H2O MOJO 모델로 점수 산출
  3. Python으로 결과 후처리 (스케일링, 등급화, 마스킹)

즉, H2O 모델이 학습에 쓰였지만 추론 파이프라인은
Python 코드로 제어해야 할 때 사용한다.
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


class PythonH2OModelExecutor(BaseExecutor):
    """
    Python + H2O 통합 executor.

    config 필수 키
    --------------
    model_id    : str   H2O 모델 식별자 (MOJO 경로 포함된 메타)
    input_path  : str   입력 데이터 경로 (.parquet)
    output_id   : str   결과 저장 식별자

    config 선택 키
    --------------
    preprocess_steps   : list  Python 전처리 스텝 목록
    postprocess_steps  : list  Python 후처리 스텝 목록
    score_col          : str   점수 컬럼명 (기본: "score")
    h2o_ip             : str   H2O 서버 IP (기본: localhost)
    h2o_port           : int   H2O 서버 포트 (기본: 54321)
    use_mojo           : bool  MOJO 사용 여부 (기본: True)
    """

    def execute(self) -> dict:
        cfg       = self.config
        score_col = cfg.get("score_col", "score")

        # 메타 로드
        meta_path = self.file_root / f"models/{cfg['model_id']}_meta.json"
        if not meta_path.exists():
            raise ExecutorException(f"모델 메타 없음: {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        # 입력 데이터 로드
        df = self._load_dataframe(cfg["input_path"])
        logger.info("Python+H2O pipeline  rows=%d", len(df))
        self._update_job_status(ExecutorStatus.RUNNING, progress=15)

        # 1. Python 전처리
        pre_steps = cfg.get("preprocess_steps", [])
        df = self._apply_preprocess(df, pre_steps)
        self._update_job_status(ExecutorStatus.RUNNING, progress=35)

        # 2. H2O 추론
        feature_cols = meta.get("feature_cols") or list(df.columns)
        X = df[[c for c in feature_cols if c in df.columns]]

        use_mojo = cfg.get("use_mojo", True)
        if use_mojo:
            scores = self._predict_mojo(meta, X, cfg)
        else:
            scores = self._predict_h2o_live(meta, X, cfg)

        df[score_col] = np.round(scores, 6)
        self._update_job_status(ExecutorStatus.RUNNING, progress=70)

        # 3. Python 후처리
        post_steps = cfg.get("postprocess_steps", [])
        df = self._apply_postprocess(df, post_steps, score_col)
        self._update_job_status(ExecutorStatus.RUNNING, progress=90)

        # 저장
        output_path = f"predict/{cfg['output_id']}_py_h2o.parquet"
        self._save_dataframe(df, output_path)

        summary = {
            "output_id":   cfg["output_id"],
            "model_id":    cfg["model_id"],
            "total_rows":  len(df),
            "output_path": output_path,
            "score_stats": _stats(df[score_col]),
        }

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  summary,
            "message": f"Python+H2O 파이프라인 완료  {len(df):,}건",
        }

    # ------------------------------------------------------------------

    def _apply_preprocess(self, df: pd.DataFrame, steps: list) -> pd.DataFrame:
        """
        steps 예시:
          [{"type": "fillna",   "columns": ["col_a"], "value": 0},
           {"type": "clip",     "columns": ["col_b"], "lower": 0, "upper": 100},
           {"type": "log1p",    "columns": ["col_c"]},
           {"type": "eval",     "name": "ratio",     "expr": "col_a / (col_b + 1)"}]
        """
        for step in steps:
            step_type = step["type"]
            cols = step.get("columns", [])
            try:
                if step_type == "fillna":
                    df[cols] = df[cols].fillna(step["value"])
                elif step_type == "clip":
                    df[cols] = df[cols].clip(lower=step.get("lower"), upper=step.get("upper"))
                elif step_type == "log1p":
                    for col in cols:
                        df[col] = np.log1p(df[col].clip(lower=0))
                elif step_type == "eval":
                    df[step["name"]] = df.eval(step["expr"])
                elif step_type == "drop":
                    df = df.drop(columns=[c for c in cols if c in df.columns])
                else:
                    logger.warning("알 수 없는 전처리 step: %s", step_type)
            except Exception as exc:
                logger.warning("전처리 step 실패: %s  reason=%s", step_type, exc)
        return df

    def _apply_postprocess(self, df: pd.DataFrame, steps: list, score_col: str) -> pd.DataFrame:
        """
        steps 예시:
          [{"type": "scale",  "method": "minmax", "col": "score"},
           {"type": "grade",  "col": "score",     "map": {"A": [800, 1000], "B": [600, 800]}},
           {"type": "round",  "col": "score",     "decimals": 0}]
        """
        for step in steps:
            step_type = step["type"]
            col = step.get("col", score_col)
            try:
                if step_type == "scale" and col in df.columns:
                    method = step.get("method", "minmax")
                    if method == "minmax":
                        mn, mx = df[col].min(), df[col].max()
                        df[col] = (df[col] - mn) / (mx - mn + 1e-9)
                    elif method == "standard":
                        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
                elif step_type == "grade" and col in df.columns:
                    grade_map = step["map"]
                    df["grade"] = df[col].apply(
                        lambda s: next((g for g, (lo, hi) in grade_map.items() if lo <= s < hi), "UNKNOWN")
                    )
                elif step_type == "round" and col in df.columns:
                    df[col] = df[col].round(step.get("decimals", 0))
                else:
                    logger.warning("알 수 없는 후처리 step: %s", step_type)
            except Exception as exc:
                logger.warning("후처리 step 실패: %s  reason=%s", step_type, exc)
        return df

    def _predict_mojo(self, meta: dict, X: pd.DataFrame, cfg: dict) -> np.ndarray:
        """H2O MOJO를 EasyPredictModelWrapper로 실행 (h2o 서버 불필요)."""
        mojo_path = str(self.file_root / meta.get("mojo_path", f"models/{meta['model_id']}/model.zip"))
        try:
            import h2o
            from h2o.estimators import H2OEstimator
            h2o.init(ip=cfg.get("h2o_ip", "localhost"), port=cfg.get("h2o_port", 54321))
            model   = h2o.import_mojo(mojo_path)
            h2oframe = h2o.H2OFrame(X)
            preds   = model.predict(h2oframe).as_data_frame()
            return preds.iloc[:, -1].values
        except Exception as exc:
            raise ExecutorException(f"H2O MOJO 추론 실패: {exc}")

    def _predict_h2o_live(self, meta: dict, X: pd.DataFrame, cfg: dict) -> np.ndarray:
        """H2O 서버에 살아있는 모델로 실시간 추론."""
        import h2o
        h2o.init(ip=cfg.get("h2o_ip", "localhost"), port=cfg.get("h2o_port", 54321))
        model_key = meta.get("h2o_model_id")
        if not model_key:
            raise ExecutorException("h2o_model_id가 메타에 없습니다.")
        model    = h2o.get_model(model_key)
        h2oframe = h2o.H2OFrame(X)
        preds    = model.predict(h2oframe).as_data_frame()
        return preds.iloc[:, -1].values


def _stats(s: pd.Series) -> dict:
    return {k: round(float(v), 4) for k, v in {
        "mean": s.mean(), "min": s.min(), "max": s.max(), "std": s.std(),
    }.items()}


# =============================================================================
# Module-level functions
# =============================================================================


def get_h2o_model_summary(model) -> str:
    """Python H2O 모델 요약 문자열 반환."""
    try:
        lines = []
        lines.append(f"model_id: {model.model_id}")
        try:
            lines.append(f"algo: {model.algo}")
        except Exception:
            pass
        try:
            perf = model.model_performance()
            lines.append(f"train_auc: {perf.auc():.4f}")
        except Exception:
            pass
        try:
            vi = model.varimp(use_pandas=True)
            if vi is not None and len(vi) > 0:
                top5 = vi.head(5)[["variable", "percentage"]].to_string(index=False)
                lines.append(f"top5_varimp:\n{top5}")
        except Exception:
            pass
        return "\n".join(lines)
    except Exception as e:
        return f"summary error: {e}"


def save_mojo_pojo_model_file(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    h2o_host, h2o_port, h2o_script_obj, root_dir,
    model_file_path_in_h2o_file_server, model_file_name, model_type="",
) -> dict:
    """H2O MOJO/POJO 모델 파일을 파일 서버에 저장."""
    import json as _json
    from pathlib import Path

    logger.info("save_mojo_pojo_model_file: %s  type=%s", model_file_name, model_type)

    import h2o
    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)

    save_dir = Path(root_dir) / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    local_path = save_dir / model_file_name

    try:
        model = h2o.get_model(model_file_path_in_h2o_file_server)
        mojo_path = model.save_mojo(str(save_dir))
        return {"result": "ok", "mojo_path": mojo_path, "model_type": model_type}
    except Exception as e:
        logger.warning("MOJO 저장 실패: %s  fallback to download", e)
        try:
            import requests
            url  = f"http://{h2o_file_server_host}:{h2o_file_server_port}/{model_file_path_in_h2o_file_server}"
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)
            return {"result": "ok", "mojo_path": str(local_path), "model_type": model_type}
        except Exception as e2:
            return {"result": "error", "message": str(e2)}


def get_h2o_frame(
    df: "pd.DataFrame",
    h2o_host: str,
    h2o_port: int,
    target_col: str = None,
    mining_type: str = "",
) -> "h2o.H2OFrame":
    """pandas DataFrame을 H2OFrame으로 변환."""
    import h2o

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    h2o_frame = h2o.H2OFrame(df)

    DISCRETE_TYPES = {"RANDOMFOREST", "DEEPLEARNING", "GRADIENTBOOSTING", "GLM",
                      "AUTOML", "AUTOMLGLM", "AUTOMLDL", "XGBOOST"}
    if target_col and target_col in h2o_frame.columns:
        if mining_type.upper() in DISCRETE_TYPES:
            h2o_frame[target_col] = h2o_frame[target_col].asfactor()

    return h2o_frame


def predict_h2o_frame(
    model,
    h2o_frame: "h2o.H2OFrame",
    reverse_prob: bool = True,
) -> "np.ndarray":
    """H2OFrame에 대해 예측 수행 후 numpy 배열 반환."""
    import numpy as np

    preds = model.predict(h2o_frame).as_data_frame()
    proba = preds.iloc[:, -1].values
    if reverse_prob:
        proba = 1.0 - proba
    return np.round(proba, 6)


def predict_anomaly(
    model,
    h2o_frame: "h2o.H2OFrame",
) -> "np.ndarray":
    """H2O AutoEncoder 이상 탐지 재구성 오차 반환."""
    import numpy as np

    recon = model.anomaly(h2o_frame).as_data_frame()
    return recon.iloc[:, 0].values.round(6)


def black_scholes_formula(x: float) -> float:
    """Black-Scholes 정규분포 누적함수 근사 (옵션 가격결정)."""
    import math

    # Abramowitz & Stegun 근사
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    k  = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))
    phi  = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly
    if x >= 0:
        return 1.0 - phi
    return phi


def process_autoencoder_with_h2o_frame(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O AutoEncoder 학습 — 재구성 오차 기반 이상 탐지."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("process_autoencoder_with_h2o_frame: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    feature_cols = json_obj.get("feature_cols")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)

    if feature_cols is None:
        feature_cols = list(df.columns)
    X = df[feature_cols].fillna(0).astype(float_dtype)

    import h2o
    from h2o.estimators.deeplearning import H2ODeepLearningEstimator

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    train_h2o = h2o.H2OFrame(X)

    hidden = json_obj.get("hidden", [50, 25, 50])
    ae = H2ODeepLearningEstimator(
        autoencoder=True,
        hidden=hidden,
        activation="Tanh",
        epochs=int(json_obj.get("epochs", 100)),
        seed=42,
    )
    ae.train(x=feature_cols, training_frame=train_h2o)

    # 재구성 오차
    recon_error = ae.anomaly(train_h2o).as_data_frame()
    df["recon_error"] = recon_error.iloc[:, 0].values.round(6)

    # 저장
    model_dir = root_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    mojo_path = ae.save_mojo(str(model_dir))

    result = {
        "result":     "ok",
        "model_id":   model_id,
        "mojo_path":  mojo_path,
        "recon_error_mean": round(float(df["recon_error"].mean()), 4),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def _apply_stacked_ae_array(ae_models: list, X: "np.ndarray") -> "np.ndarray":
    """스택형 AutoEncoder 각 레이어의 재구성 배열 반환."""
    import numpy as np

    import h2o

    outputs = []
    for ae in ae_models:
        try:
            h2o_x = h2o.H2OFrame(X)
            recon  = ae.anomaly(h2o_x).as_data_frame().values
            outputs.append(recon)
        except Exception as e:
            logger.warning("_apply_stacked_ae_array layer 실패: %s", e)
    return np.column_stack(outputs) if outputs else np.zeros((len(X), 1))


def process_multitarget_with_h2o_frame(
    multitarget_properties: list,
    h2o_frame: "h2o.H2OFrame",
    model,
    target_col: str,
    feature_cols: list,
) -> list:
    """H2OFrame 기반 멀티타겟 예측 처리."""
    results = []
    for i, prop in enumerate(multitarget_properties):
        try:
            preds = model.predict(h2o_frame).as_data_frame()
            proba = preds.iloc[:, -1].values
            results.append({
                "target_index": i,
                "score_col":    prop.get("score_col", f"score_{i}"),
                "proba":        proba.tolist(),
            })
        except Exception as e:
            logger.warning("multitarget index=%d 실패: %s", i, e)
    return results


def adjust_multitarget_result_json(
    result_json: dict,
    multitarget_properties: list,
) -> dict:
    """멀티타겟 결과 JSON 조정 — 타겟별 성능/점수 분포 정규화."""
    import numpy as np

    adjusted = dict(result_json)
    scores   = adjusted.get("scores", {})

    for i, prop in enumerate(multitarget_properties):
        col = prop.get("score_col", f"score_{i}")
        if col in scores:
            arr   = np.array(scores[col])
            norm  = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
            scores[col + "_norm"] = norm.round(6).tolist()

    adjusted["scores"]                = scores
    adjusted["multitarget_count"]     = len(multitarget_properties)
    adjusted["multitarget_properties"] = multitarget_properties
    return adjusted


def get_multitarget_h2o_frame(
    df: "pd.DataFrame",
    multitarget_properties: list,
    h2o_host: str,
    h2o_port: int,
    target_col: str = None,
) -> list:
    """멀티타겟 각각에 대한 H2OFrame 리스트 생성."""
    import h2o

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    frames = []
    for i, prop in enumerate(multitarget_properties):
        sub_target = prop.get("target_col", target_col)
        sub_df     = df.copy()
        h2o_frame  = h2o.H2OFrame(sub_df)
        if sub_target and sub_target in h2o_frame.columns:
            h2o_frame[sub_target] = h2o_frame[sub_target].asfactor()
        frames.append({"index": i, "frame": h2o_frame, "target_col": sub_target})
    return frames


def save_h2o_fitting_result(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir,
    model, json_obj,
) -> dict:
    """H2O 학습 결과 저장 — MOJO, 메타, 성능 지표."""
    import json as _json
    from pathlib import Path

    logger.info("save_h2o_fitting_result: model_id=%s", json_obj.get("model_id"))

    root_path = Path(root_dir)
    model_id  = json_obj["model_id"]

    model_dir = root_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # MOJO 저장
    try:
        mojo_path = model.save_mojo(str(model_dir))
    except Exception:
        mojo_path = ""

    # 성능 지표
    try:
        perf    = model.model_performance()
        metrics = {"auc": round(float(perf.auc()), 4), "logloss": round(float(perf.logloss()), 4)}
    except Exception:
        metrics = {}

    # 변수 중요도
    try:
        vi = model.varimp(use_pandas=True)
        varimp = vi.set_index("variable")["percentage"].head(20).round(4).to_dict() if vi is not None else {}
    except Exception:
        varimp = {}

    meta = {
        "model_id":     model_id,
        "h2o_model_id": model.model_id,
        "mojo_path":    mojo_path,
        "metrics":      metrics,
        "varimp":       varimp,
        "summary":      get_h2o_model_summary(model),
    }

    meta_file = model_dir / f"{model_id}_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        _json.dump(meta, f, ensure_ascii=False)

    result = {"result": "ok", "model_id": model_id, "metrics": metrics, "mojo_path": mojo_path}
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def model_var_imp(
    model,
    feature_cols: list = None,
    top_n: int = 20,
) -> "pd.DataFrame":
    """H2O 모델 변수 중요도 DataFrame 반환."""
    import pandas as pd

    try:
        vi = model.varimp(use_pandas=True)
        if vi is not None:
            return vi.head(top_n)
    except Exception:
        pass

    if feature_cols and hasattr(model, "feature_importances_"):
        import numpy as np
        imp = model.feature_importances_
        df  = pd.DataFrame({"variable": feature_cols, "importance": imp})
        df  = df.sort_values("importance", ascending=False).head(top_n)
        df["percentage"] = df["importance"] / df["importance"].sum()
        return df

    return pd.DataFrame(columns=["variable", "importance", "percentage"])


def model_scoring_history(model) -> "pd.DataFrame":
    """H2O 모델 scoring history DataFrame 반환."""
    import pandas as pd

    try:
        sh = model.scoring_history()
        if sh is not None:
            return sh
    except Exception:
        pass
    return pd.DataFrame()


def fit_xgboost(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O XGBoost 학습."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("fit_xgboost (python_h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    import h2o
    from h2o.estimators import H2OXGBoostEstimator

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    params = {
        "ntrees":     int(json_obj.get("ntrees", 100)),
        "max_depth":  int(json_obj.get("max_depth", 6)),
        "learn_rate": float(json_obj.get("learn_rate", 0.1)),
        "seed":       42,
    }
    model = H2OXGBoostEstimator(**params)
    model.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    result = save_h2o_fitting_result(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir,
        model, json_obj,
    )
    return result


def fit_gbm(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O GBM 학습."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("fit_gbm (python_h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    import h2o
    from h2o.estimators import H2OGradientBoostingEstimator

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    params = {
        "ntrees":     int(json_obj.get("ntrees", 100)),
        "max_depth":  int(json_obj.get("max_depth", 5)),
        "learn_rate": float(json_obj.get("learn_rate", 0.1)),
        "seed":       42,
    }
    model = H2OGradientBoostingEstimator(**params)
    model.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    return save_h2o_fitting_result(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir,
        model, json_obj,
    )


def fit_rf(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O Random Forest 학습."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("fit_rf (python_h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    import h2o
    from h2o.estimators import H2ORandomForestEstimator

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    model = H2ORandomForestEstimator(
        ntrees=int(json_obj.get("ntrees", 100)),
        max_depth=int(json_obj.get("max_depth", 20)),
        seed=42,
    )
    model.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    return save_h2o_fitting_result(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir,
        model, json_obj,
    )


def fit_dnn(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O DeepLearning 학습."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("fit_dnn (python_h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    import h2o
    from h2o.estimators import H2ODeepLearningEstimator

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    hidden = json_obj.get("hidden", [200, 200])
    model = H2ODeepLearningEstimator(
        hidden=hidden,
        epochs=float(json_obj.get("epochs", 100)),
        activation=json_obj.get("activation", "Rectifier"),
        seed=42,
    )
    model.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    return save_h2o_fitting_result(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir,
        model, json_obj,
    )


def get_bins_labels(n_bins: int, min_val: float = 0.0, max_val: float = 1.0) -> tuple:
    """n_bins 개의 구간 경계 및 레이블 반환 (Uplift 용)."""
    import numpy as np

    bins   = np.linspace(min_val, max_val, n_bins + 1)
    labels = [f"{bins[i]:.3f}-{bins[i+1]:.3f}" for i in range(n_bins)]
    return bins, labels


def is_beta_positive(x: float, y: float) -> bool:
    """Beta 파라미터 양수 여부 검사 (Uplift Bayesian)."""
    return x > 0 and y > 0


def get_uplift_dist(
    df: "pd.DataFrame",
    treatment_col: str,
    outcome_col: str,
    score_col: str,
    n_bins: int = 10,
) -> "pd.DataFrame":
    """Uplift 분포 산출 — Treatment/Control별 bad rate 및 qini 계산."""
    import numpy as np
    import pandas as pd

    df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df_sorted["decile"] = pd.qcut(df_sorted.index, q=n_bins, labels=False, duplicates="drop")

    rows = []
    n_total = len(df_sorted)
    n_treat = (df_sorted[treatment_col] == 1).sum()
    n_ctrl  = (df_sorted[treatment_col] == 0).sum()

    for decile, grp in df_sorted.groupby("decile"):
        n_t = (grp[treatment_col] == 1).sum()
        n_c = (grp[treatment_col] == 0).sum()
        y_t = grp[grp[treatment_col] == 1][outcome_col].mean() if n_t > 0 else 0
        y_c = grp[grp[treatment_col] == 0][outcome_col].mean() if n_c > 0 else 0
        uplift = y_t - y_c
        rows.append({
            "decile":    int(decile),
            "n_treat":   int(n_t),
            "n_ctrl":    int(n_c),
            "y_treat":   round(float(y_t), 4),
            "y_ctrl":    round(float(y_c), 4),
            "uplift":    round(float(uplift), 4),
        })
    return pd.DataFrame(rows)


def fit_uplift_drf(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O Uplift Random Forest 학습."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("fit_uplift_drf: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    treatment_col = json_obj.get("treatment_col", "treatment")
    feature_cols = json_obj.get("feature_cols")

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in (target_col, treatment_col)]

    import h2o
    from h2o.estimators import H2OUpliftRandomForestEstimator

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col, treatment_col]])
    train_h2o[target_col]    = train_h2o[target_col].asfactor()
    train_h2o[treatment_col] = train_h2o[treatment_col].asfactor()

    model = H2OUpliftRandomForestEstimator(
        ntrees=int(json_obj.get("ntrees", 100)),
        max_depth=int(json_obj.get("max_depth", 10)),
        treatment_column=treatment_col,
        uplift_metric="KL",
        seed=42,
    )
    model.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    return save_h2o_fitting_result(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir,
        model, json_obj,
    )


def predict_uplift_drf(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O Uplift DRF 예측."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_uplift_drf: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]
    float_dtype  = np.float32 if numpy_use_32bit_float_precision else np.float64

    meta_file = root_path / "models" / f"{model_id}_meta.json"
    with open(meta_file, encoding="utf-8") as f:
        meta = _json.load(f)

    full_path = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)

    import h2o
    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    model = h2o.import_mojo(str(root_path / meta.get("mojo_path", "")))

    feature_cols = meta.get("feature_cols", [])
    X = df[[c for c in feature_cols if c in df.columns]].fillna(0)
    h2o_frame = h2o.H2OFrame(X)
    preds = model.predict(h2o_frame).as_data_frame()
    df["uplift_score"] = preds.iloc[:, -1].values.round(6)

    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_uplift_predict.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result": "ok", "model_id": model_id,
        "total_rows": len(df), "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def fit_gridsearch(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O Grid Search 학습."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("fit_gridsearch (python_h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    algorithm    = json_obj.get("algorithm", "gbm").lower()

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    import h2o
    from h2o.grid.grid_search import H2OGridSearch
    from h2o.estimators import (
        H2OGradientBoostingEstimator, H2ORandomForestEstimator,
        H2OXGBoostEstimator, H2ODeepLearningEstimator,
    )

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    algo_map = {
        "gbm":          H2OGradientBoostingEstimator,
        "drf":          H2ORandomForestEstimator,
        "xgboost":      H2OXGBoostEstimator,
        "deeplearning": H2ODeepLearningEstimator,
    }
    estimator_cls = algo_map.get(algorithm, H2OGradientBoostingEstimator)

    hyper_params   = json_obj.get("hyper_params", {"max_depth": [3, 5, 7], "ntrees": [50, 100]})
    search_criteria = json_obj.get("search_criteria", {"strategy": "Cartesian"})

    grid = H2OGridSearch(
        model=estimator_cls(seed=42),
        hyper_params=hyper_params,
        search_criteria=search_criteria,
    )
    grid.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    best_model = grid.get_grid(sort_by="auc", decreasing=True).models[0]

    return save_h2o_fitting_result(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir,
        best_model, json_obj,
    )


def paramsearch_rf(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O Random Forest 파라미터 탐색."""
    json_obj = dict(json_obj)
    json_obj.setdefault("algorithm", "drf")
    json_obj.setdefault("hyper_params", {
        "ntrees": [50, 100, 200],
        "max_depth": [10, 20, 30],
    })
    return fit_gridsearch(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        numpy_use_32bit_float_precision,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
    )


def paramsearch_gbm(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O GBM 파라미터 탐색."""
    json_obj = dict(json_obj)
    json_obj.setdefault("algorithm", "gbm")
    json_obj.setdefault("hyper_params", {
        "ntrees": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learn_rate": [0.05, 0.1, 0.2],
    })
    return fit_gridsearch(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        numpy_use_32bit_float_precision,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
    )


def paramsearch_xgboost(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O XGBoost 파라미터 탐색."""
    json_obj = dict(json_obj)
    json_obj.setdefault("algorithm", "xgboost")
    json_obj.setdefault("hyper_params", {
        "ntrees": [50, 100, 200],
        "max_depth": [4, 6, 8],
        "learn_rate": [0.05, 0.1],
    })
    return fit_gridsearch(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        numpy_use_32bit_float_precision,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
    )


def paramsearch_dnn(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O DeepLearning 파라미터 탐색."""
    json_obj = dict(json_obj)
    json_obj.setdefault("algorithm", "deeplearning")
    json_obj.setdefault("hyper_params", {
        "hidden": [[50, 50], [100, 100], [200, 200]],
        "epochs": [50, 100],
    })
    return fit_gridsearch(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        numpy_use_32bit_float_precision,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
    )


def fit_anomaly_ae(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O AutoEncoder 이상 탐지 학습."""
    return process_autoencoder_with_h2o_frame(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        numpy_use_32bit_float_precision,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
    )


def feature_ae(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O AutoEncoder 기반 피처 추출."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("feature_ae (python_h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    feature_cols = json_obj.get("feature_cols")
    n_features   = int(json_obj.get("n_features", 10))
    hidden       = json_obj.get("hidden", [50, n_features, 50])

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = list(df.columns)
    X = df[feature_cols].fillna(0)

    import h2o
    from h2o.estimators.deeplearning import H2ODeepLearningEstimator

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    train_h2o = h2o.H2OFrame(X)

    ae = H2ODeepLearningEstimator(
        autoencoder=True,
        hidden=hidden,
        activation="Tanh",
        epochs=float(json_obj.get("epochs", 100)),
        seed=42,
    )
    ae.train(x=feature_cols, training_frame=train_h2o)

    # 인코딩 레이어 추출
    encoded = ae.deepfeatures(train_h2o, layer=len(hidden) // 2).as_data_frame()
    encoded_cols = [f"ae_feat_{i}" for i in range(encoded.shape[1])]
    encoded.columns = encoded_cols

    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_ae_features.parquet"
    encoded.to_parquet(output_file, index=False)

    result = {
        "result":      "ok",
        "model_id":    model_id,
        "n_features":  len(encoded_cols),
        "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def feature_glrm(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O GLRM(Generalized Low Rank Model) 피처 추출."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("feature_glrm (python_h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    feature_cols = json_obj.get("feature_cols")
    k            = int(json_obj.get("k", 10))

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = list(df.columns)

    import h2o
    from h2o.estimators import H2OGeneralizedLowRankEstimator

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    train_h2o = h2o.H2OFrame(df[feature_cols].fillna(0))

    glrm = H2OGeneralizedLowRankEstimator(k=k, seed=42)
    glrm.train(x=feature_cols, training_frame=train_h2o)

    x_arch = glrm.score_archetype(train_h2o).as_data_frame()
    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_glrm_features.parquet"
    x_arch.to_parquet(output_file, index=False)

    result = {
        "result":      "ok",
        "model_id":    model_id,
        "k":           k,
        "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def feature_svd(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O SVD 피처 추출."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("feature_svd (python_h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    feature_cols = json_obj.get("feature_cols")
    nv           = int(json_obj.get("nv", 10))

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = list(df.columns)

    import h2o
    from h2o.estimators import H2OSingularValueDecompositionEstimator

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    train_h2o = h2o.H2OFrame(df[feature_cols].fillna(0))

    svd = H2OSingularValueDecompositionEstimator(nv=nv, seed=42)
    svd.train(x=feature_cols, training_frame=train_h2o)

    u_frame = svd.predict(train_h2o).as_data_frame()
    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_svd_features.parquet"
    u_frame.to_parquet(output_file, index=False)

    result = {
        "result":      "ok",
        "model_id":    model_id,
        "nv":          nv,
        "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def report_runcorr(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """상관관계 보고서 생성 — 변수 간 상관계수 산출."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("report_runcorr (python_h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj.get("model_id", "corr_report")
    data_path    = json_obj["data_path"]
    feature_cols = json_obj.get("feature_cols")
    corr_method  = json_obj.get("method", "pearson")
    threshold    = float(json_obj.get("threshold", 0.8))

    full_path = root_path / data_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = list(df.select_dtypes(include=[np.number]).columns)

    corr_matrix = df[feature_cols].corr(method=corr_method)

    # 고상관 쌍 추출
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            c = abs(corr_matrix.iloc[i, j])
            if c >= threshold:
                high_corr_pairs.append({
                    "var1": feature_cols[i],
                    "var2": feature_cols[j],
                    "corr": round(float(corr_matrix.iloc[i, j]), 4),
                })

    result = {
        "result":          "ok",
        "model_id":        model_id,
        "n_high_corr":     len(high_corr_pairs),
        "high_corr_pairs": high_corr_pairs,
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def fit_under_ensemble(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O 언더샘플링 앙상블 학습."""
    import json as _json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("fit_under_ensemble (python_h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    n_models     = int(json_obj.get("n_models", 5))
    bad_val      = json_obj.get("bad_val", 1)
    algorithm    = json_obj.get("algorithm", "gbm").lower()

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    import h2o
    from h2o.estimators import H2OGradientBoostingEstimator, H2ORandomForestEstimator

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)

    bad_df  = df[df[target_col] == bad_val]
    good_df = df[df[target_col] != bad_val]
    n_bad   = len(bad_df)

    mojo_paths = []
    for i in range(n_models):
        sample_good = good_df.sample(n=min(n_bad * 3, len(good_df)), random_state=i)
        sub_df      = pd.concat([bad_df, sample_good]).sample(frac=1, random_state=i)
        train_h2o   = h2o.H2OFrame(sub_df[feature_cols + [target_col]].fillna(0))
        train_h2o[target_col] = train_h2o[target_col].asfactor()

        algo_cls = H2OGradientBoostingEstimator if algorithm == "gbm" else H2ORandomForestEstimator
        sub_model = algo_cls(ntrees=50, max_depth=5, seed=i)
        sub_model.train(x=feature_cols, y=target_col, training_frame=train_h2o)

        sub_id    = f"{model_id}_sub{i}"
        model_dir = root_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        mojo_path = sub_model.save_mojo(str(model_dir))
        mojo_paths.append(mojo_path)

    meta = {
        "model_id":     model_id,
        "sub_model_ids": [f"{model_id}_sub{i}" for i in range(n_models)],
        "mojo_paths":   mojo_paths,
        "feature_cols": feature_cols,
        "target_col":   target_col,
        "n_models":     n_models,
    }
    meta_file = root_path / "models" / f"{model_id}_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        _json.dump(meta, f, ensure_ascii=False)

    result = {"result": "ok", "model_id": model_id, "n_models": n_models, "mojo_paths": mojo_paths}
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def fit_automl(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O AutoML 학습."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    logger.info("fit_automl (python_h2o): model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    max_runtime  = int(json_obj.get("max_runtime_secs", 300))

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    import h2o
    from h2o.automl import H2OAutoML

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    aml = H2OAutoML(max_runtime_secs=max_runtime, seed=42)
    aml.train(x=feature_cols, y=target_col, training_frame=train_h2o)
    best_model = aml.leader

    return save_h2o_fitting_result(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir,
        best_model, json_obj,
    )


def fit_selected_automl_model(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """AutoML 리더보드에서 선택한 모델 재학습."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("fit_selected_automl_model: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    selected_algo = json_obj.get("selected_algo", "GBM")
    feature_cols = json_obj.get("feature_cols")

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    import h2o
    from h2o.estimators import (
        H2OGradientBoostingEstimator, H2ORandomForestEstimator,
        H2OXGBoostEstimator, H2ODeepLearningEstimator,
        H2OGeneralizedLinearEstimator,
    )

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    algo_map = {
        "GBM":          H2OGradientBoostingEstimator,
        "DRF":          H2ORandomForestEstimator,
        "XGBoost":      H2OXGBoostEstimator,
        "DeepLearning": H2ODeepLearningEstimator,
        "GLM":          H2OGeneralizedLinearEstimator,
    }
    estimator_cls = algo_map.get(selected_algo, H2OGradientBoostingEstimator)
    extra_params  = json_obj.get("model_params", {})
    model = estimator_cls(seed=42, **extra_params)
    model.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    return save_h2o_fitting_result(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir,
        model, json_obj,
    )


def fit_rulefit(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O RuleFit 학습."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("fit_rulefit: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")

    full_path = root_path / train_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    import h2o
    from h2o.estimators import H2ORuleFitEstimator

    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    train_h2o = h2o.H2OFrame(df[feature_cols + [target_col]].fillna(0))
    train_h2o[target_col] = train_h2o[target_col].asfactor()

    rulefit = H2ORuleFitEstimator(
        max_rule_length=int(json_obj.get("max_rule_length", 5)),
        max_num_rules=int(json_obj.get("max_num_rules", 100)),
        seed=42,
    )
    rulefit.train(x=feature_cols, y=target_col, training_frame=train_h2o)

    return save_h2o_fitting_result(
        service_db_info, file_server_host, file_server_port,
        h2o_file_server_host, h2o_file_server_port,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir,
        rulefit, json_obj,
    )


def predict_rulefit_ml(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O RuleFit 예측 (ML 노드)."""
    import json as _json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_rulefit_ml: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(root_dir)
    model_id     = json_obj["model_id"]
    predict_path = json_obj["predict_path"]

    meta_file = root_path / "models" / f"{model_id}_meta.json"
    with open(meta_file, encoding="utf-8") as f:
        meta = _json.load(f)

    full_path = root_path / predict_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)

    import h2o
    h2o.init(ip=h2o_host, port=h2o_port, nthreads=-1)
    model = h2o.import_mojo(str(root_path / meta.get("mojo_path", "")))

    feature_cols = meta.get("feature_cols", [])
    h2o_frame    = h2o.H2OFrame(df[[c for c in feature_cols if c in df.columns]].fillna(0))
    preds        = model.predict(h2o_frame).as_data_frame()
    df["score"]  = preds.iloc[:, -1].values.round(6)

    output_dir  = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_rulefit_predict.parquet"
    df.to_parquet(output_file, index=False)

    result = {
        "result": "ok", "model_id": model_id,
        "total_rows": len(df), "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def convert_rulefit_linear_to_category(
    rulefit_model,
    df_var_layout: "pd.DataFrame",
    threshold: float = 0.0,
) -> "pd.DataFrame":
    """RuleFit 선형 항을 카테고리 변수로 변환."""
    import pandas as pd

    rules = []
    try:
        rule_importance = rulefit_model.rule_importance()
        if rule_importance is not None:
            for _, row in rule_importance.iterrows():
                if abs(float(row.get("coefficient", 0))) > threshold:
                    rules.append({
                        "rule":        str(row.get("rule", "")),
                        "coefficient": float(row.get("coefficient", 0)),
                        "support":     float(row.get("support", 0)),
                    })
    except Exception as e:
        logger.warning("convert_rulefit_linear_to_category: %s", e)

    return pd.DataFrame(rules)


def predict_rulefit(
    service_db_info, file_server_host, file_server_port,
    h2o_file_server_host, h2o_file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf,
    h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
) -> dict:
    """H2O RuleFit 전체 예측 파이프라인."""
    return predict_rulefit_ml(
        service_db_info, file_server_host, file_server_port,
        numpy_use_32bit_float_precision,
        result_file_path_faf, done_file_path_faf,
        h2o_host, h2o_port, h2o_script_obj, root_dir, json_obj,
    )
