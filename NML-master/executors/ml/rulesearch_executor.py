"""
rulesearch_executor.py
----------------------
설명 가능한 규칙(Rule) 탐색 실행기.

의사결정트리 또는 연관규칙 방법으로 if-then 규칙 후보를 발굴한다.
신용/리스크 도메인에서 정책 룰 설계나 해석 가능한 분류 기준을 만들 때 활용된다.

탐색 방식:
  - decision_tree : 의사결정트리에서 규칙 경로 추출
  - association   : Apriori/FP-Growth 기반 연관규칙 탐색
  - woe_rule      : WOE 기반 변수 구간 규칙 탐색

출력물:
  - 규칙 목록 (조건, 지지도, 신뢰도, bad_rate 등)
  - 규칙 요약 JSON
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)


class RuleSearchExecutor(BaseExecutor):
    """
    규칙 탐색 executor.

    config 필수 키
    --------------
    source_path  : str   입력 데이터 경로 (.parquet)
    target_col   : str   타깃 컬럼 (1=Bad/Event)
    output_id    : str   결과 저장 식별자

    config 선택 키
    --------------
    method         : str   "decision_tree" | "association" | "woe_rule" (기본: "decision_tree")
    feature_cols   : list  탐색 대상 변수 목록
    max_depth      : int   트리 최대 깊이 (기본 4)
    min_support    : float 연관규칙 최소 지지도 (기본 0.05)
    min_confidence : float 연관규칙 최소 신뢰도 (기본 0.3)
    min_bad_rate   : float 규칙 필터 최소 bad rate (기본 0.1)
    top_n          : int   상위 N개 규칙 반환 (기본 50)
    """

    def execute(self) -> dict:
        cfg    = self.config
        method = cfg.get("method", "decision_tree")
        target = cfg["target_col"]

        df = self._load_dataframe(cfg["source_path"])
        feature_cols = cfg.get("feature_cols") or [c for c in df.columns if c != target]
        self._update_job_status(ExecutorStatus.RUNNING, progress=20)

        if method == "decision_tree":
            rules = self._search_tree_rules(df, feature_cols, target, cfg)
        elif method == "association":
            rules = self._search_association_rules(df, feature_cols, target, cfg)
        elif method == "woe_rule":
            rules = self._search_woe_rules(df, feature_cols, target, cfg)
        else:
            raise ExecutorException(f"지원하지 않는 method: {method}")

        self._update_job_status(ExecutorStatus.RUNNING, progress=80)

        # 필터 및 정렬
        min_bad_rate = cfg.get("min_bad_rate", 0.1)
        rules = [r for r in rules if r.get("bad_rate", 0) >= min_bad_rate]
        rules = sorted(rules, key=lambda x: x.get("lift", x.get("bad_rate", 0)), reverse=True)
        rules = rules[:cfg.get("top_n", 50)]

        output = {
            "output_id":   cfg["output_id"],
            "method":      method,
            "total_rules": len(rules),
            "rules":       rules,
        }
        output_path = f"analysis/{cfg['output_id']}_rules.json"
        self._save_json(output, output_path)

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result": {
                "output_path": output_path,
                "total_rules": len(rules),
                "top_rule":    rules[0] if rules else None,
            },
            "message": f"규칙 탐색 완료  method={method}  rules={len(rules)}개",
        }

    # ------------------------------------------------------------------

    def _search_tree_rules(self, df, feature_cols, target, cfg) -> list:
        """의사결정트리 경로에서 if-then 규칙 추출."""
        X = df[feature_cols].fillna(-9999)
        y = df[target]

        max_depth = cfg.get("max_depth", 4)
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=50, random_state=42)
        tree.fit(X, y)

        # 리프 노드별 조건 추출
        n_nodes      = tree.tree_.n_node_samples
        feature      = tree.tree_.feature
        threshold    = tree.tree_.threshold
        children_l   = tree.tree_.children_left
        children_r   = tree.tree_.children_right
        values       = tree.tree_.value

        rules = []

        def _recurse(node_id, conditions):
            if children_l[node_id] == children_r[node_id]:  # leaf
                leaf_vals = values[node_id][0]
                total = leaf_vals.sum()
                bad   = leaf_vals[1] if len(leaf_vals) > 1 else 0
                bad_rate = bad / total if total > 0 else 0
                overall_bad_rate = y.mean()
                lift = bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0
                rules.append({
                    "conditions": conditions[:],
                    "condition_str": " AND ".join(conditions),
                    "support":  round(int(total) / len(y), 4),
                    "bad_count": int(bad),
                    "total_count": int(total),
                    "bad_rate": round(float(bad_rate), 4),
                    "lift":     round(float(lift), 4),
                })
                return

            feat_name = feature_cols[feature[node_id]]
            thresh    = round(float(threshold[node_id]), 4)

            _recurse(children_l[node_id], conditions + [f"{feat_name} <= {thresh}"])
            _recurse(children_r[node_id], conditions + [f"{feat_name} > {thresh}"])

        _recurse(0, [])
        return rules

    def _search_association_rules(self, df, feature_cols, target, cfg) -> list:
        """FP-Growth 기반 연관규칙 탐색 (mlxtend 사용)."""
        from mlxtend.frequent_patterns import fpgrowth, association_rules as mlxt_ar

        # 수치형 → 구간화 → 이진 인코딩
        binary_df = pd.DataFrame()
        for col in feature_cols:
            try:
                bins = pd.qcut(df[col], q=5, duplicates="drop")
                dummies = pd.get_dummies(bins.astype(str), prefix=col)
                binary_df = pd.concat([binary_df, dummies], axis=1)
            except Exception:
                pass
        # 타깃 추가
        binary_df[f"{target}=1"] = (df[target] == 1).astype(int)

        min_support = cfg.get("min_support", 0.05)
        freq        = fpgrowth(binary_df.astype(bool), min_support=min_support, use_colnames=True)
        ar          = mlxt_ar(freq, metric="confidence", min_threshold=cfg.get("min_confidence", 0.3))

        rules = []
        for _, row in ar.iterrows():
            if f"{target}=1" in row["consequents"]:
                rules.append({
                    "condition_str": " AND ".join(sorted(row["antecedents"])),
                    "conditions":    sorted(row["antecedents"]),
                    "support":       round(float(row["support"]), 4),
                    "bad_rate":      round(float(row["confidence"]), 4),
                    "lift":          round(float(row["lift"]), 4),
                })
        return rules

    def _search_woe_rules(self, df, feature_cols, target, cfg) -> list:
        """변수 구간별 WOE 기반 bad rate 높은 규칙 탐색."""
        rules = []
        overall_bad = df[target].mean()

        for col in feature_cols:
            try:
                bins = pd.qcut(df[col], q=10, duplicates="drop")
            except Exception:
                continue
            for bin_label, group in df.groupby(bins):
                total   = len(group)
                bad     = group[target].sum()
                bad_rate = bad / total if total > 0 else 0
                lift     = bad_rate / overall_bad if overall_bad > 0 else 0
                rules.append({
                    "condition_str": f"{col} in {bin_label}",
                    "conditions":   [f"{col} in {bin_label}"],
                    "support":      round(total / len(df), 4),
                    "bad_count":    int(bad),
                    "total_count":  total,
                    "bad_rate":     round(float(bad_rate), 4),
                    "lift":         round(float(lift), 4),
                })
        return rules


# =============================================================================
# Module-level functions
# =============================================================================


def fit_rule_finder(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf, json_obj,
) -> dict:
    """규칙 탐색기(rule finder) 학습 — 의사결정트리 기반 if-then 규칙 발굴."""
    import json as _json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    logger.info("fit_rule_finder start: model_id=%s", json_obj.get("model_id"))

    root_dir     = json_obj.get("root_dir", "/data")
    root_path    = Path(root_dir)
    model_id     = json_obj.get("model_id", "rule_finder")
    train_path   = json_obj["train_path"]
    target_col   = json_obj["target_col"]
    feature_cols = json_obj.get("feature_cols")
    max_depth    = int(json_obj.get("max_depth", 4))
    min_samples_leaf = int(json_obj.get("min_samples_leaf", 50))
    good_val     = json_obj.get("good_val", 0)
    bad_val      = json_obj.get("bad_val", 1)

    float_dtype = np.float32 if numpy_use_32bit_float_precision else np.float64

    full_train = root_path / train_path
    df = pd.read_parquet(full_train) if str(full_train).endswith(".parquet") else pd.read_csv(full_train)

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].fillna(-9999).astype(float_dtype)
    y = df[target_col]

    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    tree.fit(X, y)

    # 리프 노드 규칙 추출
    n_nodes    = tree.tree_.n_node_samples
    feature    = tree.tree_.feature
    threshold  = tree.tree_.threshold
    children_l = tree.tree_.children_left
    children_r = tree.tree_.children_right
    values     = tree.tree_.value
    overall_bad_rate = float((y == bad_val).mean())

    rules = []

    def _recurse(node_id, conditions):
        if children_l[node_id] == children_r[node_id]:
            leaf_vals = values[node_id][0]
            total = int(leaf_vals.sum())
            bad_idx = list(tree.classes_).index(bad_val) if bad_val in tree.classes_ else 1
            bad = int(leaf_vals[bad_idx]) if bad_idx < len(leaf_vals) else 0
            bad_rate = bad / total if total > 0 else 0
            lift = bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0
            rules.append({
                "conditions":    conditions[:],
                "condition_str": " AND ".join(conditions),
                "support":       round(total / len(y), 4),
                "bad_count":     bad,
                "total_count":   total,
                "bad_rate":      round(float(bad_rate), 4),
                "lift":          round(float(lift), 4),
                "node_id":       int(node_id),
            })
            return
        feat_name = feature_cols[feature[node_id]]
        thresh    = round(float(threshold[node_id]), 4)
        _recurse(children_l[node_id], conditions + [f"{feat_name} <= {thresh}"])
        _recurse(children_r[node_id], conditions + [f"{feat_name} > {thresh}"])

    _recurse(0, [])

    # 저장
    model_dir = root_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"{model_id}_rule_finder.pkl"
    with open(model_file, "wb") as f:
        pickle.dump({"tree": tree, "feature_cols": feature_cols, "rules": rules,
                     "target_col": target_col, "good_val": good_val, "bad_val": bad_val}, f)

    result = {
        "result":       "ok",
        "model_id":     model_id,
        "n_rules":      len(rules),
        "model_file":   str(model_file),
        "feature_cols": feature_cols,
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()

    logger.info("fit_rule_finder done: n_rules=%d", len(rules))
    return result


def calculate_statistics_rule_finder(
    service_db_info, file_server_host, file_server_port,
    result_file_path_faf, done_file_path_faf, json_obj,
) -> dict:
    """규칙 탐색기 결과에 대한 통계 산출."""
    import json as _json
    import pickle
    from pathlib import Path

    import pandas as pd

    logger.info("calculate_statistics_rule_finder: model_id=%s", json_obj.get("model_id"))

    root_path  = Path(json_obj.get("root_dir", "/data"))
    model_id   = json_obj["model_id"]
    data_path  = json_obj.get("data_path") or json_obj.get("train_path")
    target_col = json_obj.get("target_col")
    subtarget_json = json_obj.get("subtarget_json", {})

    # 모델 로드
    model_file = root_path / "models" / f"{model_id}_rule_finder.pkl"
    with open(model_file, "rb") as f:
        model_data = pickle.load(f)

    rules        = model_data["rules"]
    feature_cols = model_data["feature_cols"]
    good_val     = model_data.get("good_val", 0)
    bad_val      = model_data.get("bad_val", 1)

    # 데이터 로드
    full_path = root_path / data_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)

    if target_col is None:
        target_col = model_data.get("target_col")

    total = len(df)
    overall_bad = int((df[target_col] == bad_val).sum())
    overall_bad_rate = overall_bad / total if total > 0 else 0

    stats = {
        "total_records":    total,
        "overall_bad_count": overall_bad,
        "overall_bad_rate": round(overall_bad_rate, 4),
        "n_rules":          len(rules),
        "top_rules": sorted(rules, key=lambda r: r.get("lift", 0), reverse=True)[:10],
    }

    # 서브타겟 처리
    if subtarget_json:
        subtarget_col = subtarget_json.get("subtarget_col")
        if subtarget_col and subtarget_col in df.columns:
            stats["subtarget_bad_rate"] = round(float(df[subtarget_col].mean()), 4)

    result = {"result": "ok", "model_id": model_id, "statistics": stats}
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def rule_count_table(
    rule_finder,
    rule_info,
    df_dev,
    df_val,
    target,
    good_val,
    bad_val,
    subtarget_json,
    subindex_json,
    val_suffix="val",
    rule_type="unit",
) -> pd.DataFrame:
    """규칙별 개발/검증 데이터 count table 생성."""
    import pandas as pd
    import numpy as np

    def _apply_rule(df, conditions):
        mask = pd.Series([True] * len(df), index=df.index)
        for cond in conditions:
            try:
                mask &= df.eval(cond)
            except Exception:
                pass
        return mask

    rows = []
    rules = rule_finder.get("rules", []) if isinstance(rule_finder, dict) else []

    if rule_info is not None:
        rules = [rule_info] if isinstance(rule_info, dict) else rule_info

    for rule in rules:
        conditions = rule.get("conditions", [])
        cond_str   = rule.get("condition_str", " AND ".join(conditions))

        for suffix, df in [("dev", df_dev), (val_suffix, df_val)]:
            if df is None:
                continue
            mask = _apply_rule(df, conditions)
            subset = df[mask]
            total  = len(subset)
            bad    = int((subset[target] == bad_val).sum())
            good   = int((subset[target] == good_val).sum())
            bad_rate = bad / total if total > 0 else 0
            overall_bad_rate = float((df[target] == bad_val).mean())
            lift = bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0
            rows.append({
                "condition_str": cond_str,
                "dataset":       suffix,
                "total":         total,
                "bad":           bad,
                "good":          good,
                "bad_rate":      round(bad_rate, 4),
                "lift":          round(lift, 4),
                "support":       round(total / len(df), 4),
                "rule_type":     rule_type,
            })

    return pd.DataFrame(rows)


def select_top_rules(
    service_db_info, file_server_host, file_server_port,
    result_file_path_faf, done_file_path_faf, json_obj,
) -> dict:
    """학습된 규칙 목록에서 조건에 맞는 상위 규칙 선택."""
    import json as _json
    import pickle
    from pathlib import Path

    logger.info("select_top_rules: model_id=%s", json_obj.get("model_id"))

    root_path  = Path(json_obj.get("root_dir", "/data"))
    model_id   = json_obj["model_id"]
    top_n      = int(json_obj.get("top_n", 20))
    min_lift   = float(json_obj.get("min_lift", 1.0))
    min_bad_rate = float(json_obj.get("min_bad_rate", 0.1))
    sort_by    = json_obj.get("sort_by", "lift")

    model_file = root_path / "models" / f"{model_id}_rule_finder.pkl"
    with open(model_file, "rb") as f:
        model_data = pickle.load(f)

    rules = model_data.get("rules", [])
    filtered = [r for r in rules
                if r.get("lift", 0) >= min_lift and r.get("bad_rate", 0) >= min_bad_rate]
    filtered = sorted(filtered, key=lambda r: r.get(sort_by, 0), reverse=True)
    top_rules = filtered[:top_n]

    result = {
        "result":     "ok",
        "model_id":   model_id,
        "n_selected": len(top_rules),
        "top_rules":  top_rules,
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result


def create_dummy_rules_combination(
    rules: list,
    max_combo: int = 2,
) -> list:
    """단일 규칙들을 AND 조합하여 복합 규칙 후보 생성."""
    from itertools import combinations

    combo_rules = []
    for r in range(1, max_combo + 1):
        for combo in combinations(rules, r):
            all_conditions = []
            for rule in combo:
                all_conditions.extend(rule.get("conditions", []))
            if all_conditions:
                combo_rules.append({
                    "conditions":    all_conditions,
                    "condition_str": " AND ".join(all_conditions),
                    "source_rules":  [r.get("condition_str") for r in combo],
                })
    return combo_rules


def fit_rule_optimizer_ga(
    service_db_info, file_server_host, file_server_port,
    result_file_path_faf, done_file_path_faf, json_obj,
) -> dict:
    """유전 알고리즘(GA) 기반 규칙 최적화기 학습."""
    import json as _json
    import pickle
    import random
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("fit_rule_optimizer_ga: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(json_obj.get("root_dir", "/data"))
    model_id     = json_obj["model_id"]
    finder_id    = json_obj.get("rule_finder_id", model_id)
    target_col   = json_obj.get("target_col")
    bad_val      = json_obj.get("bad_val", 1)
    n_generations = int(json_obj.get("n_generations", 50))
    pop_size     = int(json_obj.get("pop_size", 30))
    max_rules    = int(json_obj.get("max_rules", 5))
    data_path    = json_obj.get("train_path")

    # 1. load model from file-service
    model_file = root_path / "models" / f"{finder_id}_rule_finder.pkl"
    with open(model_file, "rb") as f:
        model_data = pickle.load(f)
    rules_pool = model_data.get("rules", [])
    if target_col is None:
        target_col = model_data.get("target_col")

    # 2. do something with loaded model (데이터 로드)
    full_path = root_path / data_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)

    def fitness(rule_indices):
        """복합 규칙의 적합도 = bad_rate * support (lift 등 조합 가능)."""
        try:
            mask = pd.Series([True] * len(df), index=df.index)
            for idx in rule_indices:
                rule = rules_pool[idx]
                for cond in rule.get("conditions", []):
                    try:
                        mask &= df.eval(cond)
                    except Exception:
                        pass
            subset = df[mask]
            if len(subset) == 0:
                return 0.0
            bad_rate = float((subset[target_col] == bad_val).mean())
            support  = len(subset) / len(df)
            return bad_rate * np.sqrt(support)
        except Exception:
            return 0.0

    n_rules = min(len(rules_pool), max_rules)
    if n_rules == 0:
        result = {"result": "ok", "model_id": model_id, "best_rules": [], "best_fitness": 0.0}
    else:
        # GA
        population = [
            random.sample(range(len(rules_pool)), random.randint(1, n_rules))
            for _ in range(pop_size)
        ]

        best_individual, best_score = population[0], 0.0
        for gen in range(n_generations):
            scored = [(ind, fitness(ind)) for ind in population]
            scored.sort(key=lambda x: x[1], reverse=True)
            population = [ind for ind, _ in scored[:pop_size // 2]]

            # 교차 및 변이
            offspring = []
            while len(offspring) < pop_size // 2:
                p1, p2 = random.sample(population[:10] if len(population) >= 10 else population, 2)
                mid   = len(p1) // 2
                child = list(set(p1[:mid] + p2[mid:]))[:n_rules] or [random.randint(0, len(rules_pool) - 1)]
                # 변이
                if random.random() < 0.1:
                    child[random.randint(0, len(child) - 1)] = random.randint(0, len(rules_pool) - 1)
                offspring.append(child)
            population += offspring

            if scored[0][1] > best_score:
                best_score      = scored[0][1]
                best_individual = scored[0][0]

        best_rules = [rules_pool[i] for i in best_individual if i < len(rules_pool)]
        # make monotonicity values to values for display
        for r in best_rules:
            r.setdefault("monotonicity", "none")

        result = {
            "result":       "ok",
            "model_id":     model_id,
            "best_rules":   best_rules,
            "best_fitness": round(best_score, 4),
            "n_generations": n_generations,
        }

    # 3. rule optimizer(GA) 저장
    opt_file = root_path / "models" / f"{model_id}_rule_optimizer_ga.pkl"
    with open(opt_file, "wb") as f:
        pickle.dump(result, f)

    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False, default=str)
    Path(done_file_path_faf).touch()
    return result


def fit_rule_optimizer_greedy(
    service_db_info, file_server_host, file_server_port,
    result_file_path_faf, done_file_path_faf, json_obj,
) -> dict:
    """Greedy 방식 규칙 최적화기 학습."""
    import json as _json
    import pickle
    from pathlib import Path

    import pandas as pd

    logger.info("fit_rule_optimizer_greedy: model_id=%s", json_obj.get("model_id"))

    root_path   = Path(json_obj.get("root_dir", "/data"))
    model_id    = json_obj["model_id"]
    finder_id   = json_obj.get("rule_finder_id", model_id)
    target_col  = json_obj.get("target_col")
    bad_val     = json_obj.get("bad_val", 1)
    max_rules   = int(json_obj.get("max_rules", 5))
    data_path   = json_obj.get("train_path")

    # 1. load model from file-service
    model_file = root_path / "models" / f"{finder_id}_rule_finder.pkl"
    with open(model_file, "rb") as f:
        model_data = pickle.load(f)
    rules_pool = model_data.get("rules", [])
    if target_col is None:
        target_col = model_data.get("target_col")

    # 2. do something with loaded model
    full_path = root_path / data_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)

    selected_rules = []
    remaining_df   = df.copy()

    for _ in range(max_rules):
        best_rule, best_score = None, -1.0
        for rule in rules_pool:
            if rule in selected_rules:
                continue
            mask = pd.Series([True] * len(remaining_df), index=remaining_df.index)
            for cond in rule.get("conditions", []):
                try:
                    mask &= remaining_df.eval(cond)
                except Exception:
                    pass
            subset = remaining_df[mask]
            if len(subset) == 0:
                continue
            bad_rate = float((subset[target_col] == bad_val).mean())
            if bad_rate > best_score:
                best_score, best_rule = bad_rate, rule

        if best_rule is None:
            break
        selected_rules.append(best_rule)

        # 선택된 규칙의 대상 레코드 제거 (greedy)
        mask = pd.Series([True] * len(remaining_df), index=remaining_df.index)
        for cond in best_rule.get("conditions", []):
            try:
                mask &= remaining_df.eval(cond)
            except Exception:
                pass
        remaining_df = remaining_df[~mask]

    # make monotonicity values to values for display
    for r in selected_rules:
        r.setdefault("monotonicity", "none")

    # 3. rule optimizer(Greedy) 저장
    result = {
        "result":       "ok",
        "model_id":     model_id,
        "best_rules":   selected_rules,
        "n_selected":   len(selected_rules),
    }
    opt_file = root_path / "models" / f"{model_id}_rule_optimizer_greedy.pkl"
    with open(opt_file, "wb") as f:
        pickle.dump(result, f)

    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False, default=str)
    Path(done_file_path_faf).touch()
    return result


def predict_rule_optimizer(
    service_db_info, file_server_host, file_server_port,
    numpy_use_32bit_float_precision,
    result_file_path_faf, done_file_path_faf, json_obj,
) -> dict:
    """규칙 최적화기를 사용한 예측 — 규칙 적중 여부 및 스코어 산출."""
    import json as _json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    logger.info("predict_rule_optimizer: model_id=%s", json_obj.get("model_id"))

    root_path    = Path(json_obj.get("root_dir", "/data"))
    model_id     = json_obj["model_id"]
    optimizer_type = json_obj.get("optimizer_type", "greedy")  # "ga" or "greedy"
    data_path    = json_obj["predict_path"]

    # 1. load model from file-service
    # 1) finder
    finder_id  = json_obj.get("rule_finder_id", model_id)
    finder_file = root_path / "models" / f"{finder_id}_rule_finder.pkl"
    with open(finder_file, "rb") as f:
        finder_data = pickle.load(f)

    # 2) optimizer
    opt_suffix = "ga" if optimizer_type == "ga" else "greedy"
    opt_file   = root_path / "models" / f"{model_id}_rule_optimizer_{opt_suffix}.pkl"
    with open(opt_file, "rb") as f:
        opt_data = pickle.load(f)

    best_rules = opt_data.get("best_rules", [])

    # make monotonicity values to values for display
    for r in best_rules:
        r.setdefault("monotonicity", "none")

    # 데이터 로드 및 예측
    full_path = root_path / data_path
    df = pd.read_parquet(full_path) if str(full_path).endswith(".parquet") else pd.read_csv(full_path)
    result_df = df.copy()

    for i, rule in enumerate(best_rules):
        col_name = f"rule_{i+1}_hit"
        mask = pd.Series([True] * len(df), index=df.index)
        for cond in rule.get("conditions", []):
            try:
                mask &= df.eval(cond)
            except Exception:
                pass
        result_df[col_name] = mask.astype(int)

    # 종합 규칙 적중 여부
    hit_cols = [f"rule_{i+1}_hit" for i in range(len(best_rules))]
    if hit_cols:
        result_df["any_rule_hit"] = result_df[hit_cols].max(axis=1)
        result_df["rule_hit_count"] = result_df[hit_cols].sum(axis=1)
    else:
        result_df["any_rule_hit"] = 0
        result_df["rule_hit_count"] = 0

    # 저장
    output_dir = root_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_id}_rule_predict.parquet"
    result_df.to_parquet(output_file, index=False)

    result = {
        "result":      "ok",
        "model_id":    model_id,
        "n_rules":     len(best_rules),
        "total_rows":  len(result_df),
        "hit_rate":    round(float(result_df["any_rule_hit"].mean()), 4),
        "output_file": str(output_file),
    }
    with open(result_file_path_faf, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False)
    Path(done_file_path_faf).touch()
    return result
