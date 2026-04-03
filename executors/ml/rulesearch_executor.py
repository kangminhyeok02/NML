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
