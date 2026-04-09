# rulesearch_executor.py — 설명 가능한 규칙 탐색

**파일:** `executors/ml/rulesearch_executor.py`  
**클래스:** `RuleSearchExecutor(BaseExecutor)`

## 개요

의사결정트리 또는 연관규칙 방법으로 if-then 규칙 후보를 발굴하는 executor.  
신용/리스크 도메인에서 정책 룰 설계나 해석 가능한 분류 기준을 만들 때 활용된다.

```
입력 데이터 로드
    ↓ method에 따라 규칙 탐색
      ├── decision_tree  → sklearn 트리 경로에서 규칙 추출
      ├── association    → Apriori/FP-Growth 연관규칙 탐색
      └── woe_rule       → WOE 기반 변수 구간 규칙 탐색
    ↓ min_bad_rate 필터 + lift/bad_rate 내림차순 정렬
    ↓ 상위 top_n개 선택 → analysis/{output_id}_rules.json 저장
```

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `source_path` | ✅ | `str` | 입력 데이터 경로 (.parquet) |
| `target_col` | ✅ | `str` | 타깃 컬럼 (1=Bad/Event) |
| `output_id` | ✅ | `str` | 결과 저장 식별자 |
| `method` | ❌ | `str` | `"decision_tree"` \| `"association"` \| `"woe_rule"` (기본: `"decision_tree"`) |
| `feature_cols` | ❌ | `list` | 탐색 대상 변수 목록 (없으면 타깃 외 전체) |
| `max_depth` | ❌ | `int` | 트리 최대 깊이 (기본: `4`) |
| `min_support` | ❌ | `float` | 연관규칙 최소 지지도 (기본: `0.05`) |
| `min_confidence` | ❌ | `float` | 연관규칙 최소 신뢰도 (기본: `0.3`) |
| `min_bad_rate` | ❌ | `float` | 규칙 필터 최소 bad rate (기본: `0.1`) |
| `top_n` | ❌ | `int` | 반환할 상위 규칙 수 (기본: `50`) |

---

## 탐색 방식별 상세

### `_search_tree_rules(df, feature_cols, target, cfg)` → `list`

sklearn `DecisionTreeClassifier`를 학습한 뒤 `export_text()`로 경로를 추출하여  
if-then 규칙 형태로 변환한다.

```python
dt = DecisionTreeClassifier(max_depth=cfg.get("max_depth", 4), random_state=42)
dt.fit(df[feature_cols], df[target])
text = export_text(dt, feature_names=feature_cols)
# 각 리프 노드의 경로 → 규칙 문자열 추출
```

규칙 구조:
```python
{
    "rule":      "income <= 50000 AND overdue_cnt > 2",
    "support":   0.082,    # 전체 중 해당 규칙 해당 비율
    "bad_rate":  0.412,    # 해당 구간의 bad rate
    "lift":      2.31,     # bad_rate / 전체 bad_rate
    "count":     8200,
}
```

---

### `_search_association_rules(df, feature_cols, target, cfg)` → `list`

`mlxtend`의 FP-Growth 알고리즘으로 연관규칙을 탐색한다.

```python
from mlxtend.frequent_patterns import fpgrowth, association_rules
freq_items = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
```

규칙 구조에 `antecedents`, `consequents`, `support`, `confidence`, `lift` 포함.

---

### `_search_woe_rules(df, feature_cols, target, cfg)` → `list`

각 변수의 WOE 구간 중 bad_rate가 높은 구간을 단순 규칙으로 추출한다.

```python
# 변수별 분위수 binning → WOE 계산 → bad_rate 기준 필터
{
    "rule":     "debt_ratio > 0.8",
    "bad_rate": 0.387,
    "lift":     2.18,
    "woe":      1.42,
    "iv":       0.089,
}
```

---

## 규칙 필터 및 정렬

탐색 완료 후 공통 후처리:

```python
rules = [r for r in rules if r.get("bad_rate", 0) >= min_bad_rate]
rules = sorted(rules, key=lambda x: x.get("lift", x.get("bad_rate", 0)), reverse=True)
rules = rules[:top_n]
```

---

## execute() 진행률

| 단계 | progress |
|------|----------|
| 데이터 로드 완료 | 20% |
| 규칙 탐색 완료 | 80% |
| 필터·정렬·저장 완료 | 95% |

---

## 반환값

```python
{
    "status": "COMPLETED",
    "result": {
        "output_path": "analysis/loan_rules_eda_rules.json",
        "total_rules": 42,
        "top_rule": {
            "rule":     "overdue_cnt > 3 AND debt_ratio > 0.7",
            "bad_rate": 0.623,
            "lift":     3.48,
            "support":  0.031,
        },
    },
    "message": "규칙 탐색 완료  method=decision_tree  rules=42개",
    "job_id":  str,
    "elapsed_sec": float,
}
```

---

## 출력 파일

| 파일 | 경로 |
|------|------|
| 규칙 결과 JSON | `analysis/{output_id}_rules.json` |

```json
{
  "output_id":   "loan_rules_eda",
  "method":      "decision_tree",
  "total_rules": 42,
  "rules": [
    {"rule": "overdue_cnt > 3 AND debt_ratio > 0.7", "bad_rate": 0.623, "lift": 3.48},
    ...
  ]
}
```

---

## StrategyExecutor와의 연계

```
RuleSearchExecutor
    → 규칙 목록 (bad_rate, lift 기준 정렬)

StrategyExecutor (override_rules에 상위 규칙 적용)
    → 고위험 구간 자동 거절
```

---

## 사용 예시

```python
config = {
    "job_id":        "rulesearch_001",
    "source_path":   "mart/loan_mart_train.parquet",
    "target_col":    "default",
    "output_id":     "loan_rules_v1",
    "method":        "decision_tree",
    "feature_cols":  ["overdue_cnt", "debt_ratio", "income", "age"],
    "max_depth":     4,
    "min_bad_rate":  0.15,
    "top_n":         30,
}

from executors.ml.rulesearch_executor import RuleSearchExecutor
result = RuleSearchExecutor(config=config).run()
```
