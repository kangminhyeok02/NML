# rulesearch_executor.py — 설명 가능한 규칙 탐색

## 개요

의사결정트리, 연관규칙, WOE 구간 분석으로 if-then 형태의 설명 가능한 규칙을 발굴하는 executor.  
신용/리스크 도메인에서 정책 룰 설계나 해석 가능한 분류 기준을 만들 때 활용된다.

---

## 탐색 방식 (`method`)

| `method` | 알고리즘 | 출력 형태 |
|---|---|---|
| `decision_tree` | CART 의사결정트리 경로 추출 | `if A<=x AND B>y → bad_rate` |
| `association` | FP-Growth 연관규칙 | `if A∈[범위] AND B∈[범위] → confidence` |
| `woe_rule` | WOE 기반 구간별 bad rate | `if col ∈ bin → bad_rate` |

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `source_path` | ✅ | `str` | 입력 데이터 경로 (.parquet) |
| `target_col` | ✅ | `str` | 타깃 컬럼 (1=Bad/Event) |
| `output_id` | ✅ | `str` | 결과 저장 식별자 |
| `method` | ❌ | `str` | 탐색 방식 (기본: `"decision_tree"`) |
| `feature_cols` | ❌ | `list` | 탐색 대상 변수 (없으면 타깃 외 전체) |
| `max_depth` | ❌ | `int` | 트리 최대 깊이 (기본: `4`) |
| `min_support` | ❌ | `float` | 연관규칙 최소 지지도 (기본: `0.05`) |
| `min_confidence` | ❌ | `float` | 연관규칙 최소 신뢰도 (기본: `0.3`) |
| `min_bad_rate` | ❌ | `float` | 규칙 필터 최소 bad rate (기본: `0.1`) |
| `top_n` | ❌ | `int` | 상위 N개 규칙 반환 (기본: `50`) |

---

## 탐색 메서드 상세

### `_search_tree_rules(df, feature_cols, target, cfg)` → `list`

```
① DecisionTreeClassifier(max_depth=4, min_samples_leaf=50) 학습
② 트리 구조 순회 (_recurse):
   - 분기 노드: 조건 스택에 "feature <= threshold" 추가
   - 리프 노드: 조건 스택 → 규칙 생성
```

**규칙 예시:**
```python
{
    "condition_str": "debt_ratio > 0.7 AND income <= 3000",
    "conditions":    ["debt_ratio > 0.7", "income <= 3000"],
    "support":       0.0823,
    "bad_count":     412,
    "total_count":   823,
    "bad_rate":      0.5006,
    "lift":          3.21,
}
```

---

### `_search_association_rules(df, feature_cols, target, cfg)` → `list`

```
① 수치형 변수 → qcut(q=5) → 이진 인코딩 (pd.get_dummies)
② FP-Growth(min_support) → 빈발 항목집합
③ association_rules(min_confidence) → 연관규칙
④ consequent에 "{target}=1" 포함 규칙만 필터
```

**출력:**
```python
{
    "condition_str": "income∈(0,3000] AND age∈(18,30]",
    "support":       0.0621,
    "bad_rate":      0.4823,  # = confidence
    "lift":          3.12,
}
```

---

### `_search_woe_rules(df, feature_cols, target, cfg)` → `list`

```
변수별 qcut(q=10) → 구간별 bad_rate, lift 산출
단변량 규칙 → 가장 단순하고 해석하기 쉬움
```

**Lift = bad_rate(구간) / bad_rate(전체)**  
Lift > 1이면 전체 평균보다 위험한 구간.

---

## 규칙 후처리

1. `bad_rate < min_bad_rate` 규칙 제거
2. `lift` 또는 `bad_rate` 기준 내림차순 정렬
3. 상위 `top_n`개만 반환

---

## 실행 흐름

```
1. source_path 데이터 로드
2. feature_cols 확정                                 [progress 20%]
3. method별 규칙 탐색                               [progress 80%]
4. min_bad_rate 필터 → lift/bad_rate 정렬 → top_n 선택
5. analysis/{output_id}_rules.json 저장
```

---

## 출력 결과

**저장 경로:** `analysis/{output_id}_rules.json`

```json
{
  "output_id":   "credit_rules_v1",
  "method":      "decision_tree",
  "total_rules": 47,
  "rules": [
    {
      "condition_str": "debt_ratio > 0.7 AND income <= 3000",
      "support":       0.0823,
      "bad_rate":      0.5006,
      "lift":          3.21,
      "bad_count":     412,
      "total_count":   823
    },
    ...
  ]
}
```

---

## StrategyExecutor와의 연계

발굴된 규칙을 `StrategyExecutor`의 `override_rules`로 활용하면  
고위험 구간에 대한 자동 거절 정책을 구현할 수 있다.

```python
override_rules = [
    {
        "condition": "debt_ratio > 0.7 and income <= 3000",
        "decision":  "REJECT",
        "reason":    "고부채비율+저소득 고위험 구간"
    }
]
```

---

## 세 방식 비교

| 항목 | `decision_tree` | `association` | `woe_rule` |
|------|----------------|---------------|-----------|
| 다변량 규칙 | ✅ (다변량 조건) | ✅ (다변량 조건) | ❌ (단변량) |
| 해석 용이성 | 높음 | 중간 | 매우 높음 |
| 계산 속도 | 빠름 | 느림 (대용량) | 빠름 |
| 추가 라이브러리 | 없음 | mlxtend | 없음 |
| 권장 상황 | 다변량 상호작용 탐색 | 장바구니 분석 유사 문제 | 변수별 위험 구간 확인 |
