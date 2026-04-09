# rulesearch_executor.py

설명 가능한 규칙(Rule) 탐색 실행기.

의사결정트리 또는 연관규칙 방법으로 if-then 규칙 후보를 발굴한다.  
신용/리스크 도메인에서 정책 룰 설계나 해석 가능한 분류 기준을 만들 때 활용된다.

---

## 클래스

### `RuleSearchExecutor(BaseExecutor)`

규칙 탐색 executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `source_path` | `str` | 입력 데이터 경로 (`.parquet`) |
| `target_col` | `str` | 타깃 컬럼 (1=Bad/Event) |
| `output_id` | `str` | 결과 저장 식별자 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|---|---|---|
| `method` | `"decision_tree"` | `"decision_tree"` \| `"association"` \| `"woe_rule"` |
| `feature_cols` | 타깃 외 전체 | 탐색 대상 변수 목록 |
| `max_depth` | `4` | 트리 최대 깊이 |
| `min_support` | `0.05` | 연관규칙 최소 지지도 |
| `min_confidence` | `0.3` | 연관규칙 최소 신뢰도 |
| `min_bad_rate` | `0.1` | 규칙 필터 최소 bad rate |
| `top_n` | `50` | 최종 반환 상위 규칙 수 |

---

### `execute() → dict`

규칙 탐색 전체 파이프라인을 실행한다.

**실행 순서**
1. 데이터 로드
2. `method`에 따라 규칙 탐색 실행
3. `min_bad_rate` 미만 규칙 제거
4. `lift` 또는 `bad_rate` 내림차순 정렬
5. 상위 `top_n`개만 선택
6. 결과를 `analysis/{output_id}_rules.json`에 저장

**반환값**
```python
{
    "status": "COMPLETED",
    "result": {
        "output_path": str,
        "total_rules": int,
        "top_rule":    dict | None,
    },
    "message": str,
}
```

---

### `_search_tree_rules(df, feature_cols, target, cfg) → list`

의사결정트리의 리프 노드 경로에서 if-then 규칙을 추출한다.

1. `DecisionTreeClassifier(max_depth=..., min_samples_leaf=50)` 학습
2. 트리를 재귀 탐색(`_recurse`)하여 각 리프 노드의 조건 경로 수집

**각 규칙 구조**
```python
{
    "conditions":    list[str],   # ["col_a <= 0.5", "col_b > 3.0"]
    "condition_str": str,         # "col_a <= 0.5 AND col_b > 3.0"
    "support":       float,       # 해당 리프의 비율
    "bad_count":     int,
    "total_count":   int,
    "bad_rate":      float,
    "lift":          float,       # bad_rate / overall_bad_rate
}
```

---

### `_search_association_rules(df, feature_cols, target, cfg) → list`

FP-Growth 기반 연관규칙을 탐색한다 (`mlxtend` 라이브러리 사용).

1. 수치형 변수를 5분위 구간으로 이진 인코딩
2. `fpgrowth(min_support=...)`로 빈발 항목집합 추출
3. `association_rules(metric="confidence", min_threshold=...)`로 규칙 필터링
4. consequent가 `{target}=1`인 규칙만 선택

**각 규칙 구조**
```python
{
    "condition_str": str,
    "conditions":    list[str],
    "support":       float,
    "bad_rate":      float,   # confidence 값
    "lift":          float,
}
```

---

### `_search_woe_rules(df, feature_cols, target, cfg) → list`

변수 구간별 WOE 기반 bad rate가 높은 규칙을 탐색한다.

1. 각 변수를 10분위(`pd.qcut`)로 구간화
2. 구간별 bad_rate와 lift 산출
3. 모든 변수/구간 조합을 규칙으로 반환

**각 규칙 구조**
```python
{
    "condition_str": "col_name in (0.1, 5.3]",
    "conditions":    list[str],
    "support":       float,
    "bad_count":     int,
    "total_count":   int,
    "bad_rate":      float,
    "lift":          float,
}
```

---

## 모듈 레벨 함수

### `fit_rule_finder(..., json_obj) → dict`

의사결정트리 기반 규칙 탐색기를 학습하고 결과를 저장한다.

- `_recurse()`로 리프 노드 규칙 추출 (bad_val 기반)
- 모델을 `models/{model_id}_rule_finder.pkl`에 저장
- `result_file_path_faf`, `done_file_path_faf`에 결과 기록

**json_obj 주요 키**: `model_id`, `train_path`, `target_col`, `feature_cols`, `max_depth`, `min_samples_leaf`, `good_val`, `bad_val`

---

### `calculate_statistics_rule_finder(..., json_obj) → dict`

학습된 규칙 탐색기 결과에 대한 통계를 산출한다.

- 전체 레코드 수, bad 건수, bad_rate
- lift 내림차순 상위 10개 규칙
- `subtarget_json`이 있으면 서브타겟 bad_rate도 포함

---

### `rule_count_table(rule_finder, rule_info, df_dev, df_val, target, good_val, bad_val, subtarget_json, subindex_json, val_suffix, rule_type) → pd.DataFrame`

규칙별 개발/검증 데이터 count table을 생성한다.

- 각 규칙의 조건을 `df.eval()`로 적용
- 개발(`dev`), 검증(`val_suffix`) 데이터셋 각각에 대해 통계 산출
- 반환 컬럼: `condition_str`, `dataset`, `total`, `bad`, `good`, `bad_rate`, `lift`, `support`, `rule_type`

---

### `select_top_rules(..., json_obj) → dict`

학습된 규칙 목록에서 조건에 맞는 상위 규칙을 선택한다.

| `json_obj` 키 | 기본값 | 설명 |
|---|---|---|
| `top_n` | `20` | 상위 N개 |
| `min_lift` | `1.0` | 최소 lift |
| `min_bad_rate` | `0.1` | 최소 bad_rate |
| `sort_by` | `"lift"` | 정렬 기준 |

---

### `create_dummy_rules_combination(rules, max_combo=2) → list`

단일 규칙들을 AND 조합하여 복합 규칙 후보를 생성한다.

- `itertools.combinations(rules, r)` for r in `1..max_combo`
- 각 조합의 conditions를 합쳐 `condition_str` 생성

---

### `fit_rule_optimizer_ga(..., json_obj) → dict`

유전 알고리즘(GA) 기반 규칙 최적화기를 학습한다.

**GA 동작**
- **적합도 함수**: `bad_rate * sqrt(support)` (bad를 많이 잡으면서 충분한 지지도)
- **선택**: 상위 50% 생존
- **교차**: 두 부모 규칙 인덱스를 절반씩 결합
- **변이**: 10% 확률로 랜덤 규칙 대체
- 최적 규칙 집합을 `models/{model_id}_rule_optimizer_ga.pkl`에 저장

**json_obj 주요 키**: `model_id`, `rule_finder_id`, `n_generations`, `pop_size`, `max_rules`, `train_path`, `target_col`, `bad_val`

---

### `fit_rule_optimizer_greedy(..., json_obj) → dict`

Greedy 방식 규칙 최적화기를 학습한다.

**Greedy 동작**
- 매 단계에서 현재 남은 데이터 기준 `bad_rate` 가장 높은 규칙 선택
- 선택된 규칙의 대상 레코드를 제거하고 다음 규칙 탐색
- `max_rules`개 규칙 선택 또는 더 이상 좋은 규칙 없을 때까지 반복
- 결과를 `models/{model_id}_rule_optimizer_greedy.pkl`에 저장

---

### `predict_rule_optimizer(..., json_obj) → dict`

규칙 최적화기를 사용하여 예측을 수행한다.

- `optimizer_type` (`"ga"` 또는 `"greedy"`)에 따라 해당 optimizer pkl 로드
- 각 규칙별 적중 여부를 `rule_{i}_hit` 컬럼으로 추가
- `any_rule_hit`: 하나라도 적중한 경우 1
- `rule_hit_count`: 적중한 규칙 수
- 결과를 `output/{model_id}_rule_predict.parquet`에 저장
