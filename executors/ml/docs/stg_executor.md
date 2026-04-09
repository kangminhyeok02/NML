# stg_executor.py (Strategy Executor)

모델 예측 결과를 실제 업무 전략(Strategy)으로 변환하는 실행기.

금융/리스크 도메인에서 모델이 산출한 점수/확률값을 바탕으로  
승인/거절/한도/등급 등의 업무 의사결정을 자동화한다.

---

## 클래스

### `StrategyExecutor(BaseExecutor)`

업무 전략 적용 executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `input_path` | `str` | 예측 점수가 포함된 데이터 경로 (`.parquet`) |
| `score_col` | `str` | 점수 컬럼명 |
| `strategy_type` | `str` | `"grade"` \| `"threshold"` \| `"tiered"` \| `"matrix"` |
| `output_id` | `str` | 결과 저장 식별자 |

#### config 선택 키

| 키 | 타입 | 설명 |
|---|---|---|
| `grade_map` | `dict` | 등급별 점수 구간. `grade` 전략에 필요 |
| `threshold` | `float` | 이진 임계값. `threshold` 전략에 필요 |
| `tiered_rules` | `list` | 다단계 정책 목록. `tiered` 전략에 필요 |
| `matrix_rules` | `dict` | 2차원 매트릭스 정의. `matrix` 전략에 필요 |
| `override_rules` | `list` | 오버라이드 룰 목록 |
| `key_cols` | `list` | 결과에 포함할 키 컬럼 목록 |

---

### `execute() → dict`

전략 적용 전체 파이프라인을 실행한다.

**실행 순서**
1. 예측 데이터 로드 및 점수 컬럼 검증
2. `strategy_type`에 따라 전략 적용
3. `override_rules` 있으면 오버라이드 적용
4. 결과 저장: `strategy/{output_id}_result.parquet`
5. 요약 저장: `strategy/{output_id}_summary.json`

**저장 컬럼**: `key_cols` + `score_col` + `grade`, `decision`, `limit_amt`, `override_flag` (존재하는 것만)

**반환값**
```python
{
    "status": "COMPLETED",
    "result": {
        "output_path": str,
        "total_rows":  int,
        "summary":     dict,
    },
    "message": str,
}
```

---

### `_apply_grade_strategy(df, score_col, grade_map) → pd.DataFrame`

점수를 등급(A/B/C/D 등)으로 매핑하고 의사결정을 추가한다.

```python
# grade_map 예시:
# {"A": [800, 1000], "B": [600, 800], "C": [400, 600], "D": [0, 400]}
```

- `grade` 컬럼: 점수 구간에 해당하는 등급 (미해당 시 `"UNKNOWN"`)
- `decision` 컬럼: A/B 등급이면 `"APPROVE"`, 그 외 `"REJECT"`

---

### `_apply_threshold_strategy(df, score_col, threshold) → pd.DataFrame`

단일 임계값으로 이진 승인/거절을 결정한다.

- `score >= threshold` → `"APPROVE"`
- `score < threshold` → `"REJECT"`
- `numpy.where`로 벡터화 처리

---

### `_apply_tiered_strategy(df, score_col, tiered_rules) → pd.DataFrame`

다단계 정책(tiered rules)을 적용한다. score_min 내림차순으로 정렬 후 각 고객에게 첫 번째 해당 규칙을 적용한다.

```python
# tiered_rules 예시:
# [
#   {"score_min": 800, "grade": "A", "limit_pct": 1.0, "rate": 3.5},
#   {"score_min": 600, "grade": "B", "limit_pct": 0.7, "rate": 5.0},
#   {"score_min": 0,   "grade": "C", "limit_pct": 0.0, "reject": True}
# ]
```

- `grade`, `decision`, `limit_pct` 컬럼 추가
- 해당 규칙 없으면 `"REJECT_ALL"`, `"REJECT"`, `0.0`

---

### `_apply_matrix_strategy(df, score_col, matrix_rules) → pd.DataFrame`

기존 등급(existing_grade)과 모델 점수의 2차원 매트릭스로 의사결정한다.

```python
# matrix_rules 예시:
# {
#   "existing_grade_col": "crif_grade",
#   "matrix": {
#     "A": {"700+": "APPROVE_FULL", "500-700": "APPROVE_PARTIAL"},
#     "B": {"700+": "APPROVE_PARTIAL", "500-700": "REVIEW"}
#   }
# }
```

점수 구간: `700+`, `500-700`, `500-`으로 3분류.  
매트릭스에 없는 조합은 `"REVIEW"` 적용.

---

### `_apply_overrides(df, override_rules) → (pd.DataFrame, int)`

정책 룰에 의한 강제 오버라이드를 적용한다.

```python
# override_rules 예시:
# [
#   {"condition": "debt_ratio > 0.9", "decision": "REJECT", "reason": "고DSR"},
#   {"condition": "fraud_flag == 1",  "decision": "REJECT", "reason": "사기이력"}
# ]
```

- `df.query(condition)`으로 대상을 찾아 `decision`을 강제 변경
- `override_flag = 1`, `override_reason` 컬럼 업데이트
- 오버라이드된 건수를 두 번째 반환값으로 반환

---

### `_build_summary(df, cfg) → dict`

전략 적용 결과 요약을 생성한다.

- 점수 기술통계 (`score_stats`)
- 등급 분포 (`grade_dist`, `grade` 컬럼 있을 때)
- 의사결정 분포 (`decision_dist`, `decision` 컬럼 있을 때)
- 오버라이드 건수 (`override_count`, `override_flag` 컬럼 있을 때)
