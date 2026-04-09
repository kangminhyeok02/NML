# stg_executor.py — 업무 전략 적용 (Strategy)

**파일:** `executors/ml/stg_executor.py`  
**클래스:** `StrategyExecutor(BaseExecutor)`

## 개요

모델 예측 결과(점수/확률)를 실제 업무 의사결정(승인·거절·등급·한도)으로 변환하는 executor.  
금융/리스크 도메인에서 모델 점수를 정책으로 연결하는 최종 ML 단계.

```
모델 점수 데이터 로드
    ↓ 전략 적용 (grade / threshold / tiered / matrix)
    ↓ 오버라이드 룰 적용 (사기이력, 고DSR 등 강제 의사결정)
    ↓ grade, decision, limit_amt, override_flag 컬럼 추가
    ↓ strategy/{output_id}_result.parquet 저장
    ↓ {output_id}_summary.json 저장
```

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `input_path` | ✅ | `str` | 예측 점수가 포함된 데이터 경로 (.parquet) |
| `score_col` | ✅ | `str` | 점수 컬럼명 |
| `strategy_type` | ✅ | `str` | `"grade"` \| `"threshold"` \| `"tiered"` \| `"matrix"` |
| `output_id` | ✅ | `str` | 결과 저장 식별자 |
| `grade_map` | 조건부 | `dict` | 등급별 점수 구간 (grade 전략 필수) |
| `threshold` | 조건부 | `float` | 승인 임계값 (threshold 전략 필수) |
| `tiered_rules` | 조건부 | `list` | 다단계 정책 목록 (tiered 전략 필수) |
| `matrix_rules` | 조건부 | `dict` | 2차원 매트릭스 정의 (matrix 전략 필수) |
| `override_rules` | ❌ | `list` | 강제 오버라이드 룰 목록 |
| `key_cols` | ❌ | `list` | 결과에 포함할 키 컬럼 목록 |

---

## 전략 유형별 상세

### `"grade"` — 점수 구간 등급화

```python
grade_map = {"A": [800, 1000], "B": [600, 800], "C": [400, 600], "D": [0, 400]}
# 결과 컬럼: grade (A/B/C/D/UNKNOWN), decision (APPROVE/REJECT)
# A, B → APPROVE / C, D → REJECT (커스텀 가능)
```

---

### `"threshold"` — 임계값 기반 승인/거절

```python
threshold = 0.3    # 확률 기준 예시
# score >= 0.3 → decision = "APPROVE"
# score <  0.3 → decision = "REJECT"
```

---

### `"tiered"` — 다단계 등급별 한도·조건

```python
tiered_rules = [
    {"score_min": 800, "grade": "A", "limit_pct": 1.0,  "rate": 3.5},
    {"score_min": 600, "grade": "B", "limit_pct": 0.7,  "rate": 5.0},
    {"score_min": 400, "grade": "C", "limit_pct": 0.3,  "rate": 8.0},
    {"score_min": 0,   "grade": "D", "reject": True},
]
# 결과 컬럼: grade, limit_pct, rate, decision
```

---

### `"matrix"` — 2차원 매트릭스

두 개 변수(예: 점수 구간 × 직종)의 교차 매트릭스로 의사결정.

```python
matrix_rules = {
    "score_bins": [0, 400, 600, 800, 1000],
    "col2":       "job_type",
    "matrix": {
        "full_time": ["REJECT", "REVIEW", "APPROVE", "APPROVE"],
        "part_time": ["REJECT", "REJECT",  "REVIEW", "APPROVE"],
    }
}
```

---

## `_apply_overrides(df, override_rules)` → `(pd.DataFrame, int)`

오버라이드 룰은 정책 강제 반영 기능으로 모델 결과를 덮어쓴다.

```python
override_rules = [
    {"condition": "fraud_flag == 1",  "decision": "REJECT", "reason": "사기이력"},
    {"condition": "dsr > 0.7",        "decision": "REJECT", "reason": "고DSR"},
    {"condition": "age < 19",         "decision": "REJECT", "reason": "미성년"},
]
# override_flag=1, override_reason 컬럼 추가
```

반환: `(결과 DataFrame, 오버라이드 적용 건수)`

---

## execute() 진행률

| 단계 | progress |
|------|----------|
| 데이터 로드 완료 | 20% |
| 전략 적용 완료 | 65% |
| 오버라이드 적용 완료 | 85% |
| 저장 완료 | 95% |

---

## 저장 컬럼 선택

```python
save_cols = key_cols + [score_col] + [
    c for c in ["grade", "decision", "limit_amt", "override_flag"]
    if c in df.columns
]
# key_cols에 지정한 컬럼만 결과 파일에 포함
```

---

## 반환값

```python
{
    "status": "COMPLETED",
    "result": {
        "output_id":    "loan_stg_202312",
        "strategy_type": "tiered",
        "total_rows":   50000,
        "output_path":  "strategy/loan_stg_202312_result.parquet",
        "summary": {
            "grade_dist":    {"A": 12000, "B": 18000, "C": 11000, "D": 9000},
            "decision_dist": {"APPROVE": 30000, "REJECT": 20000},
            "override_cnt":  850,
        },
    },
    "message": "전략 적용 완료  tiered  50,000건",
    "job_id":  str,
    "elapsed_sec": float,
}
```

---

## 출력 파일

| 파일 | 경로 |
|------|------|
| 전략 결과 parquet | `strategy/{output_id}_result.parquet` |
| 요약 JSON | `strategy/{output_id}_summary.json` |

---

## 사용 예시

```python
config = {
    "job_id":        "stg_001",
    "input_path":    "predict/loan_pred_202312_result.parquet",
    "score_col":     "ml_score",
    "strategy_type": "tiered",
    "output_id":     "loan_stg_202312",
    "key_cols":      ["cust_id"],
    "tiered_rules": [
        {"score_min": 800, "grade": "A", "limit_pct": 1.0, "rate": 3.5},
        {"score_min": 600, "grade": "B", "limit_pct": 0.7, "rate": 5.0},
        {"score_min": 0,   "grade": "C", "reject": True},
    ],
    "override_rules": [
        {"condition": "fraud_flag == 1", "decision": "REJECT", "reason": "사기이력"},
    ],
}

from executors.ml.stg_executor import StrategyExecutor
result = StrategyExecutor(config=config).run()
```
