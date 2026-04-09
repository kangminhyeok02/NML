# stg_executor.py — 업무 전략 적용 (Strategy)

## 개요

모델 예측 결과(점수/확률)를 실제 업무 의사결정(승인·거절·등급·한도)으로 변환하는 executor.  
금융/리스크 도메인에서 모델 점수를 정책으로 연결하는 최종 단계.

```
모델 점수
    ↓ 전략 적용 (grade / threshold / tiered / matrix)
    ↓ 오버라이드 룰 적용 (사기이력, 고DSR 등 강제 거절)
    ↓ grade, decision, limit_amt, override_flag 컬럼 추가
    ↓ 결과 저장
```

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `input_path` | ✅ | `str` | 예측 점수 포함 데이터 경로 (.parquet) |
| `score_col` | ✅ | `str` | 점수 컬럼명 |
| `strategy_type` | ✅ | `str` | `"grade"` \| `"threshold"` \| `"tiered"` \| `"matrix"` |
| `output_id` | ✅ | `str` | 결과 저장 식별자 |
| `grade_map` | ❌ | `dict` | 등급별 점수 구간 (grade 전략 필수) |
| `threshold` | ❌ | `float` | 승인 임계값 (threshold 전략 필수) |
| `tiered_rules` | ❌ | `list` | 다단계 정책 목록 (tiered 전략 필수) |
| `matrix_rules` | ❌ | `dict` | 2차원 매트릭스 정의 (matrix 전략 필수) |
| `override_rules` | ❌ | `list` | 강제 오버라이드 룰 목록 |
| `key_cols` | ❌ | `list` | 결과에 포함할 키 컬럼 목록 |

---

## 전략 유형별 상세

### `"grade"` — 점수 구간 등급화

```python
grade_map = {
    "A": [800, 1000],
    "B": [600, 800],
    "C": [400, 600],
    "D": [0,   400],
}
# 결과 컬럼: grade (A/B/C/D/UNKNOWN), decision (APPROVE/REJECT)
# A, B → APPROVE / C, D → REJECT
```

---

### `"threshold"` — 임계값 기반 승인/거절

```python
threshold = 0.3
# score >= 0.3 → APPROVE
# score <  0.3 → REJECT
# 결과 컬럼: decision
```

---

### `"tiered"` — 다단계 정책 (등급별 한도/금리)

```python
tiered_rules = [
    {"score_min": 800, "grade": "A", "limit_pct": 1.0, "rate": 3.5},
    {"score_min": 600, "grade": "B", "limit_pct": 0.7, "rate": 5.0},
    {"score_min": 400, "grade": "C", "limit_pct": 0.3, "rate": 8.0},
    {"score_min": 0,   "grade": "D", "limit_pct": 0.0, "reject": True},
]
# 결과 컬럼: grade, decision, limit_pct
# score_min 내림차순 정렬 후 첫 번째 만족 룰 적용
```

---

### `"matrix"` — 2차원 의사결정 매트릭스

```python
matrix_rules = {
    "existing_grade_col": "crif_grade",   # 기존 신용등급 컬럼
    "matrix": {
        "A": {"700+": "APPROVE_FULL",    "500-700": "APPROVE_PARTIAL"},
        "B": {"700+": "APPROVE_PARTIAL", "500-700": "REVIEW"},
        "C": {"700+": "REVIEW",          "500-700": "REJECT"},
    }
}
# 점수 구간: 700+ / 500-700 / 500-
# 결과 컬럼: decision (APPROVE_FULL / APPROVE_PARTIAL / REVIEW / REJECT)
```

---

## 오버라이드 룰 (`override_rules`)

전략 적용 결과를 강제로 덮어쓰는 비즈니스 룰.  
`df.eval(condition)`으로 해당 건을 찾아 `decision`을 강제 변경한다.

```python
override_rules = [
    {
        "condition": "fraud_flag == 1",
        "decision":  "REJECT",
        "reason":    "사기이력 보유"
    },
    {
        "condition": "debt_ratio > 0.9",
        "decision":  "REJECT",
        "reason":    "고DSR (총부채상환비율 90% 초과)"
    },
    {
        "condition": "bankruptcy_yn == 'Y'",
        "decision":  "REJECT",
        "reason":    "파산 이력"
    },
]
# 결과: override_flag=1, override_reason 컬럼 추가
```

---

## 실행 흐름

```
1. input_path 데이터 로드 + score_col 존재 확인      [progress 20%]
2. strategy_type별 전략 적용                         [progress 65%]
3. override_rules 적용 (있을 때만)                   [progress 85%]
4. key_cols + score_col + 결과 컬럼 선택
5. strategy/{output_id}_result.parquet 저장
6. strategy/{output_id}_summary.json 저장
```

---

## 출력 결과

**결과 파일:** `strategy/{output_id}_result.parquet`

포함 컬럼 (있을 때):
- `{key_cols}` — 식별 키
- `{score_col}` — 모델 점수
- `grade` — 등급
- `decision` — 의사결정 (APPROVE/REJECT/REVIEW/...)
- `limit_pct` — 한도 비율 (tiered 전략)
- `override_flag` — 오버라이드 여부 (0/1)
- `override_reason` — 오버라이드 사유

**요약 파일:** `strategy/{output_id}_summary.json`
```json
{
  "strategy_type":  "tiered",
  "total":          100000,
  "decision_dist":  {"APPROVE": 62000, "REJECT": 38000},
  "approve_rate":   0.62,
  "grade_dist":     {"A": 12000, "B": 25000, "C": 25000, "D": 38000},
  "override_count": 1243
}
```

---

## 파이프라인 위치

```
PredictExecutor
    ↓ score 컬럼 포함 parquet
StrategyExecutor
    ↓ grade, decision, override_flag 추가
ExportExecutor
    ↓ DB 적재 또는 API 전송
```

---

## 주요 설계 포인트

- **오버라이드 우선**: 모델 점수가 아무리 좋아도 사기·파산 이력은 무조건 거절
- **투명성**: `override_flag` + `override_reason`으로 모든 강제 변경 추적 가능
- **확장성**: `matrix` 전략으로 기존 CB 등급 + 새 모델 점수를 2차원으로 결합 가능
