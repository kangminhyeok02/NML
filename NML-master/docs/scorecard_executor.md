# scorecard_executor.py — 신용평가 스코어카드 모델

**파일:** `executors/ml/scorecard_executor.py`  
**클래스:** `ScorecardExecutor(BaseExecutor)`

## 개요

금융/리스크 도메인에서 가장 전통적인 모델링 방식.  
변수 구간화(binning) → WOE 변환 → IV 산출 → 로지스틱 회귀 → PDO 점수 스케일링  
순서로 스코어카드를 생성한다.

```
학습 데이터 로드
    ↓ _calc_woe() : 수치형 변수 분위수 binning + WOE/IV 계산
    ↓ IV 필터 : iv_threshold 미만 변수 제거
    ↓ _apply_woe() : 선택 변수에 WOE 값 적용
    ↓ StandardScaler + LogisticRegression 학습
    ↓ _build_scorecard() : PDO 기반 점수 스케일링 → 스코어카드 테이블
    ↓ _score_data() → _evaluate_scorecard() : 성능 평가
    ↓ 결과 저장 (scorecard JSON + meta JSON)
```

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `train_path` | ✅ | `str` | 학습 데이터 경로 (.parquet) |
| `target_col` | ✅ | `str` | 타깃 컬럼 (1=Bad, 0=Good) |
| `model_id` | ✅ | `str` | 모델 저장 식별자 |
| `feature_cols` | ✅ | `list` | 스코어카드에 사용할 변수 목록 |
| `valid_path` | ❌ | `str` | 검증 데이터 경로 |
| `n_bins` | ❌ | `int` | binning 구간 수 (기본: `10`) |
| `min_bin_rate` | ❌ | `float` | 최소 bin 비율 (기본: `0.05`) |
| `iv_threshold` | ❌ | `float` | IV 필터 기준값 (기본: `0.02`) |
| `base_score` | ❌ | `int` | 기준 점수 (기본: `600`) |
| `pdo` | ❌ | `int` | PDO — odds 2배당 점수 변화량 (기본: `20`) |

---

## 내부 메서드

### `_calc_woe(series, target, n_bins)` → `pd.DataFrame`

수치형 변수를 `n_bins` 구간으로 분위수 binning하고 WOE/IV를 계산한다.

```
WOE = ln(Distribution_Good / Distribution_Bad)
IV  = (Distribution_Good - Distribution_Bad) × WOE
변수 IV = Σ(구간별 IV)
```

반환 컬럼: `bin, count, good_cnt, bad_cnt, WOE, IV`

---

### `_apply_woe(df, selected_cols, woe_tables, target_col)` → `pd.DataFrame`

선택된 변수에 WOE 값을 적용하여 WOE-transformed DataFrame을 반환한다.

---

### `_build_scorecard(lr, scaler, woe_tables, selected_cols, base_score, pdo)` → `pd.DataFrame`

로지스틱 회귀 계수와 PDO 공식을 사용하여 스코어카드 포인트 테이블을 생성한다.

```
Factor = pdo / ln(2)
Offset = base_score - Factor × ln(base_odds)
점수 = Offset + Factor × (β₀/n + βᵢ × WOEᵢ)
```

반환 컬럼: `variable, bin, WOE, score_point`

---

### `_score_data(df, scorecard, selected_cols, woe_tables)` → `pd.Series`

고객별 총 스코어를 산출한다 (각 변수 구간 점수의 합).

---

### `_evaluate_scorecard(scores, y)` → `dict`

| 지표 | 산출 방법 |
|------|----------|
| `auc` | `roc_auc_score(y, scores)` |
| `gini` | `2 × AUC - 1` |
| `ks` | KS 통계량 (누적 Good/Bad 분포 최대 차이) |

---

## execute() 진행률

| 단계 | progress |
|------|----------|
| 데이터 로드 완료 | 15% |
| WOE/IV 계산 완료 | 40% |
| 로지스틱 회귀 학습 완료 | 65% |
| 성능 평가 완료 | 85% |
| 저장 완료 | 95% |

---

## 반환값

```python
{
    "status": "COMPLETED",
    "result": {
        "model_id":       "scorecard_loan_v1",
        "selected_cols":  ["income", "debt_ratio", "overdue_cnt"],
        "iv_dict":        {"income": 0.342, "debt_ratio": 0.218, "overdue_cnt": 0.156},
        "metrics": {
            "train": {"auc": 0.812, "gini": 0.624, "ks": 0.431},
            "valid": {"auc": 0.798, "gini": 0.596, "ks": 0.412},
        },
        "base_score":     600,
        "pdo":            20,
        "scorecard_path": "models/scorecard_loan_v1_scorecard.json",
        "meta_path":      "models/scorecard_loan_v1_meta.json",
    },
    "message": "스코어카드 학습 완료  변수=3개  AUC=0.812",
    "job_id":  str,
    "elapsed_sec": float,
}
```

---

## 출력 파일

| 파일 | 경로 | 내용 |
|------|------|------|
| 스코어카드 JSON | `models/{model_id}_scorecard.json` | scorecard, woe_tables, iv_dict, metrics |
| 메타 정보 JSON | `models/{model_id}_meta.json` | selected_cols, metrics, base_score, pdo |

`_scorecard.json` 구조:
```json
{
  "scorecard": [
    {"variable": "income", "bin": "(50000, 100000]", "WOE": 0.42, "score_point": 18},
    {"variable": "income", "bin": "(100000, inf]",   "WOE": 0.91, "score_point": 32}
  ],
  "woe_tables":    {"income": [...], "debt_ratio": [...]},
  "iv_dict":       {"income": 0.342, "debt_ratio": 0.218},
  "selected_cols": ["income", "debt_ratio", "overdue_cnt"],
  "metrics":       {"train": {"auc": 0.812}, "valid": {"auc": 0.798}},
  "base_score":    600,
  "pdo":           20
}
```

---

## 사용 예시

```python
config = {
    "job_id":       "scorecard_001",
    "train_path":   "mart/loan_mart_train.parquet",
    "valid_path":   "mart/loan_mart_valid.parquet",
    "target_col":   "default",
    "model_id":     "scorecard_loan_v1",
    "feature_cols": ["income", "debt_ratio", "overdue_cnt", "credit_limit", "age"],
    "n_bins":       10,
    "iv_threshold": 0.02,
    "base_score":   600,
    "pdo":          20,
}

from executors.ml.scorecard_executor import ScorecardExecutor
result = ScorecardExecutor(config=config).run()
```
