# scorecard_executor.py — 신용 스코어카드 모델

## 개요

금융/리스크 도메인의 전통적 모델링 방식인 신용평가 스코어카드를 생성하는 executor.  
변수 구간화(Binning) → WOE 변환 → IV 산출 → 로지스틱 회귀 → 점수 스케일링 순으로 동작한다.

```
원시 변수
    ↓ Binning (qcut / cut)
    ↓ WOE 변환
    ↓ IV 필터 (IV < threshold 제거)
    ↓ StandardScaler + LogisticRegression
    ↓ PDO 스케일링 → 스코어카드 포인트 테이블
    ↓ 개인별 점수 = Σ(변수별 구간 포인트)
```

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `train_path` | ✅ | `str` | 학습 데이터 경로 (.parquet) |
| `target_col` | ✅ | `str` | 타깃 컬럼 (1=Bad, 0=Good) |
| `model_id` | ✅ | `str` | 모델 저장 식별자 |
| `feature_cols` | ✅ | `list` | 스코어카드 후보 변수 목록 |
| `valid_path` | ❌ | `str` | 검증 데이터 경로 |
| `n_bins` | ❌ | `int` | Binning 구간 수 (기본: `10`) |
| `min_bin_rate` | ❌ | `float` | 최소 bin 비율 (기본: `0.05`) |
| `iv_threshold` | ❌ | `float` | IV 필터 임계값 (기본: `0.02`) |
| `base_score` | ❌ | `int` | 기준 점수 (기본: `600`) |
| `pdo` | ❌ | `int` | Points to Double the Odds (기본: `20`) |

---

## 메서드 상세

### `_calc_woe(x, y, n_bins)` → `pd.DataFrame`

수치형 변수에 대한 WOE/IV 테이블을 생성한다.

```
① pd.qcut(q=n_bins) 시도 → 실패 시 pd.cut(bins=n_bins)
② 구간별: Good/Bad 분포 계산
③ WOE = ln(Distribution_Good / Distribution_Bad)
④ IV  = (Dist_Good - Dist_Bad) × WOE
```

| 출력 컬럼 | 설명 |
|----------|------|
| `bin` | 구간 레이블 |
| `count` | 구간 내 전체 건수 |
| `good` / `bad` | Good/Bad 건수 |
| `bad_rate` | 구간 내 불량률 |
| `WOE` | Weight of Evidence |
| `IV` | Information Value (구간) |

**IV 해석 기준:**

| IV 범위 | 예측력 |
|---------|--------|
| < 0.02 | 무의미 |
| 0.02 ~ 0.1 | 약함 |
| 0.1 ~ 0.3 | 중간 |
| > 0.3 | 강함 |

---

### `_apply_woe(df, selected_cols, woe_tables, target_col)` → `pd.DataFrame`

선택된 변수에 WOE 값을 매핑하여 로지스틱 회귀 입력 데이터를 생성한다.

```python
bins = pd.qcut(df[col], q=10, duplicates="drop")
result[col] = bins.astype(str).map(woe_map).fillna(0)
```

---

### `_build_scorecard(lr, scaler, woe_tables, selected_cols, base_score, pdo)` → `pd.DataFrame`

로지스틱 회귀 계수를 PDO 방식으로 점수 포인트로 변환한다.

```
factor = PDO / ln(2)
offset = base_score - factor × ln(1)

구간 포인트 = -(coef / scale) × WOE × factor
```

**스코어카드 테이블 예시:**

| variable | bin | WOE | points |
|----------|-----|-----|--------|
| income | (0, 3000] | 0.85 | 15 |
| income | (3000, 6000] | 0.12 | 3 |
| age | (18, 30] | -0.62 | -11 |

---

### `_score_data(df, scorecard, selected_cols, woe_tables)` → `pd.Series`

각 개인의 점수 = 해당 구간 포인트의 합산.

```python
개인점수 = Σ(변수별 해당 bin의 points)
```

---

### `_evaluate_scorecard(scores, y)` → `dict`

| 지표 | 산출 방식 |
|------|----------|
| `auc` | `roc_auc_score(y, scores)` |
| `gini` | `2 × AUC - 1` |
| `ks` | `max(cum_bad_rate - cum_good_rate)` |

---

## 실행 흐름

```
1. train_path / valid_path 로드                        [progress 15%]
2. 변수별 WOE/IV 계산 (_calc_woe)                     [progress 40%]
3. IV 필터링 (iv < iv_threshold 제거)
4. WOE 변환 (_apply_woe)
5. StandardScaler + LogisticRegression 학습            [progress 65%]
6. PDO 스케일링 → 스코어카드 포인트 테이블 (_build_scorecard)
7. 개인별 점수 산출 (_score_data)
8. KS / AUC / Gini 평가 (_evaluate_scorecard)          [progress 85%]
9. 결과 저장
   - models/{model_id}_scorecard.json
   - models/{model_id}_train_scores.parquet
```

---

## 출력 결과

**스코어카드 JSON:** `models/{model_id}_scorecard.json`
```json
{
  "scorecard":     [{"variable": "income", "bin": "(0,3000]", "WOE": 0.85, "points": 15}, ...],
  "woe_tables":    {"income": [...], "age": [...]},
  "iv_dict":       {"income": 0.342, "age": 0.187, "debt_ratio": 0.089},
  "selected_cols": ["income", "age"],
  "metrics": {
    "auc": 0.8123, "gini": 0.6246, "ks": 0.4812,
    "valid": {"auc": 0.7981, "gini": 0.5962, "ks": 0.4631}
  },
  "base_score": 600,
  "pdo": 20
}
```

**학습 점수 파일:** `models/{model_id}_train_scores.parquet`

---

## PredictExecutor와의 연계

스코어카드 예측은 `PredictExecutor`를 통하지 않고  
`_score_data()`를 직접 재호출하거나, 스코어카드 JSON을 운영계에 배포한다.

```
스코어카드 JSON 배포 → 운영계에서 룰 기반 점수 계산
(H2O / Python 모델 파일 불필요 → 가장 투명한 배포 방식)
```
