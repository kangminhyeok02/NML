# scorecard_executor.py

신용평가식 스코어카드(Credit Scorecard) 모델 생성 실행기.

변수 구간화(binning) → WOE 변환 → IV 산출 → 로지스틱 회귀 → 점수 스케일링 순서로 스코어카드를 생성한다.

---

## 클래스

### `ScorecardExecutor(BaseExecutor)`

스코어카드 모델 executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `train_path` | `str` | 학습 데이터 경로 (`.parquet`) |
| `target_col` | `str` | 타깃 컬럼 (1=Bad, 0=Good) |
| `model_id` | `str` | 모델 저장 식별자 |
| `feature_cols` | `list` | 스코어카드에 사용할 변수 목록 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|---|---|---|
| `valid_path` | - | 검증 데이터 경로 |
| `n_bins` | `10` | binning 구간 수 |
| `min_bin_rate` | `0.05` | 최소 bin 비율 |
| `iv_threshold` | `0.02` | IV 필터 기준. 이 값 미만 변수는 제외 |
| `base_score` | `600` | 기준 점수 (PDO 방식) |
| `pdo` | `20` | PDO(Points to Double the Odds) |

---

### `execute() → dict`

스코어카드 생성 전체 파이프라인을 실행한다.

**실행 순서**
1. 학습/검증 데이터 로드
2. 각 변수에 대해 Binning & WOE/IV 산출 (`_calc_woe`)
3. `iv_threshold` 미만 변수 제거 (IV 필터링)
4. 선택 변수에 WOE 값 적용 (`_apply_woe`)
5. StandardScaler로 정규화 후 LogisticRegression 학습
6. PDO 방식으로 점수 스케일링 (`_build_scorecard`)
7. 학습/검증 데이터 성능 평가 (`_evaluate_scorecard`)
8. 스코어카드 JSON과 학습 점수 parquet 저장

**저장 파일**
- `models/{model_id}_scorecard.json`: 스코어카드 전체 데이터
- `models/{model_id}_train_scores.parquet`: 학습 데이터 점수

**반환값**
```python
{
    "status": "COMPLETED",
    "result": {
        "model_id":      str,
        "selected_vars": list[str],
        "iv_dict":       {col: iv},
        "metrics":       dict,
    },
    "message": str,   # KS 포함
}
```

---

### `_calc_woe(x, y, n_bins) → pd.DataFrame`

수치형 변수에 대한 WOE(Weight of Evidence) / IV(Information Value) 테이블을 생성한다.

- `pd.qcut()`으로 분위수 기반 binning. 실패 시 `pd.cut()` 사용
- 각 bin별 Good/Bad 비율로 WOE 계산: `WOE = ln(dist_good / dist_bad)`
- IV 계산: `IV = (dist_good - dist_bad) * WOE`

**반환 컬럼**: `bin`, `count`, `good`, `bad`, `bad_rate`, `WOE`, `IV`

---

### `_apply_woe(df, selected_cols, woe_tables, target_col) → pd.DataFrame`

선택된 변수들에 WOE 값을 매핑하여 변환된 데이터프레임을 반환한다.

- 각 값을 해당 bin에 매핑하여 WOE 값으로 치환
- 타깃 컬럼은 제외

---

### `_build_scorecard(lr, scaler, woe_tables, selected_cols, base_score, pdo) → pd.DataFrame`

PDO(Points to Double the Odds) 방식으로 스코어카드 포인트 테이블을 생성한다.

- 각 변수/구간별 포인트 = `-(WOE * coef * factor) + offset/n_vars`
- `factor = pdo / ln(2)`, `offset = base_score - factor * ln(odds)`

**반환 컬럼**: `variable`, `bin`, `WOE`, `score`

---

### `_score_data(df, scorecard, selected_cols, woe_tables) → pd.Series`

데이터프레임에 스코어카드를 적용하여 각 관측치의 최종 점수를 반환한다.

- 각 변수별 bin에 해당하는 score 포인트를 합산

---

### `_evaluate_scorecard(scores, y) → dict`

스코어카드 성능 지표를 산출한다.

| 지표 | 설명 |
|---|---|
| `ks` | KS 통계량 (최대 누적 분리도) |
| `auc` | ROC AUC |
| `gini` | Gini 계수 = `2 * AUC - 1` |
