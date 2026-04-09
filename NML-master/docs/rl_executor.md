# rl_executor.py — 강화학습 (Reinforcement Learning)

## 개요

강화학습 기반 의사결정 정책을 학습하거나, 학습된 정책으로 추론/배포하는 executor.  
금융/리스크 도메인에서 대출 한도 최적화, 컬렉션 전략 시퀀스 학습, 상품 추천 등에 활용된다.

---

## 지원 알고리즘 (`algorithm`)

| `algorithm` | 방식 | 상태공간 | 특징 |
|---|---|---|---|
| `q_learning` | 테이블 Q-Learning | 이산 (소규모) | 오프라인 배치 데이터로 학습 |
| `dqn` | Deep Q-Network | 연속 | stable-baselines3, gym 환경 |
| `ppo` | Proximal Policy Optimization | 연속 | stable-baselines3, gym 환경 |
| `contextual_bandit` | LinUCB 밴딧 | 연속 | 단순 1-step 의사결정, 빠른 수렴 |

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `algorithm` | ✅ | `str` | 알고리즘 선택 |
| `mode` | ✅ | `str` | `"train"` \| `"evaluate"` \| `"deploy"` |
| `model_id` | ✅ | `str` | 모델 저장 식별자 |
| `env_config` | ❌ | `dict` | 환경 설정 (gym_env_id 등) |
| `train_steps` | ❌ | `int` | DQN/PPO 학습 스텝 수 |
| `train_data_path` | ❌ | `str` | 오프라인 학습 데이터 경로 (Q-Learning, Bandit) |
| `input_path` | ❌ | `str` | 평가/배포 대상 데이터 경로 |
| `output_id` | ❌ | `str` | 결과 저장 식별자 |
| `state_cols` | ❌ | `list` | 상태 변수 컬럼 목록 |
| `action_col` | ❌ | `str` | 행동 컬럼명 |
| `reward_col` | ❌ | `str` | 보상 컬럼명 |
| `gamma` | ❌ | `float` | 할인율 (기본: `0.99`) |
| `lr` | ❌ | `float` | 학습률 (기본: `1e-3`) |
| `alpha` | ❌ | `float` | LinUCB 탐색 계수 (기본: `1.0`) |
| `state` | ❌ | `dict` | deploy 모드 단건 상태값 |

---

## mode별 동작

### `"train"` — 정책 학습

**Q-Learning (`_train_q_learning`):**
```
① KBinsDiscretizer(n_bins=10)로 연속 상태 이산화
② Q-테이블 초기화 (n_states × n_actions)
③ 오프라인 데이터 순회:
   Q[s,a] += lr × (r + γ × max(Q[s']) - Q[s,a])
```
- 출력: `{"Q": ndarray, "discretizer": ..., "action_map": {...}}`

**DQN/PPO (`_train_sb3`):**
```
stable-baselines3 DQN 또는 PPO
→ gym 환경 (gym_env_id)에서 total_timesteps 학습
→ 10 에피소드 평균 보상으로 성능 평가
```

**LinUCB Bandit (`_train_bandit`):**
```
행동별 A (n×n), b (n) 행렬 초기화
오프라인 데이터로 A, b 갱신:
    A[a] += x @ x.T
    b[a] += r × x
```
- 추론 시: θ = A⁻¹b, UCB = θ·x + α√(x·A⁻¹x)

---

### `"evaluate"` — 오프라인 평가

```
input_path 데이터 로드
→ state_cols 추출
→ _predict_actions() 로 최적 행동 추천
→ predict/{output_id}_rl_eval.parquet 저장
  (컬럼: 원본 + recommended_action [+ action_value])
```

---

### `"deploy"` — 단건 실시간 추론

```python
config = {
    "mode":      "deploy",
    "algorithm": "contextual_bandit",
    "model_id":  "bandit_v1",
    "state": {
        "age": 35, "income": 5000, "debt_ratio": 0.3
    }
}
# → {"recommended_action": 2, "action_value": 0.8234}
```

---

## `_predict_actions(model, algorithm, X)` — 공통 추론

| 알고리즘 | 추론 방식 |
|---------|---------|
| `q_learning` | 이산화 상태 → Q[s].argmax() |
| `dqn` / `ppo` | `model.predict(X, deterministic=True)` |
| `contextual_bandit` | 행동별 UCB 계산 → 최대값 행동 선택 |

---

## 출력 결과

**모델 파일:** `models/{model_id}_rl.pkl`

**메타 파일:** `models/{model_id}_meta.json`
```json
{
  "model_id":  "bandit_credit_v1",
  "algorithm": "contextual_bandit",
  "mode":      "train",
  "metrics":   {"n_actions": 3, "train_rows": 50000},
  "model_path": "models/bandit_credit_v1_rl.pkl",
  "env_config": {}
}
```

---

## 도메인 활용 예시

| 문제 | 알고리즘 | 상태(state) | 행동(action) | 보상(reward) |
|------|---------|------------|------------|------------|
| 대출 한도 결정 | Bandit | 신용점수, 소득 | 한도 구간 (3단계) | 상환 완료 여부 |
| 컬렉션 채널 선택 | Q-Learning | 연체일, 금액 | 전화/문자/방문 | 회수 금액 |
| 금리 최적화 | PPO | 시장금리, 고객 세그먼트 | 금리 설정 | 계약 성사 여부 |

---

## 주의 사항

- `q_learning`, `contextual_bandit`은 **오프라인 배치 데이터**로 학습 (simulator 불필요)
- `dqn`, `ppo`는 **gym 환경** 필수 — 커스텀 환경은 `gym.Env` 상속 구현 후 등록
- 대규모 상태공간에서 `q_learning` 사용 시 차원의 저주 주의 → `dqn` 권장
