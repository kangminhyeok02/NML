# rl_executor.py (Reinforcement Learning Executor)

강화학습(Reinforcement Learning) 기반 의사결정 정책 학습 실행기.

금융/리스크 도메인에서 대출 한도 최적화, 컬렉션 전략 학습, 추천 시스템 등에 활용된다.

---

## 클래스

### `RLExecutor(BaseExecutor)`

강화학습 executor.

#### config 필수 키

| 키 | 타입 | 설명 |
|---|---|---|
| `algorithm` | `str` | `"q_learning"` \| `"dqn"` \| `"ppo"` \| `"contextual_bandit"` |
| `mode` | `str` | `"train"` \| `"evaluate"` \| `"deploy"` |
| `model_id` | `str` | 모델 저장 식별자 |

**train 모드 추가 키**

| 키 | 타입 | 설명 |
|---|---|---|
| `env_config` | `dict` | 환경 설정 (상태공간, 행동공간, 보상함수) |
| `train_steps` | `int` | 학습 스텝 수 |

**evaluate/deploy 모드 추가 키**

| 키 | 타입 | 설명 |
|---|---|---|
| `input_path` | `str` | 평가/배포 대상 데이터 경로 |
| `output_id` | `str` | 결과 저장 식별자 |

#### config 선택 키

| 키 | 기본값 | 설명 |
|---|---|---|
| `state_cols` | 전체 컬럼 | 상태 변수 목록 |
| `action_col` | - | 행동 컬럼명 |
| `reward_col` | - | 보상 컬럼명 |
| `gamma` | `0.99` | 할인율 |
| `lr` | `1e-3` | 학습률 |

---

### `execute() → dict`

`mode`에 따라 학습/평가/배포 메서드를 호출한다.

---

### `_run_train(cfg, algorithm) → dict`

RL 모델을 학습하고 pickle 파일로 저장한다.

**실행 순서**
1. `algorithm`에 따라 해당 학습 메서드 호출
2. 모델을 `models/{model_id}_rl.pkl`에 저장
3. 메타 정보를 `models/{model_id}_meta.json`에 저장

---

### `_run_evaluate(cfg, algorithm) → dict`

학습된 정책으로 오프라인 데이터를 평가한다.

- `_load_rl_model()`로 모델 로드
- `_predict_actions()`로 행동 추천
- `recommended_action`, `action_value` 컬럼 추가
- 결과를 `predict/{output_id}_rl_eval.parquet`에 저장

---

### `_run_deploy(cfg, algorithm) → dict`

단건 상태 입력에 대해 실시간 행동을 추천한다.

- `cfg["state"]` 딕셔너리를 numpy 배열로 변환
- `_predict_actions()`로 최적 행동 추론

**반환값**
```python
{
    "status": "COMPLETED",
    "result": {
        "recommended_action": int,
        "action_value":       float | None,
    },
    "message": str,
}
```

---

### `_train_q_learning(cfg) → (model, metrics)`

테이블 기반 Q-Learning으로 오프라인 배치 데이터를 학습한다.

1. `KBinsDiscretizer(n_bins=10)`로 연속 상태를 이산화
2. Bellman 업데이트: `Q[s,a] += lr * (r + gamma * max(Q[s2]) - Q[s,a])`
3. 모델 딕셔너리 반환: `{Q, discretizer, action_map, state_cols, n_bins}`

**metrics**: `avg_q`, `max_q`

---

### `_train_sb3(cfg, algorithm) → (model, metrics)`

stable-baselines3로 DQN 또는 PPO를 학습한다.

- `gymnasium` 환경 생성 (`env_config["gym_env_id"]`, 기본: `"CartPole-v1"`)
- `total_timesteps = cfg["train_steps"]`
- 학습 후 `_evaluate_sb3()`로 평균 보상 계산

**metrics**: `mean_reward`

---

### `_train_bandit(cfg) → (model, metrics)`

LinUCB 알고리즘으로 컨텍스추얼 밴딧을 학습한다.

- 각 행동 `a`에 대해 `A[a]`(identity 행렬), `b[a]`(영벡터) 초기화
- 배치 데이터를 순회하며 `A[a] += x*xᵀ`, `b[a] += r*x` 업데이트
- 모델: `{A, b, alpha, actions, state_cols}`

**metrics**: `n_actions`, `train_rows`

---

### `_load_rl_model(model_id)`

`models/{model_id}_rl.pkl` 파일을 로드한다.  
파일이 없으면 `ExecutorException` 발생.

---

### `_predict_actions(model, algorithm, X) → (actions, values)`

알고리즘별 행동 추론을 수행한다.

| `algorithm` | 추론 방식 |
|---|---|
| `"q_learning"` | `discretizer`로 상태 이산화 후 `argmax(Q[s])` |
| `"dqn"`, `"ppo"` | `model.predict(X, deterministic=True)` |
| `"contextual_bandit"` | LinUCB: `theta = A⁻¹b`, `p = theta·x + alpha*√(xᵀA⁻¹x)` |

반환: `(actions: list, values: list | None)`

---

## 모듈 레벨 함수

### `_evaluate_sb3(model, env, n_eval_episodes=10) → (mean_reward, std_reward)`

stable-baselines3 모델을 `n_eval_episodes`번 에피소드로 평가한다.  
각 에피소드는 `terminated` 또는 `truncated` 될 때까지 실행.

---

### `rlenv(..., json_obj) → dict`

강화학습 환경(Environment) 설정 및 초기 상태를 구성한다.

- 데이터 파일이 있으면 로드하여 `n_states` 산출
- 환경 메타를 `models/{model_id}_env_meta.json`에 저장
- `result_file_path_faf`, `done_file_path_faf`에 결과 기록

**json_obj 주요 키**: `model_id`, `env_config`, `state_cols`, `action_space`, `reward_col`, `data_path`

---

### `rlagent(..., json_obj) → dict`

강화학습 에이전트(Agent)를 학습하고 정책을 저장한다.

**동작 방식**
- `algorithm == "q_learning"`:
  - `train_data_path` 있으면 오프라인 배치 Q-Learning
  - 없으면 랜덤 탐색으로 Q 테이블 초기화
- `algorithm in ("dqn", "ppo")`:
  - stable-baselines3 학습 시도
  - import 실패 시 Q-table fallback
- 에이전트를 `models/{model_id}_agent.pkl`에 저장

**metrics**: `avg_q`, `max_q`, `min_q` (Q-Learning) 또는 `mean_reward`, `std_reward` (SB3)
