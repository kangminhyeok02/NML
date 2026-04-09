# rl_executor.py — 강화학습 (Reinforcement Learning)

**파일:** `executors/ml/rl_executor.py`  
**클래스:** `RLExecutor(BaseExecutor)`

## 개요

강화학습(RL) 기반 의사결정 정책을 학습하고 평가·배포하는 executor.  
금융/리스크 도메인에서 주로 다음 목적으로 활용된다:
- 대출 한도/금리 동적 최적화
- 연체 고객 컬렉션 전략 최적 시퀀스 학습
- 상품/채널 추천 최적화
- 시뮬레이션 환경에서의 정책 탐색

```
mode = "train"    → 강화학습 정책 학습 → models/{model_id}_rl.pkl 저장
mode = "evaluate" → 오프라인 데이터 평가 → 추천 행동·보상 산출
mode = "deploy"   → 단건 실시간 추론 → recommended_action 반환
```

---

## 지원 알고리즘 (`algorithm`)

| `algorithm` | 방식 | 적합 상황 | 의존 라이브러리 |
|---|---|---|---|
| `q_learning` | 테이블 기반 Q-Learning | 소규모 이산 상태공간, 배치 오프라인 학습 | numpy |
| `dqn` | Deep Q-Network | 연속 상태공간 | `stable-baselines3`, `gymnasium` |
| `ppo` | Proximal Policy Optimization | 연속 상태, 안정적 정책 탐색 | `stable-baselines3`, `gymnasium` |
| `contextual_bandit` | LinUCB 컨텍스추얼 밴딧 | 단순 의사결정, 실시간 추론에 유리 | numpy |

---

## config 파라미터

| 키 | 필수 | 타입 | 설명 |
|----|------|------|------|
| `algorithm` | ✅ | `str` | 알고리즘 이름 |
| `mode` | ✅ | `str` | `"train"` \| `"evaluate"` \| `"deploy"` |
| `model_id` | ✅ | `str` | 모델 저장 식별자 |
| `env_config` | train 필수 | `dict` | 환경 설정 (상태공간, 행동공간, 보상함수) |
| `train_steps` | train 필수 | `int` | 학습 스텝 수 |
| `input_path` | eval/deploy | `str` | 평가·배포 대상 데이터 경로 |
| `output_id` | eval/deploy | `str` | 결과 저장 식별자 |
| `reward_col` | ❌ | `str` | 보상 컬럼명 (오프라인 학습 시) |
| `state_cols` | ❌ | `list` | 상태 변수 목록 |
| `action_col` | ❌ | `str` | 행동 컬럼명 |
| `gamma` | ❌ | `float` | 할인율 (기본: `0.99`) |
| `lr` | ❌ | `float` | 학습률 (기본: `1e-3`) |

---

## 모드별 실행 흐름

### `mode="train"` — `_run_train(cfg, algorithm)`

```
1. 환경 설정 파싱 (env_config)
2. 알고리즘에 따라 학습
   - q_learning      → _train_q_learning(cfg)
   - dqn / ppo       → _train_sb3(cfg, algorithm)    # stable-baselines3
   - contextual_bandit → _train_bandit(cfg)
3. models/{model_id}_rl.pkl 저장
4. meta JSON 저장
```

### `mode="evaluate"` — `_run_evaluate(cfg, algorithm)`

```
1. 모델 로드 (pkl)
2. input_path 데이터 로드
3. 각 행에 대해 추천 행동 예측
4. 보상 추정 및 정책 성능 평가
5. predict/{output_id}_rl_eval.parquet 저장
```

### `mode="deploy"` — `_run_deploy(cfg, algorithm)`

```
1. 모델 로드
2. 단건 또는 소배치 실시간 추론
3. recommended_action 반환
```

---

## execute() 진행률

| 단계 | progress |
|------|----------|
| 초기화 완료 | 15% |
| 학습/평가 완료 | 85% |
| 저장 완료 | 95% |

---

## 반환값 (train 모드)

```python
{
    "status": "COMPLETED",
    "result": {
        "model_id":    "rl_collection_v1",
        "algorithm":   "contextual_bandit",
        "mode":        "train",
        "model_path":  "models/rl_collection_v1_rl.pkl",
        "metrics": {
            "total_reward":    12453.2,
            "avg_reward":      0.87,
            "episodes":        1000,
        },
    },
    "message": "RL 학습 완료  contextual_bandit  steps=10000",
    "job_id":  str,
    "elapsed_sec": float,
}
```

## 반환값 (evaluate 모드)

```python
{
    "status": "COMPLETED",
    "result": {
        "output_id":   "collection_eval_202312",
        "total_rows":  5000,
        "output_path": "predict/collection_eval_202312_rl_eval.parquet",
        "eval_metrics": {
            "mean_reward":      0.83,
            "action_dist":      {"call": 0.45, "sms": 0.31, "email": 0.24},
        },
    },
    "message": "RL 평가 완료  5,000건",
    "job_id":  str,
    "elapsed_sec": float,
}
```

---

## 출력 파일

| 파일 | 경로 |
|------|------|
| RL 정책 모델 (pickle) | `models/{model_id}_rl.pkl` |
| 모델 메타 JSON | `models/{model_id}_meta.json` |
| 평가 결과 parquet | `predict/{output_id}_rl_eval.parquet` |

---

## 사용 예시

```python
# 컬렉션 전략 강화학습
config = {
    "job_id":      "rl_001",
    "algorithm":   "contextual_bandit",
    "mode":        "train",
    "model_id":    "rl_collection_v1",
    "train_steps": 10000,
    "state_cols":  ["overdue_days", "balance", "contact_cnt"],
    "action_col":  "action",       # call / sms / email
    "reward_col":  "repayment_yn", # 상환 여부
    "gamma":       0.95,
    "env_config": {
        "n_actions": 3,
        "n_states":  3,
    },
}

from executors.ml.rl_executor import RLExecutor
result = RLExecutor(config=config).run()
```
