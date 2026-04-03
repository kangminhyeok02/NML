"""
rl_executor.py  (Reinforcement Learning Executor)
--------------------------------------------------
강화학습(Reinforcement Learning) 기반 의사결정 정책 학습 실행기.

금융/리스크 도메인에서 RL은 주로 다음 목적으로 활용된다:
  - 대출 한도/금리 동적 최적화
  - 컬렉션 전략 최적 시퀀스 학습
  - 추천 시스템 (상품/채널 최적 배분)
  - 시뮬레이션 환경에서의 정책 탐색

지원 알고리즘:
  - q_learning  : 테이블 기반 Q-Learning (소규모 상태공간)
  - dqn         : Deep Q-Network (연속 상태공간, stable-baselines3)
  - ppo         : Proximal Policy Optimization (stable-baselines3)
  - contextual_bandit : 컨텍스추얼 밴딧 (단순 의사결정)
"""

import json
import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd

from executors.ml.base_executor import BaseExecutor, ExecutorException, ExecutorStatus

logger = logging.getLogger(__name__)


class RLExecutor(BaseExecutor):
    """
    강화학습 executor.

    config 필수 키
    --------------
    algorithm    : str   "q_learning" | "dqn" | "ppo" | "contextual_bandit"
    mode         : str   "train" | "evaluate" | "deploy"
    model_id     : str   모델 저장 식별자

    [train 모드]
    env_config   : dict  환경 설정 (상태공간, 행동공간, 보상함수)
    train_steps  : int   학습 스텝 수

    [evaluate/deploy 모드]
    input_path   : str   평가/배포 대상 데이터 경로
    output_id    : str   결과 저장 식별자

    config 선택 키
    --------------
    reward_col   : str   보상 컬럼명 (오프라인 학습 시)
    state_cols   : list  상태 변수 목록
    action_col   : str   행동 컬럼명
    gamma        : float 할인율 (기본 0.99)
    lr           : float 학습률 (기본 1e-3)
    """

    def execute(self) -> dict:
        cfg       = self.config
        algorithm = cfg["algorithm"]
        mode      = cfg["mode"]

        if mode == "train":
            return self._run_train(cfg, algorithm)
        elif mode == "evaluate":
            return self._run_evaluate(cfg, algorithm)
        elif mode == "deploy":
            return self._run_deploy(cfg, algorithm)
        else:
            raise ExecutorException(f"지원하지 않는 mode: {mode}")

    # ------------------------------------------------------------------

    def _run_train(self, cfg: dict, algorithm: str) -> dict:
        self._update_job_status(ExecutorStatus.RUNNING, progress=15)
        logger.info("RL 학습 시작  algorithm=%s  steps=%d", algorithm, cfg.get("train_steps", 10000))

        if algorithm == "q_learning":
            model, metrics = self._train_q_learning(cfg)
        elif algorithm in ("dqn", "ppo"):
            model, metrics = self._train_sb3(cfg, algorithm)
        elif algorithm == "contextual_bandit":
            model, metrics = self._train_bandit(cfg)
        else:
            raise ExecutorException(f"지원하지 않는 algorithm: {algorithm}")

        self._update_job_status(ExecutorStatus.RUNNING, progress=85)

        model_path = f"models/{cfg['model_id']}_rl.pkl"
        full_path  = self.file_root / model_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "wb") as f:
            pickle.dump(model, f)

        meta = {
            "model_id":  cfg["model_id"],
            "algorithm": algorithm,
            "mode":      "train",
            "metrics":   metrics,
            "model_path": model_path,
            "env_config": cfg.get("env_config", {}),
        }
        self._save_json(meta, f"models/{cfg['model_id']}_meta.json")

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result":  meta,
            "message": f"RL 학습 완료  algorithm={algorithm}  {metrics}",
        }

    def _run_evaluate(self, cfg: dict, algorithm: str) -> dict:
        """학습된 정책으로 오프라인 데이터 평가."""
        df = self._load_dataframe(cfg["input_path"])
        model = self._load_rl_model(cfg["model_id"])

        state_cols = cfg.get("state_cols", list(df.columns))
        X = df[[c for c in state_cols if c in df.columns]].fillna(0)

        actions, values = self._predict_actions(model, algorithm, X.values)
        df["recommended_action"] = actions
        if values is not None:
            df["action_value"] = values

        output_path = f"predict/{cfg.get('output_id', cfg['model_id'])}_rl_eval.parquet"
        self._save_dataframe(df, output_path)

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result": {
                "output_path":    output_path,
                "total_rows":     len(df),
                "action_dist":    pd.Series(actions).value_counts().to_dict(),
            },
            "message": f"RL 평가 완료  {len(df):,}건",
        }

    def _run_deploy(self, cfg: dict, algorithm: str) -> dict:
        """실시간 단건 의사결정 (단일 상태 → 행동 추천)."""
        state = cfg.get("state")
        if state is None:
            raise ExecutorException("deploy 모드에는 state 딕셔너리가 필요합니다.")

        model = self._load_rl_model(cfg["model_id"])
        state_array = np.array(list(state.values()), dtype=np.float32).reshape(1, -1)
        actions, values = self._predict_actions(model, algorithm, state_array)

        return {
            "status":  ExecutorStatus.COMPLETED,
            "result": {
                "recommended_action": int(actions[0]),
                "action_value":       float(values[0]) if values is not None else None,
            },
            "message": f"RL 단건 추론 완료  action={actions[0]}",
        }

    # ------------------------------------------------------------------

    def _train_q_learning(self, cfg: dict):
        """테이블 기반 Q-Learning (오프라인 배치 데이터 사용)."""
        if "train_data_path" not in cfg:
            raise ExecutorException("q_learning 학습에는 train_data_path가 필요합니다.")

        df = self._load_dataframe(cfg["train_data_path"])
        state_cols  = cfg["state_cols"]
        action_col  = cfg["action_col"]
        reward_col  = cfg["reward_col"]
        gamma       = cfg.get("gamma", 0.99)
        lr          = cfg.get("lr", 0.1)
        n_actions   = df[action_col].nunique()

        # 상태 이산화 (간단 버전: 분위수 기반 binning)
        from sklearn.preprocessing import KBinsDiscretizer
        discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
        states = discretizer.fit_transform(df[state_cols].fillna(0)).astype(int)
        n_states = int(np.prod([10] * len(state_cols)))

        Q = np.zeros((n_states, n_actions))
        action_map = {a: i for i, a in enumerate(sorted(df[action_col].unique()))}

        for i in range(len(df) - 1):
            s  = int(np.ravel_multi_index(states[i], [10] * len(state_cols), mode="clip"))
            s2 = int(np.ravel_multi_index(states[i+1], [10] * len(state_cols), mode="clip"))
            a  = action_map.get(df.iloc[i][action_col], 0)
            r  = float(df.iloc[i][reward_col])
            Q[s, a] += lr * (r + gamma * Q[s2].max() - Q[s, a])

        model = {"Q": Q, "discretizer": discretizer, "action_map": action_map,
                 "state_cols": state_cols, "n_bins": 10}
        metrics = {"avg_q": round(float(Q.mean()), 4), "max_q": round(float(Q.max()), 4)}
        return model, metrics

    def _train_sb3(self, cfg: dict, algorithm: str):
        """stable-baselines3 DQN/PPO 학습."""
        try:
            import gymnasium as gym
            from stable_baselines3 import DQN, PPO
        except ImportError:
            raise ExecutorException("stable-baselines3 및 gymnasium이 필요합니다: pip install stable-baselines3 gymnasium")

        env_id    = cfg.get("env_config", {}).get("gym_env_id", "CartPole-v1")
        env       = gym.make(env_id)
        algo_cls  = DQN if algorithm == "dqn" else PPO
        sb3_model = algo_cls("MlpPolicy", env, verbose=0,
                             gamma=cfg.get("gamma", 0.99),
                             learning_rate=cfg.get("lr", 1e-3))
        sb3_model.learn(total_timesteps=cfg.get("train_steps", 10000))

        mean_reward, _ = _evaluate_sb3(sb3_model, env, n_eval_episodes=10)
        metrics = {"mean_reward": round(float(mean_reward), 4)}
        return sb3_model, metrics

    def _train_bandit(self, cfg: dict):
        """컨텍스추얼 밴딧 (LinUCB)."""
        if "train_data_path" not in cfg:
            raise ExecutorException("bandit 학습에는 train_data_path가 필요합니다.")

        df = self._load_dataframe(cfg["train_data_path"])
        state_cols = cfg["state_cols"]
        action_col = cfg["action_col"]
        reward_col = cfg["reward_col"]

        n_features = len(state_cols)
        actions    = df[action_col].unique()
        alpha      = cfg.get("alpha", 1.0)  # LinUCB 탐색 계수

        A = {a: np.eye(n_features) for a in actions}
        b = {a: np.zeros(n_features) for a in actions}

        for _, row in df.iterrows():
            x = row[state_cols].fillna(0).values.astype(float)
            a = row[action_col]
            r = float(row[reward_col])
            A[a] += np.outer(x, x)
            b[a] += r * x

        model = {"A": A, "b": b, "alpha": alpha, "actions": list(actions), "state_cols": state_cols}
        metrics = {"n_actions": len(actions), "train_rows": len(df)}
        return model, metrics

    def _load_rl_model(self, model_id: str):
        model_path = self.file_root / f"models/{model_id}_rl.pkl"
        if not model_path.exists():
            raise ExecutorException(f"RL 모델 없음: {model_path}")
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def _predict_actions(self, model, algorithm: str, X: np.ndarray):
        if algorithm == "q_learning" and isinstance(model, dict):
            disc   = model["discretizer"]
            Q      = model["Q"]
            states = disc.transform(X).astype(int)
            n_bins = model["n_bins"]
            n_cols = X.shape[1]
            actions = []
            for s in states:
                idx = int(np.ravel_multi_index(s, [n_bins] * n_cols, mode="clip"))
                actions.append(int(np.argmax(Q[idx])))
            return actions, None

        elif algorithm in ("dqn", "ppo"):
            preds = model.predict(X, deterministic=True)
            return list(preds[0]), None

        elif algorithm == "contextual_bandit" and isinstance(model, dict):
            A, b, alpha = model["A"], model["b"], model["alpha"]
            actions, values = [], []
            for x in X:
                best_a, best_val = None, -np.inf
                for a in model["actions"]:
                    theta = np.linalg.solve(A[a], b[a])
                    p     = theta.dot(x) + alpha * np.sqrt(x.dot(np.linalg.solve(A[a], x)))
                    if p > best_val:
                        best_val, best_a = p, a
                actions.append(best_a)
                values.append(round(float(best_val), 4))
            return actions, values

        raise ExecutorException(f"predict_actions 미지원 algorithm: {algorithm}")


def _evaluate_sb3(model, env, n_eval_episodes=10):
    """stable-baselines3 evaluate_policy 간단 구현."""
    rewards = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        rewards.append(ep_reward)
    return np.mean(rewards), np.std(rewards)
