from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal


class _GaussianPolicy(nn.Module):
    """Continuous action policy for REINFORCE.

    The output action is squashed by tanh, so every action is in [-1, 1].
    That matches StockTradingEnv's action interpretation before hmax scaling.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, log_std_init: float) -> None:
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_net(obs)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std, min=-5.0, max=2.0)
        return mean, log_std.expand_as(mean)


class ReinforceAgent:
    """A minimal REINFORCE / Monte-Carlo policy-gradient agent.

    This class intentionally does not use PPO clipping, value networks, replay buffers,
    or actor-critic updates. It is meant to test whether a simpler policy-gradient
    training signal produces a clearer reward curve in this project.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        ent_coef: float = 0.001,
        max_grad_norm: float = 0.5,
        normalize_returns: bool = True,
        log_std_init: float = -0.5,
        seed: int = 0,
        device: str = "auto",
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.ent_coef = float(ent_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.normalize_returns = bool(normalize_returns)
        self.log_std_init = float(log_std_init)
        self.seed = int(seed)
        self.device = torch.device(device)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.policy = _GaussianPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            log_std_init=self.log_std_init,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.individual_stock_epoch_rewards: list[dict[str, float | int]] = []

    def _obs_tensor(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, tuple):
            obs = obs[0]
        obs_array = np.asarray(obs, dtype=np.float32)
        if obs_array.ndim == 1:
            obs_array = obs_array.reshape(1, -1)
        return torch.as_tensor(obs_array, dtype=torch.float32, device=self.device)

    def _distribution(self, obs: Any) -> Normal:
        obs_tensor = self._obs_tensor(obs)
        mean, log_std = self.policy(obs_tensor)
        return Normal(mean, log_std.exp())

    def _sample_action(self, obs: Any) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        dist = self._distribution(obs)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        return action.detach().cpu().numpy(), log_prob.squeeze(0), entropy.squeeze(0)

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        with torch.no_grad():
            dist = self._distribution(obs)
            if deterministic:
                raw_action = dist.mean
            else:
                raw_action = dist.sample()
            action = torch.tanh(raw_action).cpu().numpy()
        return action, None

    def _discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        returns: list[float] = []
        running_return = 0.0
        for reward in reversed(rewards):
            running_return = float(reward) + self.gamma * running_return
            returns.append(running_return)
        returns.reverse()

        out = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        if self.normalize_returns and out.numel() > 1:
            std = out.std(unbiased=False)
            if torch.isfinite(std) and std > 1e-8:
                out = (out - out.mean()) / (std + 1e-8)
        return out

    @staticmethod
    def _unpack_step(step_result: tuple) -> tuple[Any, np.ndarray, np.ndarray, Any]:
        if len(step_result) == 4:
            obs, rewards, dones, infos = step_result
            return obs, rewards, dones, infos
        if len(step_result) == 5:
            obs, rewards, terminated, truncated, infos = step_result
            dones = np.logical_or(terminated, truncated)
            return obs, rewards, dones, infos
        raise ValueError(f"Unsupported env.step() result length: {len(step_result)}")

    @staticmethod
    def _reset_env(env: Any) -> Any:
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            return reset_result[0]
        return reset_result

    @staticmethod
    def _scalar(value: Any) -> float:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        return float(arr[0]) if arr.size else 0.0

    @staticmethod
    def _get_env_attr(env: Any, name: str, default: Any = None) -> Any:
        """Read one attribute from a normal env or Stable-Baselines3 VecEnv."""
        try:
            if hasattr(env, "get_attr"):
                values = env.get_attr(name)
                if values:
                    return values[0]
            return getattr(env, name, default)
        except (AttributeError, IndexError, TypeError):
            return default

    @staticmethod
    def _call_env_method(env: Any, name: str) -> Any:
        """Call one method on a normal env or Stable-Baselines3 VecEnv."""
        try:
            if hasattr(env, "env_method"):
                values = env.env_method(name)
                if values:
                    return values[0]
            method = getattr(env, name, None)
            if callable(method):
                return method()
        except (AttributeError, IndexError, TypeError):
            return None
        return None

    def _record_individual_stock_epoch_rewards(self, env: Any, epoch: int) -> None:
        """Store each stock's mean daily reward contribution for one full epoch."""
        reward_components = self._call_env_method(
            env, "save_last_episode_reward_components_memory"
        )
        if not isinstance(reward_components, pd.DataFrame) or reward_components.empty:
            reward_components = self._call_env_method(env, "save_reward_components_memory")
        if not isinstance(reward_components, pd.DataFrame) or reward_components.empty:
            return

        stock_columns = [column for column in reward_components.columns if column != "date"]
        train_trading_days = len(reward_components)
        if not stock_columns or train_trading_days <= 0:
            return

        row: dict[str, float | int] = {
            "epoch": int(epoch),
            "train_trading_days": int(train_trading_days),
        }
        for ticker in stock_columns:
            daily_contributions = pd.to_numeric(
                reward_components[ticker], errors="coerce"
            ).fillna(0.0)
            row[str(ticker)] = float(daily_contributions.sum() / train_trading_days)

        self.individual_stock_epoch_rewards.append(row)

    def _policy_std_mean(self) -> float:
        """Return the mean standard deviation of the current Gaussian policy."""
        with torch.no_grad():
            log_std = torch.clamp(self.policy.log_std, min=-5.0, max=2.0)
            return float(log_std.exp().mean().detach().cpu().item())

    def _reward_v4_diagnostics(self, env: Any) -> dict[str, float]:
        """Read reward v4 episode diagnostics stored by reward_functions.py."""
        diagnostics = self._get_env_attr(env, "_reward_v4_diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}

        keys = [
            "zero_delta_ratio",
            "mean_abs_delta_a",
            "mean_abs_effective_delta_a",
            "mean_abs_future_return",
            "positive_reward_ratio",
            "positive_step_reward_ratio",
            "valid_action_count",
            "diagnostic_step_count",
        ]
        return {
            key: float(diagnostics.get(key, np.nan))
            for key in keys
        }

    def _evaluate_deterministic(self, env: Any) -> tuple[float, float, int]:
        """Evaluate the current mean policy once without exploration noise."""
        was_training = self.policy.training
        self.policy.eval()

        obs = self._reset_env(env)
        total_reward = 0.0
        steps = 0

        # StockTradingEnv performs terminal handling on the step after the final
        # tradable day. Stopping at terminal_day preserves its asset memory.
        terminal_day = self._get_env_attr(env, "terminal_day", None)
        max_steps = int(terminal_day) if terminal_day is not None else 100000
        max_steps = max(1, max_steps)

        try:
            for _ in range(max_steps):
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, dones, _infos = self._unpack_step(env.step(action))
                total_reward += self._scalar(reward)
                steps += 1
                done = bool(np.asarray(dones).reshape(-1)[0])
                if done:
                    break

            asset_memory = self._get_env_attr(env, "asset_memory", None)
            if asset_memory is not None and len(asset_memory) > 0:
                final_asset = float(np.asarray(asset_memory, dtype=np.float64).reshape(-1)[-1])
            else:
                final_asset = float("nan")
        finally:
            if was_training:
                self.policy.train()

        return float(total_reward), final_asset, steps

    def learn(
        self,
        env: Any,
        total_timesteps: int,
        progress_path: str | Path | None = None,
        eval_env: Any | None = None,
        eval_interval: int = 5,
    ) -> "ReinforceAgent":
        progress_rows: list[dict[str, float]] = []
        self.individual_stock_epoch_rewards = []
        total_steps = 0
        episode = 0
        eval_interval = max(1, int(eval_interval))
        progress_path = Path(progress_path) if progress_path is not None else None
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)

        while total_steps < int(total_timesteps):
            obs = self._reset_env(env)
            done = False
            log_probs: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            rewards: list[float] = []
            sampled_actions: list[np.ndarray] = []

            # Once an episode starts, finish it before stopping training.
            # total_steps may therefore exceed total_timesteps by at most
            # one episode, which avoids recording and training on a partial epoch.
            while not done:
                action, log_prob, entropy = self._sample_action(obs)
                obs, reward, dones, _infos = self._unpack_step(env.step(action))
                reward_value = self._scalar(reward)
                done = bool(np.asarray(dones).reshape(-1)[0])

                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards.append(reward_value)
                sampled_actions.append(np.asarray(action, dtype=np.float64).reshape(-1))
                total_steps += 1

            if not rewards:
                break

            returns = self._discounted_returns(rewards)
            log_prob_tensor = torch.stack(log_probs)
            entropy_tensor = torch.stack(entropies)
            # Average over the episode so the loss scale does not grow with
            # the number of trading days. Gradient clipping remains as a safeguard.
            policy_loss = -(log_prob_tensor * returns).mean()
            entropy_loss = -self.ent_coef * entropy_tensor.mean()
            loss = policy_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            gradient_norm_tensor = nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm,
            )
            gradient_norm = float(gradient_norm_tensor.detach().cpu().item())
            gradient_clipped = float(
                np.isfinite(gradient_norm) and gradient_norm > self.max_grad_norm
            )
            self.optimizer.step()

            episode += 1
            self._record_individual_stock_epoch_rewards(env, episode)
            episode_reward = float(np.sum(rewards))
            episode_average_reward = float(np.mean(rewards))

            action_matrix = np.vstack(sampled_actions)
            previous_actions = np.vstack(
                [np.zeros((1, action_matrix.shape[1]), dtype=np.float64), action_matrix[:-1]]
            )
            action_deltas = action_matrix - previous_actions
            mean_abs_action = float(np.mean(np.abs(action_matrix)))
            mean_abs_action_delta = float(np.mean(np.abs(action_deltas)))
            policy_std_mean = self._policy_std_mean()
            reward_v4_diagnostics = self._reward_v4_diagnostics(env)

            deterministic_reward = float("nan")
            deterministic_final_asset = float("nan")
            deterministic_steps = float("nan")
            should_evaluate = (
                eval_env is not None
                and (
                    episode == 1
                    or episode % eval_interval == 0
                    or total_steps >= int(total_timesteps)
                )
            )
            if should_evaluate:
                deterministic_reward, deterministic_final_asset, eval_steps = (
                    self._evaluate_deterministic(eval_env)
                )
                deterministic_steps = float(eval_steps)

            progress_rows.append(
                {
                    "time/total_timesteps": float(total_steps),
                    "rollout/episode_reward": episode_reward,
                    "rollout/episode_average_reward": episode_average_reward,
                    "rollout/ep_len_mean": float(len(rewards)),
                    "train/loss": float(loss.detach().cpu().item()),
                    "train/policy_loss": float(policy_loss.detach().cpu().item()),
                    "train/entropy_loss": float(entropy_loss.detach().cpu().item()),
                    "train/gradient_norm_before_clip": gradient_norm,
                    "train/gradient_clipped": gradient_clipped,
                    "train/policy_std_mean": policy_std_mean,
                    "train/mean_abs_action": mean_abs_action,
                    "train/mean_abs_action_delta": mean_abs_action_delta,
                    "reward_v4/zero_delta_ratio": reward_v4_diagnostics["zero_delta_ratio"],
                    "reward_v4/mean_abs_delta_a": reward_v4_diagnostics["mean_abs_delta_a"],
                    "reward_v4/mean_abs_effective_delta_a": reward_v4_diagnostics[
                        "mean_abs_effective_delta_a"
                    ],
                    "reward_v4/mean_abs_future_return": reward_v4_diagnostics[
                        "mean_abs_future_return"
                    ],
                    "reward_v4/positive_reward_ratio": reward_v4_diagnostics[
                        "positive_reward_ratio"
                    ],
                    "reward_v4/positive_step_reward_ratio": reward_v4_diagnostics[
                        "positive_step_reward_ratio"
                    ],
                    "reward_v4/valid_action_count": reward_v4_diagnostics[
                        "valid_action_count"
                    ],
                    "reward_v4/diagnostic_step_count": reward_v4_diagnostics[
                        "diagnostic_step_count"
                    ],
                    "eval/deterministic_episode_reward": deterministic_reward,
                    "eval/deterministic_final_asset": deterministic_final_asset,
                    "eval/deterministic_steps": deterministic_steps,
                    "episodes": float(episode),
                }
            )

            message = (
                f"REINFORCE episode {episode:04d} | "
                f"steps={total_steps} | ep_reward={episode_reward:.6f} | "
                f"ep_avg_reward={episode_average_reward:.6f} | "
                f"grad_norm={gradient_norm:.6f} | "
                f"policy_std={policy_std_mean:.6f}"
            )
            if should_evaluate:
                message += (
                    f" | deterministic_reward={deterministic_reward:.6f} | "
                    f"final_asset={deterministic_final_asset:.2f}"
                )
            print(message)

            if progress_path is not None:
                pd.DataFrame(progress_rows).to_csv(progress_path, index=False)

        return self

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        if out_path.suffix != ".pt":
            out_path = out_path.with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "ent_coef": self.ent_coef,
                "max_grad_norm": self.max_grad_norm,
                "normalize_returns": self.normalize_returns,
                "log_std_init": self.log_std_init,
                "seed": self.seed,
                "policy_state_dict": self.policy.state_dict(),
            },
            out_path,
        )
        return out_path

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "ReinforceAgent":
        load_path = Path(path)
        if load_path.suffix != ".pt":
            load_path = load_path.with_suffix(".pt")
        if not load_path.exists():
            raise FileNotFoundError(f"找不到 REINFORCE model: {load_path}")
        map_location = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        try:
            payload = torch.load(load_path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(load_path, map_location=map_location)

        agent = cls(
            state_dim=int(payload["state_dim"]),
            action_dim=int(payload["action_dim"]),
            hidden_dim=int(payload.get("hidden_dim", 256)),
            learning_rate=float(payload.get("learning_rate", 3e-4)),
            gamma=float(payload.get("gamma", 0.99)),
            ent_coef=float(payload.get("ent_coef", 0.001)),
            max_grad_norm=float(payload.get("max_grad_norm", 0.5)),
            normalize_returns=bool(payload.get("normalize_returns", True)),
            log_std_init=float(payload.get("log_std_init", -0.5)),
            seed=int(payload.get("seed", 0)),
            device=device,
        )
        agent.policy.load_state_dict(payload["policy_state_dict"])
        agent.policy.eval()
        return agent
