from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REINFORCE_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "finrl"
    / "agents"
    / "policy_gradient"
    / "reinforce.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_reinforce_module",
    _REINFORCE_MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_REINFORCE_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _REINFORCE_MODULE
_SPEC.loader.exec_module(_REINFORCE_MODULE)

ReinforceAgent = _REINFORCE_MODULE.ReinforceAgent


class _FixedLengthEnv:
    def __init__(self, episode_length: int) -> None:
        self.episode_length = int(episode_length)
        self.step_in_episode = 0

    def reset(self):
        self.step_in_episode = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, _action):
        self.step_in_episode += 1
        done = self.step_in_episode >= self.episode_length
        observation = np.array([float(self.step_in_episode)], dtype=np.float32)
        reward = np.array([1.0], dtype=np.float32)
        return observation, reward, np.array([done]), {}


def test_reinforce_finishes_started_episode(tmp_path) -> None:
    env = _FixedLengthEnv(episode_length=3)
    progress_path = tmp_path / "progress.csv"
    agent = ReinforceAgent(
        state_dim=1,
        action_dim=1,
        hidden_dim=8,
        normalize_returns=False,
        seed=7,
        device="cpu",
    )

    agent.learn(
        env=env,
        total_timesteps=5,
        progress_path=progress_path,
        eval_env=None,
    )

    progress = pd.read_csv(progress_path)

    # The first episode ends at 3 steps. Starting the second episode is allowed
    # because 3 < 5, and that episode must finish at 6 instead of stopping at 5.
    assert progress["time/total_timesteps"].tolist() == [3.0, 6.0]
    assert progress["rollout/ep_len_mean"].tolist() == [3.0, 3.0]
