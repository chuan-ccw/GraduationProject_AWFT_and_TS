"""
Stock NeurIPS2018 Part 2. Train

This series is a reproduction of paper "Deep reinforcement learning for
automated stock trading: An ensemble strategy".

Introduce how to use FinRL to make data into the gym form environment, and train DRL agents on it.
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl import config
from finrl.config import INDICATORS
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from experiment_settings import env_bool, env_int, env_json, env_str

# Runtime settings: Run_StockTrading_Examples.py can override config.py through environment variables.
REWARD_TYPE = env_str("REWARD_TYPE", getattr(config, "REWARD_TYPE", "v0"))
REWARD_PARAMS = env_json("REWARD_PARAMS", getattr(config, "REWARD_PARAMS", {}))
RESULTS_DIR = env_str("RESULTS_DIR", config.RESULTS_DIR)
TRAINED_MODEL_DIR = env_str("TRAINED_MODEL_DIR", config.TRAINED_MODEL_DIR)
EXPERIMENT_DIR = env_str("EXPERIMENT_DIR", None)
TRAIN_DATA_PATH = Path(env_str("TRAIN_DATA_PATH", "train_data.csv"))
RANDOM_SEED = env_int("RANDOM_SEED", 43)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
try:
    import torch

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
except Exception:
    pass

print("Train setting:")
print(f"  REWARD_TYPE: {REWARD_TYPE}")
print(f"  REWARD_PARAMS: {REWARD_PARAMS}")
print(f"  EXPERIMENT_DIR: {EXPERIMENT_DIR}")
print(f"  TRAIN_DATA_PATH: {TRAIN_DATA_PATH}")
print(f"  RANDOM_SEED: {RANDOM_SEED}")

# %% Part 1. Prepare directories

check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])

OUTPUT_DIR = Path(EXPERIMENT_DIR).resolve() if EXPERIMENT_DIR else Path(RESULTS_DIR).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
today_str = datetime.now().strftime("%m%d")


def plot_train_reward_curve(rewards: pd.DataFrame, out_path: Path, agent_name: str) -> None:
    if rewards.empty:
        print(f"No train reward data to plot for {agent_name}.")
        return

    plt.figure(figsize=(15, 5))
    x = rewards["date"].astype(str) if "date" in rewards.columns else np.arange(len(rewards))
    y = rewards["cumulative_reward_unscaled"]
    plt.plot(x, y)

    max_xticks = 12
    if len(rewards) > 0:
        tick_positions = np.linspace(0, len(rewards) - 1, min(max_xticks, len(rewards))).astype(int)
        if hasattr(x, "iloc"):
            tick_labels = [str(x.iloc[i]) for i in tick_positions]
        else:
            tick_labels = [str(x[i]) for i in tick_positions]
        plt.xticks(tick_positions, tick_labels, rotation=45, ha="right")

    plt.title(f"Training Cumulative Reward Curve - {agent_name.upper()} ({REWARD_TYPE})")
    plt.xlabel("Training date")
    plt.ylabel("Cumulative reward, unscaled")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_on_train_and_save_reward_curve(model, agent_name: str) -> None:
    """Run the trained model once on the training window and save its reward curve.

    The reward curve belongs to the training window, not the trade/backtest window.
    This function creates a fresh training environment so the saved curve covers
    the full training period instead of whatever partial episode SB3 last used.
    """
    if model is None:
        return

    train_eval_env = StockTradingEnv(
        df=train,
        reward_type=REWARD_TYPE,
        reward_params=REWARD_PARAMS,
        **env_kwargs,
    )
    vec_env, obs = train_eval_env.get_sb_env()
    vec_env.reset()
    max_steps = len(train_eval_env.df.index.unique()) - 1

    for i in range(len(train_eval_env.df.index.unique())):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards_step, dones, info = vec_env.step(action)
        if dones[0] or i >= max_steps - 1:
            break

    rewards = vec_env.env_method(method_name="save_reward_memory")[0]

    train_dates = pd.Series(train["date"].astype(str).unique())
    train_start_date = train_dates.iloc[0]
    train_end_date = train_dates.iloc[-1]
    suffix = f"{agent_name}_{REWARD_TYPE}_{train_start_date}_to_{train_end_date}_{today_str}"

    reward_path = OUTPUT_DIR / f"train_reward_curve_{suffix}.csv"
    plot_path = OUTPUT_DIR / f"train_reward_growth_curve_{suffix}.png"
    rewards.to_csv(reward_path, index=False, encoding="utf-8-sig")
    plot_train_reward_curve(rewards, plot_path, agent_name)

    print(f"Train reward curve saved to {reward_path}")
    print(f"Train reward curve plot saved to {plot_path}")

# %% Part 2. Build environment

train = pd.read_csv(TRAIN_DATA_PATH)
train = train.set_index(train.columns[0])
train.index.names = [""]

stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}

e_train_gym = StockTradingEnv(
    df=train,
    reward_type=REWARD_TYPE,
    reward_params=REWARD_PARAMS,
    **env_kwargs,
)
env_train, _ = e_train_gym.get_sb_env()
env_train.seed(RANDOM_SEED)
print(type(env_train))

# %% Part 3. Train DRL Agents

if_using_a2c = env_bool("IF_USING_A2C", False)
if_using_ddpg = env_bool("IF_USING_DDPG", False)
if_using_ppo = env_bool("IF_USING_PPO", True)
if_using_td3 = env_bool("IF_USING_TD3", False)
if_using_sac = env_bool("IF_USING_SAC", False)

A2C_TOTAL_TIMESTEPS = env_int("A2C_TOTAL_TIMESTEPS", 20000)
DDPG_TOTAL_TIMESTEPS = env_int("DDPG_TOTAL_TIMESTEPS", 20000)
PPO_TOTAL_TIMESTEPS = env_int("PPO_TOTAL_TIMESTEPS", 10000)
TD3_TOTAL_TIMESTEPS = env_int("TD3_TOTAL_TIMESTEPS", 20000)
SAC_TOTAL_TIMESTEPS = env_int("SAC_TOTAL_TIMESTEPS", 20000)

# --- Agent 1: A2C ---
agent = DRLAgent(env=env_train)
model_a2c = agent.get_model("a2c", seed=RANDOM_SEED)
if if_using_a2c:
    tmp_path = RESULTS_DIR + "/a2c"
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_a2c.set_logger(new_logger_a2c)

trained_a2c = (
    agent.train_model(model=model_a2c, tb_log_name="a2c", total_timesteps=A2C_TOTAL_TIMESTEPS)
    if if_using_a2c
    else None
)
if if_using_a2c:
    trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c")
    evaluate_on_train_and_save_reward_curve(trained_a2c, "a2c")

# --- Agent 2: DDPG ---
agent = DRLAgent(env=env_train)
model_ddpg = agent.get_model("ddpg", seed=RANDOM_SEED)
if if_using_ddpg:
    tmp_path = RESULTS_DIR + "/ddpg"
    new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_ddpg.set_logger(new_logger_ddpg)

trained_ddpg = (
    agent.train_model(model=model_ddpg, tb_log_name="ddpg", total_timesteps=DDPG_TOTAL_TIMESTEPS)
    if if_using_ddpg
    else None
)
if if_using_ddpg:
    trained_ddpg.save(TRAINED_MODEL_DIR + "/agent_ddpg")
    evaluate_on_train_and_save_reward_curve(trained_ddpg, "ddpg")

# --- Agent 3: PPO ---
agent = DRLAgent(env=env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS, seed=RANDOM_SEED)
if if_using_ppo:
    tmp_path = RESULTS_DIR + "/ppo"
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_ppo.set_logger(new_logger_ppo)

trained_ppo = (
    agent.train_model(model=model_ppo, tb_log_name="ppo", total_timesteps=PPO_TOTAL_TIMESTEPS)
    if if_using_ppo
    else None
)
if if_using_ppo:
    trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo")
    evaluate_on_train_and_save_reward_curve(trained_ppo, "ppo")

# --- Agent 4: TD3 ---
agent = DRLAgent(env=env_train)
TD3_PARAMS = {
    "batch_size": 100,
    "buffer_size": 1000000,
    "learning_rate": 0.001,
}
model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS, seed=RANDOM_SEED)
if if_using_td3:
    tmp_path = RESULTS_DIR + "/td3"
    new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_td3.set_logger(new_logger_td3)

trained_td3 = (
    agent.train_model(model=model_td3, tb_log_name="td3", total_timesteps=TD3_TOTAL_TIMESTEPS)
    if if_using_td3
    else None
)
if if_using_td3:
    trained_td3.save(TRAINED_MODEL_DIR + "/agent_td3")
    evaluate_on_train_and_save_reward_curve(trained_td3, "td3")

# --- Agent 5: SAC ---
agent = DRLAgent(env=env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS, seed=RANDOM_SEED)
if if_using_sac:
    tmp_path = RESULTS_DIR + "/sac"
    new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_sac.set_logger(new_logger_sac)

trained_sac = (
    agent.train_model(model=model_sac, tb_log_name="sac", total_timesteps=SAC_TOTAL_TIMESTEPS)
    if if_using_sac
    else None
)
if if_using_sac:
    trained_sac.save(TRAINED_MODEL_DIR + "/agent_sac")
    evaluate_on_train_and_save_reward_curve(trained_sac, "sac")

print("All agents trained and saved to", TRAINED_MODEL_DIR)
