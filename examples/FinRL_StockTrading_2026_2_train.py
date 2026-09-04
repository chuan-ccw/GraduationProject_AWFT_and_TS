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

from finrl.agents.policy_gradient import ReinforceAgent
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl import config
from finrl.config import ADDITIONAL_RAW_FEATURES, INDICATORS
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
PRIMARY_AGENT_NAME = env_str("PRIMARY_AGENT_NAME", "ppo").strip().lower()
REINFORCE_EVAL_INTERVAL = env_int("REINFORCE_EVAL_INTERVAL", 5)


def get_reward_scaling(reward_type: str) -> float:
    """
    Choose reward_scaling by reward definition.

    v0 is a money-difference reward, so it keeps FinRL's original 1e-4.
    v1/v2/v3/v4/v5/v6/v7 are return-level rewards, so they should not be shrunk by 1e-4.
    """
    reward_key = str(reward_type).lower()

    if reward_key in {"v0", "original", "finrl_original"}:
        return 1e-4

    if reward_key in {
        "v1",
        "our_reward_v1",
        "v2",
        "reward_v2",
        "future_day_return",
        "v3",
        "reward_v3",
        "future_avg_return",
        "v4",
        "reward_v4",
        "future_avg_delta_action_return",
        "future_avg_delta_action_reward",
        "v5",
        "reward_v5",
        "weight_future",
        "reward_weight_future",
        "weight_future_reward",
        "portfolio_weighted_future_return",
        "portfolio_weighted_future_return_reward",
        "wft",
        "weighted_future_trend",
        "weighted_future_trend_reward",
        "v6",
        "reward_v6",
        "delta_weight_future",
        "reward_delta_weight_future",
        "delta_weight_future_reward",
        "portfolio_weight_change_future_return",
        "portfolio_weight_change_future_return_reward",
        "dwft",
        "delta_weighted_future_trend",
        "delta_weighted_future_trend_reward",
        "v7",
        "reward_v7",
        "awft",
        "action_weight_future_trend",
        "action_weight_future_trend_reward",
    }:
        return 1.0

    return 1.0


REWARD_SCALING = get_reward_scaling(REWARD_TYPE)

REWARD_DISPLAY_NAMES = {
    "v0": "FinRL_reward",
    "original": "FinRL_reward",
    "finrl_original": "FinRL_reward",
    "v1": "buy_low_sell_high_reward",
    "our_reward_v1": "buy_low_sell_high_reward",
    "v2": "future_day_reward",
    "reward_v2": "future_day_reward",
    "future_day_return": "future_day_reward",
    "v3": "future_avg_reward",
    "reward_v3": "future_avg_reward",
    "future_avg_return": "future_avg_reward",
    "v4": "future_avg_delta_action_reward",
    "reward_v4": "future_avg_delta_action_reward",
    "future_avg_delta_action_return": "future_avg_delta_action_reward",
    "future_avg_delta_action_reward": "future_avg_delta_action_reward",
    "v5": "WFT",
    "reward_v5": "WFT",
    "weight_future": "WFT",
    "reward_weight_future": "WFT",
    "weight_future_reward": "WFT",
    "portfolio_weighted_future_return": "WFT",
    "portfolio_weighted_future_return_reward": "WFT",
    "wft": "WFT",
    "weighted_future_trend": "WFT",
    "weighted_future_trend_reward": "WFT",
    "v6": "DWFT",
    "reward_v6": "DWFT",
    "delta_weight_future": "DWFT",
    "reward_delta_weight_future": "DWFT",
    "delta_weight_future_reward": "DWFT",
    "portfolio_weight_change_future_return": "DWFT",
    "portfolio_weight_change_future_return_reward": "DWFT",
    "dwft": "DWFT",
    "delta_weighted_future_trend": "DWFT",
    "delta_weighted_future_trend_reward": "DWFT",
    "v7": "AWFT",
    "reward_v7": "AWFT",
    "awft": "AWFT",
    "action_weight_future_trend": "AWFT",
    "action_weight_future_trend_reward": "AWFT",
}


def reward_display_name(reward_type: str | None) -> str:
    """Return the formal display name used in plots instead of reward version codes."""
    key = str(reward_type or "").strip().lower()
    if key.startswith("reward_") and key not in REWARD_DISPLAY_NAMES:
        key = key[len("reward_"):]
    return REWARD_DISPLAY_NAMES.get(key, str(reward_type or ""))


def plot_reward_parameter_boxplot(
    rewards: pd.DataFrame,
    reward_parameters: pd.DataFrame,
    output_path: Path,
    agent_name: str,
) -> None:
    """Plot raw train-evaluation reward inputs without normalization."""
    box_values: list[np.ndarray] = []
    box_labels: list[str] = []

    if reward_parameters is not None and not reward_parameters.empty:
        for column, label in [("a", "a"), ("w", "w"), ("delta_w", "delta_w"), ("d", "d")]:
            if column not in reward_parameters.columns:
                continue
            values = pd.to_numeric(reward_parameters[column], errors="coerce").dropna()
            if not values.empty:
                box_values.append(values.to_numpy(dtype=float))
                box_labels.append(label)

    reward_values = pd.to_numeric(
        rewards.get("step_reward_unscaled", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    if not reward_values.empty:
        box_values.append(reward_values.to_numpy(dtype=float))
        box_labels.append("reward")

    if not box_values:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(box_values, showfliers=True)
    ax.set_xticks(np.arange(1, len(box_labels) + 1))
    ax.set_xticklabels(box_labels)
    ax.set_title(
        f"Train Evaluation Reward Parameter Box Plot - {agent_name.upper()} "
        f"({reward_display_name(REWARD_TYPE)})"
    )
    ax.set_xlabel("Reward value / parameter")
    ax.set_ylabel("Unscaled value")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_future_reward_train_skip_days(reward_type: str, reward_params: dict) -> int:
    """Return how many final training days should be skipped as action states.

    Future-return rewards need future prices inside the training period.
    Example: n_future=5 means the last 5 training dates are kept in df as labels,
    but the agent will not start a training step from those dates.
    """
    reward_key = str(reward_type or "").strip().lower()
    future_reward_keys = {
        "v2",
        "reward_v2",
        "future_day_return",
        "v3",
        "reward_v3",
        "future_avg_return",
        "v4",
        "reward_v4",
        "future_avg_delta_action_return",
        "future_avg_delta_action_reward",
        "v5",
        "reward_v5",
        "weight_future",
        "reward_weight_future",
        "weight_future_reward",
        "portfolio_weighted_future_return",
        "portfolio_weighted_future_return_reward",
        "wft",
        "weighted_future_trend",
        "weighted_future_trend_reward",
        "v6",
        "reward_v6",
        "delta_weight_future",
        "reward_delta_weight_future",
        "delta_weight_future_reward",
        "portfolio_weight_change_future_return",
        "portfolio_weight_change_future_return_reward",
        "dwft",
        "delta_weighted_future_trend",
        "delta_weighted_future_trend_reward",
        "v7",
        "reward_v7",
        "awft",
        "action_weight_future_trend",
        "action_weight_future_trend_reward",
    }
    if reward_key not in future_reward_keys:
        return 0
    return max(0, int((reward_params or {}).get("n_future", 0)))


def set_axis_labels_horizontal(ax, xlabel: str | None = None, ylabel: str | None = None, ylabel_pad: int = 65) -> None:
    """Keep x-axis label horizontal, but keep y-axis label in the default vertical style."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, rotation=0)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def safe_filename(value: str) -> str:
    text = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in str(value)
    )
    return text.strip("_") or "stock"


TRAIN_TERMINAL_SKIP_DAYS = get_future_reward_train_skip_days(REWARD_TYPE, REWARD_PARAMS)

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
print(f"  REWARD_SCALING: {REWARD_SCALING}")
print(f"  TRAIN_TERMINAL_SKIP_DAYS: {TRAIN_TERMINAL_SKIP_DAYS}")
print(f"  EXPERIMENT_DIR: {EXPERIMENT_DIR}")
print(f"  TRAIN_DATA_PATH: {TRAIN_DATA_PATH}")
print(f"  RANDOM_SEED: {RANDOM_SEED}")
print(f"  REINFORCE_EVAL_INTERVAL: {REINFORCE_EVAL_INTERVAL}")

# %% Part 1. Prepare directories

check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])

OUTPUT_DIR = Path(EXPERIMENT_DIR).resolve() if EXPERIMENT_DIR else Path(RESULTS_DIR).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
today_str = datetime.now().strftime("%m%d")


def save_image_source_manifest(rows: list[dict[str, object]]) -> None:
    """Write which data source each training image used."""
    if not rows:
        return
    manifest_path = OUTPUT_DIR / "image_source_manifest.csv"
    new_df = pd.DataFrame(rows)
    if manifest_path.exists():
        old_df = pd.read_csv(manifest_path)
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    if "image" in new_df.columns:
        new_df = new_df.drop_duplicates(subset=["image"], keep="last")
    new_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column that exists in df."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def save_sb3_training_reward_curve(agent_name: str) -> None:
    """Save the training reward curve used for convergence analysis.

    PPO / other SB3 agents:
      x-axis = epoch (the sequential progress.csv record number).
      y-axis = rollout/ep_rew_mean from Stable-Baselines3.

    REINFORCE:
      x-axis = epoch (one complete training episode).
      y-axis = average reward within that episode.
    """
    progress_path = Path(RESULTS_DIR) / agent_name / "progress.csv"
    if not progress_path.exists():
        print(f"No SB3 progress.csv found for {agent_name}: {progress_path}")
        return

    progress = pd.read_csv(progress_path)
    if progress.empty:
        print(f"SB3 progress.csv is empty for {agent_name}: {progress_path}")
        return

    is_reinforce = agent_name.lower() == "reinforce"
    x_col = _first_existing_column(progress, ["episodes"]) if is_reinforce else None
    timestep_col = None if is_reinforce else _first_existing_column(
        progress,
        ["time/total_timesteps", "total_timesteps", "time/iterations"],
    )
    reward_col = None if is_reinforce else _first_existing_column(
        progress,
        ["rollout/ep_rew_mean", "ep_rew_mean", "train/reward", "rollout/episode_reward"],
    )

    if (is_reinforce and x_col is None) or (not is_reinforce and reward_col is None):
        print("Cannot plot convergence reward curve because required columns are missing.")
        print(f"progress.csv columns = {list(progress.columns)}")
        return

    if is_reinforce:
        average_col = "rollout/episode_average_reward"
        total_col = "rollout/episode_reward"
        length_col = "rollout/ep_len_mean"

        if average_col in progress.columns:
            curve = progress[[x_col, average_col]].copy()
            curve = curve.rename(columns={x_col: "epoch", average_col: "average_reward"})
        elif total_col in progress.columns and length_col in progress.columns:
            # Compatibility with older progress.csv files: recover the exact
            # per-episode average without using the old moving-average column.
            curve = progress[[x_col, total_col, length_col]].copy()
            curve = curve.rename(columns={x_col: "epoch"})
            episode_total = pd.to_numeric(curve[total_col], errors="coerce")
            episode_length = pd.to_numeric(curve[length_col], errors="coerce")
            curve["average_reward"] = episode_total / episode_length.replace(0, np.nan)
            curve = curve[["epoch", "average_reward"]]
        else:
            print(
                "Cannot plot REINFORCE reward curve because per-episode reward "
                "and episode length columns are missing."
            )
            print(f"progress.csv columns = {list(progress.columns)}")
            return

        curve["epoch"] = pd.to_numeric(curve["epoch"], errors="coerce")
        curve["average_reward"] = pd.to_numeric(curve["average_reward"], errors="coerce")
        curve = curve.dropna(subset=["epoch", "average_reward"]).reset_index(drop=True)
        curve["epoch"] = curve["epoch"].astype(int)
        plot_x_col = "epoch"
        plot_y_col = "average_reward"
        x_label = "epoch"
        y_label = "average reward"
    else:
        selected_columns = [reward_col]
        if timestep_col is not None:
            selected_columns.insert(0, timestep_col)
        curve = progress[selected_columns].copy()
        curve = curve.rename(columns={reward_col: "average_reward"})
        curve["average_reward"] = pd.to_numeric(curve["average_reward"], errors="coerce")
        if timestep_col is not None:
            curve = curve.rename(columns={timestep_col: "training_timestep"})
            curve["training_timestep"] = pd.to_numeric(curve["training_timestep"], errors="coerce")
        curve = curve.dropna(subset=["average_reward"]).reset_index(drop=True)
        curve.insert(0, "epoch", np.arange(1, len(curve) + 1, dtype=int))
        plot_x_col = "epoch"
        plot_y_col = "average_reward"
        x_label = "epoch"
        y_label = "average reward"

    if curve.empty:
        print(f"No numeric reward data available in {progress_path}")
        return

    suffix = f"{agent_name}_{REWARD_TYPE}_seed_{RANDOM_SEED}_{today_str}"
    csv_path = OUTPUT_DIR / f"training_reward_curve_{suffix}.csv"
    png_path = OUTPUT_DIR / f"training_reward_curve_{suffix}.png"

    # Keep the original SB3 reward values while using a common epoch axis.
    curve.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Canonical filename requested by rolling experiment folder layout.
    if agent_name.lower() == PRIMARY_AGENT_NAME:
        curve.to_csv(OUTPUT_DIR / "reward_curve.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(
        curve[plot_x_col],
        curve[plot_y_col],
        linewidth=1.5,
        alpha=0.9,
    )

    ax.set_title(f"Training Reward Convergence Curve - {agent_name.upper()} ({reward_display_name(REWARD_TYPE)})")
    set_axis_labels_horizontal(ax, x_label, y_label, ylabel_pad=75)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    source_rows = [
        {
            "image": str(png_path.name),
            "image_path": str(png_path),
            "source_role": "reinforce_training_progress" if is_reinforce else "sb3_training_progress",
            "source_csv": str(progress_path),
            "output_csv": str(csv_path),
            "stage": "train_times",
            "reward_type": REWARD_TYPE,
            "random_seed": RANDOM_SEED,
            "training_data": str(TRAIN_DATA_PATH),
        }
    ]
    if agent_name.lower() == PRIMARY_AGENT_NAME:
        canonical_png = OUTPUT_DIR / f"training_reward_curve_{PRIMARY_AGENT_NAME}.png"
        fig.savefig(canonical_png, dpi=150, bbox_inches="tight")
        source_rows.append(
            {
                "image": canonical_png.name,
                "image_path": str(canonical_png),
                "source_role": "reinforce_training_progress" if is_reinforce else "sb3_training_progress",
                "source_csv": str(progress_path),
                "output_csv": str(csv_path),
                "stage": "train_times",
                "reward_type": REWARD_TYPE,
                "random_seed": RANDOM_SEED,
                "training_data": str(TRAIN_DATA_PATH),
            }
        )
    save_image_source_manifest(source_rows)
    plt.close(fig)

    print(f"Training reward curve saved to {csv_path}")
    print(f"Training reward plot saved to {png_path}")


def save_reinforce_individual_stock_train_reward_curves(model: ReinforceAgent) -> None:
    """Save one epoch-level training reward curve for each stock.

    For every complete REINFORCE epoch, the environment already records one
    unscaled daily reward contribution per stock. The y value saved here is:

        stock_epoch_reward = sum(daily reward contribution) / train trading days
    """
    epoch_rows = getattr(model, "individual_stock_epoch_rewards", None)
    if not epoch_rows:
        print("No REINFORCE individual-stock epoch rewards were recorded.")
        return

    epoch_df = pd.DataFrame(epoch_rows)
    if epoch_df.empty or "epoch" not in epoch_df.columns:
        print("REINFORCE individual-stock epoch reward data is empty.")
        return

    metadata_columns = {"epoch", "train_trading_days"}
    stock_columns = [
        column for column in epoch_df.columns if column not in metadata_columns
    ]
    if not stock_columns:
        print("No stock columns found in REINFORCE individual-stock epoch rewards.")
        return

    out_dir = OUTPUT_DIR / "individual_stock_train_reward_curve"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, object]] = []

    for ticker in stock_columns:
        source_df = epoch_df[["epoch", "train_trading_days", ticker]].copy()
        source_df = source_df.rename(columns={ticker: "stock_epoch_reward"})
        source_df["epoch"] = pd.to_numeric(source_df["epoch"], errors="coerce")
        source_df["train_trading_days"] = pd.to_numeric(
            source_df["train_trading_days"], errors="coerce"
        )
        source_df["stock_epoch_reward"] = pd.to_numeric(
            source_df["stock_epoch_reward"], errors="coerce"
        )
        source_df = source_df.dropna(subset=["epoch", "stock_epoch_reward"]).copy()
        if source_df.empty:
            continue

        source_df["epoch"] = source_df["epoch"].astype(int)
        csv_path = out_dir / f"{safe_filename(ticker)}_reward_curve.csv"
        png_path = out_dir / f"{safe_filename(ticker)}_reward_curve.png"
        source_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(
            source_df["epoch"],
            source_df["stock_epoch_reward"],
            linewidth=1.8,
        )
        ax.set_title(
            f"{ticker} Individual Train Reward Curve ({reward_display_name(REWARD_TYPE)})"
        )
        set_axis_labels_horizontal(ax, "Epoch", "Reward", ylabel_pad=65)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        source_rows.append(
            {
                "image": png_path.name,
                "image_path": str(png_path),
                "source_role": "reinforce_individual_stock_train_reward",
                "source_csv": str(csv_path),
                "output_csv": str(csv_path),
                "stage": "train_times",
                "reward_type": REWARD_TYPE,
                "random_seed": RANDOM_SEED,
                "training_data": str(TRAIN_DATA_PATH),
            }
        )

    save_image_source_manifest(source_rows)
    print(f"Individual stock train reward curves saved to {out_dir}")



def save_reinforce_diagnostics() -> None:
    """Copy REINFORCE training and reward-v4 diagnostics to the times folder."""
    progress_path = Path(RESULTS_DIR) / "reinforce" / "progress.csv"
    if not progress_path.exists():
        print(f"No REINFORCE progress.csv found: {progress_path}")
        return

    progress = pd.read_csv(progress_path)
    if progress.empty:
        print(f"REINFORCE progress.csv is empty: {progress_path}")
        return

    diagnostic_columns = [
        "episodes",
        "time/total_timesteps",
        "rollout/episode_reward",
        "rollout/episode_average_reward",
        "rollout/ep_len_mean",
        "train/loss",
        "train/policy_loss",
        "train/entropy_loss",
        "train/gradient_norm_before_clip",
        "train/gradient_clipped",
        "train/policy_std_mean",
        "train/mean_abs_action",
        "train/mean_abs_action_delta",
        "reward_v4/zero_delta_ratio",
        "reward_v4/mean_abs_delta_a",
        "reward_v4/mean_abs_effective_delta_a",
        "reward_v4/mean_abs_future_return",
        "reward_v4/positive_reward_ratio",
        "reward_v4/positive_step_reward_ratio",
        "reward_v4/valid_action_count",
        "reward_v4/diagnostic_step_count",
        "eval/deterministic_episode_reward",
        "eval/deterministic_final_asset",
        "eval/deterministic_steps",
    ]
    available_columns = [col for col in diagnostic_columns if col in progress.columns]
    diagnostics = progress[available_columns].copy()
    output_path = OUTPUT_DIR / "reinforce_diagnostics.csv"
    diagnostics.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"REINFORCE diagnostics saved to {output_path}")


def save_reinforce_deterministic_evaluation_curve() -> None:
    """Save the periodic deterministic evaluation curve from progress.csv."""
    progress_path = Path(RESULTS_DIR) / "reinforce" / "progress.csv"
    if not progress_path.exists():
        print(f"No REINFORCE progress.csv found: {progress_path}")
        return

    progress = pd.read_csv(progress_path)
    required = [
        "episodes",
        "eval/deterministic_episode_reward",
        "eval/deterministic_final_asset",
    ]
    if progress.empty or any(col not in progress.columns for col in required):
        print("Cannot plot deterministic evaluation curve because required columns are missing.")
        return

    curve = progress[required].copy()
    curve = curve.rename(
        columns={
            "episodes": "epoch",
            "eval/deterministic_episode_reward": "deterministic_reward",
            "eval/deterministic_final_asset": "deterministic_final_asset",
        }
    )
    for col in ["epoch", "deterministic_reward", "deterministic_final_asset"]:
        curve[col] = pd.to_numeric(curve[col], errors="coerce")
    curve = curve.dropna(subset=["epoch", "deterministic_reward"]).reset_index(drop=True)
    if curve.empty:
        print("No deterministic evaluation rows were recorded.")
        return

    curve["epoch"] = curve["epoch"].astype(int)
    suffix = f"reinforce_{REWARD_TYPE}_seed_{RANDOM_SEED}_{today_str}"
    csv_path = OUTPUT_DIR / f"deterministic_eval_curve_{suffix}.csv"
    canonical_csv_path = OUTPUT_DIR / "deterministic_eval_curve.csv"
    curve.to_csv(csv_path, index=False, encoding="utf-8-sig")
    curve.to_csv(canonical_csv_path, index=False, encoding="utf-8-sig")

    # 2026-08-08: deterministic_eval_curve 圖片暫停產出。
    # 保留 deterministic evaluation CSV，之後需要時可直接恢復繪圖程式。
    # png_path = OUTPUT_DIR / f"deterministic_eval_curve_{suffix}.png"
    # canonical_png_path = OUTPUT_DIR / "deterministic_eval_curve.png"
    #
    # fig, ax = plt.subplots(figsize=(15, 5))
    # ax.plot(curve["epoch"], curve["deterministic_reward"], linewidth=2.0)
    # ax.set_title(
    #     "Deterministic Evaluation Reward Curve - REINFORCE "
    #     f"({reward_display_name(REWARD_TYPE)})"
    # )
    # set_axis_labels_horizontal(ax, "epoch", "deterministic reward", ylabel_pad=75)
    # ax.grid(True, alpha=0.25)
    # fig.tight_layout()
    # fig.savefig(png_path, dpi=150, bbox_inches="tight")
    # fig.savefig(canonical_png_path, dpi=150, bbox_inches="tight")
    # plt.close(fig)
    #
    # save_image_source_manifest(
    #     [
    #         {
    #             "image": png_path.name,
    #             "image_path": str(png_path),
    #             "source_role": "reinforce_periodic_deterministic_evaluation",
    #             "source_csv": str(progress_path),
    #             "output_csv": str(csv_path),
    #             "stage": "train_times_diagnostic_eval",
    #             "reward_type": REWARD_TYPE,
    #             "random_seed": RANDOM_SEED,
    #             "training_data": str(TRAIN_DATA_PATH),
    #             "eval_interval_epochs": REINFORCE_EVAL_INTERVAL,
    #         },
    #         {
    #             "image": canonical_png_path.name,
    #             "image_path": str(canonical_png_path),
    #             "source_role": "reinforce_periodic_deterministic_evaluation",
    #             "source_csv": str(progress_path),
    #             "output_csv": str(canonical_csv_path),
    #             "stage": "train_times_diagnostic_eval",
    #             "reward_type": REWARD_TYPE,
    #             "random_seed": RANDOM_SEED,
    #             "training_data": str(TRAIN_DATA_PATH),
    #             "eval_interval_epochs": REINFORCE_EVAL_INTERVAL,
    #         },
    #     ]
    # )
    print(f"Deterministic evaluation curve saved to {csv_path}")


def plot_train_eval_reward_curve(rewards, plot_path, agent_name):
    plot_df = pd.DataFrame(rewards)

    # 避免 env_step 重複插入造成：
    # ValueError: cannot insert env_step, already exists
    if "env_step" not in plot_df.columns:
        plot_df.insert(0, "env_step", np.arange(1, len(plot_df) + 1))

    if "reward" in plot_df.columns:
        reward_col = "reward"
    elif "cumulative_reward" in plot_df.columns:
        reward_col = "cumulative_reward"
    else:
        numeric_cols = plot_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "env_step"]
        reward_col = numeric_cols[0]

    plot_df["cumulative_reward"] = plot_df[reward_col].cumsum()

    csv_path = Path(plot_path).with_suffix(".csv")
    plot_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(14, 5))
    plt.plot(plot_df["env_step"], plot_df["cumulative_reward"], linewidth=1.8)
    ax = plt.gca()
    ax.set_title(f"Train Evaluation Cumulative Reward by Environment Step - {agent_name.upper()} ({reward_display_name(REWARD_TYPE)})")
    set_axis_labels_horizontal(ax, "Environment Step", "Cumulative Reward", ylabel_pad=65)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    save_image_source_manifest(
        [
            {
                "image": Path(plot_path).name,
                "image_path": str(plot_path),
                "source_role": "train_environment_evaluation_rewards",
                "source_csv": str(csv_path),
                "stage": "train_times_eval",
                "reward_type": REWARD_TYPE,
                "random_seed": RANDOM_SEED,
                "training_data": str(TRAIN_DATA_PATH),
            }
        ]
    )
    plt.close()


def evaluate_on_train_and_save_reward_curve(model, agent_name: str) -> None:
    """Run the trained model once on the training window and save evaluation reward.

    Important: this is only a deterministic evaluation over the train window.
    To check convergence, use save_sb3_training_reward_curve(), which reads
    Stable-Baselines3 progress.csv and uses training timesteps on the x-axis.
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
    weights = vec_env.env_method(method_name="save_weight_memory")[0]
    dwft_parameters = vec_env.env_method(
        method_name="save_dwft_parameter_memory"
    )[0]
    awft_parameters = vec_env.env_method(
        method_name="save_awft_parameter_memory"
    )[0]
    rewards = rewards.copy()
    rewards.insert(0, "env_step", np.arange(1, len(rewards) + 1))

    train_dates = pd.Series(train["date"].astype(str).unique())
    train_start_date = train_dates.iloc[0]
    train_end_date = train_dates.iloc[-1]
    suffix = f"{agent_name}_{REWARD_TYPE}_{train_start_date}_to_{train_end_date}_{today_str}"

    reward_path = OUTPUT_DIR / f"train_eval_reward_by_step_{suffix}.csv"
    weights_path = OUTPUT_DIR / f"train_eval_post_trade_weights_{suffix}.csv"
    dwft_parameters_path = OUTPUT_DIR / f"train_eval_dwft_parameters_{suffix}.csv"
    awft_parameters_path = OUTPUT_DIR / f"train_eval_awft_parameters_{suffix}.csv"
    reward_boxplot_path = OUTPUT_DIR / f"train_eval_reward_parameter_boxplot_{suffix}.png"
    # 2026-08-08: train_eval_cumulative_reward_by_step 圖片暫停產出。
    # plot_path = OUTPUT_DIR / f"train_eval_cumulative_reward_by_step_{suffix}.png"
    rewards.to_csv(reward_path, index=False, encoding="utf-8-sig")
    weights.to_csv(weights_path, index=False, encoding="utf-8-sig")
    if dwft_parameters is not None and not dwft_parameters.empty:
        dwft_parameters.to_csv(
            dwft_parameters_path,
            index=False,
            encoding="utf-8-sig",
        )
    if awft_parameters is not None and not awft_parameters.empty:
        awft_parameters.to_csv(
            awft_parameters_path,
            index=False,
            encoding="utf-8-sig",
        )
    reward_parameters = (
        awft_parameters
        if awft_parameters is not None and not awft_parameters.empty
        else dwft_parameters
    )
    plot_reward_parameter_boxplot(
        rewards,
        reward_parameters,
        reward_boxplot_path,
        agent_name,
    )

    if agent_name.lower() == PRIMARY_AGENT_NAME:
        rewards.to_csv(OUTPUT_DIR / "train_eval_reward_curve.csv", index=False, encoding="utf-8-sig")
        weights.to_csv(
            OUTPUT_DIR / "train_eval_post_trade_weights.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if dwft_parameters is not None and not dwft_parameters.empty:
            dwft_parameters.to_csv(
                OUTPUT_DIR / "train_eval_dwft_parameters.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if awft_parameters is not None and not awft_parameters.empty:
            awft_parameters.to_csv(
                OUTPUT_DIR / "train_eval_awft_parameters.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if reward_boxplot_path.exists():
            (OUTPUT_DIR / "train_eval_reward_parameter_boxplot.png").write_bytes(
                reward_boxplot_path.read_bytes()
            )
    # plot_train_eval_reward_curve(rewards, plot_path, agent_name)

    print(f"Train evaluation reward by step saved to {reward_path}")
    if dwft_parameters is not None and not dwft_parameters.empty:
        print(f"Train evaluation DWFT parameters saved to {dwft_parameters_path}")
    if awft_parameters is not None and not awft_parameters.empty:
        print(f"Train evaluation AWFT parameters saved to {awft_parameters_path}")
    print(f"Train evaluation reward parameter box plot saved to {reward_boxplot_path}")
    print(f"Train evaluation post-trade weights saved to {weights_path}")

# %% Part 2. Build environment

train = pd.read_csv(TRAIN_DATA_PATH)
train = train.set_index(train.columns[0])
train.index.names = [""]

train_dates = pd.Series(train["date"].astype(str).unique())
if TRAIN_TERMINAL_SKIP_DAYS > 0:
    skipped_start_pos = max(0, len(train_dates) - TRAIN_TERMINAL_SKIP_DAYS)
    last_action_pos = max(0, skipped_start_pos - 1)
    guard_df = pd.DataFrame(
        [
            {
                "reward_type": REWARD_TYPE,
                "n_future": TRAIN_TERMINAL_SKIP_DAYS,
                "train_start_date": train_dates.iloc[0] if not train_dates.empty else "",
                "train_end_date_config": train_dates.iloc[-1] if not train_dates.empty else "",
                "last_train_action_date": train_dates.iloc[last_action_pos] if not train_dates.empty else "",
                "skipped_start_date": train_dates.iloc[skipped_start_pos] if skipped_start_pos < len(train_dates) else "",
                "skipped_end_date": train_dates.iloc[-1] if not train_dates.empty else "",
                "skipped_days_count": min(TRAIN_TERMINAL_SKIP_DAYS, len(train_dates)),
                "note": "Final dates stay in df only as future labels; agent does not train actions from those dates.",
            }
        ]
    )
    guard_path = OUTPUT_DIR / "train_future_window_guard.csv"
    guard_df.to_csv(guard_path, index=False, encoding="utf-8-sig")
    print("Future-reward train guard enabled:")
    print(f"  Last train action date: {guard_df.loc[0, 'last_train_action_date']}")
    print(f"  Skipped action dates   : {guard_df.loc[0, 'skipped_start_date']} ~ {guard_df.loc[0, 'skipped_end_date']}")
    print(f"  Guard file             : {guard_path}")

stock_dimension = len(train.tic.unique())
state_space = config.get_stock_trading_state_space(stock_dimension)
config.print_observation_feature_mode(stock_dimension)

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
    "additional_raw_features": ADDITIONAL_RAW_FEATURES,
    "action_space": stock_dimension,
    "reward_scaling": REWARD_SCALING,
    "terminal_skip_days": TRAIN_TERMINAL_SKIP_DAYS,
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
if_using_reinforce = env_bool("IF_USING_REINFORCE", False)
if_using_td3 = env_bool("IF_USING_TD3", False)
if_using_sac = env_bool("IF_USING_SAC", False)

A2C_TOTAL_TIMESTEPS = env_int("A2C_TOTAL_TIMESTEPS", 20000)
DDPG_TOTAL_TIMESTEPS = env_int("DDPG_TOTAL_TIMESTEPS", 20000)
PPO_TOTAL_TIMESTEPS = env_int("PPO_TOTAL_TIMESTEPS", 100000)
REINFORCE_TOTAL_TIMESTEPS = env_int("REINFORCE_TOTAL_TIMESTEPS", 100000)
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
    save_sb3_training_reward_curve("a2c")
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
    save_sb3_training_reward_curve("ddpg")
    evaluate_on_train_and_save_reward_curve(trained_ddpg, "ddpg")

# --- Agent 3: PPO ---
agent = DRLAgent(env=env_train)
PPO_PARAMS = {
    "n_steps": 1000,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 100,
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
    save_sb3_training_reward_curve("ppo")
    evaluate_on_train_and_save_reward_curve(trained_ppo, "ppo")

# --- Agent 3b: REINFORCE / Policy Gradient ---
trained_reinforce = None
if if_using_reinforce:
    reinforce_progress_path = Path(RESULTS_DIR) / "reinforce" / "progress.csv"

    # Use a separate environment so deterministic evaluation does not change
    # the training environment state or its reward-v4 diagnostics.
    reinforce_eval_gym = StockTradingEnv(
        df=train,
        reward_type=REWARD_TYPE,
        reward_params=REWARD_PARAMS,
        **env_kwargs,
    )
    reinforce_eval_env, _ = reinforce_eval_gym.get_sb_env()
    reinforce_eval_env.seed(RANDOM_SEED)

    model_reinforce = ReinforceAgent(
        state_dim=state_space,
        action_dim=stock_dimension,
        hidden_dim=env_int("REINFORCE_HIDDEN_DIM", 256),
        learning_rate=float(os.environ.get("REINFORCE_LEARNING_RATE", "0.0003")),
        gamma=float(os.environ.get("REINFORCE_GAMMA", "0.99")),
        ent_coef=float(os.environ.get("REINFORCE_ENT_COEF", "0.001")),
        max_grad_norm=float(os.environ.get("REINFORCE_MAX_GRAD_NORM", "0.5")),
        seed=RANDOM_SEED,
    )
    trained_reinforce = model_reinforce.learn(
        env=env_train,
        total_timesteps=REINFORCE_TOTAL_TIMESTEPS,
        progress_path=reinforce_progress_path,
        eval_env=reinforce_eval_env,
        eval_interval=REINFORCE_EVAL_INTERVAL,
    )
    trained_reinforce.save(TRAINED_MODEL_DIR + "/agent_reinforce")
    save_reinforce_diagnostics()
    save_sb3_training_reward_curve("reinforce")
    save_reinforce_individual_stock_train_reward_curves(trained_reinforce)
    save_reinforce_deterministic_evaluation_curve()
    evaluate_on_train_and_save_reward_curve(trained_reinforce, "reinforce")

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
    save_sb3_training_reward_curve("td3")
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
    save_sb3_training_reward_curve("sac")
    evaluate_on_train_and_save_reward_curve(trained_sac, "sac")

print("All agents trained and saved to", TRAINED_MODEL_DIR)
