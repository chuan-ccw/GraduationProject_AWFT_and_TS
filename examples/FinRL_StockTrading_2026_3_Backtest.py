"""
Stock NeurIPS2018 Part 3. Backtest

Backtest trained FinRL agents and save results under the current experiment folder.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from matplotlib.ticker import FuncFormatter, MaxNLocator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.utils import set_random_seed

from finrl.agents.policy_gradient import ReinforceAgent
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl import config
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from result_manager import make_experiment_dir, save_config
from experiment_settings import env_bool, env_int, env_json, env_str

# Runtime settings: Run_StockTrading_Examples.py can override config.py through environment variables.
REWARD_TYPE = env_str("REWARD_TYPE", getattr(config, "REWARD_TYPE", "v0"))
REWARD_PARAMS = env_json("REWARD_PARAMS", getattr(config, "REWARD_PARAMS", {}))
TRAINED_MODEL_DIR = Path(env_str("TRAINED_MODEL_DIR", config.TRAINED_MODEL_DIR))
TRADE_START_DATE = env_str("TRADE_START_DATE", config.TRADE_START_DATE)
TRADE_END_DATE = env_str("TRADE_END_DATE", config.TRADE_END_DATE)
EXPERIMENT_BASE_DIR = env_str(
    "EXPERIMENT_ROOT",
    getattr(config, "EXPERIMENT_ROOT", r"C:\Users\User\Desktop\GraduationProject\Experence_result"),
)
EXPERIMENT_DIR = env_str("EXPERIMENT_DIR", None)
TRAIN_DATA_PATH = Path(env_str("TRAIN_DATA_PATH", "train_data.csv"))
TRADE_DATA_PATH = Path(env_str("TRADE_DATA_PATH", "trade_data.csv"))
WINDOW_ID = env_str("WINDOW_ID", "")
REPEAT_INDEX = env_str("REPEAT_INDEX", "")
RANDOM_SEED = env_int("RANDOM_SEED", 0)
PREDICT_DETERMINISTIC = env_bool("PREDICT_DETERMINISTIC", False)
PRIMARY_AGENT_NAME = env_str("PRIMARY_AGENT_NAME", "ppo").strip().lower()

# Backtest uses stochastic PPO actions by default so repeated tests of the same model
# can measure policy-sampling variation. RANDOM_SEED keeps each repeat reproducible.
set_random_seed(RANDOM_SEED)


def get_reward_scaling(reward_type: str) -> float:
    """
    Choose reward_scaling by reward definition.

    v0 is a money-difference reward, so it keeps FinRL's original 1e-4.
    v1/v2/v3/v4/v5/v6 are return-level rewards, so they should not be shrunk by 1e-4.
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
        "v6",
        "reward_v6",
        "delta_weight_future",
        "reward_delta_weight_future",
        "delta_weight_future_reward",
        "portfolio_weight_change_future_return",
        "portfolio_weight_change_future_return_reward",
    }:
        return 1.0

    return 1.0


REWARD_SCALING = get_reward_scaling(REWARD_TYPE)

REWARD_DISPLAY_NAMES = {
    "v0": "original_reward",
    "original": "original_reward",
    "finrl_original": "original_reward",
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
    "v5": "portfolio_weighted_future_return_reward",
    "reward_v5": "portfolio_weighted_future_return_reward",
    "weight_future": "portfolio_weighted_future_return_reward",
    "reward_weight_future": "portfolio_weighted_future_return_reward",
    "weight_future_reward": "portfolio_weighted_future_return_reward",
    "portfolio_weighted_future_return": "portfolio_weighted_future_return_reward",
    "portfolio_weighted_future_return_reward": "portfolio_weighted_future_return_reward",
    "v6": "portfolio_weight_change_future_return_reward",
    "reward_v6": "portfolio_weight_change_future_return_reward",
    "delta_weight_future": "portfolio_weight_change_future_return_reward",
    "reward_delta_weight_future": "portfolio_weight_change_future_return_reward",
    "delta_weight_future_reward": "portfolio_weight_change_future_return_reward",
    "portfolio_weight_change_future_return": "portfolio_weight_change_future_return_reward",
    "portfolio_weight_change_future_return_reward": "portfolio_weight_change_future_return_reward",
}


def reward_display_name(reward_type: str | None) -> str:
    """Return the formal display name used in plots instead of v0/v1/v2/v3/v4/v5/v6."""
    key = str(reward_type or "").strip().lower()
    if key.startswith("reward_") and key not in REWARD_DISPLAY_NAMES:
        key = key[len("reward_"):]
    return REWARD_DISPLAY_NAMES.get(key, str(reward_type or ""))


def set_axis_labels_horizontal(ax, xlabel: str | None = None, ylabel: str | None = None, ylabel_pad: int = 65) -> None:
    """Keep x-axis label horizontal, but keep y-axis label in the default vertical style."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, rotation=0)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

# %% Part 0. Create output folder

if EXPERIMENT_DIR:
    OUTPUT_DIR = Path(EXPERIMENT_DIR).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
else:
    OUTPUT_DIR = make_experiment_dir(
        base_dir=EXPERIMENT_BASE_DIR,
        reward_type=REWARD_TYPE,
        trade_start_date=TRADE_START_DATE,
        trade_end_date=TRADE_END_DATE,
    )

print("Backtest setting:")
print(f"  REWARD_TYPE: {REWARD_TYPE}")
print(f"  REWARD_PARAMS: {REWARD_PARAMS}")
print(f"  REWARD_SCALING: {REWARD_SCALING}")
print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
print(f"  TRAIN_DATA_PATH: {TRAIN_DATA_PATH}")
print(f"  TRADE_DATA_PATH: {TRADE_DATA_PATH}")
print(f"  TRAINED_MODEL_DIR: {TRAINED_MODEL_DIR}")
print(f"  RANDOM_SEED: {RANDOM_SEED}")
print(f"  PREDICT_DETERMINISTIC: {PREDICT_DETERMINISTIC}")

save_config(
    OUTPUT_DIR,
    {
        "reward_type": REWARD_TYPE,
        "reward_params": REWARD_PARAMS,
        "reward_scaling": REWARD_SCALING,
        "trade_start_date": TRADE_START_DATE,
        "trade_end_date": TRADE_END_DATE,
        "indicators": INDICATORS,
        "trained_model_dir": str(TRAINED_MODEL_DIR),
        "train_data_path": str(TRAIN_DATA_PATH),
        "trade_data_path": str(TRADE_DATA_PATH),
        "window_id": WINDOW_ID,
        "repeat_index": REPEAT_INDEX,
        "random_seed": RANDOM_SEED,
        "predict_deterministic": PREDICT_DETERMINISTIC,
    },
)

today_str = datetime.now().strftime("%m%d")


def save_image_source_manifest(rows: list[dict[str, object]]) -> None:
    """Write which data/model source each backtest image used."""
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


def format_yyyy_mm(value) -> str:
    """Format any date-like value as YYYY-MM for plot x-axis labels.

    This avoids matplotlib/pandas auto labels such as YYYY/MM/DD or full YYYY-MM-DD.
    If the value is not date-like, keep the original text.
    """
    ts = pd.to_datetime(value, errors="coerce")
    if pd.notna(ts):
        return ts.strftime("%Y-%m")
    text = str(value)
    # Common fallback for strings such as 2025/1/03 or 2025-01-03.
    match = __import__("re").search(r"(\d{4})[-/](\d{1,2})", text)
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"
    return text


def make_month_tick_positions(length: int, max_xticks: int = 12) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=int)
    return np.linspace(0, length - 1, min(max_xticks, length)).astype(int)


def set_month_xticklabels(ax, dates, max_xticks: int = 12) -> None:
    """Use YYYY-MM labels on an axis whose x values are 0..N-1."""
    positions = make_month_tick_positions(len(dates), max_xticks=max_xticks)
    if len(positions) == 0:
        return
    labels = [format_yyyy_mm(dates[i]) for i in positions]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha="center")


def get_trade_dates(trade_df: pd.DataFrame) -> pd.Index:
    """Return real trading dates from the trade dataframe, not the numeric day index."""
    if "date" in trade_df.columns:
        raw_dates = pd.Series(trade_df["date"].astype(str)).drop_duplicates()
    else:
        raw_dates = pd.Series(pd.Index(trade_df.index.unique()).astype(str))

    parsed = pd.to_datetime(raw_dates, errors="coerce")
    valid = parsed.notna()
    return pd.Index(parsed[valid].dt.strftime("%Y-%m-%d"), name="date")


def align_account_value_to_dates(account_value: pd.DataFrame | None, trade_dates: pd.Index) -> pd.Series | None:
    """Align one agent's account_value output to the real trade-date index."""
    if account_value is None or account_value.empty or "account_value" not in account_value.columns:
        return None

    df = account_value.copy()
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"].astype(str), errors="coerce")
        if parsed.notna().any():
            df["date"] = parsed.dt.strftime("%Y-%m-%d")
            series = pd.to_numeric(df.set_index("date")["account_value"], errors="coerce")
            return series.reindex(trade_dates)

    # Fallback for unexpected numeric date output: keep sequence order, but use real trade dates.
    values = pd.to_numeric(df["account_value"], errors="coerce").reset_index(drop=True)
    length = min(len(values), len(trade_dates))
    return pd.Series(values.iloc[:length].to_numpy(), index=trade_dates[:length])


def build_dji_baseline(trade_dates, initial_amount: float = 1_000_000.0) -> pd.Series:
    """Download and align DJI close prices to the provided trade dates."""
    trade_index = pd.Index([str(x)[:10] for x in trade_dates], name="date")
    if len(trade_index) == 0:
        return pd.Series(dtype=float, name="dji")

    try:
        import yfinance as yf

        start_ts = pd.to_datetime(trade_index[0], errors="coerce") - pd.Timedelta(days=7)
        end_ts = pd.to_datetime(trade_index[-1], errors="coerce") + pd.Timedelta(days=7)
        df_dji = yf.download(
            "^DJI",
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )
        if df_dji is None or df_dji.empty:
            raise ValueError("empty yfinance result")

        close = None
        if isinstance(df_dji.columns, pd.MultiIndex):
            for candidate in ("Close", "Adj Close"):
                level0 = [str(x) for x in df_dji.columns.get_level_values(0)]
                if candidate in level0:
                    sub = df_dji.xs(candidate, axis=1, level=0)
                    close = sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
                    break
        else:
            for candidate in ["Close", "Adj Close", "close", "adj close"]:
                if candidate in df_dji.columns:
                    close = df_dji[candidate]
                    break

        if close is None:
            raise ValueError(f"cannot find Close column, columns={list(df_dji.columns)}")

        close = pd.to_numeric(close, errors="coerce").dropna()
        if close.empty:
            raise ValueError("DJI close series is empty after numeric conversion")

        close.index = pd.Index(pd.to_datetime(close.index, errors="coerce").strftime("%Y-%m-%d"), name="date")
        close = close[~close.index.duplicated(keep="last")]

        aligned = close.reindex(trade_index).ffill().bfill()
        valid = aligned.dropna()
        if valid.empty:
            raise ValueError("DJI series is empty after reindex to trade dates")

        normalized = aligned / float(valid.iloc[0]) * float(initial_amount)
        normalized.name = "dji"
        return normalized
    except Exception as exc:
        print(f"DJI baseline failed: {exc}")
        return pd.Series(index=trade_index, data=np.nan, name="dji")


# %% Part 1. Load data

train = pd.read_csv(TRAIN_DATA_PATH)
trade = pd.read_csv(TRADE_DATA_PATH)

train = train.set_index(train.columns[0])
train.index.names = [""]
trade = trade.set_index(trade.columns[0])
trade.index.names = [""]

# %% Part 2. Load trained agents

if_using_a2c = env_bool("IF_USING_A2C", False)
if_using_ddpg = env_bool("IF_USING_DDPG", False)
if_using_ppo = env_bool("IF_USING_PPO", True)
if_using_reinforce = env_bool("IF_USING_REINFORCE", False)
if_using_td3 = env_bool("IF_USING_TD3", False)
if_using_sac = env_bool("IF_USING_SAC", False)

trained_a2c = A2C.load(TRAINED_MODEL_DIR / "agent_a2c") if if_using_a2c else None
trained_ddpg = DDPG.load(TRAINED_MODEL_DIR / "agent_ddpg") if if_using_ddpg else None
trained_ppo = PPO.load(TRAINED_MODEL_DIR / "agent_ppo") if if_using_ppo else None
trained_reinforce = ReinforceAgent.load(TRAINED_MODEL_DIR / "agent_reinforce") if if_using_reinforce else None
trained_td3 = TD3.load(TRAINED_MODEL_DIR / "agent_td3") if if_using_td3 else None
trained_sac = SAC.load(TRAINED_MODEL_DIR / "agent_sac") if if_using_sac else None

# %% Part 3. Backtesting - DRL agents

stock_dimension = len(trade.tic.unique())
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
    "reward_scaling": REWARD_SCALING,
}


def make_trade_env() -> StockTradingEnv:
    return StockTradingEnv(
        df=trade,
        turbulence_threshold=70,
        risk_indicator_col="vix",
        reward_type=REWARD_TYPE,
        reward_params=REWARD_PARAMS,
        **env_kwargs,
    )


def predict_and_collect(model, agent_name: str):
    if model is None:
        return None, None, None, None, None

    env = make_trade_env()

    # IMPORTANT:
    # DRLAgent.DRL_prediction() only returns account_value and actions from the DummyVecEnv.
    # If we call env.save_holdings_memory() after prediction, it may still contain only
    # the initial snapshot, which makes the holdings heatmap show only one date.
    # Therefore, collect all diagnostic memories from the same VecEnv used for prediction.
    test_env, test_obs = env.get_sb_env()
    test_env.reset()
    max_steps = len(env.df.index.unique()) - 1

    for i in range(len(env.df.index.unique())):
        action, _states = model.predict(test_obs, deterministic=PREDICT_DETERMINISTIC)
        test_obs, rewards_step, dones, info = test_env.step(action)
        if dones[0] or i >= max_steps - 1:
            break

    account_value = test_env.env_method(method_name="save_asset_memory")[0]
    actions = test_env.env_method(method_name="save_action_memory")[0]
    holdings = test_env.env_method(method_name="save_holdings_memory")[0]
    weights = test_env.env_method(method_name="save_weight_memory")[0]
    rewards = test_env.env_method(method_name="save_reward_memory")[0]

    save_agent_diagnostics(
        agent_name,
        account_value,
        actions,
        holdings,
        weights,
        rewards,
    )
    return account_value, actions, holdings, weights, rewards


def save_agent_diagnostics(
    agent_name: str,
    account_value: pd.DataFrame,
    actions: pd.DataFrame,
    holdings: pd.DataFrame,
    weights: pd.DataFrame,
    rewards: pd.DataFrame,
) -> None:
    suffix = f"{agent_name}_{REWARD_TYPE}_{TRADE_START_DATE}_to_{TRADE_END_DATE}_{today_str}"

    actions_path = OUTPUT_DIR / f"actions_{suffix}.csv"
    actions.to_csv(actions_path, index=True, encoding="utf-8-sig")

    holdings_path = OUTPUT_DIR / f"holdings_{suffix}.csv"
    holdings.to_csv(holdings_path, index=False, encoding="utf-8-sig")

    weights_path = OUTPUT_DIR / f"post_trade_weights_{suffix}.csv"
    weights.to_csv(weights_path, index=False, encoding="utf-8-sig")

    # Backtest stage is for trade-period actions, holdings, and portfolio value.
    # Reward curve is saved in FinRL_StockTrading_2026_2_train.py, because
    # reward analysis should correspond to the training window.
    holdings_heatmap_path = OUTPUT_DIR / f"holdings_heatmap_{suffix}.png"
    plot_holdings_heatmap(holdings, holdings_heatmap_path, agent_name)
    source_rows = [
        {
            "image": holdings_heatmap_path.name,
            "image_path": str(holdings_heatmap_path),
            "source_role": "current_repeat_holdings_memory",
            "source_csv": str(holdings_path),
            "actions_csv": str(actions_path),
            "weights_csv": str(weights_path),
            "train_data": str(TRAIN_DATA_PATH),
            "trade_data": str(TRADE_DATA_PATH),
            "trained_model_dir": str(TRAINED_MODEL_DIR),
            "stage": "backtest_repeat",
            "reward_type": REWARD_TYPE,
            "window_id": WINDOW_ID,
            "repeat_index": REPEAT_INDEX,
            "random_seed": RANDOM_SEED,
            "predict_deterministic": PREDICT_DETERMINISTIC,
        }
    ]
    if agent_name.lower() == PRIMARY_AGENT_NAME and holdings_heatmap_path.exists():
        canonical_heatmap_path = OUTPUT_DIR / f"holdings_heatmap_{PRIMARY_AGENT_NAME}.png"
        canonical_heatmap_path.write_bytes(holdings_heatmap_path.read_bytes())
        source_rows.append(
            {
                "image": canonical_heatmap_path.name,
                "image_path": str(canonical_heatmap_path),
                "source_role": "current_repeat_holdings_memory",
                "source_csv": str(holdings_path),
                "actions_csv": str(actions_path),
                "weights_csv": str(weights_path),
                "train_data": str(TRAIN_DATA_PATH),
                "trade_data": str(TRADE_DATA_PATH),
                "trained_model_dir": str(TRAINED_MODEL_DIR),
                "stage": "backtest_repeat",
                "reward_type": REWARD_TYPE,
                "window_id": WINDOW_ID,
                "repeat_index": REPEAT_INDEX,
                "random_seed": RANDOM_SEED,
            }
        )
    save_image_source_manifest(source_rows)

    print(f"Actions saved to {actions_path}")
    print(f"Holdings saved to {holdings_path}")
    print(f"Post-trade weights saved to {weights_path}")
    print("Backtest reward curve is intentionally not saved here; see train_reward_curve_* from the training script.")


def plot_holdings_heatmap(holdings: pd.DataFrame, out_path: Path, agent_name: str) -> None:
    """Plot a cash heatmap with an overlaid cash line and a stock holdings heatmap.

    The cash section shows:
    1) a one-row cash heatmap labeled ``cash``
    2) a line above it that traces intraday cash movement using
       previous day close cash -> cash_after_sell -> cash_after_buy,
       so users can still see trades even if end-of-day cash is similar.
    """
    if holdings.empty:
        return
    if "date" not in holdings.columns:
        holdings = holdings.copy()
        holdings.insert(0, "date", np.arange(len(holdings)))

    dates = holdings["date"].tolist()

    cash = holdings[["cash"]].T.astype(float) if "cash" in holdings.columns else pd.DataFrame()
    cash_after_sell = holdings["cash_after_sell"].astype(float) if "cash_after_sell" in holdings.columns else None
    cash_after_buy = holdings["cash_after_buy"].astype(float) if "cash_after_buy" in holdings.columns else (holdings["cash"].astype(float) if "cash" in holdings.columns else None)

    stocks = holdings.drop(columns=["date", "cash", "cash_after_sell", "cash_after_buy"], errors="ignore").T.astype(float)

    max_xticks = 12
    tick_positions = make_month_tick_positions(len(dates), max_xticks=max_xticks)

    def money_formatter(x, pos):
        abs_x = abs(x)
        if abs_x >= 1_000_000:
            return f"{x / 1_000_000:.2f}M"
        if abs_x >= 1_000:
            return f"{x / 1_000:.0f}K"
        return f"{x:.0f}"

    stock_rows = max(len(stocks), 1)
    fig_height = max(8, 0.23 * stock_rows + 3.8)
    fig = plt.figure(figsize=(18, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[1.05, 0.7, max(5.5, 0.23 * stock_rows)],
        width_ratios=[40, 0.9],
        hspace=0.05,
        wspace=0.06,
    )

    ax_cash_line = fig.add_subplot(gs[0, 0])
    ax_cash = fig.add_subplot(gs[1, 0], sharex=ax_cash_line)
    cax_cash = fig.add_subplot(gs[0:2, 1])
    ax_stock = fig.add_subplot(gs[2, 0], sharex=ax_cash)
    cax_stock = fig.add_subplot(gs[2, 1])

    if not cash.empty:
        cash_img = ax_cash.imshow(
            cash.values,
            aspect="auto",
            interpolation="nearest",
            cmap="YlOrBr",
        )
        ax_cash.set_yticks([0])
        ax_cash.set_yticklabels(["cash"])
        ax_cash.set_ylabel("Cash")
        cbar_cash = fig.colorbar(cash_img, cax=cax_cash, orientation="vertical")
        cbar_cash.set_label("Cash balance")
        cbar_cash.ax.yaxis.set_major_formatter(FuncFormatter(money_formatter))
        cbar_cash.ax.tick_params(labelsize=9)

        if cash_after_buy is not None:
            line_x = [0.0]
            line_y = [float(cash_after_buy.iloc[0])]
            for i in range(1, len(holdings)):
                prev_end_cash = float(cash_after_buy.iloc[i - 1])
                sell_cash = float(cash_after_sell.iloc[i]) if cash_after_sell is not None else prev_end_cash
                buy_cash = float(cash_after_buy.iloc[i])
                line_x.extend([i - 0.28, i, i + 0.28])
                line_y.extend([prev_end_cash, sell_cash, buy_cash])

            ax_cash_line.plot(line_x, line_y, linewidth=2.6)
            ax_cash_line.set_ylabel("Cash")
            ax_cash_line.yaxis.set_major_formatter(FuncFormatter(money_formatter))
            ax_cash_line.grid(True, axis="y", alpha=0.25)
            ax_cash_line.set_xlim(-0.5, max(len(dates) - 0.5, 0.5))
            ax_cash_line.set_title(f"Holdings Heatmap - {agent_name.upper()} ({reward_display_name(REWARD_TYPE)})", pad=8)
            ax_cash_line.text(
                0.995,
                0.92,
                "Line: prev close cash → after sell → after buy",
                ha="right",
                va="top",
                transform=ax_cash_line.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
            )
        else:
            ax_cash_line.axis("off")
    else:
        ax_cash_line.axis("off")
        ax_cash.axis("off")
        cax_cash.axis("off")

    if not stocks.empty:
        stock_img = ax_stock.imshow(
            stocks.values,
            aspect="auto",
            interpolation="nearest",
            cmap="Blues",
        )
        ax_stock.set_yticks(np.arange(len(stocks.index)))
        ax_stock.set_yticklabels(stocks.index)
        ax_stock.set_ylabel("Stocks")
        cbar_stock = fig.colorbar(stock_img, cax=cax_stock, orientation="vertical")
        cbar_stock.set_label("Number of shares")
        cbar_stock.ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        cbar_stock.ax.tick_params(labelsize=9)
    else:
        cax_stock.axis("off")
        ax_stock.text(0.5, 0.5, "No stock holdings data", ha="center", va="center")
        ax_stock.set_yticks([])

    if len(tick_positions) > 0:
        ax_stock.set_xticks(tick_positions)
        ax_stock.set_xticklabels([format_yyyy_mm(dates[i]) for i in tick_positions], rotation=0, ha="center")
    plt.setp(ax_cash_line.get_xticklabels(), visible=False)
    plt.setp(ax_cash.get_xticklabels(), visible=False)
    ax_cash_line.tick_params(axis="x", length=0)
    ax_cash.tick_params(axis="x", length=0)
    ax_stock.set_xlabel("Date", rotation=0)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_reward_curve(rewards: pd.DataFrame, out_path: Path, agent_name: str) -> None:
    if rewards.empty:
        return
    plt.figure(figsize=(15, 5))
    dates = rewards["date"].tolist() if "date" in rewards.columns else list(range(len(rewards)))
    x = np.arange(len(rewards))
    y = rewards["cumulative_reward_unscaled"]
    plt.plot(x, y)
    max_xticks = 12
    if len(rewards) > 0:
        tick_positions = make_month_tick_positions(len(rewards), max_xticks=max_xticks)
        plt.xticks(tick_positions, [format_yyyy_mm(dates[i]) for i in tick_positions], ha="right")
    ax = plt.gca()
    ax.set_title(f"Cumulative Reward Curve - {agent_name.upper()} ({reward_display_name(REWARD_TYPE)})")
    set_axis_labels_horizontal(ax, "Date", "Cumulative reward, unscaled", ylabel_pad=80)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# Run predictions. Each agent gets a fresh environment to avoid memory contamination.
df_account_value_a2c, df_actions_a2c, _, _, _ = predict_and_collect(trained_a2c, "a2c") if if_using_a2c else (None, None, None, None, None)
df_account_value_ddpg, df_actions_ddpg, _, _, _ = predict_and_collect(trained_ddpg, "ddpg") if if_using_ddpg else (None, None, None, None, None)
df_account_value_ppo, df_actions_ppo, df_holdings_ppo, df_weights_ppo, df_rewards_ppo = predict_and_collect(trained_ppo, "ppo") if if_using_ppo else (None, None, None, None, None)
df_account_value_reinforce, df_actions_reinforce, df_holdings_reinforce, df_weights_reinforce, df_rewards_reinforce = predict_and_collect(trained_reinforce, "reinforce") if if_using_reinforce else (None, None, None, None, None)
df_account_value_td3, df_actions_td3, _, _, _ = predict_and_collect(trained_td3, "td3") if if_using_td3 else (None, None, None, None, None)
df_account_value_sac, df_actions_sac, _, _, _ = predict_and_collect(trained_sac, "sac") if if_using_sac else (None, None, None, None, None)

# %% Part 4. Baselines

# MVO baseline is intentionally removed from all future outputs.
# Keep DJI as the only market baseline for backtest_result CSV/PNG files.

# %% Part 5. DJIA index baseline

trade_dates = get_trade_dates(trade)
dji = build_dji_baseline(trade_dates, initial_amount=1_000_000.0).reindex(trade_dates).to_frame(name="dji")

# %% Part 6. Compare results


result = pd.DataFrame(index=trade_dates)
result.index.name = "date"

agent_outputs = {
    "a2c": (if_using_a2c, df_account_value_a2c),
    "ddpg": (if_using_ddpg, df_account_value_ddpg),
    "ppo": (if_using_ppo, df_account_value_ppo),
    "reinforce": (if_using_reinforce, df_account_value_reinforce),
    "td3": (if_using_td3, df_account_value_td3),
    "sac": (if_using_sac, df_account_value_sac),
}
for agent_name, (enabled, account_value) in agent_outputs.items():
    if enabled:
        result[agent_name] = align_account_value_to_dates(account_value, trade_dates)

result["dji"] = pd.to_numeric(dji["dji"], errors="coerce")

print("\n=== Backtest Results ===")
print(result)

backtest_csv_path = OUTPUT_DIR / f"backtest_result_{REWARD_TYPE}_{TRADE_START_DATE}_to_{TRADE_END_DATE}_{today_str}.csv"
result.to_csv(backtest_csv_path, index=True, encoding="utf-8-sig")
print(f"\nBacktest results saved to {backtest_csv_path}")

# %% Part 7. Plot portfolio value

fig, ax = plt.subplots(figsize=(15, 5))
x = np.arange(len(result))
for col in result.columns:
    series = pd.to_numeric(result[col], errors="coerce")
    if series.notna().any():
        ax.plot(x, series, label=col)
ax.set_title("Equity Curve")
set_axis_labels_horizontal(ax, "Date", "Asset Allocation ($)", ylabel_pad=65)
set_month_xticklabels(ax, list(result.index), max_xticks=12)
ax.legend()
fig.tight_layout()
plot_path = OUTPUT_DIR / f"backtest_result_{REWARD_TYPE}_{TRADE_START_DATE}_to_{TRADE_END_DATE}_{today_str}.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
save_image_source_manifest(
    [
        {
            "image": plot_path.name,
            "image_path": str(plot_path),
            "source_role": "current_repeat_backtest_result",
            "source_csv": str(backtest_csv_path),
            "train_data": str(TRAIN_DATA_PATH),
            "trade_data": str(TRADE_DATA_PATH),
            "trained_model_dir": str(TRAINED_MODEL_DIR),
            "stage": "backtest_repeat",
            "reward_type": REWARD_TYPE,
            "window_id": WINDOW_ID,
            "repeat_index": REPEAT_INDEX,
            "random_seed": RANDOM_SEED,
            "predict_deterministic": PREDICT_DETERMINISTIC,
        }
    ]
)
plt.close(fig)
print(f"\nPlot saved to {plot_path}")

# %% Part 8. Backtest summary for rolling aggregation

summary = {
    "reward_type": REWARD_TYPE,
    "trade_start_date": TRADE_START_DATE,
    "trade_end_date": TRADE_END_DATE,
    "window_id": WINDOW_ID,
    "repeat_index": REPEAT_INDEX,
    "random_seed": RANDOM_SEED,
    "predict_deterministic": PREDICT_DETERMINISTIC,
}

for agent_name in [c for c in result.columns if c != "dji"]:
    series = pd.to_numeric(result[agent_name], errors="coerce").dropna()
    if series.empty:
        continue
    summary[f"initial_{agent_name}"] = float(series.iloc[0])
    summary[f"final_{agent_name}"] = float(series.iloc[-1])
    summary[f"{agent_name}_return"] = float(series.iloc[-1] / series.iloc[0] - 1)
    running_max = series.cummax()
    drawdown = series / running_max - 1
    summary[f"{agent_name}_max_drawdown"] = float(drawdown.min())

if "dji" in result.columns and result["dji"].notna().any():
    summary["final_dji"] = float(result["dji"].dropna().iloc[-1])

summary_path = OUTPUT_DIR / f"backtest_summary_{REWARD_TYPE}_{TRADE_START_DATE}_to_{TRADE_END_DATE}_{today_str}.csv"
pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")
print(f"Backtest summary saved to {summary_path}")
