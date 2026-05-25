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

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl import config
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from result_manager import make_experiment_dir, save_config
from experiment_settings import env_bool, env_json, env_str

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
RANDOM_SEED = env_str("RANDOM_SEED", "")

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
print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
print(f"  TRAIN_DATA_PATH: {TRAIN_DATA_PATH}")
print(f"  TRADE_DATA_PATH: {TRADE_DATA_PATH}")
print(f"  TRAINED_MODEL_DIR: {TRAINED_MODEL_DIR}")

save_config(
    OUTPUT_DIR,
    {
        "reward_type": REWARD_TYPE,
        "reward_params": REWARD_PARAMS,
        "trade_start_date": TRADE_START_DATE,
        "trade_end_date": TRADE_END_DATE,
        "indicators": INDICATORS,
        "trained_model_dir": str(TRAINED_MODEL_DIR),
        "train_data_path": str(TRAIN_DATA_PATH),
        "trade_data_path": str(TRADE_DATA_PATH),
        "window_id": WINDOW_ID,
        "repeat_index": REPEAT_INDEX,
        "random_seed": RANDOM_SEED,
    },
)

today_str = datetime.now().strftime("%m%d")

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
if_using_td3 = env_bool("IF_USING_TD3", False)
if_using_sac = env_bool("IF_USING_SAC", False)

trained_a2c = A2C.load(TRAINED_MODEL_DIR / "agent_a2c") if if_using_a2c else None
trained_ddpg = DDPG.load(TRAINED_MODEL_DIR / "agent_ddpg") if if_using_ddpg else None
trained_ppo = PPO.load(TRAINED_MODEL_DIR / "agent_ppo") if if_using_ppo else None
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
    "reward_scaling": 1e-4,
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
        return None, None, None, None

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
        action, _states = model.predict(test_obs, deterministic=True)
        test_obs, rewards_step, dones, info = test_env.step(action)
        if dones[0] or i >= max_steps - 1:
            break

    account_value = test_env.env_method(method_name="save_asset_memory")[0]
    actions = test_env.env_method(method_name="save_action_memory")[0]
    holdings = test_env.env_method(method_name="save_holdings_memory")[0]
    rewards = test_env.env_method(method_name="save_reward_memory")[0]

    save_agent_diagnostics(agent_name, account_value, actions, holdings, rewards)
    return account_value, actions, holdings, rewards


def save_agent_diagnostics(
    agent_name: str,
    account_value: pd.DataFrame,
    actions: pd.DataFrame,
    holdings: pd.DataFrame,
    rewards: pd.DataFrame,
) -> None:
    suffix = f"{agent_name}_{REWARD_TYPE}_{TRADE_START_DATE}_to_{TRADE_END_DATE}_{today_str}"

    actions_path = OUTPUT_DIR / f"actions_{suffix}.csv"
    actions.to_csv(actions_path, index=True, encoding="utf-8-sig")

    holdings_path = OUTPUT_DIR / f"holdings_{suffix}.csv"
    holdings.to_csv(holdings_path, index=False, encoding="utf-8-sig")

    # Backtest stage is for trade-period actions, holdings, and portfolio value.
    # Reward curve is saved in FinRL_StockTrading_2026_2_train.py, because
    # reward analysis should correspond to the training window.
    plot_holdings_heatmap(holdings, OUTPUT_DIR / f"holdings_heatmap_{suffix}.png", agent_name)

    print(f"Actions saved to {actions_path}")
    print(f"Holdings saved to {holdings_path}")
    print("Backtest reward curve is intentionally not saved here; see train_reward_curve_* from the training script.")


def plot_holdings_heatmap(holdings: pd.DataFrame, out_path: Path, agent_name: str) -> None:
    """Plot cash and stock holdings using two independent, readable color scales.

    Cash is a dollar amount, while stock holdings are share counts.
    They must not share one color scale; otherwise cash dominates the visualization.
    Layout:
      - top thin band: cash balance
      - middle horizontal colorbar: cash scale
      - bottom block: stock share holdings
      - right vertical colorbar: stock share scale
    """
    if holdings.empty:
        return
    if "date" not in holdings.columns:
        holdings = holdings.copy()
        holdings.insert(0, "date", np.arange(len(holdings)))

    dates = holdings["date"].astype(str).tolist()
    cash = holdings[["cash"]].T.astype(float) if "cash" in holdings.columns else pd.DataFrame()
    stocks = holdings.drop(columns=["date", "cash"], errors="ignore").T.astype(float)

    max_xticks = 12
    tick_positions = (
        np.linspace(0, len(dates) - 1, min(max_xticks, len(dates))).astype(int)
        if len(dates) > 0
        else np.array([], dtype=int)
    )

    def money_formatter(x, pos):
        abs_x = abs(x)
        if abs_x >= 1_000_000:
            return f"{x / 1_000_000:.2f}M"
        if abs_x >= 1_000:
            return f"{x / 1_000:.0f}K"
        return f"{x:.0f}"

    fig_height = max(8, 0.23 * max(len(stocks), 1) + 2.8)
    fig = plt.figure(figsize=(18, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[0.55, 0.22, max(5.5, 0.23 * max(len(stocks), 1))],
        width_ratios=[40, 1.0],
        hspace=0.08,
        wspace=0.03,
    )

    ax_cash = fig.add_subplot(gs[0, 0])
    cax_cash = fig.add_subplot(gs[1, 0])
    ax_stock = fig.add_subplot(gs[2, 0], sharex=ax_cash)
    cax_stock = fig.add_subplot(gs[2, 1])
    fig.add_subplot(gs[0, 1]).axis("off")
    fig.add_subplot(gs[1, 1]).axis("off")

    if not cash.empty:
        cash_img = ax_cash.imshow(cash.values, aspect="auto", interpolation="nearest", cmap="YlOrBr")
        ax_cash.set_yticks([0])
        ax_cash.set_yticklabels(["cash"])
        ax_cash.set_ylabel("Cash", labelpad=12)
        cbar_cash = fig.colorbar(cash_img, cax=cax_cash, orientation="horizontal")
        cbar_cash.set_label("Cash balance", labelpad=4)
        cbar_cash.ax.xaxis.set_major_formatter(FuncFormatter(money_formatter))
        cbar_cash.ax.tick_params(labelsize=9, pad=2)
    else:
        ax_cash.axis("off")
        cax_cash.axis("off")

    if not stocks.empty:
        stock_img = ax_stock.imshow(stocks.values, aspect="auto", interpolation="nearest", cmap="Blues")
        ax_stock.set_yticks(np.arange(len(stocks.index)))
        ax_stock.set_yticklabels(stocks.index)
        ax_stock.set_ylabel("Stocks")
        cbar_stock = fig.colorbar(stock_img, cax=cax_stock)
        cbar_stock.set_label("Number of shares", labelpad=10)
        cbar_stock.ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        cbar_stock.ax.tick_params(labelsize=9)
    else:
        cax_stock.axis("off")
        ax_stock.text(0.5, 0.5, "No stock holdings data", ha="center", va="center")
        ax_stock.set_yticks([])

    if len(tick_positions) > 0:
        ax_stock.set_xticks(tick_positions)
        ax_stock.set_xticklabels([dates[i] for i in tick_positions], rotation=45, ha="right")
    plt.setp(ax_cash.get_xticklabels(), visible=False)
    ax_cash.tick_params(axis="x", length=0)

    ax_cash.set_title(f"Holdings Heatmap - {agent_name.upper()} ({REWARD_TYPE})", pad=8)
    ax_stock.set_xlabel("Date")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_reward_curve(rewards: pd.DataFrame, out_path: Path, agent_name: str) -> None:
    if rewards.empty:
        return
    plt.figure(figsize=(15, 5))
    x = rewards["date"].astype(str) if "date" in rewards.columns else np.arange(len(rewards))
    y = rewards["cumulative_reward_unscaled"]
    plt.plot(x, y)
    max_xticks = 12
    if len(rewards) > 0:
        tick_positions = np.linspace(0, len(rewards) - 1, min(max_xticks, len(rewards))).astype(int)
        plt.xticks(tick_positions, [str(x.iloc[i]) for i in tick_positions], rotation=45, ha="right")
    plt.title(f"Cumulative Reward Curve - {agent_name.upper()} ({REWARD_TYPE})")
    plt.xlabel("Date")
    plt.ylabel("Cumulative reward, unscaled")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# Run predictions. Each agent gets a fresh environment to avoid memory contamination.
df_account_value_a2c, df_actions_a2c, _, _ = predict_and_collect(trained_a2c, "a2c") if if_using_a2c else (None, None, None, None)
df_account_value_ddpg, df_actions_ddpg, _, _ = predict_and_collect(trained_ddpg, "ddpg") if if_using_ddpg else (None, None, None, None)
df_account_value_ppo, df_actions_ppo, df_holdings_ppo, df_rewards_ppo = predict_and_collect(trained_ppo, "ppo") if if_using_ppo else (None, None, None, None)
df_account_value_td3, df_actions_td3, _, _ = predict_and_collect(trained_td3, "td3") if if_using_td3 else (None, None, None, None)
df_account_value_sac, df_actions_sac, _, _ = predict_and_collect(trained_sac, "sac") if if_using_sac else (None, None, None, None)

# %% Part 4. Mean Variance Optimization baseline


def process_df_for_mvo(df):
    return df.pivot(index="date", columns="tic", values="close")


def StockReturnsComputing(StockPrice, Rows, Columns):
    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):
        for i in range(Rows - 1):
            StockReturn[i, j] = ((StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]) * 100
    return StockReturn


StockData = process_df_for_mvo(train)
TradeData = process_df_for_mvo(trade)

arStockPrices = np.asarray(StockData)
[Rows, Cols] = arStockPrices.shape
arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

meanReturns = np.mean(arReturns, axis=0)
covReturns = np.cov(arReturns, rowvar=False)

np.set_printoptions(precision=3, suppress=True)
print("Mean returns of assets in portfolio\n", meanReturns)

try:
    from pypfopt.efficient_frontier import EfficientFrontier

    ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
    raw_weights_mean = ef_mean.max_sharpe()
    cleaned_weights_mean = ef_mean.clean_weights()
    mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))])
    LastPrice = np.array([1 / p for p in StockData.tail(1).to_numpy()[0]])
    Initial_Portfolio = np.multiply(mvo_weights, LastPrice)
    Portfolio_Assets = TradeData @ Initial_Portfolio
    MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])
except Exception as exc:
    print(f"MVO baseline failed: {exc}")
    MVO_result = pd.DataFrame(index=TradeData.index, data={"Mean Var": np.nan})

# %% Part 5. DJIA index baseline

try:
    import yfinance as yf

    df_dji = yf.download("^DJI", start=TRADE_START_DATE, end=TRADE_END_DATE)
    df_dji = df_dji[["Close"]].reset_index()
    df_dji.columns = ["date", "close"]
    df_dji["date"] = df_dji["date"].astype(str)
    fst_day = df_dji["close"].iloc[0]
    dji = pd.merge(
        df_dji["date"],
        df_dji["close"].div(fst_day).mul(1000000),
        how="outer",
        left_index=True,
        right_index=True,
    ).set_index("date")
except Exception as exc:
    print(f"DJI baseline failed: {exc}")
    dji = pd.DataFrame(index=TradeData.index, data={"close": np.nan})

# %% Part 6. Compare results


def set_account_index(df):
    return df.set_index(df.columns[0]) if df is not None else None


df_result_a2c = set_account_index(df_account_value_a2c)
df_result_ddpg = set_account_index(df_account_value_ddpg)
df_result_ppo = set_account_index(df_account_value_ppo)
df_result_td3 = set_account_index(df_account_value_td3)
df_result_sac = set_account_index(df_account_value_sac)

result = pd.DataFrame(
    {
        "a2c": df_result_a2c["account_value"] if if_using_a2c else None,
        "ddpg": df_result_ddpg["account_value"] if if_using_ddpg else None,
        "ppo": df_result_ppo["account_value"] if if_using_ppo else None,
        "td3": df_result_td3["account_value"] if if_using_td3 else None,
        "sac": df_result_sac["account_value"] if if_using_sac else None,
        "mvo": MVO_result["Mean Var"],
        "dji": dji["close"],
    }
)

print("\n=== Backtest Results ===")
print(result)

backtest_csv_path = OUTPUT_DIR / f"backtest_result_{REWARD_TYPE}_{TRADE_START_DATE}_to_{TRADE_END_DATE}_{today_str}.csv"
result.to_csv(backtest_csv_path, index=True, encoding="utf-8-sig")
print(f"\nBacktest results saved to {backtest_csv_path}")

# %% Part 7. Plot portfolio value

plt.rcParams["figure.figsize"] = (15, 5)
plt.figure()
result.plot()
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plot_path = OUTPUT_DIR / f"backtest_result_{REWARD_TYPE}_{TRADE_START_DATE}_to_{TRADE_END_DATE}_{today_str}.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {plot_path}")

# %% Part 8. Backtest summary for rolling aggregation

summary = {
    "reward_type": REWARD_TYPE,
    "trade_start_date": TRADE_START_DATE,
    "trade_end_date": TRADE_END_DATE,
    "window_id": WINDOW_ID,
    "repeat_index": REPEAT_INDEX,
    "random_seed": RANDOM_SEED,
}

if if_using_ppo and "ppo" in result.columns and result["ppo"].notna().any():
    ppo = result["ppo"].dropna()
    summary["initial_ppo"] = float(ppo.iloc[0])
    summary["final_ppo"] = float(ppo.iloc[-1])
    summary["ppo_return"] = float(ppo.iloc[-1] / ppo.iloc[0] - 1)
    running_max = ppo.cummax()
    drawdown = ppo / running_max - 1
    summary["ppo_max_drawdown"] = float(drawdown.min())

if "mvo" in result.columns and result["mvo"].notna().any():
    summary["final_mvo"] = float(result["mvo"].dropna().iloc[-1])
if "dji" in result.columns and result["dji"].notna().any():
    summary["final_dji"] = float(result["dji"].dropna().iloc[-1])

summary_path = OUTPUT_DIR / f"backtest_summary_{REWARD_TYPE}_{TRADE_START_DATE}_to_{TRADE_END_DATE}_{today_str}.csv"
pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")
print(f"Backtest summary saved to {summary_path}")
