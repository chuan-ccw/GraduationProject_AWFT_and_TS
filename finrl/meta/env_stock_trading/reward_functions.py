from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class RewardContext:
    """Information needed by reward functions for one environment step."""

    env: Any
    begin_total_asset: float
    end_total_asset: float
    raw_actions: np.ndarray
    current_prices_before_trade: np.ndarray
    cost_before_trade: float
    current_day: int
    tic_list: list


def reward_v0(ctx: RewardContext, params: Dict[str, Any] | None = None) -> float:
    """
    Original FinRL reward.

    Physical meaning:
        reward = portfolio value after trade - portfolio value before trade
    """
    return ctx.end_total_asset - ctx.begin_total_asset


def reward_v1(ctx: RewardContext, params: Dict[str, Any] | None = None) -> float:
    """
    Jason reward v1.

    reward = simple_return
             + alpha * buy_low_score
             + beta  * sell_high_score

    The returned value is the unscaled reward. env_stocktrading.py will multiply
    it by env.reward_scaling later, exactly like the original FinRL flow.
    """
    params = params or {}

    eps = params.get("eps", 1e-8)
    alpha = params.get("alpha", 0.05)
    beta = params.get("beta", 0.05)
    n_recent = params.get("n_recent", 14)

    simple_return = (ctx.end_total_asset - ctx.begin_total_asset) / (
        ctx.begin_total_asset + eps
    )

    buy_low_score_total = 0.0
    sell_high_score_total = 0.0

    start_day = max(0, ctx.current_day - n_recent + 1)

    for i, action in enumerate(ctx.raw_actions):
        price = ctx.current_prices_before_trade[i]
        tic = ctx.tic_list[i]

        recent_data = ctx.env.df.loc[start_day: ctx.current_day]
        recent_data = recent_data[recent_data["tic"] == tic]

        if recent_data.empty:
            continue

        if "high" in recent_data.columns and "low" in recent_data.columns:
            recent_high = recent_data["high"].max()
            recent_low = recent_data["low"].min()
        else:
            recent_high = recent_data["close"].max()
            recent_low = recent_data["close"].min()

        price_range = recent_high - recent_low + eps

        buy_low_score = (recent_high - price) / price_range
        sell_high_score = (price - recent_low) / price_range

        buy_low_score = np.clip(buy_low_score, 0, 1)
        sell_high_score = np.clip(sell_high_score, 0, 1)

        action_strength = abs(action) / (ctx.env.hmax + eps)

        if action > 0:
            buy_low_score_total += buy_low_score * action_strength
        elif action < 0:
            sell_high_score_total += sell_high_score * action_strength

    buy_low_score_total /= ctx.env.stock_dim
    sell_high_score_total /= ctx.env.stock_dim

    return simple_return + alpha * buy_low_score_total + beta * sell_high_score_total


def _action_to_unit_array(actions: np.ndarray, hmax: float, eps: float = 1e-8) -> np.ndarray:
    """
    Convert actions to [-1, 1].

    In this env, the model outputs actions in [-1, 1], but env.step() multiplies
    them by hmax before the reward function is called. Therefore raw_actions are
    usually in [-hmax, hmax]. This function converts them back to [-1, 1].
    """
    actions = np.asarray(actions, dtype=float)

    # If actions have already been scaled by hmax, convert them back.
    if np.nanmax(np.abs(actions)) > 1.0:
        actions = actions / (hmax + eps)

    return np.clip(actions, -1.0, 1.0)


def _get_price_matrix(
    env: Any,
    tic_list: list,
    price_col: str = "close",
) -> tuple[np.ndarray, dict[Any, int]]:
    """
    Build and cache a price matrix for fast future-return reward calculation.

    Return:
        price_matrix: shape = (num_days, stock_dim)
        day_to_pos: maps env day label to row position in price_matrix

    This avoids doing df.loc/filter for every stock on every reward step, which
    is much faster for PPO training.
    """
    if price_col not in env.df.columns:
        price_col = "close"

    tic_tuple = tuple(str(tic) for tic in tic_list)
    cache_key = (price_col, tic_tuple)

    if not hasattr(env, "_reward_price_cache"):
        env._reward_price_cache = {}

    if cache_key in env._reward_price_cache:
        return env._reward_price_cache[cache_key]

    day_labels = list(env.df.index.unique())
    day_to_pos = {day: pos for pos, day in enumerate(day_labels)}

    tmp = env.df.reset_index()
    first_col = tmp.columns[0]
    tmp = tmp.rename(columns={first_col: "_day_index"})

    price_table = tmp.pivot(index="_day_index", columns="tic", values=price_col)
    price_table = price_table.reindex(index=day_labels, columns=list(tic_list))
    price_matrix = price_table.to_numpy(dtype=float)

    env._reward_price_cache[cache_key] = (price_matrix, day_to_pos)
    return price_matrix, day_to_pos


def reward_v2(ctx: RewardContext, params: Dict[str, Any] | None = None) -> float:
    """
    Jason reward v2.

    使用未來第 t+n 天價格計算 simple return。

    Formula:
        R_t = sum_i a(i,t) * r(i,t)

        r(i,t) = (p(i,t+n) - p(i,t)) / p(i,t)

    其中：
        a(i,t): 第 i 檔股票在第 t 天的 action value, range = [-1, 1]
        p(i,t): 第 i 檔股票在第 t 天的價格
        p(i,t+n): 第 i 檔股票在第 t+n 天的價格

    Meaning:
        action > 0 and future price up   -> positive reward
        action < 0 and future price down -> positive reward
        action direction wrong           -> negative reward
    """
    params = params or {}

    eps = params.get("eps", 1e-8)
    n_future = int(params.get("n_future", 5))
    price_col = params.get("price_col", "close")

    price_matrix, day_to_pos = _get_price_matrix(ctx.env, ctx.tic_list, price_col)

    current_pos = day_to_pos.get(ctx.current_day)
    if current_pos is None:
        return 0.0

    future_pos = current_pos + n_future
    if future_pos >= price_matrix.shape[0]:
        return 0.0

    current_prices = np.asarray(ctx.current_prices_before_trade, dtype=float)
    future_prices = price_matrix[future_pos]

    simple_returns = (future_prices - current_prices) / (current_prices + eps)
    action_units = _action_to_unit_array(ctx.raw_actions, ctx.env.hmax, eps)

    valid = (
        np.isfinite(current_prices)
        & np.isfinite(future_prices)
        & np.isfinite(simple_returns)
        & (current_prices > eps)
        & (future_prices > eps)
    )

    if not np.any(valid):
        return 0.0

    return float(np.sum(action_units[valid] * simple_returns[valid]))


def reward_v3(ctx: RewardContext, params: Dict[str, Any] | None = None) -> float:
    """
    Jason reward v3.

    使用未來 n 天平均價格計算 simple return。

    Formula:
        R_t = sum_i a(i,t) * r(i,t)

        r(i,t) = (avg_p(i,t)^n - p(i,t)) / p(i,t)

        avg_p(i,t)^n = 1/n * sum_{k=t+1}^{t+n} p(i,k)

    其中：
        a(i,t): 第 i 檔股票在第 t 天的 action value, range = [-1, 1]
        p(i,t): 第 i 檔股票在第 t 天的價格
        avg_p(i,t)^n: 第 i 檔股票未來 n 天平均價格

    Meaning:
        v2 only looks at the price on t+n.
        v3 uses the average price from t+1 to t+n, so it is smoother.
    """
    params = params or {}

    eps = params.get("eps", 1e-8)
    n_future = int(params.get("n_future", 5))
    price_col = params.get("price_col", "close")

    price_matrix, day_to_pos = _get_price_matrix(ctx.env, ctx.tic_list, price_col)

    current_pos = day_to_pos.get(ctx.current_day)
    if current_pos is None:
        return 0.0

    start_pos = current_pos + 1
    end_pos = current_pos + n_future + 1
    if end_pos > price_matrix.shape[0]:
        return 0.0

    future_window = price_matrix[start_pos:end_pos]
    current_prices = np.asarray(ctx.current_prices_before_trade, dtype=float)
    avg_future_prices = np.nanmean(future_window, axis=0)

    simple_returns = (avg_future_prices - current_prices) / (current_prices + eps)
    action_units = _action_to_unit_array(ctx.raw_actions, ctx.env.hmax, eps)

    valid = (
        np.isfinite(current_prices)
        & np.isfinite(avg_future_prices)
        & np.isfinite(simple_returns)
        & np.all(np.isfinite(future_window), axis=0)
        & (current_prices > eps)
        & (avg_future_prices > eps)
    )

    if not np.any(valid):
        return 0.0

    return float(np.sum(action_units[valid] * simple_returns[valid]))


REWARD_FUNCTIONS = {
    "v0": reward_v0,
    "original": reward_v0,
    "finrl_original": reward_v0,

    "v1": reward_v1,
    "our_reward_v1": reward_v1,

    "v2": reward_v2,
    "reward_v2": reward_v2,
    "future_day_return": reward_v2,

    "v3": reward_v3,
    "reward_v3": reward_v3,
    "future_avg_return": reward_v3,
}


def calculate_reward(
    reward_type: str,
    ctx: RewardContext,
    params: Dict[str, Any] | None = None,
) -> float:
    """Dispatch reward calculation by reward_type."""
    key = str(reward_type).lower()

    if key not in REWARD_FUNCTIONS:
        available = ", ".join(sorted(REWARD_FUNCTIONS.keys()))
        raise ValueError(f"Unknown reward_type={reward_type!r}. Available: {available}")

    return REWARD_FUNCTIONS[key](ctx, params)
