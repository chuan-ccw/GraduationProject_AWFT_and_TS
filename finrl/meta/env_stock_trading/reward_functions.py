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
    stock_weights_after_trade: np.ndarray
    cost_before_trade: float
    current_day: int
    tic_list: list
    previous_stock_weights_after_trade: np.ndarray | None = None


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


def calculate_stock_weights_after_trade(
    stock_prices: np.ndarray,
    stock_holdings: np.ndarray,
    cash_after_trade: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Calculate each stock's market-value weight immediately after trading.

    Formula:
        weight_i = price_i * shares_i / total_asset
        total_asset = cash + sum_j(price_j * shares_j)

    Cash is included in total_asset, so stock weights can sum to less than 1.
    """
    prices = np.asarray(stock_prices, dtype=float)
    holdings = np.asarray(stock_holdings, dtype=float)

    if prices.shape != holdings.shape:
        raise ValueError(
            "stock_prices and stock_holdings must have the same shape, "
            f"got {prices.shape} and {holdings.shape}."
        )

    stock_values = prices * holdings
    total_asset = float(cash_after_trade) + float(np.sum(stock_values))

    if not np.isfinite(total_asset) or total_asset <= eps:
        return np.zeros_like(stock_values, dtype=float)

    weights = stock_values / total_asset
    return np.where(np.isfinite(weights), weights, 0.0)


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



def _reset_reward_v4_diagnostics(env: Any) -> dict[str, float]:
    """Reset per-episode statistics used to diagnose reward v4."""
    diagnostics = {
        "diagnostic_step_count": 0.0,
        "valid_action_count": 0.0,
        "zero_delta_count": 0.0,
        "sum_abs_delta_a": 0.0,
        "sum_abs_effective_delta_a": 0.0,
        "sum_abs_future_return": 0.0,
        "positive_reward_count": 0.0,
        "positive_step_reward_count": 0.0,
        "zero_delta_ratio": 0.0,
        "mean_abs_delta_a": 0.0,
        "mean_abs_effective_delta_a": 0.0,
        "mean_abs_future_return": 0.0,
        "positive_reward_ratio": 0.0,
        "positive_step_reward_ratio": 0.0,
    }
    env._reward_v4_diagnostics = diagnostics
    return diagnostics


def _update_reward_v4_diagnostics(
    env: Any,
    delta_a: np.ndarray,
    effective_delta_a: np.ndarray,
    future_avg_returns: np.ndarray,
    zero_delta_mask: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Accumulate reward v4 diagnostics for the current episode.

    positive_reward_ratio is the proportion of valid stock-level reward
    contributions whose value is greater than zero.
    """
    diagnostics = getattr(env, "_reward_v4_diagnostics", None)
    if not isinstance(diagnostics, dict):
        diagnostics = _reset_reward_v4_diagnostics(env)

    valid_count = int(np.sum(valid))
    if valid_count <= 0:
        return

    contributions = effective_delta_a[valid] * future_avg_returns[valid]
    step_reward = float(np.sum(contributions))

    diagnostics["diagnostic_step_count"] += 1.0
    diagnostics["valid_action_count"] += float(valid_count)
    diagnostics["zero_delta_count"] += float(np.sum(zero_delta_mask & valid))
    diagnostics["sum_abs_delta_a"] += float(np.sum(np.abs(delta_a[valid])))
    diagnostics["sum_abs_effective_delta_a"] += float(
        np.sum(np.abs(effective_delta_a[valid]))
    )
    diagnostics["sum_abs_future_return"] += float(
        np.sum(np.abs(future_avg_returns[valid]))
    )
    diagnostics["positive_reward_count"] += float(np.sum(contributions > 0.0))
    diagnostics["positive_step_reward_count"] += float(step_reward > 0.0)

    action_count = diagnostics["valid_action_count"]
    step_count = diagnostics["diagnostic_step_count"]
    diagnostics["zero_delta_ratio"] = diagnostics["zero_delta_count"] / action_count
    diagnostics["mean_abs_delta_a"] = diagnostics["sum_abs_delta_a"] / action_count
    diagnostics["mean_abs_effective_delta_a"] = (
        diagnostics["sum_abs_effective_delta_a"] / action_count
    )
    diagnostics["mean_abs_future_return"] = (
        diagnostics["sum_abs_future_return"] / action_count
    )
    diagnostics["positive_reward_ratio"] = (
        diagnostics["positive_reward_count"] / action_count
    )
    diagnostics["positive_step_reward_ratio"] = (
        diagnostics["positive_step_reward_count"] / step_count
    )


def reward_v4(ctx: RewardContext, params: Dict[str, Any] | None = None) -> float:
    """
    Jason reward v4.

    使用有效 action value 變化量與未來 n 天平均價格報酬計算 reward。

    Formula:
        R_t = sum_i effective_delta_a(i,t) * future_avg_returns(i,t)

        effective_delta_a(i,t) = delta_a(i,t),
                                 if abs(delta_a(i,t)) > eps

        effective_delta_a(i,t) = c * sign(a(i,t)),
                                 if abs(delta_a(i,t)) <= eps

        delta_a(i,t) = a(i,t) - a(i,t-1)

        future_avg_returns(i,t)
            = (avg_p(i,t)^n - p(i,t)) / p(i,t)

        avg_p(i,t)^n = 1/n * sum_{k=t+1}^{t+n} p(i,k)

    其中：
        a(i,t): 第 i 檔股票在第 t 天的 action value, range = [-1, 1]
        delta_a(i,t): 第 i 檔股票在第 t 天相對於第 t-1 天的 action value 變化量
        c: delta_a(i,t) 接近 0 時使用的小幅有效 action 變化量
        p(i,t): 第 i 檔股票在第 t 天的價格
        avg_p(i,t)^n: 第 i 檔股票未來 n 天平均價格

    Meaning:
        If delta_a(i,t) changes, keep using delta_a(i,t).
        If delta_a(i,t) is approximately zero, use c times the sign of the
        current action only for reward calculation. The actual action sent to
        the environment is not modified.
    """
    params = params or {}

    eps = params.get("eps", 1e-8)
    n_future = int(params.get("n_future", 5))
    price_col = params.get("price_col", "close")
    c = float(params.get("c", 0.05))

    if not 0.0 <= c <= 1.0:
        raise ValueError("reward_v4 parameter 'c' must be in [0, 1].")

    current_action_units = _action_to_unit_array(ctx.raw_actions, ctx.env.hmax, eps)
    previous_action_units = getattr(ctx.env, "_reward_v4_previous_action_units", None)

    if (
        ctx.current_day == 0
        or previous_action_units is None
        or len(previous_action_units) != len(current_action_units)
    ):
        previous_action_units = np.zeros_like(current_action_units, dtype=float)
    else:
        previous_action_units = np.asarray(previous_action_units, dtype=float)

    delta_a = current_action_units - previous_action_units

    # Keep current action for the next step. This must happen even when the
    # future window is unavailable near the end of the episode.
    ctx.env._reward_v4_previous_action_units = current_action_units.copy()

    # current_day == 0 means a new training/evaluation episode has started.
    if ctx.current_day == 0:
        _reset_reward_v4_diagnostics(ctx.env)

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

    future_avg_returns = (avg_future_prices - current_prices) / (current_prices + eps)

    # delta_a 接近 0 時，只在 reward 計算中使用 c * sign(current action)。
    # 不修改實際送進環境的 action。
    zero_delta_mask = np.isclose(delta_a, 0.0, atol=eps, rtol=0.0)
    effective_delta_a = np.where(
        zero_delta_mask,
        c * np.sign(current_action_units),
        delta_a,
    )

    valid = (
        np.isfinite(current_prices)
        & np.isfinite(avg_future_prices)
        & np.isfinite(future_avg_returns)
        & np.isfinite(effective_delta_a)
        & np.all(np.isfinite(future_window), axis=0)
        & (current_prices > eps)
        & (avg_future_prices > eps)
    )

    if not np.any(valid):
        return 0.0

    _update_reward_v4_diagnostics(
        env=ctx.env,
        delta_a=delta_a,
        effective_delta_a=effective_delta_a,
        future_avg_returns=future_avg_returns,
        zero_delta_mask=zero_delta_mask,
        valid=valid,
    )

    return float(np.sum(effective_delta_a[valid] * future_avg_returns[valid]))


def reward_v5(
    ctx: RewardContext,
    params: Dict[str, Any] | None = None,
) -> float:
    """Weighted-Future-Trend (WFT).

    The stock weights are calculated immediately after the agent's trades have
    been executed, using the current-day prices and the actual resulting share
    holdings.

    Formula:
        R_t = sum_i w(i,t) * future_avg_returns(i,t)

        w(i,t) = market_value(i,t) / total_asset(t)

        market_value(i,t) = shares(i,t) * p(i,t)

        future_avg_returns(i,t)
            = (avg_p(i,t)^n - p(i,t)) / p(i,t)

        avg_p(i,t)^n = 1/n * sum_{k=t+1}^{t+n} p(i,k)

    Cash is included in total_asset(t), so sum_i w(i,t) may be smaller than 1.
    """
    params = params or {}

    eps = float(params.get("eps", 1e-8))
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
    stock_weights = np.asarray(ctx.stock_weights_after_trade, dtype=float)
    avg_future_prices = np.nanmean(future_window, axis=0)
    future_avg_returns = (avg_future_prices - current_prices) / (
        current_prices + eps
    )

    if stock_weights.shape != current_prices.shape:
        raise ValueError(
            "stock_weights_after_trade and current_prices_before_trade must "
            f"have the same shape, got {stock_weights.shape} and "
            f"{current_prices.shape}."
        )

    valid = (
        np.isfinite(current_prices)
        & np.isfinite(avg_future_prices)
        & np.isfinite(future_avg_returns)
        & np.isfinite(stock_weights)
        & np.all(np.isfinite(future_window), axis=0)
        & (current_prices > eps)
        & (avg_future_prices > eps)
    )

    if not np.any(valid):
        return 0.0

    return float(np.sum(stock_weights[valid] * future_avg_returns[valid]))


def reward_v6(
    ctx: RewardContext,
    params: Dict[str, Any] | None = None,
) -> float:
    """Delta-Weighted-Future-Trend (DWFT).

    Formula:
        R_t = sum_i delta_w(i,t) * future_signal(i,t)

        delta_w(i,t) = w(i,t) - w(i,t-1)

        w(i,t) = market_value(i,t) / total_asset(t)

        future_avg_returns(i,t)
            = (avg_p(i,t)^n - p(i,t)) / p(i,t)

        avg_p(i,t)^n = 1/n * sum_{k=t+1}^{t+n} p(i,k)

        future_direction(i,t)
            = direction(i,t) * (P_max(i,t) - P_min(i,t)) / p(i,t)

        direction(i,t) = +1, if P_min occurs before P_max
                         -1, if P_min occurs after P_max
                          0, if both occur at the same position

    ``future_variable`` selects the future signal used by DWFT:
        "future_direction"  -> delta_w * future_direction
        "future_avg_return" -> delta_w * future_avg_returns

    Both w(i,t) and w(i,t-1) are post-trade stock weights. At the first
    step of an episode, w(i,t-1) is initialized from the starting portfolio
    before the first trade. For the usual all-cash reset, those weights are 0.
    """
    params = params or {}

    eps = float(params.get("eps", 1e-8))
    n_future = int(params.get("n_future", 5))
    price_col = params.get("price_col", "close")
    future_variable = str(
        params.get("future_variable", "future_direction")
    ).strip().lower()

    future_variable_aliases = {
        "future_direction": "future_direction",
        "direction": "future_direction",
        "future_avg_return": "future_avg_return",
        "future_average": "future_avg_return",
        "future_avg_returns": "future_avg_return",
    }
    if future_variable not in future_variable_aliases:
        available = ", ".join(sorted(future_variable_aliases))
        raise ValueError(
            "reward_v6 parameter 'future_variable' must be one of: "
            f"{available}. Got {future_variable!r}."
        )
    future_variable = future_variable_aliases[future_variable]

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
    current_weights = np.asarray(ctx.stock_weights_after_trade, dtype=float)

    if ctx.previous_stock_weights_after_trade is None:
        raise ValueError(
            "previous_stock_weights_after_trade is required for reward v6."
        )
    previous_weights = np.asarray(
        ctx.previous_stock_weights_after_trade,
        dtype=float,
    )

    if current_weights.shape != current_prices.shape:
        raise ValueError(
            "stock_weights_after_trade and current_prices_before_trade must "
            f"have the same shape, got {current_weights.shape} and "
            f"{current_prices.shape}."
        )
    if previous_weights.shape != current_weights.shape:
        raise ValueError(
            "previous_stock_weights_after_trade and stock_weights_after_trade "
            f"must have the same shape, got {previous_weights.shape} and "
            f"{current_weights.shape}."
        )

    avg_future_prices = np.nanmean(future_window, axis=0)
    future_avg_returns = (avg_future_prices - current_prices) / (
        current_prices + eps
    )

    future_max_prices = np.nanmax(future_window, axis=0)
    future_min_prices = np.nanmin(future_window, axis=0)
    future_max_positions = np.argmax(future_window, axis=0)
    future_min_positions = np.argmin(future_window, axis=0)
    future_direction_signs = np.sign(
        future_max_positions - future_min_positions
    ).astype(float)
    future_direction = (
        future_direction_signs
        * (future_max_prices - future_min_prices)
        / (current_prices + eps)
    )

    future_signal = (
        future_direction
        if future_variable == "future_direction"
        else future_avg_returns
    )
    delta_weights = current_weights - previous_weights

    valid = (
        np.isfinite(current_prices)
        & np.isfinite(avg_future_prices)
        & np.isfinite(future_avg_returns)
        & np.isfinite(future_max_prices)
        & np.isfinite(future_min_prices)
        & np.isfinite(future_direction)
        & np.isfinite(future_signal)
        & np.isfinite(current_weights)
        & np.isfinite(previous_weights)
        & np.isfinite(delta_weights)
        & np.all(np.isfinite(future_window), axis=0)
        & (current_prices > eps)
        & (avg_future_prices > eps)
    )

    if not np.any(valid):
        return 0.0

    return float(np.sum(delta_weights[valid] * future_signal[valid]))


# Backward-compatible function name for code created before the v5 alias.
reward_weight_future = reward_v5

# Formal and compatibility function names for reward v6.
reward_weight_change_future = reward_v6


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

    "v4": reward_v4,
    "reward_v4": reward_v4,
    "future_avg_delta_action_return": reward_v4,
    "future_avg_delta_action_reward": reward_v4,

    "v5": reward_v5,
    "reward_v5": reward_v5,
    "weight_future": reward_v5,
    "reward_weight_future": reward_v5,
    "weight_future_reward": reward_v5,
    "portfolio_weighted_future_return": reward_v5,
    "portfolio_weighted_future_return_reward": reward_v5,
    "wft": reward_v5,
    "weighted_future_trend": reward_v5,
    "weighted_future_trend_reward": reward_v5,

    "v6": reward_v6,
    "reward_v6": reward_v6,
    "delta_weight_future": reward_v6,
    "reward_delta_weight_future": reward_v6,
    "delta_weight_future_reward": reward_v6,
    "portfolio_weight_change_future_return": reward_v6,
    "portfolio_weight_change_future_return_reward": reward_v6,
    "dwft": reward_v6,
    "delta_weighted_future_trend": reward_v6,
    "delta_weighted_future_trend_reward": reward_v6,
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
