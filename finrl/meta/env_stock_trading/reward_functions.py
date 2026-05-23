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
             - gamma * transaction_cost_rate (已經扣除所以不用)
             + alpha * buy_low_score
             + beta  * sell_high_score

    The returned value is the unscaled reward. env_stocktrading.py will multiply
    it by env.reward_scaling later, exactly like the original FinRL flow.
    """
    params = params or {}

    eps = params.get("eps", 1e-8)
    alpha = params.get("alpha", 0.05)
    beta = params.get("beta", 0.05)
    # gamma = params.get("gamma", 1.0)
    n_recent = params.get("n_recent", 14)

    simple_return = (ctx.end_total_asset - ctx.begin_total_asset) / (
        ctx.begin_total_asset + eps
    )

    # transaction_cost = ctx.env.cost - ctx.cost_before_trade
    # transaction_cost_rate = transaction_cost / (ctx.begin_total_asset + eps)

    buy_low_score_total = 0.0
    sell_high_score_total = 0.0

    start_day = max(0, ctx.current_day - n_recent + 1)

    for i, action in enumerate(ctx.raw_actions):
        price = ctx.current_prices_before_trade[i]
        tic = ctx.tic_list[i]

        recent_data = ctx.env.df.loc[start_day : ctx.current_day]
        recent_data = recent_data[recent_data["tic"] == tic]

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

    return (
        simple_return
        # - gamma * transaction_cost_rate
        + alpha * buy_low_score_total
        + beta * sell_high_score_total
    )


REWARD_FUNCTIONS = {
    "v0": reward_v0,
    "original": reward_v0,
    "finrl_original": reward_v0,
    "v1": reward_v1,
    "our_reward_v1": reward_v1,
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
