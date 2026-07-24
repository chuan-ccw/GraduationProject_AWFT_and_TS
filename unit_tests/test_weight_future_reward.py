from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_REWARD_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "finrl"
    / "meta"
    / "env_stock_trading"
    / "reward_functions.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_reward_functions_module",
    _REWARD_MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_REWARD_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _REWARD_MODULE
_SPEC.loader.exec_module(_REWARD_MODULE)

RewardContext = _REWARD_MODULE.RewardContext
calculate_stock_weights_after_trade = (
    _REWARD_MODULE.calculate_stock_weights_after_trade
)
calculate_reward = _REWARD_MODULE.calculate_reward
reward_v5 = _REWARD_MODULE.reward_v5
reward_v6 = _REWARD_MODULE.reward_v6


def _make_price_df() -> pd.DataFrame:
    rows = []
    prices = {
        0: {"AAA": 10.0, "BBB": 20.0},
        1: {"AAA": 11.0, "BBB": 18.0},
        2: {"AAA": 13.0, "BBB": 22.0},
    }
    for day, day_prices in prices.items():
        for tic, close in day_prices.items():
            rows.append({"day": day, "tic": tic, "close": close})
    return pd.DataFrame(rows).set_index("day")


def test_calculate_stock_weights_after_trade_includes_cash() -> None:
    weights = calculate_stock_weights_after_trade(
        stock_prices=np.array([10.0, 20.0]),
        stock_holdings=np.array([20.0, 10.0]),
        cash_after_trade=400.0,
    )

    np.testing.assert_allclose(weights, np.array([0.25, 0.25]))
    assert np.isclose(np.sum(weights), 0.5)


def test_reward_v5_uses_post_trade_weights() -> None:
    env = SimpleNamespace(df=_make_price_df())
    ctx = RewardContext(
        env=env,
        begin_total_asset=800.0,
        end_total_asset=800.0,
        raw_actions=np.array([100.0, -100.0]),
        current_prices_before_trade=np.array([10.0, 20.0]),
        stock_weights_after_trade=np.array([0.25, 0.25]),
        cost_before_trade=0.0,
        current_day=0,
        tic_list=["AAA", "BBB"],
    )

    reward = reward_v5(
        ctx,
        {"n_future": 2, "price_col": "close"},
    )

    # AAA future average = 12 -> +20%; BBB future average = 20 -> 0%.
    # reward = 0.25 * 0.20 + 0.25 * 0.00 = 0.05.
    assert np.isclose(reward, 0.05)


def test_v5_and_formal_name_dispatch_to_same_reward() -> None:
    env = SimpleNamespace(df=_make_price_df())
    ctx = RewardContext(
        env=env,
        begin_total_asset=800.0,
        end_total_asset=800.0,
        raw_actions=np.array([100.0, -100.0]),
        current_prices_before_trade=np.array([10.0, 20.0]),
        stock_weights_after_trade=np.array([0.25, 0.25]),
        cost_before_trade=0.0,
        current_day=0,
        tic_list=["AAA", "BBB"],
    )
    params = {"n_future": 2, "price_col": "close"}

    reward_by_v5 = calculate_reward("v5", ctx, params)
    reward_by_formal_name = calculate_reward(
        "portfolio_weighted_future_return_reward",
        ctx,
        params,
    )

    assert np.isclose(reward_by_v5, 0.05)
    assert np.isclose(reward_by_formal_name, reward_by_v5)



def test_reward_v6_uses_change_in_post_trade_weights() -> None:
    env = SimpleNamespace(df=_make_price_df())
    ctx = RewardContext(
        env=env,
        begin_total_asset=800.0,
        end_total_asset=800.0,
        raw_actions=np.array([100.0, -100.0]),
        current_prices_before_trade=np.array([10.0, 20.0]),
        stock_weights_after_trade=np.array([0.40, 0.10]),
        previous_stock_weights_after_trade=np.array([0.25, 0.25]),
        cost_before_trade=0.0,
        current_day=0,
        tic_list=["AAA", "BBB"],
    )

    reward = reward_v6(
        ctx,
        {"n_future": 2, "price_col": "close"},
    )

    # delta_w = [+0.15, -0.15].
    # AAA future return = +20%; BBB future return = 0%.
    # reward = 0.15 * 0.20 + (-0.15) * 0.00 = 0.03.
    assert np.isclose(reward, 0.03)


def test_v6_and_formal_name_dispatch_to_same_reward() -> None:
    env = SimpleNamespace(df=_make_price_df())
    ctx = RewardContext(
        env=env,
        begin_total_asset=800.0,
        end_total_asset=800.0,
        raw_actions=np.array([100.0, -100.0]),
        current_prices_before_trade=np.array([10.0, 20.0]),
        stock_weights_after_trade=np.array([0.40, 0.10]),
        previous_stock_weights_after_trade=np.array([0.25, 0.25]),
        cost_before_trade=0.0,
        current_day=0,
        tic_list=["AAA", "BBB"],
    )
    params = {"n_future": 2, "price_col": "close"}

    reward_by_v6 = calculate_reward("v6", ctx, params)
    reward_by_formal_name = calculate_reward(
        "portfolio_weight_change_future_return_reward",
        ctx,
        params,
    )

    assert np.isclose(reward_by_v6, 0.03)
    assert np.isclose(reward_by_formal_name, reward_by_v6)
