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
reward_v7 = _REWARD_MODULE.reward_v7
calculate_individual_reward_components = (
    _REWARD_MODULE.calculate_individual_reward_components
)
reward_v3 = _REWARD_MODULE.reward_v3


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


def _make_direction_price_df() -> pd.DataFrame:
    rows = []
    prices = {
        0: {"AAA": 10.0, "BBB": 20.0},
        1: {"AAA": 8.0, "BBB": 24.0},
        2: {"AAA": 12.0, "BBB": 16.0},
    }
    for day, day_prices in prices.items():
        for tic, close in day_prices.items():
            rows.append({"day": day, "tic": tic, "close": close})
    return pd.DataFrame(rows).set_index("day")


def _make_v3_price_df() -> pd.DataFrame:
    rows = []
    prices = {
        0: {"AAA": 10.0, "BBB": 20.0},
        1: {"AAA": 11.0, "BBB": 18.0},
        2: {"AAA": 13.0, "BBB": 18.0},
    }
    for day, day_prices in prices.items():
        for tic, close in day_prices.items():
            rows.append({"day": day, "tic": tic, "close": close})
    return pd.DataFrame(rows).set_index("day")


def test_reward_v3_uses_final_executed_actions() -> None:
    env = SimpleNamespace(df=_make_v3_price_df(), hmax=100, stock_dim=2)
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
        executed_actions=np.array([20.0, -10.0]),
    )

    reward = reward_v3(ctx, {"n_future": 2, "price_col": "close"})

    # Executed action units = [+0.20, -0.10].
    # AAA future average = 12 -> +20%; BBB future average = 18 -> -10%.
    # reward = 0.20 * 0.20 + (-0.10) * (-0.10) = 0.05.
    assert np.isclose(reward, 0.05)


def test_reward_v3_normalizes_one_executed_share_by_hmax() -> None:
    env = SimpleNamespace(df=_make_v3_price_df(), hmax=100, stock_dim=2)
    ctx = RewardContext(
        env=env,
        begin_total_asset=800.0,
        end_total_asset=800.0,
        raw_actions=np.array([100.0, 0.0]),
        current_prices_before_trade=np.array([10.0, 20.0]),
        stock_weights_after_trade=np.array([0.25, 0.25]),
        cost_before_trade=0.0,
        current_day=0,
        tic_list=["AAA", "BBB"],
        executed_actions=np.array([1.0, 0.0]),
    )

    reward = reward_v3(ctx, {"n_future": 2, "price_col": "close"})

    # One executed share means action unit = 1 / 100 = 0.01, not +1.0.
    # reward = 0.01 * 0.20 = 0.002.
    assert np.isclose(reward, 0.002)


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
        "weighted_future_trend",
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
        {
            "n_future": 2,
            "price_col": "close",
            "future_variable": "future_avg_return",
        },
    )

    # delta_w = [+0.15, -0.15].
    # AAA future return = +20%; BBB future return = 0%.
    # reward = 0.15 * 0.20 + (-0.15) * 0.00 = 0.03.
    assert np.isclose(reward, 0.03)


def test_reward_v6_uses_same_price_delta_weight() -> None:
    env = SimpleNamespace(df=_make_price_df())
    ctx = RewardContext(
        env=env,
        begin_total_asset=800.0,
        end_total_asset=800.0,
        raw_actions=np.array([100.0, -100.0]),
        current_prices_before_trade=np.array([10.0, 20.0]),
        stock_weights_after_trade=np.array([0.40, 0.10]),
        previous_stock_weights_after_trade=np.array([0.10, 0.40]),
        stock_weights_before_trade=np.array([0.25, 0.25]),
        cost_before_trade=0.0,
        current_day=0,
        tic_list=["AAA", "BBB"],
    )

    reward = reward_v6(
        ctx,
        {
            "n_future": 2,
            "price_col": "close",
            "future_variable": "future_avg_return",
            "delta_weight_variable": "same_price_delta_weight",
        },
    )

    # same_price_delta_weight = post-trade weight - pre-trade weight
    #                         = [+0.15, -0.15].
    # The previous day's weights are intentionally different and must not be used.
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

    params["future_variable"] = "future_avg_return"

    reward_by_v6 = calculate_reward("v6", ctx, params)
    reward_by_formal_name = calculate_reward(
        "delta_weighted_future_trend",
        ctx,
        params,
    )

    assert np.isclose(reward_by_v6, 0.03)
    assert np.isclose(reward_by_formal_name, reward_by_v6)


def test_reward_v6_uses_future_direction_by_default() -> None:
    env = SimpleNamespace(df=_make_direction_price_df())
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
    # AAA: min 8 occurs before max 12 -> +(12 - 8) / 10 = +0.4.
    # BBB: min 16 occurs after max 24 -> -(24 - 16) / 20 = -0.4.
    # reward = 0.15 * 0.4 + (-0.15) * (-0.4) = 0.12.
    assert np.isclose(reward, 0.12)


def test_reward_v6_rejects_unknown_future_variable() -> None:
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

    try:
        reward_v6(
            ctx,
            {
                "n_future": 2,
                "price_col": "close",
                "future_variable": "unknown",
            },
        )
    except ValueError as exc:
        assert "future_variable" in str(exc)
    else:
        raise AssertionError("reward_v6 should reject an unknown future_variable")


def test_reward_v6_rejects_unknown_delta_weight_variable() -> None:
    env = SimpleNamespace(df=_make_price_df())
    ctx = RewardContext(
        env=env,
        begin_total_asset=800.0,
        end_total_asset=800.0,
        raw_actions=np.array([100.0, -100.0]),
        current_prices_before_trade=np.array([10.0, 20.0]),
        stock_weights_after_trade=np.array([0.40, 0.10]),
        previous_stock_weights_after_trade=np.array([0.25, 0.25]),
        stock_weights_before_trade=np.array([0.25, 0.25]),
        cost_before_trade=0.0,
        current_day=0,
        tic_list=["AAA", "BBB"],
    )

    try:
        reward_v6(
            ctx,
            {
                "n_future": 2,
                "price_col": "close",
                "delta_weight_variable": "unknown",
            },
        )
    except ValueError as exc:
        assert "delta_weight_variable" in str(exc)
    else:
        raise AssertionError(
            "reward_v6 should reject an unknown delta_weight_variable"
        )



def test_reward_v7_applies_scale_only_to_trade_branch() -> None:
    env = SimpleNamespace(df=_make_v3_price_df(), hmax=100, stock_dim=2)
    ctx = RewardContext(
        env=env,
        begin_total_asset=800.0,
        end_total_asset=800.0,
        raw_actions=np.array([0.0, 0.0]),
        current_prices_before_trade=np.array([10.0, 20.0]),
        stock_weights_after_trade=np.array([0.40, 0.25]),
        previous_stock_weights_after_trade=np.array([0.40, 0.25]),
        cost_before_trade=0.0,
        current_day=0,
        tic_list=["AAA", "BBB"],
        executed_actions=np.array([2.0, 0.0]),
    )

    reward = reward_v7(
        ctx,
        {"n_future": 2, "price_col": "close", "scale": 3.0},
    )

    # AAA trades: 2 * 20% * 3 = 1.20.
    # BBB is held without trading: 0.25 * (-10%) = -0.025.
    assert np.isclose(reward, 1.175)


def test_reward_v7_individual_components_match_reward_formula() -> None:
    env = SimpleNamespace(df=_make_v3_price_df(), hmax=100, stock_dim=2)
    ctx = RewardContext(
        env=env,
        begin_total_asset=800.0,
        end_total_asset=800.0,
        raw_actions=np.array([0.0, 0.0]),
        current_prices_before_trade=np.array([10.0, 20.0]),
        stock_weights_after_trade=np.array([0.40, 0.25]),
        previous_stock_weights_after_trade=np.array([0.40, 0.25]),
        cost_before_trade=0.0,
        current_day=0,
        tic_list=["AAA", "BBB"],
        executed_actions=np.array([2.0, 0.0]),
    )
    params = {"n_future": 2, "price_col": "close", "scale": 3.0}

    components = calculate_individual_reward_components("v7", ctx, params)

    np.testing.assert_allclose(components, np.array([1.20, -0.025]))
    assert np.isclose(np.sum(components), reward_v7(ctx, params))
