import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "finrl"
    / "meta"
    / "env_stock_trading"
    / "trading_score.py"
)
SPEC = importlib.util.spec_from_file_location("trading_score_for_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load Trading Score module: {MODULE_PATH}")
trading_score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trading_score)

build_action_sign_matrix = trading_score.build_action_sign_matrix
build_runlen_factor = trading_score.build_runlen_factor
build_runlen_matrix = trading_score.build_runlen_matrix
build_stock_trend_score = trading_score.build_stock_trend_score
calculate_annual_ts = trading_score.calculate_annual_ts
calculate_annual_ts_versions = trading_score.calculate_annual_ts_versions
calculate_cumulative_ts = trading_score.calculate_cumulative_ts
calculate_cumulative_ts_versions = trading_score.calculate_cumulative_ts_versions


def test_runlen_and_ts_v2_reciprocal() -> None:
    prices = pd.DataFrame(
        {"StockA": [100.0, 102.0, 102.0, 105.0, 90.0, 80.0]},
        index=["t", "t+1", "t+2", "t+3", "t+4", "t+5"],
    )

    runlen = build_runlen_matrix(prices)
    np.testing.assert_allclose(runlen["StockA"], [1, 2, 2, 3, 1, 2])

    runlen_v2 = build_runlen_factor(runlen, version="v2")
    np.testing.assert_allclose(
        runlen_v2["StockA"],
        [1, 1 / 2, 1 / 2, 1 / 3, 1, 1 / 2],
    )


def test_ts_v2_only_changes_runlen_factor() -> None:
    index = ["d1", "d2", "d3"]
    d_matrix = pd.DataFrame({"A": [0.10, 0.10, -0.20]}, index=index)
    runlen = pd.DataFrame({"A": [1.0, 2.0, 4.0]}, index=index)

    s_v1 = build_stock_trend_score(d_matrix, runlen, version="v1")
    s_v2 = build_stock_trend_score(d_matrix, runlen, version="v2")

    np.testing.assert_allclose(s_v1["A"], [10.0, 20.0, -80.0])
    np.testing.assert_allclose(s_v2["A"], [10.0, 5.0, -5.0])


def test_annual_and_cumulative_ts_use_same_normalized_formula() -> None:
    index = ["d1", "d2"]
    stock_trend_score = pd.DataFrame(
        {
            "A": [10.0, -20.0],
            "B": [30.0, 40.0],
        },
        index=index,
    )
    executed_actions = pd.DataFrame(
        {
            "A": [5.0, -3.0],
            "B": [0.0, -2.0],
        },
        index=index,
    )
    act = build_action_sign_matrix(executed_actions)

    _, _, annual_ts = calculate_annual_ts(stock_trend_score, act)
    _, _, cumulative_ts = calculate_cumulative_ts(stock_trend_score, act)

    # Numerator = 10 + 20 - 40 = -10; denominator = 10 + 20 + 30 + 40 = 100.
    assert annual_ts == -10.0
    assert cumulative_ts == -10.0


def test_annual_ts_versions_return_v1_and_v2() -> None:
    action_dates = ["2021-01-01", "2021-01-02", "2021-01-03"]
    actions = pd.DataFrame({"A": [1.0, 0.0, -1.0]}, index=action_dates)
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 120.0, 110.0]},
        index=[*action_dates, "2021-01-04"],
    )

    scores, daily, detail = calculate_annual_ts_versions(
        action_values=actions,
        price_pivot=prices,
        n_future=1,
    )

    assert set(scores) == {"Annual_TS", "Annual_TS_v2"}
    assert np.isclose(
        scores["Annual_TS"],
        daily["Daily_TS"].sum() / daily["Max_TS"].sum() * 100.0,
    )
    assert np.isclose(
        scores["Annual_TS_v2"],
        daily["Daily_TS_v2"].sum() / daily["Max_TS_v2"].sum() * 100.0,
    )
    np.testing.assert_allclose(
        detail["runlen_factor_v2"],
        1.0 / detail["runlen"],
    )


def test_cumulative_ts_runlen_continues_across_year_boundary() -> None:
    action_dates = [
        "2021-12-30",
        "2021-12-31",
        "2022-01-03",
        "2022-01-04",
    ]
    actions = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=action_dates)
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=[*action_dates, "2022-01-05"],
    )

    scores, _, detail = calculate_cumulative_ts_versions(
        action_values=actions,
        price_pivot=prices,
        n_future=1,
    )

    assert set(scores) == {"Cumulative_TS", "Cumulative_TS_v2"}
    np.testing.assert_allclose(detail["runlen"], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        detail["runlen_factor_v2"],
        [1.0, 1 / 2, 1 / 3, 1 / 4],
    )
