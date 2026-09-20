from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

TSVersion = Literal["v1", "v2"]
DEFAULT_TS_VERSION: TSVersion = "v1"
ACTION_EPS = 1e-12


def normalize_ts_version(version: str | None) -> TSVersion:
    """Normalize supported Trading Score versions."""
    value = str(version or DEFAULT_TS_VERSION).strip().lower()
    aliases = {
        "v1": "v1",
        "ts": "v1",
        "original": "v1",
        "v2": "v2",
        "ts_v2": "v2",
        "ts-v2": "v2",
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise ValueError(
            f"Unsupported TS version: {version!r}. Expected 'v1' or 'v2'."
        )
    return normalized  # type: ignore[return-value]


def build_d_matrix(
    price_pivot: pd.DataFrame,
    n_future: int,
) -> pd.DataFrame:
    """Calculate the future average-return signal d(i,t).

    d(i,t) = (mean(P^c_{i,t+1:t+n}) - P^c_{i,t}) / P^c_{i,t}
    """
    if int(n_future) <= 0:
        raise ValueError("n_future must be a positive integer")

    prices = price_pivot.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    d_values = np.full_like(prices, np.nan, dtype=float)

    n_rows, _ = prices.shape
    for row_idx in range(n_rows):
        end_idx = row_idx + int(n_future) + 1
        if end_idx > n_rows:
            continue

        current = prices[row_idx]
        future_window = prices[row_idx + 1 : end_idx]
        valid = (
            np.isfinite(current)
            & (current > 0)
            & np.all(np.isfinite(future_window), axis=0)
        )
        if not np.any(valid):
            continue

        future_average = np.mean(future_window, axis=0)
        d_values[row_idx, valid] = (
            future_average[valid] - current[valid]
        ) / current[valid]

    return pd.DataFrame(
        d_values,
        index=price_pivot.index,
        columns=price_pivot.columns,
    )


def build_runlen_matrix(price_pivot: pd.DataFrame) -> pd.DataFrame:
    """Calculate same-direction price-movement persistence length.

    The first valid day is runlen = 1. A flat day keeps the previous runlen
    and previous non-flat direction, so runlen never becomes 0 inside a valid
    price sequence.

    Example:
        price  = [100, 102, 102, 105, 90, 80]
        runlen = [  1,   2,   2,   3,  1,  2]
    """
    prices = price_pivot.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n_rows, n_stocks = prices.shape
    runlen = np.full((n_rows, n_stocks), np.nan, dtype=float)

    for stock_idx in range(n_stocks):
        previous_direction = 0
        previous_runlen = 0.0

        for row_idx in range(n_rows):
            current = prices[row_idx, stock_idx]
            if not np.isfinite(current):
                previous_direction = 0
                previous_runlen = 0.0
                continue

            if row_idx == 0 or not np.isfinite(prices[row_idx - 1, stock_idx]):
                direction = 0
                current_runlen = 1.0
            else:
                diff = current - prices[row_idx - 1, stock_idx]
                direction = int(np.sign(diff))

                if direction == 0:
                    current_runlen = previous_runlen
                elif previous_direction == 0:
                    current_runlen = previous_runlen + 1.0
                elif direction == previous_direction:
                    current_runlen = previous_runlen + 1.0
                else:
                    current_runlen = 1.0

            runlen[row_idx, stock_idx] = current_runlen
            if direction != 0:
                previous_direction = direction
            previous_runlen = current_runlen

    return pd.DataFrame(
        runlen,
        index=price_pivot.index,
        columns=price_pivot.columns,
    )


def build_runlen_factor(
    runlen_matrix: pd.DataFrame,
    version: str = DEFAULT_TS_VERSION,
) -> pd.DataFrame:
    """Return the runlen factor used by the selected TS version.

    TS / v1:
        runlen_factor(i,t) = runlen(i,t)

    TS_v2 / v2:
        runlen_factor(i,t) = 1 / runlen(i,t)

    Therefore TS_v2 gives the largest runlen weight (=1) at a new rise/fall
    segment and progressively smaller weights as the same-direction trend
    continues.
    """
    normalized_version = normalize_ts_version(version)
    runlen = runlen_matrix.apply(pd.to_numeric, errors="coerce").astype(float)

    if normalized_version == "v1":
        return runlen.copy()

    invalid = runlen.notna() & (runlen <= 0)
    if invalid.any().any():
        raise ValueError("runlen must be positive before applying TS_v2 reciprocal")

    return 1.0 / runlen


def build_stock_trend_score(
    d_matrix: pd.DataFrame,
    runlen_matrix: pd.DataFrame,
    version: str = DEFAULT_TS_VERSION,
) -> pd.DataFrame:
    """Calculate stock trend score s(i,t) for TS or TS_v2.

    TS / v1:
        s(i,t) = sign(d(i,t)) * |d(i,t)| * 100 * runlen(i,t)

    TS_v2 / v2:
        s_v2(i,t) = sign(d(i,t)) * |d(i,t)| * 100 / runlen(i,t)
    """
    d_aligned, runlen_aligned = d_matrix.align(runlen_matrix, join="inner")
    runlen_factor = build_runlen_factor(runlen_aligned, version=version)
    return np.sign(d_aligned) * d_aligned.abs() * 100.0 * runlen_factor


def build_action_sign_matrix(
    action_values: pd.DataFrame,
    eps: float = ACTION_EPS,
) -> pd.DataFrame:
    """Map executed action values to Buy/Hold/Sell = +1/0/-1."""
    numeric = action_values.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Action values contain missing/non-numeric values")

    act = pd.DataFrame(0.0, index=numeric.index, columns=numeric.columns)
    act[numeric > float(eps)] = 1.0
    act[numeric < -float(eps)] = -1.0
    return act


def calculate_period_ts(
    stock_trend_score: pd.DataFrame,
    act: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Calculate per-stock TS, Daily TS, Max TS and normalized period score.

    TS(i,t)      = act(i,t) * s(i,t)
    DailyTS(t)   = sum_i TS(i,t)
    MaxTS(t)     = sum_i |s(i,t)|
    Period TS    = sum_t DailyTS(t) / sum_t MaxTS(t) * 100

    The same normalized period formula is used for:
      * Annual TS: one annual Test period
      * Cumulative TS: the complete continuous five-year Test period
    """
    s_aligned = stock_trend_score.reindex(index=act.index, columns=act.columns)
    a_aligned = act.apply(pd.to_numeric, errors="coerce")

    if s_aligned.isna().any().any():
        raise ValueError("Stock trend score contains missing values in the score period")
    if a_aligned.isna().any().any():
        raise ValueError("Action sign matrix contains missing values in the score period")

    ts = a_aligned * s_aligned
    daily_ts = ts.sum(axis=1)
    max_ts = s_aligned.abs().sum(axis=1)

    denominator = float(max_ts.sum())
    if denominator <= 0:
        raise ValueError("TS denominator is zero; Max TS has no positive total")

    period_ts = float(daily_ts.sum() / denominator * 100.0)
    daily = pd.DataFrame(
        {
            "date": act.index,
            "Daily_TS": daily_ts.to_numpy(dtype=float),
            "Max_TS": max_ts.to_numpy(dtype=float),
        }
    )
    return ts, daily, period_ts


def calculate_annual_ts(
    stock_trend_score: pd.DataFrame,
    act: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Calculate Annual TS using the unified normalized TS formula."""
    return calculate_period_ts(stock_trend_score=stock_trend_score, act=act)


def calculate_cumulative_ts(
    stock_trend_score: pd.DataFrame,
    act: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Calculate Cumulative TS across the complete continuous test period."""
    return calculate_period_ts(stock_trend_score=stock_trend_score, act=act)


def calculate_single_stock_period_ts(
    stock_trend_score: pd.Series,
    act: pd.Series,
) -> float:
    """Calculate the normalized TS for one stock over one period."""
    s = pd.to_numeric(stock_trend_score, errors="coerce")
    a = pd.to_numeric(act, errors="coerce")

    if s.isna().any() or a.isna().any():
        raise ValueError("Single-stock TS input contains missing values")

    denominator = float(s.abs().sum())
    if denominator <= 0:
        return float("nan")

    numerator = float((a * s).sum())
    return numerator / denominator * 100.0


def calculate_single_stock_annual_ts(
    stock_trend_score: pd.Series,
    act: pd.Series,
) -> float:
    """Calculate Annual TS for one stock."""
    return calculate_single_stock_period_ts(stock_trend_score, act)


def calculate_single_stock_cumulative_ts(
    stock_trend_score: pd.Series,
    act: pd.Series,
) -> float:
    """Calculate Cumulative TS for one stock."""
    return calculate_single_stock_period_ts(stock_trend_score, act)


def _calculate_ts_versions(
    action_values: pd.DataFrame,
    price_pivot: pd.DataFrame,
    n_future: int,
    cumulative: bool,
    eps: float = ACTION_EPS,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Calculate both TS versions from one action matrix and one price matrix.

    Parameters
    ----------
    action_values:
        Executed action values. Index = action date, columns = ticker.
    price_pivot:
        Close-price matrix. Index = trading date, columns = ticker. For
        Cumulative TS, this should contain the complete continuous period so
        runlen is continuous across annual window boundaries.
    n_future:
        Number of future trading days used by d(i,t).
    cumulative:
        False -> Annual TS; True -> Cumulative TS.
    """
    if action_values.empty:
        raise ValueError("No action values were provided for TS calculation")

    actions = action_values.copy()
    actions.index = actions.index.astype(str)
    actions = actions[~actions.index.duplicated(keep="last")].sort_index()
    tickers = [str(column) for column in actions.columns]
    actions.columns = tickers

    prices = price_pivot.copy()
    prices.index = prices.index.astype(str)
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    prices = prices.reindex(columns=tickers)

    d_matrix = build_d_matrix(prices, n_future=n_future)
    runlen_matrix = build_runlen_matrix(prices)
    action_dates = actions.index.tolist()

    d_valid = d_matrix.reindex(index=action_dates, columns=tickers)
    runlen_valid = runlen_matrix.reindex(index=action_dates, columns=tickers)
    current_prices = prices.reindex(index=action_dates, columns=tickers)

    if d_valid.isna().any().any():
        missing_dates = d_valid.index[d_valid.isna().any(axis=1)].tolist()
        preview = ", ".join(missing_dates[:5])
        if len(missing_dates) > 5:
            preview += ", ..."
        raise ValueError(
            f"TS needs {int(n_future)} future trading days for every action date. "
            f"Missing d(i,t) date(s): {preview}"
        )
    if runlen_valid.isna().any().any():
        raise ValueError("runlen(i,t) contains missing values in the TS period")
    if current_prices.isna().any().any():
        raise ValueError("Close prices contain missing values in the TS period")

    act = build_action_sign_matrix(actions, eps=eps)
    score_function = calculate_cumulative_ts if cumulative else calculate_annual_ts

    s_v1 = build_stock_trend_score(d_valid, runlen_valid, version="v1")
    ts_v1, daily_v1, score_v1 = score_function(s_v1, act)

    s_v2 = build_stock_trend_score(d_valid, runlen_valid, version="v2")
    ts_v2, daily_v2, score_v2 = score_function(s_v2, act)

    daily = daily_v1.copy()
    daily["Daily_TS_v2"] = daily_v2["Daily_TS"].to_numpy(dtype=float)
    daily["Max_TS_v2"] = daily_v2["Max_TS"].to_numpy(dtype=float)

    runlen_factor_v2 = build_runlen_factor(runlen_valid, version="v2")
    stock_rows: list[pd.DataFrame] = []
    for ticker in tickers:
        stock_rows.append(
            pd.DataFrame(
                {
                    "date": action_dates,
                    "tic": ticker,
                    "close": current_prices[ticker].to_numpy(dtype=float),
                    "d": d_valid[ticker].to_numpy(dtype=float),
                    "runlen": runlen_valid[ticker].to_numpy(dtype=float),
                    "runlen_factor_v2": runlen_factor_v2[ticker].to_numpy(dtype=float),
                    "stock_trend_score": s_v1[ticker].to_numpy(dtype=float),
                    "stock_trend_score_v2": s_v2[ticker].to_numpy(dtype=float),
                    "executed_action": actions[ticker].to_numpy(dtype=float),
                    "act": act[ticker].to_numpy(dtype=float),
                    "TS": ts_v1[ticker].to_numpy(dtype=float),
                    "TS_v2": ts_v2[ticker].to_numpy(dtype=float),
                }
            )
        )

    score_prefix = "Cumulative" if cumulative else "Annual"
    scores = {
        f"{score_prefix}_TS": score_v1,
        f"{score_prefix}_TS_v2": score_v2,
    }
    return scores, daily, pd.concat(stock_rows, ignore_index=True)


def calculate_annual_ts_versions(
    action_values: pd.DataFrame,
    price_pivot: pd.DataFrame,
    n_future: int,
    eps: float = ACTION_EPS,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Calculate Annual TS and Annual TS_v2 with one unified implementation."""
    return _calculate_ts_versions(
        action_values=action_values,
        price_pivot=price_pivot,
        n_future=n_future,
        cumulative=False,
        eps=eps,
    )


def calculate_cumulative_ts_versions(
    action_values: pd.DataFrame,
    price_pivot: pd.DataFrame,
    n_future: int,
    eps: float = ACTION_EPS,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Calculate Cumulative TS and Cumulative TS_v2 for a continuous period."""
    return _calculate_ts_versions(
        action_values=action_values,
        price_pivot=price_pivot,
        n_future=n_future,
        cumulative=True,
        eps=eps,
    )
