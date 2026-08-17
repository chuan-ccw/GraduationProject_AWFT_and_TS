from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd


RANKING_CSV_NAME = "future_ranking_top5_bottom5.csv"
RANKING_TOP5_PNG_NAME = "future_ranking_top5.png"
RANKING_BOTTOM5_PNG_NAME = "future_ranking_bottom5.png"
FUTURE_RANKING_PRICE_DATA_NAME = "future_ranking_price_data.csv"
TIMES_SCORE_CSV_NAME = "future_ranking_scores.csv"
WINDOW_SUMMARY_CSV_NAME = "future_ranking_score_summary.csv"

PERCENT_METRICS = [
    "Buy Score",
    "Buy Precision",
    "Holding Score",
    "Holding Precision",
    "Sell Score",
    "Sell Precision",
]


def _normalize_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str).str.strip(), errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def _normalize_reward_type(reward_type: str | None) -> str:
    key = str(reward_type or "").strip().lower()
    aliases = {
        "reward_v2": "v2",
        "future_day_return": "v2",
        "reward_v3": "v3",
        "future_avg_return": "v3",
        "reward_v4": "v4",
        "future_avg_delta_action_return": "v4",
        "future_avg_delta_action_reward": "v4",
        "reward_v5": "v5",
        "weight_future": "v5",
        "reward_weight_future": "v5",
        "weight_future_reward": "v5",
        "portfolio_weighted_future_return": "v5",
        "portfolio_weighted_future_return_reward": "v5",
        "wft": "v5",
        "weighted_future_trend": "v5",
        "weighted_future_trend_reward": "v5",
        "reward_v6": "v6",
        "delta_weight_future": "v6",
        "reward_delta_weight_future": "v6",
        "delta_weight_future_reward": "v6",
        "portfolio_weight_change_future_return": "v6",
        "portfolio_weight_change_future_return_reward": "v6",
        "dwft": "v6",
        "delta_weighted_future_trend": "v6",
        "delta_weighted_future_trend_reward": "v6",
    }
    return aliases.get(key, key)


def _future_metric_name(reward_type: str, reward_params: dict[str, Any]) -> str:
    explicit_metric = str(reward_params.get("future_metric", "")).strip().lower()
    if explicit_metric:
        aliases = {
            "future_day_return": "future_day_return",
            "future_avg_return": "future_avg_return",
            "future_average": "future_avg_return",
            "future_avg_returns": "future_avg_return",
            "future_direction": "future_direction",
            "direction": "future_direction",
        }
        if explicit_metric not in aliases:
            raise ValueError(
                f"Unsupported Future Ranking future_metric={explicit_metric!r}."
            )
        return aliases[explicit_metric]

    reward_key = _normalize_reward_type(reward_type)
    if reward_key == "v2":
        return "future_day_return"
    if reward_key in {"v3", "v4", "v5"}:
        return "future_avg_return"
    if reward_key == "v6":
        future_variable = str(
            reward_params.get("future_variable", "future_direction")
        ).strip().lower()
        aliases = {
            "future_direction": "future_direction",
            "direction": "future_direction",
            "future_avg_return": "future_avg_return",
            "future_average": "future_avg_return",
            "future_avg_returns": "future_avg_return",
        }
        if future_variable not in aliases:
            raise ValueError(
                "Future Ranking cannot match reward v6 because "
                f"future_variable={future_variable!r} is unsupported."
            )
        return aliases[future_variable]
    raise ValueError(
        "Future Ranking requires a reward with an explicit future-price signal. "
        f"Current reward_type={reward_type!r}."
    )


def _load_trade_price_table(
    trade_data_path: Path,
    price_col: str,
) -> pd.DataFrame:
    trade = pd.read_csv(trade_data_path)
    if trade.empty or "date" not in trade.columns or "tic" not in trade.columns:
        raise ValueError(f"Invalid trade data for Future Ranking: {trade_data_path}")

    actual_price_col = price_col
    if actual_price_col not in trade.columns and "close" in trade.columns:
        actual_price_col = "close"
    if actual_price_col not in trade.columns:
        raise ValueError(
            f"Future Ranking price column {price_col!r} not found in {trade_data_path}."
        )

    price_data = trade[["date", "tic", actual_price_col]].copy()
    price_data["date"] = _normalize_date_series(price_data["date"])
    price_data = price_data[price_data["date"].notna()].copy()
    price_data["tic"] = price_data["tic"].astype(str)
    price_data[actual_price_col] = pd.to_numeric(
        price_data[actual_price_col], errors="coerce"
    )

    date_order = price_data["date"].drop_duplicates().tolist()
    ticker_order = price_data["tic"].drop_duplicates().tolist()
    price_table = price_data.pivot_table(
        index="date",
        columns="tic",
        values=actual_price_col,
        aggfunc="last",
    )
    return price_table.reindex(index=date_order, columns=ticker_order)


def _compute_future_signal(
    current_prices: np.ndarray,
    future_window: np.ndarray,
    metric_name: str,
    eps: float = 1e-8,
) -> np.ndarray:
    if metric_name == "future_day_return":
        future_prices = future_window[-1]
        return (future_prices - current_prices) / (current_prices + eps)

    if metric_name == "future_avg_return":
        avg_future_prices = np.nanmean(future_window, axis=0)
        return (avg_future_prices - current_prices) / (current_prices + eps)

    if metric_name == "future_direction":
        future_max_prices = np.nanmax(future_window, axis=0)
        future_min_prices = np.nanmin(future_window, axis=0)
        future_max_positions = np.argmax(future_window, axis=0)
        future_min_positions = np.argmin(future_window, axis=0)
        direction_signs = np.sign(
            future_max_positions - future_min_positions
        ).astype(float)
        return (
            direction_signs
            * (future_max_prices - future_min_prices)
            / (current_prices + eps)
        )

    raise ValueError(f"Unsupported Future Ranking metric: {metric_name}")


def compute_future_ranking(
    trade_data_path: Path,
    reward_type: str,
    reward_params: dict[str, Any],
) -> pd.DataFrame:
    """Compute Future Ranking for Test dates using extra future price data."""
    n_future = int(reward_params.get("n_future", 5))
    if n_future <= 0:
        raise ValueError(f"n_future must be positive, got {n_future}.")

    price_col = str(reward_params.get("price_col", "close"))
    metric_name = _future_metric_name(reward_type, reward_params)
    test_price_table = _load_trade_price_table(trade_data_path, price_col)
    if test_price_table.empty:
        raise ValueError(f"Future Ranking price table is empty: {trade_data_path}")

    future_price_path = trade_data_path.with_name(FUTURE_RANKING_PRICE_DATA_NAME)
    if future_price_path.exists():
        price_table = _load_trade_price_table(future_price_path, price_col)
    else:
        # Backward compatibility for older experiment folders. Their final
        # n_future Test days remain invalid because no post-Test prices exist.
        price_table = test_price_table.copy()

    dates = list(test_price_table.index)
    tickers = [str(column) for column in test_price_table.columns]
    price_table = price_table.reindex(columns=tickers)
    source_dates = list(price_table.index)
    source_date_to_pos = {date: index for index, date in enumerate(source_dates)}
    prices = price_table.to_numpy(dtype=float)
    stock_count = len(tickers)
    rows: list[dict[str, Any]] = []

    for date in dates:
        current_pos = source_date_to_pos.get(date)
        if current_pos is None:
            has_full_future_window = False
        else:
            start_pos = current_pos + 1
            end_pos = current_pos + n_future + 1
            has_full_future_window = end_pos <= len(source_dates)

        if current_pos is None:
            finite_prices = False
        elif has_full_future_window:
            current_prices = prices[current_pos]
            future_window = prices[start_pos:end_pos]
            finite_prices = (
                np.all(np.isfinite(current_prices))
                and np.all(np.isfinite(future_window))
                and np.all(current_prices > 1e-8)
            )
        else:
            finite_prices = False

        if not has_full_future_window or not finite_prices:
            for ticker in tickers:
                rows.append(
                    {
                        "Date": date,
                        "Ticker": ticker,
                        "Future Return": np.nan,
                        "Rank": np.nan,
                        "Top/Bottom Category": "Invalid",
                        "Rank Score": np.nan,
                        "Valid Day": False,
                        "Future Metric": metric_name,
                        "n_future": n_future,
                    }
                )
            continue

        future_signal = _compute_future_signal(
            current_prices=current_prices,
            future_window=future_window,
            metric_name=metric_name,
        )
        if not np.all(np.isfinite(future_signal)):
            for ticker in tickers:
                rows.append(
                    {
                        "Date": date,
                        "Ticker": ticker,
                        "Future Return": np.nan,
                        "Rank": np.nan,
                        "Top/Bottom Category": "Invalid",
                        "Rank Score": np.nan,
                        "Valid Day": False,
                        "Future Metric": metric_name,
                        "n_future": n_future,
                    }
                )
            continue

        sorted_indices = np.argsort(-future_signal, kind="stable")
        ranks = np.empty(stock_count, dtype=int)
        ranks[sorted_indices] = np.arange(1, stock_count + 1)

        for stock_index, ticker in enumerate(tickers):
            rank = int(ranks[stock_index])
            if rank <= 5:
                category = f"Top{rank}"
                rank_score = (6 - rank) / 15.0
            elif rank > stock_count - 5:
                bottom_rank = stock_count - rank + 1
                category = f"Bottom{bottom_rank}"
                rank_score = (6 - bottom_rank) / 15.0
            else:
                category = "Other"
                rank_score = 0.0

            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Future Return": float(future_signal[stock_index]),
                    "Rank": rank,
                    "Top/Bottom Category": category,
                    "Rank Score": rank_score,
                    "Valid Day": True,
                    "Future Metric": metric_name,
                    "n_future": n_future,
                }
            )

    return pd.DataFrame(rows)


def _plot_future_ranking(
    ranking: pd.DataFrame,
    out_path: Path,
    ranking_side: str,
) -> None:
    if ranking.empty:
        return

    if ranking_side not in {"top", "bottom"}:
        raise ValueError(f"Unsupported ranking_side: {ranking_side!r}")

    dates = ranking["Date"].drop_duplicates().tolist()
    tickers = ranking["Ticker"].drop_duplicates().tolist()
    date_to_pos = {date: index for index, date in enumerate(dates)}
    ticker_to_pos = {ticker: index for index, ticker in enumerate(tickers)}
    heatmap = np.zeros((len(tickers), len(dates)), dtype=int)

    prefix = "Top" if ranking_side == "top" else "Bottom"
    category_values = {
        f"{prefix}5": 1,
        f"{prefix}4": 2,
        f"{prefix}3": 3,
        f"{prefix}2": 4,
        f"{prefix}1": 5,
    }
    for _, row in ranking.iterrows():
        category = str(row["Top/Bottom Category"])
        value = category_values.get(category, 0)
        if value == 0:
            continue
        date = str(row["Date"])
        ticker = str(row["Ticker"])
        if date in date_to_pos and ticker in ticker_to_pos:
            heatmap[ticker_to_pos[ticker], date_to_pos[date]] = value

    color_map = plt.cm.Reds if ranking_side == "top" else plt.cm.Blues
    colors = [(1.0, 1.0, 1.0, 1.0)] + [
        color_map(value) for value in np.linspace(0.35, 0.9, 5)
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), cmap.N)

    fig_height = max(8, 0.24 * max(len(tickers), 1) + 1.8)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    image = ax.imshow(
        heatmap,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    ax.set_yticks(np.arange(len(tickers)))
    ax.set_yticklabels(tickers)

    tick_count = min(12, len(dates))
    if tick_count > 0:
        positions = np.linspace(0, len(dates) - 1, tick_count).astype(int)
        labels = [
            pd.to_datetime(dates[index], errors="coerce").strftime("%Y-%m")
            for index in positions
        ]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=0, ha="center")

    metric_name = str(ranking["Future Metric"].iloc[0]) if "Future Metric" in ranking.columns else "future"
    n_future = ranking["n_future"].iloc[0] if "n_future" in ranking.columns else "?"
    title_side = "Top5" if ranking_side == "top" else "Bottom5"
    ax.set_title(f"Future Ranking - {title_side} ({metric_name}, n={n_future})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stocks")

    ticks = [1, 2, 3, 4, 5]
    labels = [f"{prefix}{rank}" for rank in [5, 4, 3, 2, 1]]
    cbar = fig.colorbar(
        image,
        ax=ax,
        orientation="vertical",
        ticks=ticks,
        fraction=0.03,
        pad=0.02,
    )
    cbar.ax.set_yticklabels(labels)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ensure_window_future_ranking(
    window_dir: Path,
    trade_data_path: Path,
    reward_type: str,
    reward_params: dict[str, Any],
) -> pd.DataFrame:
    """Create the Window-level ranking CSV/PNG once and reuse them for all times."""
    ranking_csv_path = window_dir / RANKING_CSV_NAME
    ranking_top5_png_path = window_dir / RANKING_TOP5_PNG_NAME
    ranking_bottom5_png_path = window_dir / RANKING_BOTTOM5_PNG_NAME

    if ranking_csv_path.exists():
        ranking = pd.read_csv(ranking_csv_path)
    else:
        ranking = compute_future_ranking(
            trade_data_path=trade_data_path,
            reward_type=reward_type,
            reward_params=reward_params,
        )
        ranking.to_csv(ranking_csv_path, index=False, encoding="utf-8-sig")

    if not ranking_top5_png_path.exists():
        _plot_future_ranking(ranking, ranking_top5_png_path, "top")
    if not ranking_bottom5_png_path.exists():
        _plot_future_ranking(ranking, ranking_bottom5_png_path, "bottom")

    return ranking


def _read_diagnostic_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = _normalize_date_series(df["date"])
    return df[df["date"].notna()].reset_index(drop=True)


def _latest_matching_file(directory: Path, pattern: str) -> Path | None:
    candidates = list(directory.glob(pattern))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _format_percent(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.2f}%"


def _format_count(value: int | float) -> str:
    if not np.isfinite(float(value)):
        return "N/A"
    return str(int(value))


def _build_score_output(metrics: dict[str, float | int]) -> pd.DataFrame:
    rows = []
    for metric in PERCENT_METRICS:
        rows.append({"Metric": metric, "Value": _format_percent(float(metrics[metric]))})
    for metric in [
        "Total Buy Events",
        "Correct Buy Events",
        "Total Holding Events",
        "Correct Holding Events",
        "Total Sell Events",
        "Correct Sell Events",
        "Valid Days",
    ]:
        rows.append({"Metric": metric, "Value": _format_count(metrics[metric])})
    return pd.DataFrame(rows)


def score_times_future_ranking(
    experiment_dir: Path,
    ranking: pd.DataFrame,
    agent_name: str,
    reward_type: str,
) -> dict[str, float | int]:
    """Score one times using executed actions and post-trade weights."""
    actions_path = _latest_matching_file(
        experiment_dir,
        f"actions_{agent_name}_{reward_type}_*.csv",
    )
    weights_path = _latest_matching_file(
        experiment_dir,
        f"post_trade_weights_{agent_name}_{reward_type}_*.csv",
    )
    if actions_path is None or weights_path is None:
        missing = []
        if actions_path is None:
            missing.append("executed actions")
        if weights_path is None:
            missing.append("post-trade weights")
        raise FileNotFoundError(
            "Future Ranking score cannot be computed because "
            + ", ".join(missing)
            + f" are missing in {experiment_dir}."
        )

    actions = _read_diagnostic_csv(actions_path)
    weights = _read_diagnostic_csv(weights_path)
    if actions.empty or weights.empty:
        raise ValueError(f"Empty action/weight source in {experiment_dir}.")

    required_ranking_columns = {
        "Date",
        "Ticker",
        "Top/Bottom Category",
        "Rank Score",
        "Valid Day",
    }
    if not required_ranking_columns.issubset(ranking.columns):
        missing = sorted(required_ranking_columns - set(ranking.columns))
        raise ValueError(f"Future Ranking CSV missing columns: {missing}")

    ranking = ranking.copy()
    ranking["Date"] = _normalize_date_series(ranking["Date"])
    ranking["Ticker"] = ranking["Ticker"].astype(str)
    ranking["Rank Score"] = pd.to_numeric(ranking["Rank Score"], errors="coerce")
    if ranking["Valid Day"].dtype != bool:
        ranking["Valid Day"] = (
            ranking["Valid Day"].astype(str).str.strip().str.lower() == "true"
        )

    valid_dates = set(ranking.loc[ranking["Valid Day"], "Date"].dropna().unique())
    valid_day_count = len(valid_dates)
    lookup = ranking.set_index(["Date", "Ticker"])[
        ["Top/Bottom Category", "Rank Score", "Valid Day"]
    ]

    action_stock_columns = [column for column in actions.columns if column != "date"]
    weight_metadata = {
        "date",
        "cash_after_trade",
        "total_asset_after_trade",
        "cash_weight",
        "stock_weight_sum",
        "total_weight",
    }
    weight_stock_columns = [
        column for column in weights.columns if column not in weight_metadata
    ]
    ranking_tickers = set(ranking["Ticker"].unique())
    stock_columns = [
        column
        for column in action_stock_columns
        if column in weight_stock_columns and column in ranking_tickers
    ]
    if not stock_columns:
        raise ValueError(f"No common stock columns for Future Ranking score in {experiment_dir}.")

    actions = actions[actions["date"].isin(valid_dates)].copy()
    weights = weights[weights["date"].isin(valid_dates)].copy()
    actions = actions.drop_duplicates(subset=["date"], keep="last").set_index("date")
    weights = weights.drop_duplicates(subset=["date"], keep="last").set_index("date")

    total_buy = 0
    correct_buy = 0
    total_holding = 0
    correct_holding = 0
    total_sell = 0
    correct_sell = 0
    buy_score_sum = 0.0
    holding_score_sum = 0.0
    sell_score_sum = 0.0

    for date in sorted(valid_dates):
        action_row = actions.loc[date] if date in actions.index else None
        weight_row = weights.loc[date] if date in weights.index else None

        for ticker in stock_columns:
            try:
                rank_row = lookup.loc[(date, ticker)]
            except KeyError:
                continue
            category = str(rank_row["Top/Bottom Category"])
            rank_score = float(rank_row["Rank Score"])

            if action_row is not None:
                action_value = pd.to_numeric(
                    pd.Series([action_row[ticker]]), errors="coerce"
                ).iloc[0]
                if pd.notna(action_value) and action_value > 0:
                    total_buy += 1
                    if category.startswith("Top"):
                        correct_buy += 1
                        buy_score_sum += rank_score
                elif pd.notna(action_value) and action_value < 0:
                    total_sell += 1
                    if category.startswith("Bottom"):
                        correct_sell += 1
                        sell_score_sum += rank_score

            if weight_row is not None:
                weight_value = pd.to_numeric(
                    pd.Series([weight_row[ticker]]), errors="coerce"
                ).iloc[0]
                if pd.notna(weight_value) and weight_value > 0:
                    total_holding += 1
                    if category.startswith("Top"):
                        correct_holding += 1
                        holding_score_sum += rank_score

    buy_score = buy_score_sum / valid_day_count * 100.0 if valid_day_count else np.nan
    holding_score = (
        holding_score_sum / valid_day_count * 100.0 if valid_day_count else np.nan
    )
    sell_score = sell_score_sum / valid_day_count * 100.0 if valid_day_count else np.nan

    return {
        "Buy Score": buy_score,
        "Buy Precision": correct_buy / total_buy * 100.0 if total_buy else np.nan,
        "Holding Score": holding_score,
        "Holding Precision": (
            correct_holding / total_holding * 100.0 if total_holding else np.nan
        ),
        "Sell Score": sell_score,
        "Sell Precision": correct_sell / total_sell * 100.0 if total_sell else np.nan,
        "Total Buy Events": total_buy,
        "Correct Buy Events": correct_buy,
        "Total Holding Events": total_holding,
        "Correct Holding Events": correct_holding,
        "Total Sell Events": total_sell,
        "Correct Sell Events": correct_sell,
        "Valid Days": valid_day_count,
    }


def build_times_future_ranking_evaluation(
    window_dir: Path,
    experiment_dir: Path,
    agent_name: str,
    reward_type: str,
    reward_params: dict[str, Any],
) -> None:
    """Create/reuse Window ranking and save one times' score CSV."""
    trade_data_path = experiment_dir / "data" / "trade_data.csv"
    if not trade_data_path.exists():
        print(
            "Future Ranking skipped: missing trade_data.csv in "
            f"{experiment_dir / 'data'}"
        )
        return

    try:
        ranking = ensure_window_future_ranking(
            window_dir=window_dir,
            trade_data_path=trade_data_path,
            reward_type=reward_type,
            reward_params=reward_params,
        )
        metrics = score_times_future_ranking(
            experiment_dir=experiment_dir,
            ranking=ranking,
            agent_name=agent_name,
            reward_type=reward_type,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Future Ranking skipped: {exc}")
        return

    score_path = experiment_dir / TIMES_SCORE_CSV_NAME
    _build_score_output(metrics).to_csv(
        score_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Future Ranking score saved to: {score_path}")


def _parse_percent(value: Any) -> float:
    text = str(value).strip()
    if text.lower() in {"nan", "n/a", "na", ""}:
        return np.nan
    if text.endswith("%"):
        text = text[:-1]
    return float(text)


def build_window_future_ranking_summary(window_dir: Path) -> None:
    """Create the 5-times Window summary after all times have finished."""
    times_dirs = sorted(
        path
        for path in window_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"times_\d{2}_seed_\d+", path.name)
    )
    rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, float]] = []

    for times_dir in times_dirs:
        score_path = times_dir / TIMES_SCORE_CSV_NAME
        if not score_path.exists():
            continue
        score_df = pd.read_csv(score_path)
        if not {"Metric", "Value"}.issubset(score_df.columns):
            continue
        values = dict(zip(score_df["Metric"], score_df["Value"]))
        match = re.match(r"(times_\d{2})_seed_\d+", times_dir.name)
        times_label = match.group(1) if match else times_dir.name

        display_row: dict[str, Any] = {"Times": times_label}
        numeric_row: dict[str, float] = {}
        for metric in PERCENT_METRICS:
            numeric_value = _parse_percent(values.get(metric, "N/A"))
            numeric_row[metric] = numeric_value
            display_row[metric] = _format_percent(numeric_value)
        rows.append(display_row)
        numeric_rows.append(numeric_row)

    if not rows:
        print(f"Future Ranking summary skipped: no times scores in {window_dir}")
        return

    average_row: dict[str, Any] = {"Times": "Average"}
    for metric in PERCENT_METRICS:
        values = np.asarray([row[metric] for row in numeric_rows], dtype=float)
        average = float(np.mean(values)) if len(values) and np.all(np.isfinite(values)) else np.nan
        average_row[metric] = _format_percent(average)
    rows.append(average_row)

    summary_path = window_dir / WINDOW_SUMMARY_CSV_NAME
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Future Ranking 5-times summary saved to: {summary_path}")
