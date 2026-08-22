from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# =====================================================
# 1. User setting
# =====================================================

# Only change this folder name when analyzing another completed model.
INPUT_FOLDER = r"rolling_2018-2025_train3y_test1y_shift1y_20260819_220402_reinforce"

# Same experiment root style used by the current Compare scripts.
EXPERIMENT_ROOT = Path(
    os.environ.get(
        "EXPERIMENT_ROOT",
        r"C:\Users\User\Desktop\GraduationProject\Experence_result",
    )
)

# Current Future Ranking definition uses the average price of the next 60
# trading days. If a saved config explicitly records another ranking horizon,
# that value is preferred automatically.
DEFAULT_FUTURE_RANKING_DAYS = 60

ACTION_EPS = 1e-12
EXPECTED_TIMES = 5


# =====================================================
# 2. Folder helpers
# =====================================================

WINDOW_PATTERN = re.compile(
    r"^window_(?P<id>\d{2})_train_(?P<train_start>\d{4})-(?P<train_end>\d{4})_"
    r"test_(?P<test_start>\d{4})-(?P<test_end>\d{4})$"
)
TIMES_PATTERN = re.compile(r"^times_(?P<id>\d{2})_seed_\d+$")


def resolve_input_folder() -> Path:
    direct = Path(INPUT_FOLDER)
    if direct.is_dir():
        return direct.resolve()

    direct_under_root = EXPERIMENT_ROOT / INPUT_FOLDER
    if direct_under_root.is_dir():
        return direct_under_root.resolve()

    if not EXPERIMENT_ROOT.exists():
        raise FileNotFoundError(
            f"Experiment root does not exist: {EXPERIMENT_ROOT}\n"
            "Set EXPERIMENT_ROOT environment variable or update the default path."
        )

    matches = [
        path
        for path in EXPERIMENT_ROOT.rglob(INPUT_FOLDER)
        if path.is_dir() and path.name == INPUT_FOLDER
    ]
    if not matches:
        raise FileNotFoundError(
            f"Cannot find input folder '{INPUT_FOLDER}' under {EXPERIMENT_ROOT}"
        )

    if len(matches) > 1:
        matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        print(
            f"Warning: found {len(matches)} folders named '{INPUT_FOLDER}'. "
            f"Using newest: {matches[0]}"
        )

    return matches[0].resolve()


def find_windows(model_dir: Path) -> list[Path]:
    windows = [
        path
        for path in model_dir.iterdir()
        if path.is_dir() and WINDOW_PATTERN.fullmatch(path.name)
    ]
    return sorted(
        windows,
        key=lambda path: int(WINDOW_PATTERN.fullmatch(path.name).group("id")),
    )


def find_times(window_dir: Path) -> list[Path]:
    times_dirs = [
        path
        for path in window_dir.iterdir()
        if path.is_dir() and TIMES_PATTERN.fullmatch(path.name)
    ]
    return sorted(
        times_dirs,
        key=lambda path: int(TIMES_PATTERN.fullmatch(path.name).group("id")),
    )


def window_id(window_dir: Path) -> str:
    match = WINDOW_PATTERN.fullmatch(window_dir.name)
    if match is None:
        raise ValueError(f"Invalid window folder name: {window_dir.name}")
    return f"window_{int(match.group('id')):02d}"


def times_id(times_dir: Path) -> str:
    match = TIMES_PATTERN.fullmatch(times_dir.name)
    if match is None:
        raise ValueError(f"Invalid times folder name: {times_dir.name}")
    return f"times_{int(match.group('id')):02d}"


def test_date_range(window_dir: Path) -> tuple[str, str]:
    match = WINDOW_PATTERN.fullmatch(window_dir.name)
    if match is None:
        raise ValueError(f"Invalid window folder name: {window_dir.name}")
    return (
        f"{match.group('test_start')}-01-01",
        f"{match.group('test_end')}-12-31",
    )


# =====================================================
# 3. CSV loading helpers
# =====================================================

def normalize_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str).str.strip(), errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def normalize_wide_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "date" not in out.columns and len(out.columns) > 0:
        first_col = out.columns[0]
        parsed = pd.to_datetime(out[first_col], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            out = out.rename(columns={first_col: "date"})

    if "date" not in out.columns:
        raise ValueError("CSV has no usable date column")

    out["date"] = normalize_date_series(out["date"])
    out = out[out["date"].notna()].copy()
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True)


def read_csv_header(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, nrows=5)
    except Exception:
        return None


def looks_like_wide_stock_table(path: Path) -> bool:
    df = read_csv_header(path)
    if df is None or df.empty or len(df.columns) < 5:
        return False

    try:
        normalized = normalize_wide_table(df)
    except ValueError:
        return False

    value_cols = [col for col in normalized.columns if col != "date"]
    if len(value_cols) < 3:
        return False

    numeric = normalized[value_cols].apply(pd.to_numeric, errors="coerce")
    return bool(numeric.notna().any().sum() >= 3)


def find_role_csv(times_dir: Path, role: str) -> Path:
    csv_paths = [
        path
        for path in times_dir.rglob("*.csv")
        if "overall_score_analysis" not in path.parts
    ]

    if role == "actions":
        keywords = ("action", "actions")
        excluded = (
            "agent_action",
            "agent_actions",
            "policy_action",
            "raw_action",
            "train_action",
        )
    elif role == "holdings":
        keywords = ("holding", "holdings", "position")
        excluded = ("continuous",)
    else:
        raise ValueError(f"Unknown role: {role}")

    candidates: list[tuple[int, float, Path]] = []
    for path in csv_paths:
        name = path.name.lower()
        if not any(keyword in name for keyword in keywords):
            continue
        if any(token in name for token in excluded):
            continue
        if not looks_like_wide_stock_table(path):
            continue

        score = 0
        if role == "actions":
            if name.startswith("actions_") or name == "actions.csv":
                score += 50
            if "executed" in name or "final" in name:
                score += 20
            if "stock_turnover" in name:
                score += 10
        else:
            if name.startswith("holdings_") or name == "holdings.csv":
                score += 50
            if "backtest" in name:
                score += 5

        # Prefer files near the times root instead of nested diagnostic copies.
        score -= len(path.relative_to(times_dir).parts)
        candidates.append((score, path.stat().st_mtime, path))

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find a usable {role} CSV under {times_dir}"
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def load_actions(times_dir: Path) -> tuple[pd.DataFrame, Path]:
    # Latest Annual Reset output writes final executed actions directly here.
    preferred = sorted(times_dir.glob("actions_window_*.csv"))
    path = max(preferred, key=lambda p: p.stat().st_mtime) if preferred else find_role_csv(times_dir, "actions")
    df = normalize_wide_table(pd.read_csv(path))
    return df, path


def load_holdings(times_dir: Path) -> tuple[pd.DataFrame, Path]:
    # Latest Annual Reset output writes post-trade share holdings directly here.
    preferred = sorted(times_dir.glob("holdings_window_*.csv"))
    path = max(preferred, key=lambda p: p.stat().st_mtime) if preferred else find_role_csv(times_dir, "holdings")
    df = normalize_wide_table(pd.read_csv(path))
    return df, path


def load_future_ranking_price_data(times_dir: Path) -> tuple[pd.DataFrame, Path]:
    preferred = times_dir / "data" / "future_ranking_price_data.csv"
    if preferred.exists():
        path = preferred
    else:
        matches = sorted(times_dir.rglob("future_ranking_price_data.csv"))
        if not matches:
            raise FileNotFoundError(
                f"Cannot find future_ranking_price_data.csv under {times_dir}"
            )
        path = matches[0]

    df = pd.read_csv(path)
    required = {"date", "tic", "close"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Future Ranking source is missing columns {sorted(required - set(df.columns))}: {path}"
        )

    out = df[["date", "tic", "close"]].copy()
    out["date"] = normalize_date_series(out["date"])
    out["tic"] = out["tic"].astype(str)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "tic", "close"])
    out = out.drop_duplicates(subset=["date", "tic"], keep="last")
    return out.reset_index(drop=True), path


def find_saved_ranking_horizon(times_dir: Path) -> int | None:
    keys = {
        "future_ranking_days",
        "future_ranking_n_days",
        "future_ranking_n",
        "ranking_n_future",
        "ranking_horizon",
    }

    json_paths = list(times_dir.glob("*.json")) + list((times_dir / "data").glob("*.json"))
    for path in json_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        stack = [payload]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                for key, value in current.items():
                    if str(key).lower() in keys:
                        try:
                            value_int = int(value)
                        except (TypeError, ValueError):
                            continue
                        if value_int > 0:
                            return value_int
                    if isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(current, list):
                stack.extend(item for item in current if isinstance(item, (dict, list)))

    return None


# =====================================================
# 4. Future Ranking
# =====================================================

def build_future_ranking_matrix(
    price_data: pd.DataFrame,
    stock_order: list[str],
    n_future: int,
) -> pd.DataFrame:
    price_pivot = price_data.pivot(index="date", columns="tic", values="close")
    price_pivot = price_pivot.sort_index().reindex(columns=stock_order)

    prices = price_pivot.to_numpy(dtype=float)
    ranking = np.zeros_like(prices, dtype=float)
    ranking[:] = np.nan

    n_rows, n_stocks = prices.shape
    for row_idx in range(n_rows):
        end_idx = row_idx + n_future + 1
        if end_idx > n_rows:
            continue

        current_prices = prices[row_idx]
        future_window = prices[row_idx + 1 : end_idx]
        valid = (
            np.isfinite(current_prices)
            & (current_prices > 0)
            & np.all(np.isfinite(future_window), axis=0)
        )
        if int(np.sum(valid)) < 10:
            continue

        avg_future_prices = np.mean(future_window, axis=0)
        future_returns = np.full(n_stocks, np.nan, dtype=float)
        future_returns[valid] = (
            avg_future_prices[valid] - current_prices[valid]
        ) / current_prices[valid]

        valid_indices = np.where(np.isfinite(future_returns))[0]
        descending = valid_indices[np.argsort(future_returns[valid_indices])[::-1]]
        ascending = valid_indices[np.argsort(future_returns[valid_indices])]

        row_scores = np.zeros(n_stocks, dtype=float)
        for rank_index, stock_index in enumerate(descending[:5]):
            row_scores[stock_index] = 5 - rank_index
        for rank_index, stock_index in enumerate(ascending[:5]):
            row_scores[stock_index] = -5 + rank_index

        ranking[row_idx] = row_scores

    return pd.DataFrame(ranking, index=price_pivot.index, columns=stock_order)


# =====================================================
# 5. Trading behavior and score
# =====================================================

def stock_columns_from_actions(
    actions: pd.DataFrame,
    holdings: pd.DataFrame,
    ranking_source: pd.DataFrame,
) -> list[str]:
    metadata = {
        "date",
        "cash",
        "cash_after_sell",
        "cash_after_buy",
        "account_value",
        "total_asset",
        "total_asset_after_trade",
        "cash_weight",
        "stock_weight_sum",
        "total_weight",
    }
    ranking_tickers = set(ranking_source["tic"].astype(str).unique())
    holdings_columns = set(holdings.columns)

    return [
        str(column)
        for column in actions.columns
        if column not in metadata
        and str(column) in ranking_tickers
        and column in holdings_columns
    ]


def build_action_matrix(
    actions: pd.DataFrame,
    holdings: pd.DataFrame,
    stock_order: list[str],
    valid_dates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    action_values = actions.set_index("date").reindex(valid_dates)[stock_order]
    holding_values = holdings.set_index("date").reindex(valid_dates)[stock_order]

    action_values = action_values.apply(pd.to_numeric, errors="coerce")
    holding_values = holding_values.apply(pd.to_numeric, errors="coerce")

    score_action = pd.DataFrame(
        0.0,
        index=valid_dates,
        columns=stock_order,
    )
    behavior = pd.DataFrame(
        "None",
        index=valid_dates,
        columns=stock_order,
        dtype=object,
    )

    buy_mask = action_values > ACTION_EPS
    sell_mask = action_values < -ACTION_EPS
    hold_mask = (~buy_mask) & (~sell_mask) & (holding_values > ACTION_EPS)

    score_action[buy_mask] = 1.0
    score_action[hold_mask] = 1.0
    score_action[sell_mask] = -1.0

    behavior[buy_mask] = "Buy"
    behavior[hold_mask] = "Hold"
    behavior[sell_mask] = "Sell"

    return score_action, behavior


def calculate_daily_scores(
    ranking: pd.DataFrame,
    score_action: pd.DataFrame,
) -> pd.DataFrame:
    ranking_aligned = ranking.reindex(
        index=score_action.index,
        columns=score_action.columns,
    )
    cell_score = ranking_aligned * score_action

    top_score = cell_score.where(ranking_aligned > 0, 0.0).sum(axis=1)
    bottom_score = cell_score.where(ranking_aligned < 0, 0.0).sum(axis=1)
    daily_score = cell_score.sum(axis=1)

    result = pd.DataFrame(
        {
            "date": score_action.index,
            "top_score": top_score.to_numpy(),
            "bottom_score": bottom_score.to_numpy(),
            "daily_score": daily_score.to_numpy(),
        }
    )

    for column in ["top_score", "bottom_score", "daily_score"]:
        result[column] = result[column].round(10)
        if np.allclose(result[column], np.round(result[column]), equal_nan=True):
            result[column] = np.round(result[column]).astype(int)

    return result


def calculate_overall_score(daily_scores: pd.DataFrame) -> tuple[float, float]:
    if daily_scores.empty:
        raise ValueError("No valid daily score rows")

    average_daily_score = float(
        pd.to_numeric(daily_scores["daily_score"], errors="coerce").mean()
    )
    overall_score = (average_daily_score + 30.0) / 60.0 * 100.0
    return average_daily_score, overall_score


def valid_test_dates(
    actions: pd.DataFrame,
    holdings: pd.DataFrame,
    ranking: pd.DataFrame,
    test_start: str,
    test_end: str,
) -> list[str]:
    action_dates = set(actions["date"].dropna().astype(str))
    holding_dates = set(holdings["date"].dropna().astype(str))
    ranking_dates = set(ranking.dropna(how="all").index.astype(str))

    common = action_dates & holding_dates & ranking_dates
    return sorted(
        date
        for date in common
        if test_start <= date <= test_end
    )


# =====================================================
# 6. Combined plot
# =====================================================

RANK_COLORS = {
    -5: "#006d2c",  # Bottom1
    -4: "#238b45",
    -3: "#41ab5d",
    -2: "#74c476",
    -1: "#bae4b3",  # Bottom5
    0: "#ffffff",
    1: "#fcbba1",   # Top5
    2: "#fc9272",
    3: "#fb6a4a",
    4: "#de2d26",
    5: "#a50f15",   # Top1
}


def month_tick_positions(dates: list[str]) -> tuple[list[int], list[str]]:
    positions: list[int] = []
    labels: list[str] = []
    seen: set[str] = set()
    for index, date in enumerate(dates):
        month = str(date)[:7]
        if month not in seen:
            seen.add(month)
            positions.append(index)
            labels.append(month)
    return positions, labels


def plot_combined_heatmap(
    ranking: pd.DataFrame,
    behavior: pd.DataFrame,
    stock_order: list[str],
    dates: list[str],
    output_path: Path,
    window_label: str,
    times_label: str,
    overall_score: float,
) -> None:
    ranking_plot = ranking.reindex(index=dates, columns=stock_order).T
    behavior_plot = behavior.reindex(index=dates, columns=stock_order).T

    colors = [RANK_COLORS[value] for value in range(-5, 6)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-5.5, 6.5, 1.0), cmap.N)

    fig_width = max(18.0, min(28.0, len(dates) / 10.0))
    fig_height = max(9.0, len(stock_order) * 0.36)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.imshow(
        ranking_plot.to_numpy(dtype=float),
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    marker_styles = {
        "Buy": {
            "marker": "^",
            "s": 30,
            "facecolors": "#111111",
            "edgecolors": "white",
            "linewidths": 0.45,
            "alpha": 1.0,
        },
        "Hold": {
            "marker": "o",
            "s": 10,
            "facecolors": "#222222",
            "edgecolors": "none",
            "linewidths": 0.0,
            "alpha": 0.45,
        },
        "Sell": {
            "marker": "v",
            "s": 30,
            "facecolors": "#111111",
            "edgecolors": "white",
            "linewidths": 0.45,
            "alpha": 1.0,
        },
    }
    for behavior_name, style in marker_styles.items():
        y_indices, x_indices = np.where(
            behavior_plot.to_numpy(dtype=object) == behavior_name
        )
        if len(x_indices) == 0:
            continue
        ax.scatter(
            x_indices,
            y_indices,
            marker=style["marker"],
            s=style["s"],
            facecolors=style["facecolors"],
            edgecolors=style["edgecolors"],
            linewidths=style["linewidths"],
            alpha=style["alpha"],
            zorder=3,
        )

    ax.set_yticks(np.arange(len(stock_order)))
    ax.set_yticklabels(stock_order)

    tick_positions, tick_labels = month_tick_positions(dates)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="left")

    ax.set_xlabel("Test Date")
    ax.set_ylabel("Stock")
    ax.set_title(
        "Future Ranking × Trading Behavior\n"
        f"{window_label} | {times_label} | Overall Score: {overall_score:.2f}"
    )

    behavior_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_styles[name]["marker"],
            linestyle="None",
            markerfacecolor=marker_styles[name]["facecolors"],
            markeredgecolor=marker_styles[name]["edgecolors"],
            markeredgewidth=0.6 if name != "Hold" else 0.0,
            markersize=10 if name != "Hold" else 7,
            alpha=marker_styles[name]["alpha"],
            label=name,
        )
        for name in ["Buy", "Hold", "Sell"]
    ]

    # Keep the heatmap strictly inside the available Test dates.
    ax.set_xlim(-0.5, len(dates) - 0.5)
    ax.margins(x=0)
    ax.set_ylim(len(stock_order) - 0.5, -0.5)

    # Reserve a narrow area on the right for Action and Future Ranking.
    fig.tight_layout(rect=[0, 0, 0.92, 1])

    # Action remains a normal figure-level legend.
    fig.legend(
        handles=behavior_handles,
        title="Action",
        loc="upper left",
        bbox_to_anchor=(0.925, 0.93),
        borderaxespad=0.0,
        ncol=1,
        frameon=True,
        fontsize=10,
        title_fontsize=11,
        labelspacing=0.6,
        borderpad=0.6,
    )

    # Future Ranking uses a vertical colorbar so its width and height can be
    # controlled directly by [left, bottom, width, height].
    colorbar_ax = fig.add_axes([0.935, 0.05, 0.012, 0.75])
    rank_colorbar = ColorbarBase(
        colorbar_ax,
        cmap=cmap,
        norm=norm,
        boundaries=np.arange(-5.5, 6.5, 1.0),
        ticks=np.arange(-5, 6),
        orientation="vertical",
    )

    rank_colorbar.ax.set_yticklabels(
        [
            "Bottom1",
            "Bottom2",
            "Bottom3",
            "Bottom4",
            "Bottom5",
            "",
            "Top5",
            "Top4",
            "Top3",
            "Top2",
            "Top1",
        ]
    )
    rank_colorbar.ax.tick_params(labelsize=8, pad=3, length=3)
    rank_colorbar.ax.set_title("Future Ranking", fontsize=9, pad=6)
    rank_colorbar.outline.set_linewidth(1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# 7. Per-times / per-window processing
# =====================================================

def process_times(
    window_dir: Path,
    times_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    actions, actions_path = load_actions(times_dir)
    holdings, holdings_path = load_holdings(times_dir)
    ranking_source, ranking_source_path = load_future_ranking_price_data(times_dir)

    stock_order = stock_columns_from_actions(actions, holdings, ranking_source)
    if len(stock_order) < 10:
        raise ValueError(
            f"Only found {len(stock_order)} common stock columns among actions, holdings, and Future Ranking data"
        )

    saved_horizon = find_saved_ranking_horizon(times_dir)
    n_future = saved_horizon or DEFAULT_FUTURE_RANKING_DAYS
    ranking = build_future_ranking_matrix(
        ranking_source,
        stock_order=stock_order,
        n_future=n_future,
    )

    test_start, test_end = test_date_range(window_dir)
    dates = valid_test_dates(
        actions,
        holdings,
        ranking,
        test_start=test_start,
        test_end=test_end,
    )
    if not dates:
        raise ValueError(
            "No common valid dates among Test period, Future Ranking, actions, and holdings"
        )

    score_action, behavior = build_action_matrix(
        actions,
        holdings,
        stock_order=stock_order,
        valid_dates=dates,
    )
    ranking_valid = ranking.reindex(index=dates, columns=stock_order)
    daily_scores = calculate_daily_scores(ranking_valid, score_action)
    average_daily_score, overall_score = calculate_overall_score(daily_scores)

    if not daily_scores["daily_score"].between(-30, 30).all():
        raise ValueError("Daily Score exceeded theoretical range [-30, 30]")

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_scores.to_csv(
        output_dir / "future_ranking_daily_score.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.2f",
    )

    plot_combined_heatmap(
        ranking=ranking_valid,
        behavior=behavior,
        stock_order=stock_order,
        dates=dates,
        output_path=output_dir / "future_ranking_trading_behavior.png",
        window_label=window_id(window_dir),
        times_label=times_id(times_dir),
        overall_score=overall_score,
    )

    return {
        "times": times_id(times_dir),
        "average_daily_score": average_daily_score,
        "overall_score": overall_score,
        "valid_days": len(dates),
        "future_ranking_days": n_future,
        "future_ranking_source": str(ranking_source_path),
        "actions_source": str(actions_path),
        "holdings_source": str(holdings_path),
    }


def save_window_summary(rows: list[dict[str, object]], output_path: Path) -> dict[str, float]:
    if not rows:
        raise ValueError("No valid times rows for this window")

    summary = pd.DataFrame(rows)[
        ["times", "average_daily_score", "overall_score"]
    ].copy()
    summary = summary.sort_values("times").reset_index(drop=True)

    average_row = {
        "times": "Average",
        "average_daily_score": float(summary["average_daily_score"].mean()),
        "overall_score": float(summary["overall_score"].mean()),
    }
    summary = pd.concat([summary, pd.DataFrame([average_row])], ignore_index=True)
    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.2f",
    )
    return average_row


def save_model_summaries(
    model_rows: list[dict[str, object]],
    analysis_dir: Path,
) -> None:
    all_windows_rows: list[dict[str, object]] = []
    window_average_rows: list[dict[str, object]] = []

    for window_row in model_rows:
        row: dict[str, object] = {"window": window_row["window"]}
        scores: dict[str, float] = window_row["times_scores"]
        for index in range(1, EXPECTED_TIMES + 1):
            key = f"times_{index:02d}"
            row[key] = scores.get(key, np.nan)
        row["average"] = window_row["average_overall_score"]
        all_windows_rows.append(row)

        window_average_rows.append(
            {
                "window": window_row["window"],
                "average_overall_score": window_row["average_overall_score"],
            }
        )

    pd.DataFrame(all_windows_rows).to_csv(
        analysis_dir / "future_ranking_overall_score_all_windows.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.2f",
    )
    pd.DataFrame(window_average_rows).to_csv(
        analysis_dir / "future_ranking_window_average_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.2f",
    )


# =====================================================
# 8. Main
# =====================================================

def main() -> None:
    model_dir = resolve_input_folder()
    windows = find_windows(model_dir)
    if not windows:
        raise FileNotFoundError(f"No window folders found under {model_dir}")

    analysis_dir = model_dir / "overall_score_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("Plot layout: Action legend + vertical Future Ranking colorbar")
    print(f"Input:\n{model_dir.name}\n")

    model_rows: list[dict[str, object]] = []

    for window_dir in windows:
        current_window_id = window_id(window_dir)
        print(f"Processing {current_window_id}")

        times_dirs = find_times(window_dir)
        if len(times_dirs) < EXPECTED_TIMES:
            print(
                f"Warning: {current_window_id} only found {len(times_dirs)} times folder(s)."
            )

        valid_rows: list[dict[str, object]] = []
        for times_dir in times_dirs:
            current_times_id = times_id(times_dir)
            times_output_dir = analysis_dir / current_window_id / current_times_id
            try:
                row = process_times(window_dir, times_dir, times_output_dir)
            except Exception as exc:
                print(f"  Warning: {current_times_id} skipped: {exc}")
                continue

            valid_rows.append(row)
            print(
                f"  {current_times_id} ... Overall Score = {float(row['overall_score']):.2f}"
            )

        if not valid_rows:
            print(f"  Warning: no valid times found for {current_window_id}.\n")
            continue

        if len(valid_rows) < EXPECTED_TIMES:
            print(
                f"Warning: {current_window_id} only found {len(valid_rows)} valid times."
            )

        window_output_dir = analysis_dir / current_window_id
        average_row = save_window_summary(
            valid_rows,
            window_output_dir / "future_ranking_overall_score_summary.csv",
        )
        window_average = float(average_row["overall_score"])
        print(f"  Window Average = {window_average:.2f}\n")

        model_rows.append(
            {
                "window": current_window_id,
                "times_scores": {
                    str(row["times"]): float(row["overall_score"])
                    for row in valid_rows
                },
                "average_overall_score": window_average,
            }
        )

    if not model_rows:
        raise RuntimeError("No valid Window × times results were produced")

    save_model_summaries(model_rows, analysis_dir)

    print("Model Summary:")
    for row in model_rows:
        print(
            f"{row['window']} = {float(row['average_overall_score']):.2f}"
        )

    print("\nDone.")
    print(f"\nOutput:\n{analysis_dir}")


if __name__ == "__main__":
    main()
