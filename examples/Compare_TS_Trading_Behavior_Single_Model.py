from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

import TS_turnover_graph as ts_graph
from result_manager import load_experiment_metadata
from finrl.meta.env_stock_trading.trading_score import (
    calculate_annual_ts,
    calculate_single_stock_annual_ts,
    normalize_ts_version,
)

# 單一模型的 TS 交易行為圖：30 檔大圖及單股股價走勢圖

# =====================================================
# 1. User setting
# =====================================================
# 平常只需要修改下面這個 rolling 資料夾名稱。
# 可填 rolling_* 資料夾名稱，也可填完整路徑。
SOURCE_FOLDER = (
    r"rolling_2018-2025_train3y_test1y_shift1y_20260912_000936_reinforce"
)

EXPERIMENT_ROOT = Path(
    os.environ.get(
        "EXPERIMENT_ROOT",
        r"C:\Users\User\Desktop\GraduationProject\Experence_result",
    )
)

EXPECTED_TIMES = 5

# v1 = TS: runlen; v2 = TS_v2: 1 / runlen
TS_VERSION = "v2"

REWARD_DISPLAY_NAMES = {
    "v0": "FinRL",
    "v1": "Buy-Low-Sell-High",
    "v2": "Future-Day",
    "v3": "Future-Avg",
    "v4": "Future-Avg-Delta-Action",
    "v5": "WFT",
    "v6": "DWFT",
    "v7": "AWFT",
}


# =====================================================
# 2. Source / folder helpers
# =====================================================
def _safe_name(text: str, max_len: int = 90) -> str:
    text = re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(text).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "output")[:max_len]


def resolve_folder(folder_value: str, source_name: str) -> Path:
    text = str(folder_value or "").strip().strip('"').strip("'")
    if not text:
        raise ValueError(f"{source_name} 尚未設定 rolling 資料夾。")

    direct = Path(text).expanduser()
    if direct.is_dir():
        return direct.resolve()

    direct_under_root = EXPERIMENT_ROOT / text
    if direct_under_root.is_dir():
        return direct_under_root.resolve()

    if not EXPERIMENT_ROOT.exists():
        raise FileNotFoundError(
            f"Experiment root does not exist: {EXPERIMENT_ROOT}\n"
            "Set EXPERIMENT_ROOT environment variable or update the default path."
        )

    matches = [
        path
        for path in EXPERIMENT_ROOT.rglob(text)
        if path.is_dir() and path.name == text
    ]
    if not matches:
        raise FileNotFoundError(
            f"Cannot find {source_name} folder '{text}' under {EXPERIMENT_ROOT}"
        )

    if len(matches) > 1:
        matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        print(
            f"Warning: found {len(matches)} folders named '{text}' for "
            f"{source_name}. Using newest: {matches[0]}"
        )

    return matches[0].resolve()


def reward_display_name(rolling_root: Path) -> tuple[str, str]:
    metadata = load_experiment_metadata(rolling_root)
    reward_type = str(metadata.get("reward_type", "") or "").strip().lower()
    if reward_type.startswith("reward_"):
        reward_type = reward_type[len("reward_") :]

    display_name = REWARD_DISPLAY_NAMES.get(reward_type)
    if display_name:
        return display_name, reward_type

    # Fallback for old folders without metadata.
    for part in rolling_root.parts[::-1]:
        match = re.fullmatch(r"reward_(.+)", part, flags=re.IGNORECASE)
        if match:
            reward_type = match.group(1).lower()
            return REWARD_DISPLAY_NAMES.get(reward_type, match.group(1)), reward_type

    return rolling_root.name, reward_type


def window_year(window_dir: Path) -> str:
    test_start, test_end = ts_graph.test_date_range(window_dir)
    start_year = test_start[:4]
    end_year = test_end[:4]
    if start_year == end_year:
        return start_year
    return f"{start_year}-{end_year}"


def windows_by_year(rolling_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for window_dir in ts_graph.find_windows(rolling_root):
        year = window_year(window_dir)
        if year in result:
            raise ValueError(
                f"Duplicate Test year {year} under {rolling_root}: "
                f"{result[year].name}, {window_dir.name}"
            )
        result[year] = window_dir
    return result


def times_by_id(window_dir: Path) -> dict[str, Path]:
    return {
        ts_graph.times_id(times_dir): times_dir
        for times_dir in ts_graph.find_times(window_dir)
    }


def make_output_dir(source_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        EXPERIMENT_ROOT
        / "single_ts_trading_behavior"
        / f"{_safe_name(source_name)}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def append_experiment_history(
    output_root: Path,
    source_root: Path,
    source_name: str,
    reward_type: str,
) -> Path:
    """Append one single-model execution to its own experiment history CSV."""
    history_path = (
        EXPERIMENT_ROOT
        / "single_ts_trading_behavior"
        / "single_experiment_history.csv"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)

    history_row = pd.DataFrame(
        [
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source_name": source_name,
                "reward_type": reward_type,
                "source_folder": source_root.name,
                "source_path": str(source_root),
                "output_folder": output_root.name,
                "output_path": str(output_root),
            }
        ]
    )
    history_row.to_csv(
        history_path,
        mode="a",
        header=not history_path.exists(),
        index=False,
        encoding="utf-8-sig",
    )
    return history_path


# =====================================================
# 3. Load one source for one Window x times
# =====================================================
def stock_columns_from_action_holdings(
    actions: pd.DataFrame,
    holdings: pd.DataFrame,
) -> list[str]:
    """Get stock columns without depending on a source-specific price file.

    Stock columns are taken from actions and holdings, not the price CSV.
    """
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
    holdings_columns = {str(column) for column in holdings.columns}

    return [
        str(column)
        for column in actions.columns
        if str(column) not in metadata
        and str(column) in holdings_columns
    ]


def build_plot_behavior_matrix(
    actions: pd.DataFrame,
    holdings: pd.DataFrame,
    stock_order: list[str],
    dates: list[str],
) -> pd.DataFrame:
    """Build plot-only action labels without changing Trading Score actions.

    FinRL saves actions with the trade-date label, but holdings_memory starts
    with the pre-trade state. Therefore the holdings row at the next trading
    date represents the position remaining after the current day's action.

    For each stock:
    - Buy: action > 0.
    - Partially Sell: action < 0 and shares remain after the trade.
    - Sell All: action < 0 and shares become zero after the trade.
    Hold/None are intentionally not plotted.
    """
    action_values = (
        actions.set_index("date")
        .reindex(dates)[stock_order]
        .apply(pd.to_numeric, errors="coerce")
    )

    holdings_by_date = (
        holdings.set_index("date")[stock_order]
        .apply(pd.to_numeric, errors="coerce")
    )

    # holdings(date=t) is the position before Action(t).
    # The next holdings row is the position after Action(t).
    post_trade_holdings = holdings_by_date.shift(-1).reindex(dates)

    action_eps = float(getattr(ts_graph, "ACTION_EPS", 1e-12))
    holding_zero_eps = max(action_eps, 1e-6)

    behavior = pd.DataFrame(
        "None",
        index=dates,
        columns=stock_order,
        dtype=object,
    )

    buy_mask = action_values > action_eps
    sell_mask = action_values < -action_eps
    remaining_mask = post_trade_holdings > holding_zero_eps
    cleared_mask = post_trade_holdings.abs() <= holding_zero_eps

    partially_sell_mask = sell_mask & remaining_mask
    sell_all_mask = sell_mask & cleared_mask

    behavior[buy_mask] = "Buy"
    behavior[partially_sell_mask] = "Partially Sell"
    behavior[sell_all_mask] = "Sell All"
    return behavior


def load_source_times(
    window_dir: Path,
    times_dir: Path,
) -> dict[str, object]:
    actions, actions_path = ts_graph.load_actions(times_dir)
    holdings, holdings_path = ts_graph.load_holdings(times_dir)
    price_data, price_source_path = ts_graph.load_price_data(times_dir)

    # Use actions/holdings to determine stocks; the price CSV may be incomplete.
    stock_order = stock_columns_from_action_holdings(actions, holdings)
    if not stock_order:
        raise ValueError(f"No common action/holding stock columns found under {times_dir}")

    n_future = (
        ts_graph.find_saved_future_days(times_dir)
        or ts_graph.DEFAULT_FUTURE_DAYS
    )

    test_start, test_end = ts_graph.test_date_range(window_dir)
    dates = ts_graph.test_action_dates(
        actions,
        test_start=test_start,
        test_end=test_end,
    )

    act, _ = ts_graph.build_action_matrix(
        actions,
        holdings,
        stock_order=stock_order,
        dates=dates,
    )
    behavior = build_plot_behavior_matrix(
        actions,
        holdings,
        stock_order=stock_order,
        dates=dates,
    )

    return {
        "stock_order": stock_order,
        "dates": dates,
        "n_future": n_future,
        "price_data": price_data,
        "act": act,
        "behavior": behavior,
        "actions_source": actions_path,
        "holdings_source": holdings_path,
        "price_source": price_source_path,
    }


def _build_trend_from_price_data(
    price_data: pd.DataFrame,
    stock_order: list[str],
    dates: list[str],
    n_future: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and validate d, runlen and s for the full Test period."""
    price_pivot = ts_graph.build_price_pivot(
        price_data,
        stock_order=stock_order,
    )
    d_matrix = ts_graph.build_d_matrix(
        price_pivot,
        n_future=n_future,
    )
    runlen_matrix = ts_graph.build_runlen_matrix(price_pivot)

    ts_graph.validate_full_ts_period(
        d_matrix=d_matrix,
        runlen_matrix=runlen_matrix,
        stock_order=stock_order,
        action_dates=dates,
        n_future=n_future,
    )

    d_valid = d_matrix.reindex(index=dates, columns=stock_order)
    runlen_valid = runlen_matrix.reindex(index=dates, columns=stock_order)
    s_valid = ts_graph.build_stock_trend_score(
        d_valid,
        runlen_valid,
        version=TS_VERSION,
    )
    return d_valid, runlen_valid, s_valid


# =====================================================
# 4. Annual TS calculation
# =====================================================
def single_stock_score_table(
    stock_trend_score: pd.DataFrame,
    act: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker in stock_trend_score.columns:
        rows.append(
            {
                "tic": ticker,
                f"{source_name}_Annual_TS": calculate_single_stock_annual_ts(
                    stock_trend_score[ticker],
                    act[ticker],
                ),
            }
        )
    return pd.DataFrame(rows)


def save_year_average_annual_ts(
    summary_df: pd.DataFrame,
    output_path: Path,
    source_name: str,
) -> pd.DataFrame:
    """Save yearly Average Annual TS only if all expected runs are valid."""
    source_col = f"{source_name}_Annual_TS"
    rows: list[dict[str, object]] = []
    for year, year_df in summary_df.groupby("year", sort=True):
        times_count = int(year_df["times"].nunique())
        complete = times_count == EXPECTED_TIMES
        rows.append(
            {
                "year": year,
                "times_count": times_count,
                "expected_times": EXPECTED_TIMES,
                "complete_5_times": complete,
                f"{source_name}_Average_Annual_TS": (
                    float(year_df[source_col].mean()) if complete else np.nan
                ),
            }
        )
        if not complete:
            print(
                f"Warning: {year} has {times_count}/{EXPECTED_TIMES} valid "
                "times; yearly 5-times average Annual TS is left blank."
            )

    result = pd.DataFrame(rows)
    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f",
    )
    return result


# =====================================================
# 5. Plot helpers
# =====================================================
RISE_COLOR = np.array([214, 39, 40], dtype=float) / 255.0
FALL_COLOR = np.array([44, 160, 44], dtype=float) / 255.0
WHITE = np.array([1.0, 1.0, 1.0], dtype=float)

MARKER_STYLES = {
    "Buy": {
        "marker": "^",
        "s": 32,
        "line_s": 72,
        "facecolors": "#FF0000",
        "edgecolors": "white",
        "linewidths": 0.45,
        "alpha": 1.0,
    },
    "Partially Sell": {
        "marker": "v",
        "s": 32,
        "line_s": 72,
        "facecolors": "#006eff",
        "edgecolors": "white",  
        "linewidths": 0.45,
        "alpha": 1.0,
    },
    "Sell All": {
        "marker": "X",
        "s": 32,
        "line_s": 72,
        "facecolors": "#00d40b",
        "edgecolors": "white",
        "linewidths": 0.45,
        "alpha": 1.0,
    },
}


def _style_box(ax) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#aaaaaa")
        spine.set_linewidth(0.9)


def _add_right_legends(
    fig,
    main_axes,
    source_name: str,
) -> None:
    if isinstance(main_axes, (list, tuple, np.ndarray)):
        positions = [ax.get_position() for ax in main_axes]
        main_pos = Bbox.union(positions)
    else:
        main_pos = main_axes.get_position()
    left = main_pos.x1 + 0.008
    right = 0.985
    width = max(0.055, right - left)

    # Keep the right-side legend column narrow and tall, similar to the
    # turnover graph layout instead of using large square boxes.
    row_ax = fig.add_axes([left, 0.77, width, 0.17])
    action_ax = fig.add_axes([left, 0.57, width, 0.18])
    trend_ax = fig.add_axes([left, 0.08, width, 0.45])

    for legend_ax in [row_ax, action_ax, trend_ax]:
        _style_box(legend_ax)

    # Row Labels box: title on top, with a separate 2 x 1 table underneath.
    row_ax.text(
        0.5,
        0.88,
        "Row Labels",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="semibold",
    )

    table_left = 0.08
    table_right = 0.92
    table_bottom = 0.08
    table_top = 0.73
    table_height = table_top - table_bottom
    table_edgecolor = "#aaaaaa"
    table_linewidth = 0.9

    row_ax.add_patch(
        Rectangle(
            (table_left, table_bottom),
            table_right - table_left,
            table_height,
            fill=False,
            edgecolor=table_edgecolor,
            linewidth=table_linewidth,
        )
    )
    row_ax.hlines(
        [table_bottom + table_height / 2.0],
        xmin=table_left,
        xmax=table_right,
        colors=table_edgecolor,
        linewidth=table_linewidth,
    )

    row_ax.text(
        0.5,
        table_bottom + table_height * 3.0 / 4.0,
        "Trend",
        ha="center",
        va="center",
        fontsize=12.5,
    )
    row_ax.text(
        0.5,
        table_bottom + table_height * 1.0 / 4.0,
        source_name,
        ha="center",
        va="center",
        fontsize=11.5,
        wrap=True,
    )

    # Action box.
    action_ax.text(
        0.5,
        0.86,
        "Action",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="semibold",
    )
    action_y = {"Buy": 0.61, "Partially Sell": 0.39, "Sell All": 0.17}
    for action_name in ["Buy", "Partially Sell", "Sell All"]:
        style = MARKER_STYLES[action_name]
        action_ax.scatter(
            [0.24],
            [action_y[action_name]],
            marker=style["marker"],
            s=185,
            facecolors=style["facecolors"],
            edgecolors=style["edgecolors"],
            linewidths=0.8,
            alpha=style["alpha"],
        )
        action_ax.text(
            0.43,
            action_y[action_name],
            action_name,
            ha="left",
            va="center",
            fontsize=11.5,
        )

    # Trend box.
    trend_ax.text(
        0.5,
        0.94,
        "Trend",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="semibold",
    )

    gradient_steps = 128
    light_rise = WHITE * 0.78 + RISE_COLOR * 0.22
    light_fall = WHITE * 0.78 + FALL_COLOR * 0.22
    rise_gradient = np.linspace(RISE_COLOR, light_rise, gradient_steps)
    fall_gradient = np.linspace(light_fall, FALL_COLOR, gradient_steps)
    trend_gradient = np.vstack([rise_gradient, fall_gradient]).reshape(
        gradient_steps * 2,
        1,
        3,
    )

    bar_left = 0.36
    bar_right = 0.64
    bar_bottom = 0.10
    bar_top = 0.80

    trend_ax.text(
        0.5,
        bar_top + 0.045,
        "Rise",
        ha="center",
        va="bottom",
        fontsize=12.5,
    )
    trend_ax.imshow(
        trend_gradient,
        extent=[bar_left, bar_right, bar_bottom, bar_top],
        aspect="auto",
        interpolation="bilinear",
        origin="upper",
        zorder=2,
    )
    trend_ax.add_patch(
        Rectangle(
            (bar_left, bar_bottom),
            bar_right - bar_left,
            bar_top - bar_bottom,
            fill=False,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
    )
    trend_ax.text(
        0.5,
        bar_bottom - 0.035,
        "Fall",
        ha="center",
        va="top",
        fontsize=12.5,
    )


def _plot_action_markers(
    ax,
    behavior: pd.DataFrame,
    row_offset: int,
) -> None:
    behavior_values = behavior.to_numpy(dtype=object)
    for behavior_name, style in MARKER_STYLES.items():
        stock_indices, x_indices = np.where(
            behavior_values == behavior_name
        )
        if len(x_indices) == 0:
            continue

        y_indices = stock_indices * 2 + row_offset
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


def _plot_single_stock_action_line(
    ax,
    price_series: pd.Series,
    behavior_series: pd.Series,
    row_label: str,
    show_xticklabels: bool,
) -> None:
    x = np.arange(len(price_series))
    y = pd.to_numeric(price_series, errors="coerce").to_numpy(dtype=float)

    ax.plot(
        x,
        y,
        color="#000000",
        linewidth=1,
        zorder=2,
    )

    behavior_aligned = behavior_series.reindex(price_series.index)
    for behavior_name, style in MARKER_STYLES.items():
        mask = (behavior_aligned == behavior_name).to_numpy(dtype=bool)
        if not np.any(mask):
            continue
        ax.scatter(
            x[mask],
            y[mask],
            marker=style["marker"],
            s=style["line_s"],
            facecolors=style["facecolors"],
            edgecolors=style["edgecolors"],
            linewidths=style["linewidths"],
            alpha=style["alpha"],
            zorder=3,
        )

    ax.set_ylabel(row_label)
    ax.grid(True, axis="both", color="#d0d0d0", linewidth=0.35, alpha=0.9)
    ax.set_xlim(-0.5, len(price_series) - 0.5)
    ax.margins(x=0)

    if not show_xticklabels:
        ax.tick_params(axis="x", labelbottom=False)


def plot_single_stock(
    trend_rgb: np.ndarray,
    price_series: pd.Series,
    behavior: pd.Series,
    ticker: str,
    dates: list[str],
    output_path: Path,
    year_label: str,
    times_label: str,
    source_name: str,
    annual_ts: float,
) -> None:
    n_dates = len(dates)
    if trend_rgb.shape != (1, n_dates, 3):
        raise ValueError(
            "Single-stock trend RGB shape mismatch: "
            f"expected {(1, n_dates, 3)}, got {trend_rgb.shape}"
        )

    price_series = pd.to_numeric(price_series.reindex(dates), errors="coerce")
    behavior = behavior.reindex(dates)
    if price_series.isna().any():
        raise ValueError(f"Single-stock price series contains missing values: {ticker}")

    fig_width = max(18.0, min(28.0, len(dates) / 10.0))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_width, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [0.30, 1.0]},
    )
    trend_ax, action_ax = axes

    trend_ax.imshow(
        trend_rgb,
        aspect="auto",
        interpolation="nearest",
        extent=[-0.5, n_dates - 0.5, 0.5, -0.5],
    )
    trend_ax.set_yticks([0])
    trend_ax.set_yticklabels([ticker])
    trend_ax.tick_params(axis="x", labelbottom=False)

    _plot_single_stock_action_line(
        action_ax,
        price_series=price_series,
        behavior_series=behavior,
        row_label=source_name,
        show_xticklabels=True,
    )

    tick_positions, tick_labels = ts_graph.month_tick_positions(dates)
    for axis in axes:
        axis.set_xticks(tick_positions)
        axis.set_xlim(-0.5, n_dates - 0.5)
        axis.margins(x=0)
    action_ax.set_xticklabels(tick_labels, rotation=0, ha="left")

    trend_ax.set_title(
        "Stock Trend x Agent Action\n"
        f"{year_label} | {times_label} | "
        f"{source_name} Annual TS: {annual_ts:.2f}",
        fontsize=18,
        fontweight="normal",
        pad=14,
    )
    action_ax.set_xlabel("Test Date")

    fig.tight_layout(rect=[0, 0, 0.91, 1])
    _add_right_legends(fig, axes, source_name=source_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_single_model_heatmap(
    trend_rgb: np.ndarray,
    behavior: pd.DataFrame,
    stock_order: list[str],
    dates: list[str],
    output_path: Path,
    year_label: str,
    times_label: str,
    source_name: str,
    annual_ts: float,
) -> None:
    n_stocks = len(stock_order)
    n_dates = len(dates)
    if trend_rgb.shape != (n_stocks, n_dates, 3):
        raise ValueError(
            "Trend RGB shape mismatch: "
            f"expected {(n_stocks, n_dates, 3)}, got {trend_rgb.shape}"
        )

    # Each ticker uses two rows: Trend + the selected model's actions.
    plot_rgb = np.ones((n_stocks * 2, n_dates, 3), dtype=float)
    plot_rgb[0::2, :, :] = trend_rgb

    fig_width = max(18.0, min(28.0, len(dates) / 10.0))
    fig_height = 5.0 if n_stocks == 1 else max(10.0, n_stocks * 0.52)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(plot_rgb, aspect="auto", interpolation="nearest")

    behavior_plot = behavior.reindex(index=dates, columns=stock_order).T
    _plot_action_markers(ax, behavior_plot, row_offset=1)

    # Stock names remain on the first row (Trend row) of each stock block.
    ax.set_yticks(np.arange(n_stocks) * 2)
    ax.set_yticklabels(stock_order)

    tick_positions, tick_labels = ts_graph.month_tick_positions(dates)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="left")

    # Separator between Trend / Model rows.
    for row_idx in range(1, n_stocks * 2):
        ax.axhline(
            row_idx - 0.5,
            color="#aaaaaa",
            linewidth=0.60,
            zorder=4,
        )

    # Boundaries between different stocks match the main plot frame.
    main_border_color = ax.spines["left"].get_edgecolor()
    main_border_width = ax.spines["left"].get_linewidth()
    for stock_idx in range(1, n_stocks):
        ax.axhline(
            stock_idx * 2 - 0.5,
            color=main_border_color,
            linewidth=main_border_width,
            zorder=5,
        )

    ax.set_xlabel("Test Date")
    ax.set_ylabel("Stock")
    ax.set_title(
        "Stock Trend x Agent Action\n"
        f"{year_label} | {times_label} | "
        f"{source_name} Average Annual TS: {annual_ts:.2f}",
        fontsize=18,
        fontweight="normal",
        pad=14,
    )

    ax.set_xlim(-0.5, n_dates - 0.5)
    ax.margins(x=0)
    ax.set_ylim(n_stocks * 2 - 0.5, -0.5)

    fig.tight_layout(rect=[0, 0, 0.91, 1])
    _add_right_legends(fig, ax, source_name=source_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# 6. Process one Window x times
# =====================================================
def process_one(
    window_dir: Path,
    times_dir: Path,
    output_dir: Path,
    source_name: str,
) -> dict[str, object]:
    source = load_source_times(window_dir, times_dir)
    stock_order = list(source["stock_order"])
    dates = list(source["dates"])
    year_label = window_year(window_dir)
    current_times_id = ts_graph.times_id(times_dir)

    if len(stock_order) != 30:
        print(f"  Warning: expected 30 stocks, but found {len(stock_order)} stocks.")

    price_data = source["price_data"]
    act = source["act"]
    behavior = source["behavior"]
    if not all(isinstance(frame, pd.DataFrame) for frame in [price_data, act, behavior]):
        raise TypeError("Expected DataFrame for price_data, act and behavior")

    d_matrix, _, stock_trend_score = _build_trend_from_price_data(
        price_data,
        stock_order=stock_order,
        dates=dates,
        n_future=int(source["n_future"]),
    )
    _, _, annual_ts = calculate_annual_ts(
        stock_trend_score=stock_trend_score,
        act=act,
    )
    single_scores = single_stock_score_table(stock_trend_score, act, source_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    single_scores.to_csv(
        output_dir / "single_stock_annual_ts.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f",
    )
    pd.DataFrame(
        [{"year": year_label, "times": current_times_id,
          "source": source_name, "Annual_TS": annual_ts}]
    ).to_csv(
        output_dir / "overall_annual_ts.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f",
    )

    # All stock charts reuse the same trend color normalization.
    d_plot_all = d_matrix.reindex(index=dates, columns=stock_order).T
    trend_rgb_all, _ = ts_graph.d_to_rgb(d_plot_all.to_numpy(dtype=float))
    price_pivot = ts_graph.build_price_pivot(
        price_data,
        stock_order=stock_order,
    ).reindex(index=dates, columns=stock_order)

    plot_single_model_heatmap(
        trend_rgb=trend_rgb_all,
        behavior=behavior,
        stock_order=stock_order,
        dates=dates,
        output_path=output_dir / "00_all_stocks.png",
        year_label=year_label,
        times_label=current_times_id,
        source_name=source_name,
        annual_ts=annual_ts,
    )

    single_dir = output_dir / "single_stocks"
    single_dir.mkdir(parents=True, exist_ok=True)
    score_lookup = single_scores.set_index("tic")
    for stock_idx, ticker in enumerate(stock_order):
        plot_single_stock(
            trend_rgb=trend_rgb_all[stock_idx : stock_idx + 1, :, :],
            price_series=price_pivot[ticker],
            behavior=behavior[ticker],
            ticker=ticker,
            dates=dates,
            output_path=single_dir / f"{_safe_name(ticker)}.png",
            year_label=year_label,
            times_label=current_times_id,
            source_name=source_name,
            annual_ts=float(score_lookup.loc[ticker, f"{source_name}_Annual_TS"]),
        )

    return {
        "year": year_label,
        "times": current_times_id,
        "stocks": len(stock_order),
        "images": 1 + len(stock_order),
        f"{source_name}_Annual_TS": annual_ts,
    }


# =====================================================
# 7. Main
# =====================================================
def main() -> None:
    source_root = resolve_folder(SOURCE_FOLDER, "SOURCE")
    source_name, reward_type = reward_display_name(source_root)
    output_root = make_output_dir(source_name)

    print("Single Model Stock Trend x Agent Action")
    print(f"Source: {source_name}\n  {source_root}")
    print(f"Output:\n  {output_root}\n")

    manifest = {
        "source": {
            "name": source_name,
            "reward_type": reward_type,
            "rolling_root": str(source_root),
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ts_version": normalize_ts_version(TS_VERSION),
        "note": (
            "Each Window x times outputs one all-stock figure plus one "
            "figure per stock for the selected model."
        ),
    }
    (output_root / "single_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    history_path = append_experiment_history(
        output_root=output_root,
        source_root=source_root,
        source_name=source_name,
        reward_type=reward_type,
    )
    print(f"History:\n  {history_path}\n")

    windows = windows_by_year(source_root)
    if not windows:
        raise RuntimeError("No Test year found in the selected source")

    summary_rows: list[dict[str, object]] = []
    for year, window_dir in sorted(windows.items()):
        print(f"Processing {year}")
        times_map = times_by_id(window_dir)
        if len(times_map) < EXPECTED_TIMES:
            print(
                f"  Warning: only {len(times_map)}/{EXPECTED_TIMES} "
                f"times folder(s) found for {year}."
            )
        for current_times_id, times_dir in sorted(times_map.items()):
            output_dir = output_root / year / current_times_id
            try:
                row = process_one(
                    window_dir=window_dir,
                    times_dir=times_dir,
                    output_dir=output_dir,
                    source_name=source_name,
                )
            except Exception as exc:
                print(f"  Warning: {current_times_id} skipped: {exc}")
                continue

            summary_rows.append(row)
            print(
                f"  {current_times_id}: {source_name} Annual TS = "
                f"{float(row[f'{source_name}_Annual_TS']):.2f}, "
                f"images = {int(row['images'])}"
            )
        print()

    if not summary_rows:
        raise RuntimeError("No valid single-model figures were produced")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        output_root / "single_annual_ts_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f",
    )
    save_year_average_annual_ts(
        summary_df=summary_df,
        output_path=output_root / "single_year_average_annual_ts.csv",
        source_name=source_name,
    )

    print("Done.")
    print(f"Processed Window x times: {len(summary_rows)}")
    print(f"Total images: {int(summary_df['images'].sum())}")
    print(f"Output:\n{output_root}")


if __name__ == "__main__":
    main()
