from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

#畫Stock Trend x Agent Action (換手圖)

# =====================================================
# 1. User setting
# =====================================================

# Only change this folder name when analyzing another completed model.
INPUT_FOLDER = r"rolling_2018-2025_train3y_test1y_shift1y_20260903_000326_reinforce" #v0
               # r"rolling_2018-2025_train3y_test1y_shift1y_20260903_000326_reinforce" #v0
               # r"rolling_2018-2025_train3y_test1y_shift1y_20260904_234026_reinforce"


# Same experiment root style used by the current Compare / score scripts.
EXPERIMENT_ROOT = Path(
    os.environ.get(
        "EXPERIMENT_ROOT",
        r"C:\Users\User\Desktop\GraduationProject\Experence_result",
    )
)

# d(i,t) uses the average close price of the next n trading days.
# If a saved config explicitly records another horizon, that value is
# preferred automatically.
DEFAULT_FUTURE_DAYS = 60

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


def find_action_csv(times_dir: Path) -> Path:
    csv_paths = [
        path
        for path in times_dir.rglob("*.csv")
        if "atds_analysis" not in path.parts
    ]

    keywords = ("action", "actions")
    excluded = (
        "agent_action",
        "agent_actions",
        "policy_action",
        "raw_action",
        "train_action",
    )

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
        if name.startswith("actions_") or name == "actions.csv":
            score += 50
        if "executed" in name or "final" in name:
            score += 20
        if "stock_turnover" in name:
            score += 10

        # Prefer files near the times root instead of nested diagnostic copies.
        score -= len(path.relative_to(times_dir).parts)
        candidates.append((score, path.stat().st_mtime, path))

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find a usable actions CSV under {times_dir}"
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def load_actions(times_dir: Path) -> tuple[pd.DataFrame, Path]:
    # Latest Annual Reset output writes final executed actions directly here.
    preferred = sorted(times_dir.glob("actions_window_*.csv"))
    path = (
        max(preferred, key=lambda p: p.stat().st_mtime)
        if preferred
        else find_action_csv(times_dir)
    )
    df = normalize_wide_table(pd.read_csv(path))
    return df, path


def find_holdings_csv(times_dir: Path) -> Path:
    csv_paths = [
        path
        for path in times_dir.rglob("*.csv")
        if "atds_analysis" not in path.parts
    ]

    candidates: list[tuple[int, float, Path]] = []
    for path in csv_paths:
        name = path.name.lower()
        if not any(keyword in name for keyword in ("holding", "holdings", "position")):
            continue
        if "continuous" in name:
            continue
        if not looks_like_wide_stock_table(path):
            continue

        score = 0
        if name.startswith("holdings_") or name == "holdings.csv":
            score += 50
        if "backtest" in name:
            score += 5

        # Prefer files near the times root instead of nested diagnostic copies.
        score -= len(path.relative_to(times_dir).parts)
        candidates.append((score, path.stat().st_mtime, path))

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find a usable holdings CSV under {times_dir}"
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def load_holdings(times_dir: Path) -> tuple[pd.DataFrame, Path]:
    # Latest Annual Reset output writes post-trade share holdings directly here.
    preferred = sorted(times_dir.glob("holdings_window_*.csv"))
    path = (
        max(preferred, key=lambda p: p.stat().st_mtime)
        if preferred
        else find_holdings_csv(times_dir)
    )
    df = normalize_wide_table(pd.read_csv(path))
    return df, path


def load_price_data(times_dir: Path) -> tuple[pd.DataFrame, Path]:
    # Reuse the same price source produced for Future Ranking because it
    # already contains the Test period plus the extra future trading days.
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
            f"Price source is missing columns "
            f"{sorted(required - set(df.columns))}: {path}"
        )

    out = df[["date", "tic", "close"]].copy()
    out["date"] = normalize_date_series(out["date"])
    out["tic"] = out["tic"].astype(str)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "tic", "close"])
    out = out.drop_duplicates(subset=["date", "tic"], keep="last")
    return out.reset_index(drop=True), path


def find_saved_future_days(times_dir: Path) -> int | None:
    keys = {
        "future_ranking_days",
        "future_ranking_n_days",
        "future_ranking_n",
        "ranking_n_future",
        "ranking_horizon",
        "future_days",
        "n_future",
        "dit_n",
        "d_n",
    }

    json_paths = list(times_dir.glob("*.json")) + list(
        (times_dir / "data").glob("*.json")
    )
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
                stack.extend(
                    item
                    for item in current
                    if isinstance(item, (dict, list))
                )

    return None


# =====================================================
# 4. ATDS parameters
# =====================================================

def stock_columns_from_actions(
    actions: pd.DataFrame,
    holdings: pd.DataFrame,
    price_data: pd.DataFrame,
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
    available_tickers = set(price_data["tic"].astype(str).unique())
    holdings_columns = set(holdings.columns)

    return [
        str(column)
        for column in actions.columns
        if column not in metadata
        and str(column) in available_tickers
        and column in holdings_columns
    ]


def build_price_pivot(
    price_data: pd.DataFrame,
    stock_order: list[str],
) -> pd.DataFrame:
    price_pivot = price_data.pivot(
        index="date",
        columns="tic",
        values="close",
    )
    price_pivot = price_pivot.sort_index().reindex(columns=stock_order)
    return price_pivot.apply(pd.to_numeric, errors="coerce")


def build_d_matrix(
    price_pivot: pd.DataFrame,
    n_future: int,
) -> pd.DataFrame:
    """Calculate d(i,t) from the average close of the next n trading days.

    d(i,t) = (mean(P^c_{i,t+1:t+n}) - P^c_{i,t}) / P^c_{i,t}
    """
    prices = price_pivot.to_numpy(dtype=float)
    d_values = np.full_like(prices, np.nan, dtype=float)

    n_rows, n_stocks = prices.shape
    for row_idx in range(n_rows):
        end_idx = row_idx + n_future + 1
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
    """Calculate consecutive same-direction price movement length.

    The first valid day is runlen = 1. If the same rise/fall direction
    continues, runlen increases by 1. A flat day keeps the previous runlen
    and does not reset the previous non-flat direction.
    """
    prices = price_pivot.to_numpy(dtype=float)
    n_rows, n_stocks = prices.shape
    runlen = np.full((n_rows, n_stocks), np.nan, dtype=float)

    for stock_idx in range(n_stocks):
        previous_direction = 0
        previous_runlen = 0

        for row_idx in range(n_rows):
            current = prices[row_idx, stock_idx]
            if not np.isfinite(current):
                previous_direction = 0
                previous_runlen = 0
                continue

            if row_idx == 0 or not np.isfinite(prices[row_idx - 1, stock_idx]):
                direction = 0
                current_runlen = 1
            else:
                diff = current - prices[row_idx - 1, stock_idx]
                direction = int(np.sign(diff))

                # Match the ATDS example:
                # price = [100, 102, 105, 90, 80]
                # runlen = [1, 2, 3, 1, 2]
                #
                # Flat days keep the previous runlen and previous non-flat
                # direction. Therefore, a trend continuing after a flat day
                # resumes from the same streak, while a reversal restarts at 1.
                if direction == 0:
                    current_runlen = previous_runlen
                elif previous_direction == 0:
                    current_runlen = previous_runlen + 1
                elif direction == previous_direction:
                    current_runlen = previous_runlen + 1
                else:
                    current_runlen = 1

            runlen[row_idx, stock_idx] = current_runlen
            if direction != 0:
                previous_direction = direction
            previous_runlen = current_runlen

    return pd.DataFrame(
        runlen,
        index=price_pivot.index,
        columns=price_pivot.columns,
    )


def build_stock_trend_score(
    d_matrix: pd.DataFrame,
    runlen_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """s(i,t) = sign(d(i,t)) * |d(i,t)| * 100 * runlen(i,t)."""
    d_aligned, runlen_aligned = d_matrix.align(runlen_matrix, join="inner")
    return np.sign(d_aligned) * d_aligned.abs() * 100.0 * runlen_aligned


def build_action_matrix(
    actions: pd.DataFrame,
    holdings: pd.DataFrame,
    stock_order: list[str],
    dates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    action_values = actions.set_index("date").reindex(dates)[stock_order]
    holding_values = holdings.set_index("date").reindex(dates)[stock_order]

    action_values = action_values.apply(pd.to_numeric, errors="coerce")
    holding_values = holding_values.apply(pd.to_numeric, errors="coerce")

    if holding_values.isna().any().any():
        missing_dates = holding_values.index[holding_values.isna().any(axis=1)].tolist()
        date_preview = ", ".join(missing_dates[:5])
        if len(missing_dates) > 5:
            date_preview += ", ..."
        raise ValueError(
            "Holdings data is incomplete for the Test period. "
            f"Missing date(s): {date_preview}"
        )

    act = pd.DataFrame(
        0.0,
        index=dates,
        columns=stock_order,
    )
    behavior = pd.DataFrame(
        "None",
        index=dates,
        columns=stock_order,
        dtype=object,
    )

    buy_mask = action_values > ACTION_EPS
    sell_mask = action_values < -ACTION_EPS
    hold_mask = (~buy_mask) & (~sell_mask) & (holding_values > ACTION_EPS)

    # ATDS scoring is unchanged: Buy = +1, Sell = -1, Hold/None = 0.
    act[buy_mask] = 1.0
    act[sell_mask] = -1.0

    behavior[buy_mask] = "Buy"
    behavior[hold_mask] = "Hold"
    behavior[sell_mask] = "Sell"

    return act, behavior


# =====================================================
# 5. ATDS calculation
# =====================================================

def test_action_dates(
    actions: pd.DataFrame,
    test_start: str,
    test_end: str,
) -> list[str]:
    return sorted(
        date
        for date in actions["date"].dropna().astype(str).unique()
        if test_start <= date <= test_end
    )


def validate_full_atds_period(
    d_matrix: pd.DataFrame,
    runlen_matrix: pd.DataFrame,
    stock_order: list[str],
    action_dates: list[str],
    n_future: int,
) -> None:
    if not action_dates:
        raise ValueError("No action dates found inside the Test period")

    d_test = d_matrix.reindex(
        index=action_dates,
        columns=stock_order,
    )
    runlen_test = runlen_matrix.reindex(
        index=action_dates,
        columns=stock_order,
    )

    missing_d = d_test.isna()
    missing_runlen = runlen_test.isna()

    if missing_d.any().any():
        missing_cells = int(missing_d.sum().sum())
        missing_dates = d_test.index[missing_d.any(axis=1)].tolist()

        date_preview = ", ".join(missing_dates[:5])
        if len(missing_dates) > 5:
            date_preview += ", ..."

        raise ValueError(
            f"ATDS cannot cover the full Test period because d(i,t) has "
            f"{missing_cells} missing cell(s). Missing date(s): {date_preview}. "
            f"d(i,t) uses the next {n_future} trading days, so the price file "
            f"must contain at least {n_future} additional future trading days "
            f"after the final Test date."
        )

    if missing_runlen.any().any():
        missing_cells = int(missing_runlen.sum().sum())
        raise ValueError(
            f"runlen(i,t) has {missing_cells} missing cell(s) in the Test period."
        )


def calculate_atds(
    stock_trend_score: pd.DataFrame,
    act: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Calculate TDS, DailyTDS, MaxTDS, and ATDS.

    tds(i,t) = act(i,t) * s(i,t)
    DailyTDS(t) = sum_i tds(i,t)
    MaxTDS(t) = sum_i |s(i,t)|
    ATDS = sum_t DailyTDS(t) / sum_t MaxTDS(t) * 100
    """
    s_aligned = stock_trend_score.reindex(
        index=act.index,
        columns=act.columns,
    )

    tds = act * s_aligned
    daily_tds = tds.sum(axis=1)
    max_tds = s_aligned.abs().sum(axis=1)

    denominator = float(max_tds.sum())
    if denominator <= 0:
        raise ValueError("ATDS denominator is zero; MaxTDS has no positive total")

    atds = float(daily_tds.sum() / denominator * 100.0)

    daily = pd.DataFrame(
        {
            "date": act.index,
            "DailyTDS": daily_tds.to_numpy(dtype=float),
            "MaxTDS": max_tds.to_numpy(dtype=float),
        }
    )

    return tds, daily, atds


# =====================================================
# 6. Combined ATDS plot
# =====================================================

RISE_COLOR = np.array([214, 39, 40], dtype=float) / 255.0
FALL_COLOR = np.array([44, 160, 44], dtype=float) / 255.0
WHITE = np.array([1.0, 1.0, 1.0], dtype=float)


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


def d_to_rgb(
    d_values: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Map d(i,t) to red/green shades using |d(i,t)| as color intensity."""
    finite_abs = np.abs(d_values[np.isfinite(d_values)])
    if finite_abs.size == 0:
        raise ValueError("No finite d(i,t) values available for plotting")

    max_abs_d = float(np.max(finite_abs))
    if max_abs_d <= 0:
        max_abs_d = 1.0

    intensity = np.clip(np.abs(d_values) / max_abs_d, 0.0, 1.0)

    # Keep very small non-zero values visible while preserving deeper colors
    # for larger rise/fall magnitudes.
    visible_intensity = np.where(
        np.isfinite(d_values) & (np.abs(d_values) > 0),
        0.12 + 0.88 * intensity,
        0.0,
    )

    rgb = np.ones((*d_values.shape, 3), dtype=float)

    positive = d_values > 0
    negative = d_values < 0

    for mask, target_color in (
        (positive, RISE_COLOR),
        (negative, FALL_COLOR),
    ):
        alpha = visible_intensity[mask][:, None]
        rgb[mask] = WHITE * (1.0 - alpha) + target_color * alpha

    return rgb, max_abs_d


def plot_atds_heatmap(
    d_matrix: pd.DataFrame,
    behavior: pd.DataFrame,
    stock_order: list[str],
    dates: list[str],
    output_path: Path,
    window_label: str,
    times_label: str,
    atds: float,
) -> None:
    d_plot = d_matrix.reindex(index=dates, columns=stock_order).T
    behavior_plot = behavior.reindex(index=dates, columns=stock_order).T

    d_values = d_plot.to_numpy(dtype=float)
    trend_rgb, _ = d_to_rgb(d_values)

    n_stocks = len(stock_order)
    n_dates = len(dates)

    # Every stock occupies two rows:
    #   2*i     -> d(i,t) rise/fall color
    #   2*i + 1 -> Agent action marker
    plot_rgb = np.ones((n_stocks * 2, n_dates, 3), dtype=float)
    plot_rgb[0::2, :, :] = trend_rgb

    fig_width = max(18.0, min(28.0, len(dates) / 10.0))
    fig_height = max(11.0, len(stock_order) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.imshow(
        plot_rgb,
        aspect="auto",
        interpolation="nearest",
    )

    marker_styles = {
        "Buy": {
            "marker": "^",
            "s": 32,
            "facecolors": "#FF0000",
            "edgecolors": "white",
            "linewidths": 0.45,
            "alpha": 1.0,
        },
        "Hold": {
            "marker": "o",
            "s": 11,
            "facecolors": "#222222",
            "edgecolors": "none",
            "linewidths": 0.0,
            "alpha": 0.50,
        },
        "Sell": {
            "marker": "v",
            "s": 32,
            "facecolors": "#008000",
            "edgecolors": "white",
            "linewidths": 0.45,
            "alpha": 1.0,
        },
    }

    behavior_values = behavior_plot.to_numpy(dtype=object)
    for behavior_name, style in marker_styles.items():
        stock_indices, x_indices = np.where(
            behavior_values == behavior_name
        )
        if len(x_indices) == 0:
            continue

        # Put action symbols on the lower row of each stock pair.
        y_indices = stock_indices * 2 + 1

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

    # Place each stock label in the center of its two-row block.
    stock_tick_positions = np.arange(n_stocks) * 2 + 0.5
    ax.set_yticks(stock_tick_positions)
    ax.set_yticklabels(stock_order)

    tick_positions, tick_labels = month_tick_positions(dates)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="left")

    # Thin separator between each stock pair.
    for stock_idx in range(1, n_stocks):
        ax.axhline(
            stock_idx * 2 - 0.5,
            color="#c7c7c7",
            linewidth=0.6,
            zorder=4,
        )

    ax.set_xlabel("Test Date")
    ax.set_ylabel("Stock")
    ax.set_title(
        "Stock Trend x Agent Action\n"
        f"{window_label} | {times_label} | ATDS: {atds:.2f}"
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
            markersize=13 if name != "Hold" else 10,
            alpha=marker_styles[name]["alpha"],
            label=name,
        )
        for name in ["Buy", "Hold", "Sell"]
    ]

    ax.set_xlim(-0.5, n_dates - 0.5)
    ax.margins(x=0)
    ax.set_ylim(n_stocks * 2 - 0.5, -0.5)

    # Reserve more right-side space for larger Action and Trend legends.
    fig.tight_layout(rect=[0, 0, 0.88, 1])

    action_legend = fig.legend(
        handles=behavior_handles,
        title="Action",
        loc="upper left",
        bbox_to_anchor=(0.89, 0.93),
        borderaxespad=0.0,
        ncol=1,
        frameon=True,
        fontsize=13,
        title_fontsize=14,
        labelspacing=1.0,
        borderpad=1.1,
        handlelength=2.2,
        handletextpad=0.9,
    )

    # Keep the Action legend and use its actual rendered width for the
    # Trend box so both right-side boxes are exactly the same width.
    fig.add_artist(action_legend)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    action_bbox = action_legend.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    )

    # Move the whole right-side legend block left so its distance to the
    # main plot is close to its distance to the final image right boundary.
    # Only the block position changes; the overall figure size is unchanged.
    legend_block_gap_from_plot = 0.004
    legend_block_left = ax.get_position().x1 + legend_block_gap_from_plot
    action_legend.set_bbox_to_anchor(
        (legend_block_left, 0.93),
        transform=fig.transFigure,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    action_bbox = action_legend.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    )
    action_title_bbox = action_legend.get_title().get_window_extent(
        renderer
    ).transformed(fig.transFigure.inverted())
    action_title_top_padding = action_bbox.y1 - action_title_bbox.y1

    trend_gap = 0.025
    trend_bottom = 0.08

    # First calculate the original Trend title position, then move only the
    # Trend top border downward until its title-to-border spacing matches
    # the Action legend's title-to-border spacing.
    original_trend_top = action_bbox.y0 - trend_gap
    original_trend_height = max(0.20, original_trend_top - trend_bottom)
    trend_title_figure_y = (
        trend_bottom + 0.95 * original_trend_height
    )

    trend_top = trend_title_figure_y + action_title_top_padding
    trend_height = max(0.20, trend_top - trend_bottom)
    trend_title_y = (
        trend_title_figure_y - trend_bottom
    ) / trend_height

    trend_ax = fig.add_axes(
        [
            action_bbox.x0,
            trend_bottom,
            action_bbox.width,
            trend_height,
        ]
    )
    trend_ax.set_xlim(0.0, 1.0)
    trend_ax.set_ylim(0.0, 1.0)
    trend_ax.set_xticks([])
    trend_ax.set_yticks([])
    trend_ax.set_facecolor("white")

    # Match the Trend frame to the Action legend frame.
    # Matplotlib Axes also draws four spines, so update them as well;
    # otherwise the default black spines would cover the patch edge.
    action_frame = action_legend.get_frame()
    frame_edgecolor = action_frame.get_edgecolor()
    frame_linewidth = action_frame.get_linewidth()

    trend_ax.patch.set_edgecolor(frame_edgecolor)
    trend_ax.patch.set_linewidth(frame_linewidth)
    for spine in trend_ax.spines.values():
        spine.set_edgecolor(frame_edgecolor)
        spine.set_linewidth(frame_linewidth)

    trend_ax.text(
        0.5,
        trend_title_y,
        "Trend",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="normal",
    )

    # Vertical trend color bar:
    # top    = strongest rise (deep red)
    # center = weakest rise/fall (light red / light green)
    # bottom = strongest fall (deep green)
    gradient_steps = 128
    light_rise = WHITE * 0.78 + RISE_COLOR * 0.22
    light_fall = WHITE * 0.78 + FALL_COLOR * 0.22

    rise_gradient = np.linspace(
        RISE_COLOR,
        light_rise,
        gradient_steps,
    )
    fall_gradient = np.linspace(
        light_fall,
        FALL_COLOR,
        gradient_steps,
    )
    trend_gradient = np.vstack(
        [rise_gradient, fall_gradient]
    ).reshape(gradient_steps * 2, 1, 3)

    bar_left = 0.30
    bar_right = 0.70
    bar_bottom = 0.08
    bar_top = 0.83

    trend_ax.text(
        0.5,
        bar_top + 0.035,
        "Rise",
        ha="center",
        va="bottom",
        fontsize=13,
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
        fontsize=13,
    )

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
    price_data, price_source_path = load_price_data(times_dir)

    stock_order = stock_columns_from_actions(actions, holdings, price_data)
    if len(stock_order) < 10:
        raise ValueError(
            f"Only found {len(stock_order)} common stock columns "
            "between actions and price data"
        )

    n_future = find_saved_future_days(times_dir) or DEFAULT_FUTURE_DAYS

    price_pivot = build_price_pivot(
        price_data,
        stock_order=stock_order,
    )
    d_matrix = build_d_matrix(
        price_pivot,
        n_future=n_future,
    )
    runlen_matrix = build_runlen_matrix(price_pivot)
    stock_trend_score = build_stock_trend_score(
        d_matrix,
        runlen_matrix,
    )

    test_start, test_end = test_date_range(window_dir)
    dates = test_action_dates(
        actions,
        test_start=test_start,
        test_end=test_end,
    )

    validate_full_atds_period(
        d_matrix=d_matrix,
        runlen_matrix=runlen_matrix,
        stock_order=stock_order,
        action_dates=dates,
        n_future=n_future,
    )

    act, behavior = build_action_matrix(
        actions,
        holdings,
        stock_order=stock_order,
        dates=dates,
    )

    d_valid = d_matrix.reindex(index=dates, columns=stock_order)
    runlen_valid = runlen_matrix.reindex(index=dates, columns=stock_order)
    s_valid = stock_trend_score.reindex(index=dates, columns=stock_order)

    tds, daily_scores, atds = calculate_atds(
        stock_trend_score=s_valid,
        act=act,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    daily_scores.to_csv(
        output_dir / "atds_daily_score.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f",
    )

    # Save the exact per-stock values used in ATDS so the score can be audited.
    parameter_rows: list[pd.DataFrame] = []
    for tic in stock_order:
        parameter_rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "tic": tic,
                    "d": d_valid[tic].to_numpy(dtype=float),
                    "runlen": runlen_valid[tic].to_numpy(dtype=float),
                    "stock_trend_score": s_valid[tic].to_numpy(dtype=float),
                    "action": act[tic].to_numpy(dtype=float),
                    "tds": tds[tic].to_numpy(dtype=float),
                }
            )
        )

    pd.concat(parameter_rows, ignore_index=True).to_csv(
        output_dir / "atds_parameters.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f",
    )

    plot_atds_heatmap(
        d_matrix=d_valid,
        behavior=behavior,
        stock_order=stock_order,
        dates=dates,
        output_path=output_dir / "atds_trading_behavior.png",
        window_label=window_id(window_dir),
        times_label=times_id(times_dir),
        atds=atds,
    )

    return {
        "times": times_id(times_dir),
        "ATDS": atds,
        "valid_days": len(dates),
        "future_days": n_future,
        "price_source": str(price_source_path),
        "actions_source": str(actions_path),
        "holdings_source": str(holdings_path),
    }


def save_window_summary(
    rows: list[dict[str, object]],
    output_path: Path,
) -> dict[str, float]:
    if not rows:
        raise ValueError("No valid times rows for this window")

    summary = pd.DataFrame(rows)[
        ["times", "ATDS"]
    ].copy()
    summary = summary.sort_values("times").reset_index(drop=True)

    average_row = {
        "times": "Average",
        "ATDS": float(summary["ATDS"].mean()),
    }
    summary = pd.concat(
        [summary, pd.DataFrame([average_row])],
        ignore_index=True,
    )

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
        row: dict[str, object] = {
            "window": window_row["window"],
        }
        scores: dict[str, float] = window_row["times_scores"]

        for index in range(1, EXPECTED_TIMES + 1):
            key = f"times_{index:02d}"
            row[key] = scores.get(key, np.nan)

        row["average"] = window_row["average_ATDS"]
        all_windows_rows.append(row)

        window_average_rows.append(
            {
                "window": window_row["window"],
                "average_ATDS": window_row["average_ATDS"],
            }
        )

    pd.DataFrame(all_windows_rows).to_csv(
        analysis_dir / "atds_all_windows.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.2f",
    )

    pd.DataFrame(window_average_rows).to_csv(
        analysis_dir / "atds_window_average_summary.csv",
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
        raise FileNotFoundError(
            f"No window folders found under {model_dir}"
        )

    analysis_dir = model_dir / "atds_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("Score: Annual Trading Decision Score (ATDS)")
    print(
        "Plot layout: each stock uses two rows "
        "(Rise/Fall color + Action symbol)"
    )
    print(f"Input:\n{model_dir.name}\n")

    model_rows: list[dict[str, object]] = []

    for window_dir in windows:
        current_window_id = window_id(window_dir)
        print(f"Processing {current_window_id}")

        times_dirs = find_times(window_dir)
        if len(times_dirs) < EXPECTED_TIMES:
            print(
                f"Warning: {current_window_id} only found "
                f"{len(times_dirs)} times folder(s)."
            )

        valid_rows: list[dict[str, object]] = []

        for times_dir in times_dirs:
            current_times_id = times_id(times_dir)
            times_output_dir = (
                analysis_dir
                / current_window_id
                / current_times_id
            )

            try:
                row = process_times(
                    window_dir,
                    times_dir,
                    times_output_dir,
                )
            except Exception as exc:
                print(
                    f"  Warning: {current_times_id} skipped: {exc}"
                )
                continue

            valid_rows.append(row)
            print(
                f"  {current_times_id} ... "
                f"ATDS = {float(row['ATDS']):.2f}"
            )

        if not valid_rows:
            print(
                f"  Warning: no valid times found for "
                f"{current_window_id}.\n"
            )
            continue

        if len(valid_rows) < EXPECTED_TIMES:
            print(
                f"Warning: {current_window_id} only found "
                f"{len(valid_rows)} valid times."
            )

        window_output_dir = analysis_dir / current_window_id
        average_row = save_window_summary(
            valid_rows,
            window_output_dir / "atds_summary.csv",
        )

        window_average = float(average_row["ATDS"])
        print(
            f"  Window Average ATDS = {window_average:.2f}\n"
        )

        model_rows.append(
            {
                "window": current_window_id,
                "times_scores": {
                    str(row["times"]): float(row["ATDS"])
                    for row in valid_rows
                },
                "average_ATDS": window_average,
            }
        )

    if not model_rows:
        raise RuntimeError(
            "No valid Window × times ATDS results were produced"
        )

    save_model_summaries(
        model_rows,
        analysis_dir,
    )

    print("Model Summary:")
    for row in model_rows:
        print(
            f"{row['window']} = "
            f"{float(row['average_ATDS']):.2f}"
        )

    print("\nDone.")
    print(f"\nOutput:\n{analysis_dir}")


if __name__ == "__main__":
    main()
