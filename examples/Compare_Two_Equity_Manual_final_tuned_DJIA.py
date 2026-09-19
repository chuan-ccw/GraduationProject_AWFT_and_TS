from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Compare_V0_V1_Manual as legacy_compare


# =====================================================
# Manual 2-source equity compare settings
# =====================================================
# 使用方式：
# 1. 手動填入 SOURCE_1 / SOURCE_2 的 rolling 資料夾。
# 2. rolling 可以填完整路徑，也可以只填 rolling_* 資料夾名稱。
# 3. reward_type 要對應來源：v0 填 "v0"；DWFT 填 "v6"。
# 4. label 只影響圖片與 CSV 顯示名稱，可自行修改。
#
# 目前預設用途：手動比較兩組實驗結果。

EXPERIMENT_ROOT = os.environ.get(
    "EXPERIMENT_ROOT",
    getattr(
        legacy_compare.config,
        "EXPERIMENT_ROOT",
        r"C:\Users\User\Desktop\GraduationProject\Experence_result",
    ),
)

# 可填完整路徑，例如：
# r"C:\Users\User\Desktop\GraduationProject\Experence_result\reward_v0\rolling_..."
# 也可只填 rolling_* 資料夾名稱。
SOURCE_1 = {
    "label": "FinRL_reward-REINFORCE",
    "table_label": "FinRL",
    "reward_type": "v0",
    "rolling": r"rolling_2018-2025_train3y_test1y_shift1y_20260912_000936_reinforce",
    "color": "blue",
}

SOURCE_2 = {
    "label": "AWFT_reward-REINFORCE",
    "table_label": "AWFT",
    "reward_type": "v7",
    "rolling": r"rolling_2018-2025_train3y_test1y_shift1y_20260904_234026_reinforce",
    "color": "red",
}

SOURCES = [SOURCE_1, SOURCE_2]

# "annual_reset" 或 "continuous_5y"
COMPARE_DATA_MODE = "annual_reset"

# 留空會自動使用時間戳記建立資料夾。
COMPARE_NAME = ""



# =====================================================
# Helpers
# =====================================================
def _safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if len(text) > max_len else text


def _validate_sources() -> None:
    if len(SOURCES) != 2:
        raise ValueError("SOURCES 必須剛好有 2 個來源。")

    for index, source in enumerate(SOURCES, start=1):
        if not str(source.get("rolling", "")).strip():
            raise ValueError(
                f"SOURCE_{index} 的 rolling 尚未填寫。"
                "請填 rolling_* 資料夾名稱或完整路徑。"
            )
        if not str(source.get("reward_type", "")).strip():
            raise ValueError(f"SOURCE_{index} 的 reward_type 尚未填寫。")
        if not str(source.get("label", "")).strip():
            raise ValueError(f"SOURCE_{index} 的 label 尚未填寫。")


def _resolve_source(source: dict[str, str], index: int) -> tuple[Path, pd.DataFrame]:
    # Reuse the old compare loader so the data source and normalization rules
    # remain identical to the previous equity comparison program.
    legacy_compare.EXPERIMENT_ROOT = EXPERIMENT_ROOT

    reward_type = legacy_compare.normalize_reward_type(source["reward_type"])
    rolling_root = legacy_compare.resolve_rolling_root(
        f"SOURCE_{index}",
        reward_type,
        source["rolling"],
    )
    df = legacy_compare.collect_compare_series(
        rolling_root,
        reward_type,
        COMPARE_DATA_MODE,
    )
    return rolling_root, df


def _select_and_rename(df: pd.DataFrame, key: str) -> pd.DataFrame:
    selected = pd.DataFrame({"date": df["date"].astype(str)})
    selected[f"{key}_mean"] = pd.to_numeric(df["strategy_mean_norm"], errors="coerce")

    if "strategy_min_norm" in df.columns:
        selected[f"{key}_min"] = pd.to_numeric(df["strategy_min_norm"], errors="coerce")
    else:
        selected[f"{key}_min"] = selected[f"{key}_mean"]

    if "strategy_max_norm" in df.columns:
        selected[f"{key}_max"] = pd.to_numeric(df["strategy_max_norm"], errors="coerce")
    else:
        selected[f"{key}_max"] = selected[f"{key}_mean"]

    if "dji_mean_norm" in df.columns:
        selected[f"{key}_dji"] = pd.to_numeric(df["dji_mean_norm"], errors="coerce")

    return selected


def build_compare_dataframe(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    for index, df in enumerate(dataframes, start=1):
        current = _select_and_rename(df, f"source{index}")
        if merged is None:
            merged = current
        else:
            merged = pd.merge(merged, current, on="date", how="outer")

    if merged is None:
        raise ValueError("沒有可比較的資料。")

    merged = merged.sort_values("date").reset_index(drop=True)

    # Prefer the DJIA series already stored in source outputs.
    merged["dji_mean_norm"] = np.nan
    for index in range(1, 3):
        dji_col = f"source{index}_dji"
        if dji_col in merged.columns and pd.to_numeric(merged[dji_col], errors="coerce").notna().any():
            merged["dji_mean_norm"] = pd.to_numeric(merged[dji_col], errors="coerce")
            break

    # Keep old behavior: if no stored DJIA is available, try to build it from trade dates.
    if not pd.to_numeric(merged["dji_mean_norm"], errors="coerce").notna().any():
        dji = legacy_compare.build_dji_baseline(
            merged["date"].tolist(),
            initial_amount=1_000_000.0,
        )
        if dji.notna().any():
            dji = pd.to_numeric(
                dji.reindex(pd.Index(merged["date"].astype(str), name="date")),
                errors="coerce",
            )
            merged["dji_mean_norm"] = legacy_compare._normalize_series(dji).to_numpy()

    return merged


def _strategy_return_percent(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    return (float(valid.iloc[-1]) - 1.0) * 100.0


def _annual_return_percent(
    series: pd.Series,
    dates: pd.Series,
    year: int,
) -> float:
    """Return one annual-reset year's performance in percent."""
    parsed_dates = pd.to_datetime(dates, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric[parsed_dates.dt.year == year].dropna()
    if valid.empty:
        return float("nan")
    return (float(valid.iloc[-1]) - 1.0) * 100.0


def _add_annual_performance_table(ax, merged: pd.DataFrame) -> None:
    """Add the 4x6 annual performance table inside the 2021 lower-left area."""
    parsed_dates = pd.to_datetime(merged["date"], errors="coerce")
    years = sorted(int(year) for year in parsed_dates.dt.year.dropna().unique())

    # The current experiment is a 5-year test, so this becomes a 4x6 table:
    # header row + two reward rows + DJIA row, and label column + five year columns.
    years = years[:5]
    if not years:
        return

    table_rows: list[list[str]] = [["Reward"] + [str(year) for year in years]]

    for index, source in enumerate(SOURCES, start=1):
        mean_col = f"source{index}_mean"
        row = [str(source.get("table_label", source["label"]))]
        for year in years:
            ret = _annual_return_percent(merged[mean_col], merged["date"], year)
            row.append("N/A" if not np.isfinite(ret) else f"{ret:.2f}%")
        table_rows.append(row)

    # Add DJIA benchmark annual cumulative returns to the same table.
    dji = pd.to_numeric(merged["dji_mean_norm"], errors="coerce")
    row = ["DJIA"]
    for year in years:
        ret = _annual_return_percent(dji, merged["date"], year)
        row.append("N/A" if not np.isfinite(ret) else f"{ret:.2f}%")
    table_rows.append(row)

    # Keep the table below the y=1.0 line and to the left of the 2022 boundary.
    ax.text(
        0.006,
        0.185,
        "Annual Cumulative Return",
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
        va="bottom",
        ha="left",
        zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
    )

    table = ax.table(
        cellText=table_rows,
        cellLoc="center",
        bbox=[0.006, 0.04, 0.19, 0.13],
        zorder=5,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("#666666")
        cell.set_facecolor("white")
        if row == 0 or col == 0:
            cell.set_text_props(weight="bold")

    # Match the strategy names in the table with their equity-line colors.
    table[(1, 0)].get_text().set_color(SOURCE_1["color"])
    table[(2, 0)].get_text().set_color(SOURCE_2["color"])
    if (3, 0) in table.get_celld():
        table[(3, 0)].get_text().set_color("tab:green")


def _plot_equity_line(
    ax,
    x: np.ndarray,
    dates: pd.Series,
    values: pd.Series,
    *,
    color: str,
    linewidth: float,
    label: str,
) -> None:
    """Plot annual-reset years as separate segments so year boundaries stay disconnected."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mode = str(COMPARE_DATA_MODE or "").strip().lower()

    if mode not in {"annual_reset", "times"}:
        ax.plot(x, numeric, color=color, linewidth=linewidth, label=label)
        return

    years = pd.to_datetime(dates, errors="coerce").dt.year.to_numpy()
    label_used = False
    for year in pd.unique(years[pd.notna(years)]):
        mask = (years == year) & np.isfinite(numeric)
        positions = np.flatnonzero(mask)
        if len(positions) == 0:
            continue
        ax.plot(
            x[positions],
            numeric[positions],
            color=color,
            linewidth=linewidth,
            label=label if not label_used else "_nolegend_",
        )
        label_used = True


def _plot_annual_reset_boundaries(ax, dates: pd.Series) -> None:
    """Draw each year boundary at the first data point of the new year."""
    mode = str(COMPARE_DATA_MODE or "").strip().lower()
    if mode not in {
        "annual_reset",
        "times",
        "continuous",
        "continuous_5y",
        "continuous5y",
    }:
        return

    years = pd.to_datetime(dates, errors="coerce").dt.year.to_numpy()
    for position in range(1, len(years)):
        previous_year = years[position - 1]
        current_year = years[position]
        if pd.notna(previous_year) and pd.notna(current_year) and previous_year != current_year:
            ax.axvline(
                position,
                color="gray",
                linestyle="--",
                linewidth=1.1,
                zorder=0,
            )


def _set_jan_may_sep_xticks(ax, dates: pd.Series) -> None:
    """Show Jan/May/Sep at each month's first available data point."""
    parsed_dates = pd.to_datetime(dates, errors="coerce")

    tick_positions: list[int] = []
    tick_labels: list[str] = []
    seen_year_month: set[tuple[int, int]] = set()

    for position, date in enumerate(parsed_dates):
        if pd.isna(date) or date.month not in {1, 5, 9}:
            continue

        year_month = (date.year, date.month)
        if year_month in seen_year_month:
            continue

        seen_year_month.add(year_month)
        tick_positions.append(position)
        tick_labels.append(f"{date.year}-{date.month:02d}")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    # Left alignment keeps the label text starting exactly at the month's
    # first plotted data point. January therefore aligns with the year line.
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("left")


def make_compare_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if COMPARE_NAME.strip():
        folder_name = _safe_name(COMPARE_NAME.strip())
    else:
        folder_name = f"compare_two_equity_{timestamp}"

    compare_dir = Path(EXPERIMENT_ROOT) / "compare_two_equity" / folder_name
    compare_dir.mkdir(parents=True, exist_ok=False)
    return compare_dir


def plot_compare(merged: pd.DataFrame, compare_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(merged))

    mode = str(COMPARE_DATA_MODE or "").strip().lower()
    is_annual_reset = mode in {"annual_reset", "times"}

    for index, source in enumerate(SOURCES, start=1):
        key = f"source{index}"
        mean = pd.to_numeric(merged[f"{key}_mean"], errors="coerce")
        ret = _strategy_return_percent(mean)

        # Annual-reset performance is shown in the table below, so the legend
        # only keeps the series name. Continuous 5-year mode keeps its total return.
        legend_label = (
            source["label"]
            if is_annual_reset
            else f"{source['label']} mean ({ret:.2f}%)"
        )

        _plot_equity_line(
            ax,
            x,
            merged["date"],
            mean,
            color=source["color"],
            linewidth=2.2,
            label=legend_label,
        )

    dji = pd.to_numeric(merged["dji_mean_norm"], errors="coerce")
    if dji.notna().any():
        ret = _strategy_return_percent(dji)
        dji_label = "DJIA" if is_annual_reset else f"DJIA ({ret:.2f}%)"
        _plot_equity_line(
            ax,
            x,
            merged["date"],
            dji,
            color="tab:green",
            linewidth=1.8,
            label=dji_label,
        )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.1)
    _plot_annual_reset_boundaries(ax, merged["date"])
    _set_jan_may_sep_xticks(ax, merged["date"])

    # Reduce only the right-side x-axis padding so the right border moves left.
    ax.set_xlim(float(x[0]), float(x[-1]))

    title_mode = (
        "Continuous 5-Year Test"
        if mode in {"continuous", "continuous_5y", "continuous5y"}
        else "Annual Reset"
    )
    title_labels = " vs ".join(str(source["label"]) for source in SOURCES)
    ax.set_title(f"{title_labels} {title_mode} Equity Curve")
    legacy_compare.set_axis_labels_horizontal(
        ax,
        "Date",
        "Equity",
        ylabel_pad=95,
    )

    if is_annual_reset:
        _add_annual_performance_table(ax, merged)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.83, 0.03),
            borderaxespad=0.0,
            frameon=True,
        )
    else:
        ax.legend()

    plt.tight_layout()

    fig.savefig(
        compare_dir / "compare_portfolio_value.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_outputs(
    merged: pd.DataFrame,
    compare_dir: Path,
    roots: list[Path],
    dataframes: list[pd.DataFrame],
) -> None:
    merged.to_csv(
        compare_dir / "compare_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []
    for index, source in enumerate(SOURCES, start=1):
        mean_col = f"source{index}_mean"
        values = pd.to_numeric(merged[mean_col], errors="coerce").dropna()
        summary_rows.append(
            {
                "strategy": source["label"],
                "reward_type": source["reward_type"],
                "rolling_root": str(roots[index - 1]),
                "final_normalized_value": float(values.iloc[-1]) if not values.empty else np.nan,
                "return_percent": _strategy_return_percent(merged[mean_col]),
            }
        )

    dji_values = pd.to_numeric(merged["dji_mean_norm"], errors="coerce").dropna()
    if not dji_values.empty:
        summary_rows.append(
            {
                "strategy": "DJIA",
                "reward_type": "benchmark",
                "rolling_root": "",
                "final_normalized_value": float(dji_values.iloc[-1]),
                "return_percent": _strategy_return_percent(merged["dji_mean_norm"]),
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        compare_dir / "compare_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest_rows = []
    for index, (source, root, df) in enumerate(zip(SOURCES, roots, dataframes), start=1):
        for source_csv in df.attrs.get("source_csvs", []):
            manifest_rows.append(
                {
                    "plot": "compare_portfolio_value.png",
                    "source_index": index,
                    "label": source["label"],
                    "reward_type": source["reward_type"],
                    "rolling_root": str(root),
                    "source_csv": source_csv,
                    "merged_rows": len(merged),
                }
            )

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(
            compare_dir / "image_source_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )

    config_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_root": str(EXPERIMENT_ROOT),
        "compare_data_mode": COMPARE_DATA_MODE,
        "compare_dir": str(compare_dir),
        "sources": [
            {
                "label": source["label"],
                "reward_type": source["reward_type"],
                "rolling_setting": source["rolling"],
                "rolling_path": str(root),
                "color": source["color"],
            }
            for source, root in zip(SOURCES, roots)
        ],
        "x_axis_format": "YYYY-MM",
        "plot": "compare_portfolio_value.png",
    }
    (compare_dir / "compare_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    _validate_sources()

    roots: list[Path] = []
    dataframes: list[pd.DataFrame] = []

    for index, source in enumerate(SOURCES, start=1):
        root, df = _resolve_source(source, index)
        roots.append(root)
        dataframes.append(df)

    merged = build_compare_dataframe(dataframes)
    compare_dir = make_compare_dir()
    plot_compare(merged, compare_dir)
    write_outputs(merged, compare_dir, roots, dataframes)

    print("2-source equity compare finished.")
    for index, (source, root) in enumerate(zip(SOURCES, roots), start=1):
        print(f"SOURCE_{index}: {source['label']} -> {root}")
    print(f"compare mode: {COMPARE_DATA_MODE}")
    print(f"compare output: {compare_dir}")


if __name__ == "__main__":
    main()
