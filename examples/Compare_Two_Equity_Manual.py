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
    "label": "DWFT_reward_8-REINFORCE",
    "reward_type": "v6",
    "rolling": r"rolling_2018-2025_train3y_test1y_shift1y_20260819_220402_reinforce",
    "color": "blue",
}

SOURCE_2 = {
    "label": "DWFT_reward_25-REINFORCE",
    "reward_type": "v6",
    "rolling": r"rolling_2018-2025_train3y_test1y_shift1y_20260818_193551_reinforce",
    "color": "red",
}

SOURCES = [SOURCE_1, SOURCE_2]

# "annual_reset" 或 "continuous_5y"
COMPARE_DATA_MODE = "annual_reset"

# 留空會自動使用時間戳記建立資料夾。
COMPARE_NAME = ""

MAX_XTICKS = 14


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

    # Prefer the DJI series already stored in source outputs.
    merged["dji_mean_norm"] = np.nan
    for index in range(1, 3):
        dji_col = f"source{index}_dji"
        if dji_col in merged.columns and pd.to_numeric(merged[dji_col], errors="coerce").notna().any():
            merged["dji_mean_norm"] = pd.to_numeric(merged[dji_col], errors="coerce")
            break

    # Keep old behavior: if no stored DJI is available, try to build it from trade dates.
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

    for index, source in enumerate(SOURCES, start=1):
        key = f"source{index}"
        mean = pd.to_numeric(merged[f"{key}_mean"], errors="coerce")
        min_v = pd.to_numeric(merged[f"{key}_min"], errors="coerce")
        max_v = pd.to_numeric(merged[f"{key}_max"], errors="coerce")
        ret = _strategy_return_percent(mean)

        ax.fill_between(
            x,
            min_v,
            max_v,
            color=source["color"],
            alpha=0.14,
        )
        ax.plot(
            x,
            mean,
            color=source["color"],
            linewidth=2.2,
            label=f"{source['label']} mean ({ret:.2f}%)",
        )

    dji = pd.to_numeric(merged["dji_mean_norm"], errors="coerce")
    if dji.notna().any():
        ret = _strategy_return_percent(dji)
        ax.plot(
            x,
            dji,
            color="tab:green",
            linewidth=1.8,
            label=f"DJI ({ret:.2f}%)",
        )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.1)
    legacy_compare.set_month_xticklabels(
        ax,
        merged["date"].tolist(),
        max_xticks=MAX_XTICKS,
    )

    title_mode = (
        "Continuous 5-Year Test"
        if COMPARE_DATA_MODE in {"continuous", "continuous_5y", "continuous5y"}
        else "Annual Reset"
    )
    title_labels = " vs ".join(str(source["label"]) for source in SOURCES)
    ax.set_title(f"{title_labels} {title_mode} Equity Curve")
    legacy_compare.set_axis_labels_horizontal(
        ax,
        "Date",
        "Asset Allocation (initial = 1.0)",
        ylabel_pad=95,
    )
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
                "strategy": "DJI",
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
