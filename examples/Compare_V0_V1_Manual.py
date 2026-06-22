from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finrl import config

# =====================================================
# Manual v0/v1 compare settings
# =====================================================
# 使用流程：
# 1. 先執行 Run_StockTrading_Examples.py 跑 v0
# 2. 再把 REWARD_TYPE 改成 v1，執行 Run_StockTrading_Examples.py 跑 v1
# 3. 回到本檔案，指定要比較的 v0 / v1 rolling 資料夾名稱後執行

EXPERIMENT_ROOT = getattr(
    config,
    "EXPERIMENT_ROOT",
    r"C:\Users\User\Desktop\GraduationProject\Experence_result",
)

# 只要填 rolling 資料夾名稱即可，不用填完整路徑。
# 例如："rolling_2018-2025_train3y_test1y_shift1y_20260530"
V0_ROLLING_NAME = "rolling_2018-2025_train3y_test1y_shift1y_20260603"
V1_ROLLING_NAME = "rolling_2018-2025_train3y_test1y_shift1y_20260602_183322"

# 輸出資料夾名稱。留空字串時，程式會自動加入時間戳與 v0/v1 資料夾名稱，避免覆蓋舊比較結果。
COMPARE_NAME = ""

# 圖片設定
MAX_XTICKS = 14


def _safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if len(text) > max_len else text


def resolve_rolling_root(reward_type: str, rolling_name_or_path: str) -> Path:
    """Resolve a manually selected rolling folder.

    Supports:
    1) absolute/full path
    2) EXPERIMENT_ROOT/reward_v?/rolling_name
    3) EXPERIMENT_ROOT/reward_v?/rolling_window/rolling_name  # older layout
    """
    value = str(rolling_name_or_path).strip().strip('"').strip("'")
    if not value or "YYYYMMDD" in value:
        raise ValueError(
            f"請先設定 {reward_type} 的 rolling 資料夾名稱，例如："
            "rolling_2018-2025_train3y_test1y_shift1y_20260530"
        )

    direct = Path(value)
    if direct.exists():
        return direct.resolve()

    experiment_root = Path(EXPERIMENT_ROOT)
    candidates = [
        experiment_root / f"reward_{reward_type}" / value,
        experiment_root / f"reward_{reward_type}" / "rolling_window" / value,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"找不到 reward_{reward_type} rolling folder: {value}\n"
        + "已嘗試：\n"
        + "\n".join(str(p) for p in candidates)
    )


def collect_repeat_mean_series(rolling_root: Path, reward_type: str) -> pd.DataFrame:
    """Read all window repeat_mean backtest_result.csv files and concatenate them."""
    csv_paths = sorted(rolling_root.glob(f"window_*/repeat_mean_{reward_type}/backtest_result.csv"))
    if not csv_paths:
        # Fallback: support aggregate CSV filenames if backtest_result.csv was not copied.
        csv_paths = sorted(rolling_root.glob(f"window_*/repeat_mean_{reward_type}/backtest_result_repeat_mean_{reward_type}_*.csv"))

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df = df.copy()
        df["source_window"] = csv_path.parents[1].name
        df["source_csv"] = str(csv_path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"{rolling_root} 裡找不到 window_*/repeat_mean_{reward_type}/backtest_result.csv"
        )

    out = pd.concat(frames, ignore_index=True)
    if "date" not in out.columns:
        raise KeyError(f"repeat_mean_{reward_type} 的 backtest_result.csv 缺少 date 欄位")
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out.sort_values("date").reset_index(drop=True)


def _strategy_return_percent(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    return (float(valid.iloc[-1]) - 1.0) * 100.0


def make_compare_dir(root_v0: Path, root_v1: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if COMPARE_NAME.strip():
        folder_name = _safe_name(COMPARE_NAME.strip())
    else:
        folder_name = (
            f"compare_v0_{_safe_name(root_v0.name, 42)}"
            f"__v1_{_safe_name(root_v1.name, 42)}"
            f"__{timestamp}"
        )
    compare_root = Path(EXPERIMENT_ROOT) / "compare_v0_v1"
    compare_dir = compare_root / folder_name
    compare_dir.mkdir(parents=True, exist_ok=False)
    return compare_dir


def build_compare_dataframe(df_v0: pd.DataFrame, df_v1: pd.DataFrame) -> pd.DataFrame:
    v0_cols = [
        c for c in ["date", "ppo_mean_norm", "ppo_min_norm", "ppo_max_norm", "dji_mean_norm"]
        if c in df_v0.columns
    ]
    v1_cols = [
        c for c in ["date", "ppo_mean_norm", "ppo_min_norm", "ppo_max_norm", "dji_mean_norm"]
        if c in df_v1.columns
    ]
    merged = pd.merge(
        df_v0[v0_cols],
        df_v1[v1_cols],
        on="date",
        how="outer",
        suffixes=("_v0", "_v1"),
    ).sort_values("date").reset_index(drop=True)

    # Prefer DJI from v0. If it is unavailable, use DJI from v1.
    if "dji_mean_norm_v0" in merged.columns:
        merged["dji_mean_norm"] = merged["dji_mean_norm_v0"]
    elif "dji_mean_norm_v1" in merged.columns:
        merged["dji_mean_norm"] = merged["dji_mean_norm_v1"]
    elif "dji_mean_norm" not in merged.columns:
        merged["dji_mean_norm"] = np.nan

    return merged


def plot_compare(merged: pd.DataFrame, compare_dir: Path, root_v0: Path, root_v1: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(merged))

    def add_strategy(prefix: str, color: str, label: str) -> None:
        mean_col = f"ppo_mean_norm_{prefix}"
        min_col = f"ppo_min_norm_{prefix}"
        max_col = f"ppo_max_norm_{prefix}"
        if mean_col not in merged.columns:
            return
        mean = pd.to_numeric(merged[mean_col], errors="coerce")
        min_v = pd.to_numeric(merged[min_col], errors="coerce") if min_col in merged.columns else mean
        max_v = pd.to_numeric(merged[max_col], errors="coerce") if max_col in merged.columns else mean
        ret = _strategy_return_percent(mean)
        ax.fill_between(x, min_v, max_v, color=color, alpha=0.14, label=f"{label} range")
        ax.plot(x, mean, color=color, linewidth=2.2, label=f"{label} mean ({ret:.2f}%)")

    add_strategy("v0", "blue", "v0")
    add_strategy("v1", "red", "v1")

    if "dji_mean_norm" in merged.columns:
        dji = pd.to_numeric(merged["dji_mean_norm"], errors="coerce")
        if dji.notna().any():
            ret = _strategy_return_percent(dji)
            ax.plot(x, dji, color="tab:green", linewidth=1.8, label=f"DJI ({ret:.2f}%)")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.1, label="portfolio value = 1.0")
    if len(merged) > 0:
        tick_positions = np.linspace(0, len(merged) - 1, min(MAX_XTICKS, len(merged))).astype(int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(merged["date"].iloc[i]) for i in tick_positions], rotation=45, ha="right")

    ax.set_title("Manual v0 vs v1 Repeat-Mean Portfolio Value (MVO excluded)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio Value (initial = 1.0)")
    ax.text(
        0.01,
        0.02,
        f"v0: {root_v0.name}\nv1: {root_v1.name}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.75, edgecolor="none"),
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(compare_dir / "compare_portfolio_value.png", dpi=150, bbox_inches="tight")
    fig.savefig(compare_dir / "compare_report.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Keep the requested filenames. They are aliases of the same comparison chart unless you later customize them.
    for name in ["compare_cumulative_return.png", "compare_reward_curve.png"]:
        (compare_dir / name).write_bytes((compare_dir / "compare_portfolio_value.png").read_bytes())


def write_summary(merged: pd.DataFrame, compare_dir: Path, root_v0: Path, root_v1: Path) -> None:
    rows = []
    for label, col in [
        ("v0", "ppo_mean_norm_v0"),
        ("v1", "ppo_mean_norm_v1"),
        ("dji", "dji_mean_norm"),
    ]:
        if col in merged.columns:
            ret = _strategy_return_percent(merged[col])
            final_value = pd.to_numeric(merged[col], errors="coerce").dropna()
            rows.append(
                {
                    "strategy": label,
                    "final_normalized_value": float(final_value.iloc[-1]) if not final_value.empty else np.nan,
                    "return_percent": ret,
                }
            )
    pd.DataFrame(rows).to_csv(compare_dir / "compare_summary.csv", index=False, encoding="utf-8-sig")

    config_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_root": str(EXPERIMENT_ROOT),
        "v0_rolling_name": V0_ROLLING_NAME,
        "v1_rolling_name": V1_ROLLING_NAME,
        "v0_rolling_path": str(root_v0),
        "v1_rolling_path": str(root_v1),
        "compare_dir": str(compare_dir),
    }
    (compare_dir / "compare_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    root_v0 = resolve_rolling_root("v0", V0_ROLLING_NAME)
    root_v1 = resolve_rolling_root("v1", V1_ROLLING_NAME)

    df_v0 = collect_repeat_mean_series(root_v0, "v0")
    df_v1 = collect_repeat_mean_series(root_v1, "v1")
    merged = build_compare_dataframe(df_v0, df_v1)

    compare_dir = make_compare_dir(root_v0, root_v1)
    merged.to_csv(compare_dir / "compare_metrics.csv", index=False, encoding="utf-8-sig")
    plot_compare(merged, compare_dir, root_v0, root_v1)
    write_summary(merged, compare_dir, root_v0, root_v1)

    print("Manual compare finished.")
    print(f"v0 source: {root_v0}")
    print(f"v1 source: {root_v1}")
    print(f"compare output: {compare_dir}")


if __name__ == "__main__":
    main()
