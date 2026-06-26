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

try:
    from finrl import config
except Exception:
    # Compare script only needs EXPERIMENT_ROOT / REWARD_TYPE.
    # This fallback lets the script run even if optional FinRL trading dependencies
    # such as gymnasium are not installed in the current Python environment.
    class _FallbackConfig:
        EXPERIMENT_ROOT = os.environ.get(
            "EXPERIMENT_ROOT",
            r"C:\\Users\\User\\Desktop\\GraduationProject\\Experence_result",
        )
        REWARD_TYPE = os.environ.get("REWARD_TYPE", "v1")

    config = _FallbackConfig()

# =====================================================
# Manual v0 vs target reward compare settings
# =====================================================
# 使用流程：
# 1. 先跑 v0 rolling 實驗。
# 2. 再跑任一新版 reward，例如 v1 / v2 / v3 / v4。
# 3. 直接執行本檔案。程式會「永遠用 v0 當 baseline」，target 版本會依資料夾/資料自動偵測。
#
# 重要：
# - V0_ROLLING_NAME 留空：自動找最新 reward_v0/rolling_*。
# - TARGET_ROLLING_NAME 留空：依 TARGET_REWARD_TYPE 自動找最新 reward_<target>/rolling_*。
# - TARGET_REWARD_TYPE 留空：優先讀環境變數 REWARD_TYPE，其次讀 config.REWARD_TYPE。
# - 若 TARGET_ROLLING_NAME 是完整路徑，程式會從該路徑自動偵測 v1/v2/v3/v4。

EXPERIMENT_ROOT = os.environ.get(
    "EXPERIMENT_ROOT",
    getattr(
        config,
        "EXPERIMENT_ROOT",
        r"C:\Users\User\Desktop\GraduationProject\Experence_result",
    ),
)

# 可填 rolling 資料夾名稱或完整路徑；留空 = 自動找最新。
V0_ROLLING_NAME = ""
TARGET_ROLLING_NAME = ""

# 可填 "v1" / "v2" / "v3" / "v4"；留空 = 用目前 REWARD_TYPE/config.REWARD_TYPE，再由資料夾校正。
TARGET_REWARD_TYPE = 'v4'
#TARGET_REWARD_TYPE = os.environ.get("TARGET_REWARD_TYPE", os.environ.get("REWARD_TYPE", getattr(config, "REWARD_TYPE", "v1")))

# 輸出資料夾名稱。留空時自動產生 compare_v0_<target>_...。
COMPARE_NAME = ""

MAX_XTICKS = 14

REWARD_DISPLAY_NAMES = {
    "v0": "original_reward",
    "original": "original_reward",
    "finrl_original": "original_reward",
    "v1": "buy_low_sell_high_reward",
    "our_reward_v1": "buy_low_sell_high_reward",
    "v2": "future_day_reward",
    "reward_v2": "future_day_reward",
    "future_day_return": "future_day_reward",
    "v3": "future_avg_reward",
    "reward_v3": "future_avg_reward",
    "future_avg_return": "future_avg_reward",
    "v4": "future_avg_delta_action_reward",
    "reward_v4": "future_avg_delta_action_reward",
    "future_avg_delta_action_return": "future_avg_delta_action_reward",
    "future_avg_delta_action_reward": "future_avg_delta_action_reward",
}


def reward_display_name(reward_type: str | None) -> str:
    """Return the display name used in plots instead of v0/v1/v2/v3/v4."""
    key = str(reward_type or "").strip().lower()
    if key.startswith("reward_") and key not in REWARD_DISPLAY_NAMES:
        key = key[len("reward_"):]
    return REWARD_DISPLAY_NAMES.get(key, str(reward_type or ""))


def set_axis_labels_horizontal(ax, xlabel: str | None = None, ylabel: str | None = None, ylabel_pad: int = 65) -> None:
    """Keep x-axis label horizontal, but keep y-axis label in the default vertical style."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, rotation=0)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def _safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if len(text) > max_len else text


def normalize_reward_type(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("reward_"):
        text = text[len("reward_"):]
    aliases = {
        "original": "v0",
        "finrl_original": "v0",
        "our_reward_v1": "v1",
        "reward_v2": "v2",
        "future_day_return": "v2",
        "reward_v3": "v3",
        "future_avg_return": "v3",
        "reward_v4": "v4",
        "future_avg_delta_action_return": "v4",
        "future_avg_delta_action_reward": "v4",
    }
    return aliases.get(text, text)


def format_yyyy_mm(value) -> str:
    """Format plot x-axis date labels as YYYY-MM, never YYYY/MM."""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.notna(ts):
        return ts.strftime("%Y-%m")
    text = str(value)
    match = re.search(r"(\d{4})[-/](\d{1,2})", text)
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"
    return text


def make_month_tick_positions(length: int, max_xticks: int = MAX_XTICKS) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=int)
    return np.linspace(0, length - 1, min(max_xticks, length)).astype(int)


def set_month_xticklabels(ax, dates, max_xticks: int = MAX_XTICKS) -> None:
    positions = make_month_tick_positions(len(dates), max_xticks=max_xticks)
    if len(positions) == 0:
        return
    ax.set_xticks(positions)
    ax.set_xticklabels([format_yyyy_mm(dates[i]) for i in positions], rotation=0, ha="center")


def find_latest_rolling_root(reward_type: str) -> Path | None:
    reward_type = normalize_reward_type(reward_type)
    experiment_root = Path(EXPERIMENT_ROOT)
    candidates: list[Path] = []
    for base in [experiment_root / f"reward_{reward_type}", experiment_root / f"reward_{reward_type}" / "rolling_window"]:
        if base.exists():
            candidates.extend([p for p in base.iterdir() if p.is_dir() and p.name.startswith("rolling_")])
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def find_rolling_root_by_name_any_reward(rolling_name: str) -> Path | None:
    experiment_root = Path(EXPERIMENT_ROOT)
    candidates: list[Path] = []
    for reward_root in experiment_root.glob("reward_*"):
        if not reward_root.is_dir():
            continue
        for candidate in [reward_root / rolling_name, reward_root / "rolling_window" / rolling_name]:
            if candidate.exists():
                candidates.append(candidate.resolve())
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def resolve_rolling_root(label: str, reward_type: str | None, rolling_name_or_path: str) -> Path:
    """Resolve a rolling folder from explicit path, name, or latest reward folder."""
    reward_type = normalize_reward_type(reward_type)
    value = str(rolling_name_or_path or "").strip().strip('"').strip("'")

    if value:
        direct = Path(value)
        if direct.exists():
            return direct.resolve()

        if reward_type:
            experiment_root = Path(EXPERIMENT_ROOT)
            candidates = [
                experiment_root / f"reward_{reward_type}" / value,
                experiment_root / f"reward_{reward_type}" / "rolling_window" / value,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate.resolve()

        any_match = find_rolling_root_by_name_any_reward(value)
        if any_match is not None:
            return any_match

        raise FileNotFoundError(f"找不到 {label} rolling folder: {value}")

    if not reward_type:
        raise ValueError(f"{label} 沒有提供 rolling folder，也無法判斷 reward_type。")

    latest = find_latest_rolling_root(reward_type)
    if latest is None:
        raise FileNotFoundError(f"找不到最新 reward_{reward_type} rolling folder。")
    return latest.resolve()


def detect_reward_type_from_rolling_root(rolling_root: Path, default: str | None = None) -> str:
    """Detect reward version from config, parent folder, repeat_mean folder, or CSV filename."""
    config_path = rolling_root / "rolling_config.json"
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            detected = normalize_reward_type(payload.get("reward_type"))
            if detected:
                return detected
        except Exception:
            pass

    for part in [rolling_root.parent.name, rolling_root.parent.parent.name if rolling_root.parent.parent else ""]:
        m = re.fullmatch(r"reward_(.+)", part)
        if m:
            detected = normalize_reward_type(m.group(1))
            if detected:
                return detected

    for p in rolling_root.glob("window_*/repeat_mean_*"):
        detected = normalize_reward_type(p.name.replace("repeat_mean_", ""))
        if detected:
            return detected

    for p in rolling_root.glob("window_*/repeat_mean_*/backtest_result_repeat_mean_*.csv"):
        m = re.search(r"backtest_result_repeat_mean_(.+?)_\d{4}-\d{2}-\d{2}_to_", p.name)
        if m:
            detected = normalize_reward_type(m.group(1))
            if detected:
                return detected

    detected = normalize_reward_type(default)
    if detected:
        return detected
    raise ValueError(f"無法從 {rolling_root} 判斷 reward type。")


def collect_repeat_mean_series(rolling_root: Path, reward_type: str) -> pd.DataFrame:
    """Read all window repeat_mean backtest_result.csv files and concatenate them."""
    reward_type = normalize_reward_type(reward_type)
    patterns = [
        f"window_*/repeat_mean_{reward_type}/backtest_result.csv",
        f"window_*/repeat_mean_{reward_type}/backtest_result_repeat_mean_{reward_type}_*.csv",
    ]

    csv_paths: list[Path] = []
    for pattern in patterns:
        csv_paths.extend(sorted(rolling_root.glob(pattern)))

    if not csv_paths:
        # Last-resort fallback: if target folder was selected but reward type name changed, use any repeat_mean_* folder.
        csv_paths = sorted(rolling_root.glob("window_*/repeat_mean_*/backtest_result.csv"))

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})
        df = df.copy()
        df["source_window"] = csv_path.parents[1].name
        df["source_csv"] = str(csv_path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"{rolling_root} 裡找不到 repeat_mean_{reward_type}/backtest_result.csv")

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


def make_compare_dir(root_v0: Path, root_target: Path, target_reward_type: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if COMPARE_NAME.strip():
        folder_name = _safe_name(COMPARE_NAME.strip())
    else:
        folder_name = (
            f"compare_v0_{_safe_name(root_v0.name, 36)}"
            f"__{target_reward_type}_{_safe_name(root_target.name, 36)}"
            f"__{timestamp}"
        )
    compare_root = Path(EXPERIMENT_ROOT) / f"compare_v0_{target_reward_type}"
    compare_dir = compare_root / folder_name
    compare_dir.mkdir(parents=True, exist_ok=False)
    return compare_dir


def _select_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in ["date", "ppo_mean_norm", "ppo_min_norm", "ppo_max_norm", "dji_mean_norm"]
        if c in df.columns
    ]


def build_compare_dataframe(df_v0: pd.DataFrame, df_target: pd.DataFrame, target_reward_type: str) -> pd.DataFrame:
    target_suffix = f"_{target_reward_type}"
    merged = pd.merge(
        df_v0[_select_cols(df_v0)],
        df_target[_select_cols(df_target)],
        on="date",
        how="outer",
        suffixes=("_v0", target_suffix),
    ).sort_values("date").reset_index(drop=True)

    # If a column did not overlap, pandas may leave it unsuffixed. Normalize names for downstream plotting.
    for base in ["ppo_mean_norm", "ppo_min_norm", "ppo_max_norm"]:
        plain = base
        target_name = f"{base}_{target_reward_type}"
        if plain in merged.columns and target_name not in merged.columns:
            merged = merged.rename(columns={plain: target_name})

    if "dji_mean_norm_v0" in merged.columns:
        merged["dji_mean_norm"] = merged["dji_mean_norm_v0"]
    elif f"dji_mean_norm_{target_reward_type}" in merged.columns:
        merged["dji_mean_norm"] = merged[f"dji_mean_norm_{target_reward_type}"]
    elif "dji_mean_norm" not in merged.columns:
        merged["dji_mean_norm"] = np.nan

    return merged


def plot_compare(merged: pd.DataFrame, compare_dir: Path, root_v0: Path, root_target: Path, target_reward_type: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(merged))
    baseline_name = reward_display_name("v0")
    target_name = reward_display_name(target_reward_type)

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

    add_strategy("v0", "blue", baseline_name)
    add_strategy(target_reward_type, "red", target_name)

    if "dji_mean_norm" in merged.columns:
        dji = pd.to_numeric(merged["dji_mean_norm"], errors="coerce")
        if dji.notna().any():
            ret = _strategy_return_percent(dji)
            ax.plot(x, dji, color="tab:green", linewidth=1.8, label=f"DJI ({ret:.2f}%)")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.1, label="portfolio value = 1.0")
    set_month_xticklabels(ax, merged["date"].tolist(), max_xticks=MAX_XTICKS)

    ax.set_title(f"{baseline_name} vs {target_name} Repeat-Mean Portfolio Value")
    set_axis_labels_horizontal(ax, "Date", "Normalized Portfolio Value (initial = 1.0)", ylabel_pad=95)
    ax.text(
        0.01,
        0.02,
        f"{baseline_name}: {root_v0.name}\n{target_name}: {root_target.name}",
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

    for name in ["compare_cumulative_return.png", "compare_reward_curve.png"]:
        (compare_dir / name).write_bytes((compare_dir / "compare_portfolio_value.png").read_bytes())


def write_summary(merged: pd.DataFrame, compare_dir: Path, root_v0: Path, root_target: Path, target_reward_type: str) -> None:
    rows = []
    for label, col in [
        ("v0", "ppo_mean_norm_v0"),
        (target_reward_type, f"ppo_mean_norm_{target_reward_type}"),
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
        "baseline_reward_type": "v0",
        "target_reward_type": target_reward_type,
        "v0_rolling_name_setting": V0_ROLLING_NAME,
        "target_rolling_name_setting": TARGET_ROLLING_NAME,
        "v0_rolling_path": str(root_v0),
        "target_rolling_path": str(root_target),
        "compare_dir": str(compare_dir),
        "x_axis_format": "YYYY-MM",
    }
    (compare_dir / "compare_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    # Baseline is always v0.
    root_v0 = resolve_rolling_root("v0 baseline", "v0", V0_ROLLING_NAME)

    requested_target = normalize_reward_type(TARGET_REWARD_TYPE)
    root_target = resolve_rolling_root("target", requested_target, TARGET_ROLLING_NAME)
    target_reward_type = detect_reward_type_from_rolling_root(root_target, requested_target)

    if target_reward_type == "v0":
        raise ValueError("target_reward_type 偵測為 v0；compare 需要 v0 以外的版本，例如 v1/v2/v3/v4。")

    df_v0 = collect_repeat_mean_series(root_v0, "v0")
    df_target = collect_repeat_mean_series(root_target, target_reward_type)
    merged = build_compare_dataframe(df_v0, df_target, target_reward_type)

    compare_dir = make_compare_dir(root_v0, root_target, target_reward_type)
    merged.to_csv(compare_dir / "compare_metrics.csv", index=False, encoding="utf-8-sig")
    plot_compare(merged, compare_dir, root_v0, root_target, target_reward_type)
    write_summary(merged, compare_dir, root_v0, root_target, target_reward_type)

    print("Manual compare finished.")
    print(f"v0 source: {root_v0}")
    print(f"target reward: {target_reward_type}")
    print(f"target source: {root_target}")
    print(f"compare output: {compare_dir}")


if __name__ == "__main__":
    main()
