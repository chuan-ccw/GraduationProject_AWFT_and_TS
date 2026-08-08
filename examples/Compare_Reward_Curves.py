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

from result_manager import append_compare_experiment_record, load_experiment_metadata


# =====================================================
# Manual Reward Curve compare settings
# =====================================================
# 可填：
# 1. rolling 實驗總資料夾
# 2. 單一 window 資料夾
# 3. reward_curve_average_v* 資料夾
#
# rolling 資料夾會依 window 配對，每個 window 各輸出一張比較圖。
# 若只想比較一個 window，可填 WINDOW_FILTER，例如 "window_01"。
EXPERIMENT_ROOT = os.environ.get(
    "EXPERIMENT_ROOT",
    r"C:\Users\User\Desktop\GraduationProject\Experence_result",
)
SOURCE_A_FOLDER = os.environ.get("REWARD_CURVE_SOURCE_A", "rolling_2018-2025_train3y_test1y_shift1y_20260719_reinforce")
SOURCE_B_FOLDER = os.environ.get("REWARD_CURVE_SOURCE_B", "rolling_2018-2025_train3y_test1y_shift1y_20260805_005125_reinforce")
WINDOW_FILTER = os.environ.get("REWARD_CURVE_WINDOW_FILTER", "")
COMPARE_NAME = os.environ.get("REWARD_CURVE_COMPARE_NAME", "v0_vs_v6(n=60)")

REWARD_DISPLAY_NAMES = {
    "v0": "original_reward",
    "v1": "buy_low_sell_high_reward",
    "v2": "future_day_reward",
    "v3": "future_avg_reward",
    "v4": "future_avg_delta_action_reward",
    "v5": "WFT",
    "v6": "DWFT",
}


def reward_display_name(reward_type: str | None) -> str:
    key = str(reward_type or "").strip().lower()
    if key.startswith("reward_"):
        key = key[len("reward_") :]
    return REWARD_DISPLAY_NAMES.get(key, key or "unknown_reward")


def safe_name(value: str, max_length: int = 70) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "compare")[:max_length]


def resolve_folder(value: str, label: str) -> Path:
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        raise ValueError(
            f"請先設定 {label}。可直接修改 SOURCE_A_FOLDER / SOURCE_B_FOLDER，"
            "或設定環境變數 REWARD_CURVE_SOURCE_A / REWARD_CURVE_SOURCE_B。"
        )

    direct = Path(text).expanduser()
    if direct.exists():
        return direct.resolve()

    root = Path(EXPERIMENT_ROOT)
    candidates = [path for path in root.rglob(text) if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"找不到 {label}: {text}")
    if len(candidates) > 1:
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        print(f"{label} 找到多個同名資料夾，使用最新：{candidates[0]}")
    return candidates[0].resolve()


def _window_key_from_path(csv_path: Path) -> str:
    for parent in csv_path.parents:
        if re.match(r"window_\d+", parent.name):
            return parent.name
    return csv_path.parent.name


def discover_reward_curves(folder: Path) -> dict[str, Path]:
    """Find 5-times average reward_curve.csv files under one selected folder."""
    candidates: list[Path] = []

    direct_csv = folder / "reward_curve.csv"
    if direct_csv.exists():
        candidates.append(direct_csv)

    candidates.extend(folder.glob("reward_curve_average_*/reward_curve.csv"))
    candidates.extend(folder.glob("window_*/reward_curve_average_*/reward_curve.csv"))

    unique_candidates = sorted({path.resolve() for path in candidates})
    if not unique_candidates:
        raise FileNotFoundError(
            f"{folder} 找不到 reward_curve.csv。"
            "預期位置為 window_*/reward_curve_average_*/reward_curve.csv。"
        )

    curves: dict[str, Path] = {}
    for csv_path in unique_candidates:
        key = _window_key_from_path(csv_path)
        if WINDOW_FILTER and WINDOW_FILTER.lower() not in key.lower():
            continue
        curves[key] = csv_path

    if not curves:
        raise FileNotFoundError(f"套用 WINDOW_FILTER={WINDOW_FILTER!r} 後沒有 reward curve。")
    return curves


def _window_number(key: str) -> str | None:
    match = re.search(r"window_(\d+)", key)
    return match.group(1) if match else None


def pair_reward_curves(
    curves_a: dict[str, Path],
    curves_b: dict[str, Path],
) -> list[tuple[str, Path, Path]]:
    common_keys = sorted(set(curves_a) & set(curves_b))
    if common_keys:
        return [(key, curves_a[key], curves_b[key]) for key in common_keys]

    if len(curves_a) == 1 and len(curves_b) == 1:
        return [("selected_window", next(iter(curves_a.values())), next(iter(curves_b.values())))]

    by_number_a = {_window_number(key): path for key, path in curves_a.items() if _window_number(key)}
    by_number_b = {_window_number(key): path for key, path in curves_b.items() if _window_number(key)}
    common_numbers = sorted(set(by_number_a) & set(by_number_b), key=int)
    if common_numbers:
        return [
            (f"window_{number}", by_number_a[number], by_number_b[number])
            for number in common_numbers
        ]

    raise ValueError(
        "兩個資料夾找不到可配對的 window。"
        f"\nA windows: {list(curves_a)}"
        f"\nB windows: {list(curves_b)}"
    )


def load_reward_curve(csv_path: Path) -> pd.DataFrame:
    curve = pd.read_csv(csv_path)
    if curve.empty:
        raise ValueError(f"Reward curve 是空的：{csv_path}")

    epoch_col = next((column for column in ["epoch", "episodes"] if column in curve.columns), None)
    mean_col = next(
        (
            column
            for column in [
                "average_reward",
                "rollout/episode_average_reward",
                "episode_reward_mean",
                "rollout/ep_rew_mean",
            ]
            if column in curve.columns
        ),
        None,
    )
    if mean_col is None:
        raise KeyError(f"{csv_path} 缺少 average reward 欄位。columns={list(curve.columns)}")

    epoch = (
        pd.to_numeric(curve[epoch_col], errors="coerce")
        if epoch_col
        else pd.Series(np.arange(1, len(curve) + 1), index=curve.index)
    )
    mean = pd.to_numeric(curve[mean_col], errors="coerce")
    minimum = pd.to_numeric(curve.get("min_reward", mean), errors="coerce")
    maximum = pd.to_numeric(curve.get("max_reward", mean), errors="coerce")

    normalized = pd.DataFrame(
        {
            "epoch": epoch,
            "average_reward": mean,
            "min_reward": minimum,
            "max_reward": maximum,
        }
    ).dropna(subset=["epoch", "average_reward"])
    normalized["epoch"] = normalized["epoch"].astype(int)
    normalized = normalized.drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")

    if "times_count" in curve.columns:
        times_count = pd.to_numeric(curve["times_count"], errors="coerce").dropna()
        if not times_count.empty and int(times_count.iloc[0]) != 5:
            print(f"Warning: {csv_path} 的 times_count={int(times_count.iloc[0])}，不是 5。")
    elif {"min_reward", "max_reward"}.issubset(curve.columns) is False:
        print(f"Warning: {csv_path} 沒有 Range 欄位，將以平均線本身作為上下界。")

    return normalized


def make_compare_dir(source_a: Path, source_b: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = safe_name(COMPARE_NAME) if COMPARE_NAME.strip() else (
        f"{safe_name(source_a.name, 35)}__vs__{safe_name(source_b.name, 35)}__{timestamp}"
    )
    compare_dir = Path(EXPERIMENT_ROOT) / "compare_reward_curve" / folder_name
    compare_dir.mkdir(parents=True, exist_ok=False)
    return compare_dir


def source_label(metadata: dict[str, object], fallback: str) -> str:
    reward_name = reward_display_name(str(metadata.get("reward_type", "")))
    model = str(metadata.get("model", "") or "").upper()
    base = f"{reward_name} - {model}" if model else reward_name

    params = metadata.get("reward_params", {})
    if isinstance(params, dict) and params:
        preferred_order = [
            "delta_weight_variable",
            "future_variable",
            "n_future",
            "c",
            "alpha",
            "beta",
            "price_col",
        ]
        ordered_keys = [key for key in preferred_order if key in params]
        ordered_keys.extend(key for key in params if key not in ordered_keys)
        parts = [f"{key}={params[key]}" for key in ordered_keys]
        return f"{base} [{', '.join(parts)}]"

    return base if base != "unknown_reward" else fallback


def plot_one_comparison(
    key: str,
    curve_a: pd.DataFrame,
    curve_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    compare_dir: Path,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.fill_between(
        curve_a["epoch"],
        curve_a["min_reward"],
        curve_a["max_reward"],
        color="blue",
        alpha=0.14,
    )
    ax.plot(
        curve_a["epoch"],
        curve_a["average_reward"],
        color="blue",
        linewidth=2.1,
        label=label_a,
    )

    ax.fill_between(
        curve_b["epoch"],
        curve_b["min_reward"],
        curve_b["max_reward"],
        color="red",
        alpha=0.14,
    )
    ax.plot(
        curve_b["epoch"],
        curve_b["average_reward"],
        color="red",
        linewidth=2.1,
        label=label_b,
    )

    ax.set_title(f"Training Reward Curve Comparison - {key.replace('_', ' ')}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("average reward")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    image_path = compare_dir / f"compare_reward_curve_{safe_name(key)}.png"
    fig.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    merged = pd.merge(
        curve_a.rename(
            columns={
                "average_reward": "source_a_average_reward",
                "min_reward": "source_a_min_reward",
                "max_reward": "source_a_max_reward",
            }
        ),
        curve_b.rename(
            columns={
                "average_reward": "source_b_average_reward",
                "min_reward": "source_b_min_reward",
                "max_reward": "source_b_max_reward",
            }
        ),
        on="epoch",
        how="outer",
    ).sort_values("epoch")
    csv_path = compare_dir / f"compare_reward_curve_{safe_name(key)}.csv"
    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return image_path, csv_path


def main() -> None:
    source_a = resolve_folder(SOURCE_A_FOLDER, "SOURCE_A_FOLDER")
    source_b = resolve_folder(SOURCE_B_FOLDER, "SOURCE_B_FOLDER")
    curves_a = discover_reward_curves(source_a)
    curves_b = discover_reward_curves(source_b)
    pairs = pair_reward_curves(curves_a, curves_b)

    metadata_a = load_experiment_metadata(source_a)
    metadata_b = load_experiment_metadata(source_b)
    label_a = source_label(metadata_a, source_a.name)
    label_b = source_label(metadata_b, source_b.name)
    compare_dir = make_compare_dir(source_a, source_b)

    manifest_rows: list[dict[str, object]] = []
    for key, csv_a, csv_b in pairs:
        curve_a = load_reward_curve(csv_a)
        curve_b = load_reward_curve(csv_b)
        image_path, merged_csv = plot_one_comparison(
            key,
            curve_a,
            curve_b,
            label_a,
            label_b,
            compare_dir,
        )
        manifest_rows.append(
            {
                "window": key,
                "image": str(image_path),
                "merged_csv": str(merged_csv),
                "source_a_csv": str(csv_a),
                "source_b_csv": str(csv_b),
                "source_a_label": label_a,
                "source_b_label": label_b,
                "range_definition": "min-max across repeated runs",
            }
        )

    pd.DataFrame(manifest_rows).to_csv(
        compare_dir / "image_source_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    config_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "compare_type": "training_reward_curve",
        "source_a_folder": str(source_a),
        "source_b_folder": str(source_b),
        "source_a_metadata": metadata_a,
        "source_b_metadata": metadata_b,
        "window_filter": WINDOW_FILTER,
        "matched_windows": [key for key, _, _ in pairs],
        "compare_dir": str(compare_dir),
        "line_colors": {"source_a": "blue", "source_b": "red"},
        "range_colors": {"source_a": "light_blue", "source_b": "light_red"},
        "legend": "average lines only",
    }
    (compare_dir / "compare_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    history_csv = append_compare_experiment_record(
        EXPERIMENT_ROOT,
        compare_dir,
        source_a,
        source_b,
        compare_type="training_reward_curve",
        extra={"window_filter": WINDOW_FILTER, "matched_window_count": len(pairs)},
    )

    print("Reward Curve compare finished.")
    print(f"Source A: {source_a}")
    print(f"Source B: {source_b}")
    print(f"Matched windows: {len(pairs)}")
    print(f"Output: {compare_dir}")
    print(f"Compare experiment history updated: {history_csv}")


if __name__ == "__main__":
    main()
