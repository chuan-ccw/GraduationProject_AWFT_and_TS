from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# 計算AWFT reward value (抓 a, w, d 用 scale 在 action 的 AWFT 計算 reward)

# =====================================================
# 1. User setting
# =====================================================
# 平常只需要修改 TARGET_FOLDER。
EXPERIMENT_ROOT = Path(
    r"C:\Users\User\Desktop\GraduationProject\Experence_result"
)

TARGET_FOLDER = (
    "rolling_2018-2025_train3y_test1y_shift1y_20260903_104709_reinforce"
)

EXPECTED_TIMES_PER_WINDOW = 5
WINDOW_PATTERN = re.compile(r"^window_(?P<id>\d+)(?:_|$)")
TIMES_PATTERN = re.compile(r"^times_(?P<id>\d+)_seed_(?P<seed>\d+)$")

CASE_DEFINITIONS = [
    ("buy_up", "買入 (a>0) / 上漲", "positive"),
    ("buy_down", "買入 (a>0) / 下跌", "negative"),
    ("sell_up", "賣出 (a<0) / 上漲", "negative"),
    ("sell_down", "賣出 (a<0) / 下跌", "positive"),
    ("hold_up", "持有 (a=0,w>0) / 上漲", "positive"),
    ("hold_down", "持有 (a=0,w>0) / 下跌", "negative"),
    ("no_trade_no_holding", "無交易、無持有 (a=0,w=0)", "zero"),
]


# =====================================================
# 2. Folder helpers
# =====================================================
def find_target_folder(search_root: Path, target_folder: str) -> Path | None:
    direct = Path(target_folder)
    if direct.exists() and direct.is_dir():
        return direct.resolve()

    if not search_root.exists() or not search_root.is_dir():
        print("ERROR: Experiment root folder not found.")
        print(f"\nSearch Root:\n{search_root}")
        return None

    matches = sorted(
        {
            path.resolve()
            for path in search_root.rglob(target_folder)
            if path.is_dir() and path.name == target_folder
        },
        key=lambda path: str(path).lower(),
    )
    if not matches:
        print("ERROR: Target rolling experiment folder not found.")
        print(f"\nTarget:\n{target_folder}")
        return None
    if len(matches) > 1:
        print("ERROR: Multiple matching rolling experiment folders found.")
        for index, path in enumerate(matches, start=1):
            print(f"{index}. {path}")
        return None
    return matches[0]


def _folder_sort_key(path: Path, pattern: re.Pattern[str]) -> tuple[int, str]:
    match = pattern.match(path.name)
    if match is None:
        return 10**9, path.name.lower()
    return int(match.group("id")), path.name.lower()


def find_windows(rolling_root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in rolling_root.iterdir()
            if path.is_dir() and WINDOW_PATTERN.match(path.name)
        ],
        key=lambda path: _folder_sort_key(path, WINDOW_PATTERN),
    )


def find_times(window_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in window_dir.iterdir()
            if path.is_dir() and TIMES_PATTERN.fullmatch(path.name)
        ],
        key=lambda path: _folder_sort_key(path, TIMES_PATTERN),
    )


def short_window_name(window_dir: Path) -> str:
    match = WINDOW_PATTERN.match(window_dir.name)
    if match is None:
        raise ValueError(f"Invalid window folder: {window_dir.name}")
    return f"window_{int(match.group('id')):02d}"


def times_info(times_dir: Path) -> tuple[str, int]:
    match = TIMES_PATTERN.fullmatch(times_dir.name)
    if match is None:
        raise ValueError(f"Invalid times folder: {times_dir.name}")
    return f"times_{int(match.group('id')):02d}", int(match.group("seed"))


# =====================================================
# 3. AWFT helpers
# =====================================================
def _latest_awft_parameter_csv(times_dir: Path) -> Path:
    candidates = [
        path for path in times_dir.glob("awft_parameters_*_v7_*.csv") if path.is_file()
    ]
    if not candidates:
        candidates = [
            path for path in times_dir.glob("awft_parameters_*.csv") if path.is_file()
        ]
    if not candidates:
        raise FileNotFoundError(f"No AWFT parameter CSV found in {times_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_awft_scale(times_dir: Path) -> float:
    config_path = times_dir / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing experiment_config.json: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    reward_type = str(payload.get("reward_type", "")).strip().lower()
    if reward_type not in {"v7", "reward_v7", "awft"}:
        raise ValueError(
            f"{times_dir} is not an AWFT/v7 experiment (reward_type={reward_type!r})."
        )

    reward_params = payload.get("reward_params", {}) or {}
    if "scale" not in reward_params:
        raise ValueError(f"AWFT scale is missing in {config_path}")
    return float(reward_params["scale"])


def _classify_and_calculate(data: pd.DataFrame, scale: float) -> pd.DataFrame:
    required = {"a", "w", "d"}
    if not required.issubset(data.columns):
        missing = sorted(required - set(data.columns))
        raise ValueError(f"AWFT parameter CSV missing columns: {missing}")

    out = data.copy()
    for column in ["a", "w", "d"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["a", "w", "d"]).copy()

    a = out["a"]
    w = out["w"]
    d = out["d"]

    # Corrected AWFT formula:
    #   a != 0          -> a * d * scale
    #   a == 0, w > 0   -> w * d
    #   a == 0, w == 0  -> 0
    out["reward_value"] = 0.0
    trade_mask = a != 0
    hold_mask = (a == 0) & (w > 0)
    out.loc[trade_mask, "reward_value"] = (
        a[trade_mask] * d[trade_mask] * scale
    )
    out.loc[hold_mask, "reward_value"] = w[hold_mask] * d[hold_mask]

    out["case"] = pd.NA
    out.loc[(a > 0) & (d > 0), "case"] = "buy_up"
    out.loc[(a > 0) & (d < 0), "case"] = "buy_down"
    out.loc[(a < 0) & (d > 0), "case"] = "sell_up"
    out.loc[(a < 0) & (d < 0), "case"] = "sell_down"
    out.loc[(a == 0) & (w > 0) & (d > 0), "case"] = "hold_up"
    out.loc[(a == 0) & (w > 0) & (d < 0), "case"] = "hold_down"
    out.loc[(a == 0) & (w == 0), "case"] = "no_trade_no_holding"
    return out


def summarize_one_times(
    times_dir: Path,
    window_label: str,
) -> tuple[list[dict[str, object]], int]:
    times_label, seed = times_info(times_dir)
    scale = _load_awft_scale(times_dir)
    parameter_path = _latest_awft_parameter_csv(times_dir)
    data = pd.read_csv(parameter_path)
    classified = _classify_and_calculate(data, scale)

    unclassified_count = int(classified["case"].isna().sum())
    rows: list[dict[str, object]] = []

    for case_code, case_name, expected_sign in CASE_DEFINITIONS:
        values = pd.to_numeric(
            classified.loc[classified["case"] == case_code, "reward_value"],
            errors="coerce",
        ).dropna()
        rows.append(
            {
                "window": window_label,
                "times": times_label,
                "seed": seed,
                "case": case_code,
                "case_name": case_name,
                "expected_reward_sign": expected_sign,
                "sample_count": int(len(values)),
                "mean_reward": float(values.mean()) if not values.empty else np.nan,
                "scale": scale,
                "source_csv": str(parameter_path),
            }
        )

    return rows, unclassified_count


# =====================================================
# 4. Output
# =====================================================
def main() -> None:
    print("AWFT Reward Case Summary")
    print(f"\nTarget Folder:\n{TARGET_FOLDER}")
    print(f"\nSearching under:\n{EXPERIMENT_ROOT}")

    rolling_root = find_target_folder(EXPERIMENT_ROOT, TARGET_FOLDER)
    if rolling_root is None:
        return

    windows = find_windows(rolling_root)
    if not windows:
        print(f"ERROR: No window folders found in {rolling_root}")
        return

    output_root = rolling_root / "awft_reward_value_case_summary"
    output_root.mkdir(parents=True, exist_ok=True)

    all_times_rows: list[dict[str, object]] = []
    all_window_rows: list[dict[str, object]] = []

    for window_dir in windows:
        window_label = short_window_name(window_dir)
        times_dirs = find_times(window_dir)
        print(f"\nProcessing {window_label}: {len(times_dirs)} times")
        if len(times_dirs) != EXPECTED_TIMES_PER_WINDOW:
            print(
                f"WARNING: expected {EXPECTED_TIMES_PER_WINDOW} times, "
                f"found {len(times_dirs)}."
            )

        window_rows: list[dict[str, object]] = []
        for times_dir in times_dirs:
            times_label, _ = times_info(times_dir)
            try:
                rows, unclassified_count = summarize_one_times(
                    times_dir,
                    window_label,
                )
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                print(f"  {times_label}: ERROR: {exc}")
                continue

            window_rows.extend(rows)
            all_times_rows.extend(rows)
            print(f"  {times_label}: OK")
            if unclassified_count:
                print(
                    "    WARNING: "
                    f"{unclassified_count} rows have d == 0 or do not match the 7 requested cases."
                )

        if not window_rows:
            continue

        window_df = pd.DataFrame(window_rows)
        window_output = output_root / window_label
        window_output.mkdir(parents=True, exist_ok=True)
        window_df.to_csv(
            window_output / "AWFT_reward_case_times.csv",
            index=False,
            encoding="utf-8-sig",
        )

        window_average_rows: list[dict[str, object]] = []
        for case_code, case_name, expected_sign in CASE_DEFINITIONS:
            case_rows = window_df[window_df["case"] == case_code]
            means = pd.to_numeric(case_rows["mean_reward"], errors="coerce")
            valid_means = means.dropna()
            average_reward = (
                float(valid_means.mean()) if not valid_means.empty else np.nan
            )
            summary_row = {
                "window": window_label,
                "case": case_code,
                "case_name": case_name,
                "expected_reward_sign": expected_sign,
                "mean_reward_across_times": average_reward,
                "times_count": int(len(case_rows)),
                "times_with_data": int(len(valid_means)),
            }
            window_average_rows.append(summary_row)
            all_window_rows.append(summary_row)

        pd.DataFrame(window_average_rows).to_csv(
            window_output / "AWFT_reward_case_window_average.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if all_times_rows:
        pd.DataFrame(all_times_rows).to_csv(
            output_root / "AWFT_reward_case_all_times.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if all_window_rows:
        pd.DataFrame(all_window_rows).to_csv(
            output_root / "AWFT_reward_case_all_window_average.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\nAWFT reward case summary completed.")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
