from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

#舊分數算法(被打槍)(統計csv)

# =====================================================
# Future Ranking 5-window summary settings
# =====================================================
# 使用方式：
# 1. 可直接在這裡填入 rolling_* 根資料夾完整路徑。
# 2. 或執行程式時把 rolling_* 根資料夾當成第一個參數傳入。
#
# Example:
# ROLLING_ROOT = r"C:\Users\User\Desktop\GraduationProject\Experence_result\reward_v6\rolling_2018-2025_train3y_test1y_shift1y_20260818_003131_reinforce"

ROLLING_ROOT = r"C:\Users\User\Desktop\GraduationProject\Experence_result\reward_v6\rolling_2018-2025_train3y_test1y_shift1y_20260818_193551_reinforce"

INPUT_CSV_NAME = "future_ranking_score_summary.csv"
OUTPUT_CSV_NAME = "future_ranking_score_summary.csv"

METRICS = [
    "Buy Score",
    "Buy Precision",
    "Holding Score",
    "Holding Precision",
    "Sell Score",
    "Sell Precision",
]

WINDOW_PATTERN = re.compile(
    r"window_(\d+)_train_(\d{4}-\d{4})_test_(\d{4}-\d{4})$"
)


def _resolve_rolling_root() -> Path:
    """Resolve rolling root from command line first, then ROLLING_ROOT setting."""
    raw_path = sys.argv[1].strip() if len(sys.argv) >= 2 else ROLLING_ROOT.strip()
    if not raw_path:
        raise ValueError(
            "尚未指定 rolling 根資料夾。請設定 ROLLING_ROOT，"
            "或在執行程式時傳入 rolling_* 資料夾路徑。"
        )

    rolling_root = Path(raw_path).expanduser()
    if not rolling_root.exists() or not rolling_root.is_dir():
        raise FileNotFoundError(f"找不到 rolling 根資料夾：{rolling_root}")
    return rolling_root


def _window_sort_key(path: Path) -> tuple[int, str]:
    match = WINDOW_PATTERN.fullmatch(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def _find_window_dirs(rolling_root: Path) -> list[Path]:
    window_dirs = [
        path
        for path in rolling_root.iterdir()
        if path.is_dir() and WINDOW_PATTERN.fullmatch(path.name)
    ]
    return sorted(window_dirs, key=_window_sort_key)


def _read_window_average(window_dir: Path) -> dict[str, str] | None:
    summary_path = window_dir / INPUT_CSV_NAME
    if not summary_path.exists():
        print(f"Skip: 找不到 {summary_path}")
        return None

    summary = pd.read_csv(summary_path, dtype=str, keep_default_na=False)
    if "Times" not in summary.columns:
        print(f"Skip: {summary_path} 缺少 Times 欄位")
        return None

    missing_metrics = [metric for metric in METRICS if metric not in summary.columns]
    if missing_metrics:
        print(f"Skip: {summary_path} 缺少欄位 {missing_metrics}")
        return None

    times = summary["Times"].astype(str).str.strip().str.lower()
    average_rows = summary.loc[times == "average"]
    if average_rows.empty:
        print(f"Skip: {summary_path} 找不到 Average 列")
        return None

    average_row = average_rows.iloc[0]
    match = WINDOW_PATTERN.fullmatch(window_dir.name)
    if match is None:
        return None

    result = {
        "Window": f"window_{int(match.group(1)):02d}",
        "Train Period": match.group(2),
        "Test Period": match.group(3),
    }
    for metric in METRICS:
        result[metric] = str(average_row[metric]).strip()
    return result


def build_future_ranking_score_summary(rolling_root: Path) -> Path:
    """Collect each Window's Average row into one rolling-level CSV."""
    window_dirs = _find_window_dirs(rolling_root)
    if not window_dirs:
        raise ValueError(f"找不到 window_* 資料夾：{rolling_root}")

    rows: list[dict[str, str]] = []
    for window_dir in window_dirs:
        row = _read_window_average(window_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError("沒有找到可統整的 Future Ranking Average 資料。")

    output_path = rolling_root / OUTPUT_CSV_NAME
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"Future Ranking Window summary saved to: {output_path}")
    print(f"Windows collected: {len(rows)} / {len(window_dirs)}")
    print("=" * 80)
    return output_path


def main() -> None:
    rolling_root = _resolve_rolling_root()
    build_future_ranking_score_summary(rolling_root)


if __name__ == "__main__":
    main()
