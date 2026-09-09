from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# 從
# reward_by_step_reinforce_v0_*.csv
# dwft_parameters_reinforce_v6_*.csv
# reward_by_step_reinforce_v6_*.csv
# awft_parameters_reinforce_v7_*.csv
# reward_by_step_reinforce_v7_*.csv
# 蒐集資料統整至reward_data_collection資料夾

# =====================================================
# 1. User setting
# =====================================================
# 平常只需要修改 TARGET_FOLDER。
EXPERIMENT_ROOT = Path(
    r"C:\Users\User\Desktop\GraduationProject\Experence_result"
)

TARGET_FOLDER = (
    "rolling_2018-2025_train3y_test1y_shift1y_20260904_234026_reinforce"
)


# =====================================================
# 2. Reward data source settings
# =====================================================
# 只蒐集本次明確指定的 Reward Parameter，不自動加入其他欄位。
REWARD_DATA_SOURCES = {
    "v0": {
        "reward_by_step_reinforce_v0_*.csv": {
            "reward": "step_reward_unscaled",
        },
    },
    "v6": {
        "dwft_parameters_reinforce_v6_*.csv": {
            "w": "w",
            "delta_w": "delta_w",
            "d": "d",
        },
        "reward_by_step_reinforce_v6_*.csv": {
            "reward": "step_reward_unscaled",
        },
    },
    "v7": {
        "awft_parameters_reinforce_v7_*.csv": {
            "a": "a",
            "w": "w",
            "d": "d",
        },
        "reward_by_step_reinforce_v7_*.csv": {
            "reward": "step_reward_unscaled",
        },
    },
}

WINDOW_PATTERN = re.compile(r"^window_(?P<id>\d+)(?:_|$)")
TIMES_PATTERN = re.compile(r"^times_(?P<id>\d+)_seed_.+$")
REWARD_PARENT_PATTERN = re.compile(r"^reward_(?P<version>v\d+)$", re.IGNORECASE)
SOURCE_VERSION_PATTERN = re.compile(
    r"^(?:reward_by_step|dwft_parameters|awft_parameters)_reinforce_(?P<version>v\d+)_.*\.csv$",
    re.IGNORECASE,
)


# =====================================================
# 3. Folder helpers
# =====================================================
def find_target_folder(search_root: Path, target_folder: str) -> Path | None:
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
        print(f"\nSearch Root:\n{search_root}")
        return None

    if len(matches) > 1:
        print("ERROR: Multiple matching rolling experiment folders found.")
        print(f"\nTarget:\n{target_folder}")
        print("\nMatches:")
        for index, path in enumerate(matches, start=1):
            print(f"{index}. {path}")
        return None

    return matches[0]


def detect_reward_version(rolling_root: Path) -> str | None:
    parent_match = REWARD_PARENT_PATTERN.fullmatch(rolling_root.parent.name)
    if parent_match is None:
        print("ERROR: Unable to determine Reward Version from parent folder.")
        print(f"  Rolling Folder: {rolling_root}")
        print(f"  Parent Folder: {rolling_root.parent.name}")
        print("  Expected Parent: reward_v0, reward_v6, or reward_v7")
        return None

    reward_version = parent_match.group("version").lower()
    if reward_version not in REWARD_DATA_SOURCES:
        print("ERROR: Unsupported Reward Version.")
        print(f"  Reward Version: {reward_version}")
        print(f"  Supported: {', '.join(REWARD_DATA_SOURCES)}")
        return None

    # 用實際存在的 Reward Data CSV 檔名再次驗證父資料夾版本。
    csv_versions: set[str] = set()
    version_files: list[Path] = []
    for csv_path in rolling_root.rglob("*.csv"):
        match = SOURCE_VERSION_PATTERN.fullmatch(csv_path.name)
        if match is None:
            continue
        csv_versions.add(match.group("version").lower())
        version_files.append(csv_path)

    inconsistent_versions = sorted(version for version in csv_versions if version != reward_version)
    if inconsistent_versions:
        print("ERROR: Reward Version does not match existing Reward Data CSV filenames.")
        print(f"  Parent Version: {reward_version}")
        print(f"  CSV Versions: {', '.join(sorted(csv_versions))}")
        print("  Matching Reward Data Files:")
        for path in sorted(version_files, key=lambda item: str(item).lower()):
            print(f"    {path}")
        return None

    return reward_version


def _numeric_folder_sort_key(path: Path, pattern: re.Pattern[str]) -> tuple[int, str]:
    match = pattern.match(path.name)
    if match is None:
        return 10**9, path.name.lower()
    return int(match.group("id")), path.name.lower()


def find_windows(rolling_root: Path) -> list[Path]:
    windows = [
        path
        for path in rolling_root.iterdir()
        if path.is_dir() and WINDOW_PATTERN.match(path.name)
    ]
    return sorted(windows, key=lambda path: _numeric_folder_sort_key(path, WINDOW_PATTERN))


def find_times(window_dir: Path) -> list[Path]:
    times_dirs = [
        path
        for path in window_dir.iterdir()
        if path.is_dir() and TIMES_PATTERN.fullmatch(path.name)
    ]
    return sorted(times_dirs, key=lambda path: _numeric_folder_sort_key(path, TIMES_PATTERN))


def window_id(window_dir: Path) -> str:
    match = WINDOW_PATTERN.match(window_dir.name)
    if match is None:
        raise ValueError(f"Invalid window folder name: {window_dir.name}")
    return f"window_{int(match.group('id')):02d}"


def times_id(times_dir: Path) -> str:
    match = TIMES_PATTERN.fullmatch(times_dir.name)
    if match is None:
        raise ValueError(f"Invalid times folder name: {times_dir.name}")
    return f"times_{int(match.group('id')):02d}"


# =====================================================
# 4. CSV helpers
# =====================================================
def find_single_matching_csv(
    window_dir: Path,
    times_dir: Path,
    pattern: str,
) -> Path | None:
    matches = sorted(
        [path for path in times_dir.glob(pattern) if path.is_file()],
        key=lambda path: path.name.lower(),
    )

    if not matches:
        print("WARNING: Reward data file missing.")
        print(f"  Window: {window_dir.name}")
        print(f"  Times: {times_dir.name}")
        print(f"  Expected: {pattern}")
        return None

    if len(matches) > 1:
        print("WARNING: Multiple matching reward data files found.")
        print(f"  Window: {window_dir.name}")
        print(f"  Times: {times_dir.name}")
        print(f"  Pattern: {pattern}")
        print("  Files:")
        for path in matches:
            print(f"    {path.name}")
        return None

    return matches[0]


def _print_parameter_missing(csv_path: Path, parameter: str) -> None:
    print("WARNING: Reward parameter column missing.")
    print(f"  File: {csv_path}")
    print(f"  Parameter: {parameter}")


def _print_parameter_no_data(csv_path: Path, parameter: str) -> None:
    print("WARNING: No reward parameter data found.")
    print(f"  File: {csv_path}")
    print(f"  Parameter: {parameter}")


def collect_window_data(
    window_dir: Path,
    source_config: dict[str, dict[str, str]],
) -> tuple[dict[str, list[pd.DataFrame]], dict[str, int]]:
    parameters = [
        output_parameter
        for parameter_mapping in source_config.values()
        for output_parameter in parameter_mapping
    ]
    collected = {parameter: [] for parameter in parameters}
    source_counts = {parameter: 0 for parameter in parameters}

    times_dirs = find_times(window_dir)
    if not times_dirs:
        print("WARNING: No Times folders found.")
        print(f"  Window: {window_dir.name}")
        return collected, source_counts

    for times_dir in times_dirs:
        label = times_id(times_dir)
        had_warning = False

        for pattern, parameter_mapping in source_config.items():
            csv_path = find_single_matching_csv(window_dir, times_dir, pattern)
            if csv_path is None:
                had_warning = True
                continue

            try:
                data = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                had_warning = True
                for output_parameter in parameter_mapping:
                    _print_parameter_no_data(csv_path, output_parameter)
                continue
            except Exception as exc:
                had_warning = True
                print("WARNING: Failed to read reward data CSV.")
                print(f"  File: {csv_path}")
                print(f"  Error: {exc}")
                continue

            for output_parameter, source_column in parameter_mapping.items():
                if source_column not in data.columns:
                    had_warning = True
                    _print_parameter_missing(csv_path, source_column)
                    continue

                if data.empty:
                    had_warning = True
                    _print_parameter_no_data(csv_path, output_parameter)
                    continue

                values = data[source_column].copy()
                if not values.notna().any():
                    # 原始欄位全部是 NaN 時仍保留這些列，不自行 fill 或刪除。
                    had_warning = True
                    _print_parameter_no_data(csv_path, output_parameter)

                frame = pd.DataFrame(
                    {
                        "times": [label] * len(values),
                        "value": values.to_numpy(copy=True),
                    }
                )
                collected[output_parameter].append(frame)
                source_counts[output_parameter] += len(frame)

        status = "completed with warning(s)" if had_warning else "OK"
        print(f"  {label}: {status}")

    return collected, source_counts


def save_parameter_csv(
    output_window_dir: Path,
    window_label: str,
    parameter: str,
    frames: list[pd.DataFrame],
    expected_count: int,
) -> bool:
    if not frames:
        print("WARNING: No collected reward parameter data to save.")
        print(f"  Window: {window_label}")
        print(f"  Parameter: {parameter}")
        return False

    output = pd.concat(frames, ignore_index=True)

    # 先檢查記憶體中的合併筆數，再寫檔。
    if len(output) != expected_count:
        print("ERROR: Reward data count mismatch before saving.")
        print(f"  Window: {window_label}")
        print(f"  Parameter: {parameter}")
        print(f"  Expected: {expected_count}")
        print(f"  Collected: {len(output)}")
        return False

    output_window_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_window_dir / f"{parameter}.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    try:
        saved_count = len(pd.read_csv(output_path))
    except Exception as exc:
        print("ERROR: Failed to verify saved reward data CSV.")
        print(f"  File: {output_path}")
        print(f"  Error: {exc}")
        return False

    if saved_count != expected_count:
        print("ERROR: Reward data count mismatch.")
        print(f"  Window: {window_label}")
        print(f"  Parameter: {parameter}")
        print(f"  Expected: {expected_count}")
        print(f"  Saved: {saved_count}")
        return False

    print("\nSaved:")
    print(f"  {output_path}")
    print(f"  Rows: {saved_count}")
    return True


# =====================================================
# 5. Main
# =====================================================
def main() -> None:
    print("Reward Data Collection")
    print(f"\nTarget Folder:\n{TARGET_FOLDER}")
    print(f"\nSearching under:\n{EXPERIMENT_ROOT}")

    rolling_root = find_target_folder(EXPERIMENT_ROOT, TARGET_FOLDER)
    if rolling_root is None:
        return

    print(f"\nFound:\n{rolling_root}")

    reward_version = detect_reward_version(rolling_root)
    if reward_version is None:
        return

    print(f"\nReward Version: {reward_version}")

    windows = find_windows(rolling_root)
    if not windows:
        print("ERROR: No Window folders found.")
        print(f"  Rolling Folder: {rolling_root}")
        return

    output_root = rolling_root / "reward_data_collection"
    source_config = REWARD_DATA_SOURCES[reward_version]

    for window_dir in windows:
        short_window = window_id(window_dir)
        print(f"\nProcessing {short_window}...")

        collected, source_counts = collect_window_data(window_dir, source_config)
        output_window_dir = output_root / short_window

        for parameter in collected:
            save_parameter_csv(
                output_window_dir=output_window_dir,
                window_label=short_window,
                parameter=parameter,
                frames=collected[parameter],
                expected_count=source_counts[parameter],
            )

    print("\nReward data collection completed.")
    print(f"\nOutput:\n{output_root}")


if __name__ == "__main__":
    main()
