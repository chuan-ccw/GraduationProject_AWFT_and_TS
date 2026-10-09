from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from finrl.meta.env_stock_trading.trading_score import (
    calculate_annual_ts_versions,
    calculate_cumulative_ts_versions,
)

# 計算 Annual TS / Cumulative TS，輸出至 ts_evaluation folder

# =====================================================
# 1. User setting
# =====================================================
# 平常只需要修改 TARGET_FOLDER。
EXPERIMENT_ROOT = Path(
    r"C:\Users\User\Desktop\GraduationProject\Experence_result"
)

TARGET_FOLDER = (
    "rolling_2018-2025_train3y_test1y_shift1y_20260912_000936_reinforce"
    # "rolling_2018-2025_train3y_test1y_shift1y_20260903_000326_reinforce" #v0
)

# TS 定義固定使用未來 60 個交易日。
N_FUTURE = 60
PRICE_COL = "close"
EXPECTED_TIMES_PER_WINDOW = 5

WINDOW_PATTERN = re.compile(r"^window_(?P<id>\d+)(?:_|$)")
TIMES_PATTERN = re.compile(r"^times_(?P<id>\d+)_seed_(?P<seed>\d+)$")
FUTURE_PRICE_FILE = "future_ranking_price_data.csv"
TS_PRICE_FILE = "ts_price_data.csv"
LEGACY_ATDS_PRICE_FILE = "atds_price_data.csv"


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
        print(f"\nSearch Root:\n{search_root}")
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
# 3. CSV / price helpers
# =====================================================
def _normalize_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str).str.strip(), errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def _latest_action_csv(times_dir: Path) -> Path:
    candidates = [path for path in times_dir.glob("actions_*.csv") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No executed action CSV found in {times_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_actions(path: Path) -> pd.DataFrame:
    actions = pd.read_csv(path)
    if actions.empty:
        raise ValueError(f"Action CSV is empty: {path}")

    if "date" not in actions.columns:
        first_column = actions.columns[0]
        actions = actions.rename(columns={first_column: "date"})

    actions["date"] = _normalize_date(actions["date"])
    actions = actions[actions["date"].notna()].copy()
    actions = actions.drop_duplicates(subset=["date"], keep="last")
    actions = actions.sort_values("date").reset_index(drop=True)

    stock_columns = [
        column
        for column in actions.columns
        if column != "date" and not str(column).startswith("Unnamed:")
    ]
    if not stock_columns:
        raise ValueError(f"Action CSV has no stock columns: {path}")

    for column in stock_columns:
        actions[column] = pd.to_numeric(actions[column], errors="coerce")
    if actions[stock_columns].isna().any().any():
        raise ValueError(f"Action CSV contains non-numeric action values: {path}")

    return actions[["date", *stock_columns]]


def _load_price_table(path: Path, tickers: list[str]) -> pd.DataFrame:
    data = pd.read_csv(path)
    if data.empty or "date" not in data.columns or "tic" not in data.columns:
        raise ValueError(f"Invalid price data CSV: {path}")

    actual_price_col = PRICE_COL
    if actual_price_col not in data.columns and "close" in data.columns:
        actual_price_col = "close"
    if actual_price_col not in data.columns:
        raise ValueError(f"Price column {PRICE_COL!r} not found in {path}")

    data = data[["date", "tic", actual_price_col]].copy()
    data["date"] = _normalize_date(data["date"])
    data = data[data["date"].notna()].copy()
    data["tic"] = data["tic"].astype(str)
    data[actual_price_col] = pd.to_numeric(data[actual_price_col], errors="coerce")
    data = data.sort_values(["date", "tic"])

    price_table = data.pivot_table(
        index="date",
        columns="tic",
        values=actual_price_col,
        aggfunc="last",
    ).sort_index()
    return price_table.reindex(columns=tickers)


def _has_required_future_data(
    price_table: pd.DataFrame,
    action_dates: list[str],
    tickers: list[str],
) -> bool:
    if price_table.empty or not action_dates:
        return False
    if any(ticker not in price_table.columns for ticker in tickers):
        return False

    date_to_position = {date: index for index, date in enumerate(price_table.index)}
    last_action_date = action_dates[-1]
    last_position = date_to_position.get(last_action_date)
    if last_position is None:
        return False
    if last_position + N_FUTURE >= len(price_table):
        return False

    required_dates = action_dates
    if any(date not in date_to_position for date in required_dates):
        return False

    selected = price_table.reindex(index=required_dates, columns=tickers)
    if selected.isna().any().any():
        return False

    future_block = price_table.iloc[
        last_position + 1 : last_position + N_FUTURE + 1
    ].reindex(columns=tickers)
    return len(future_block) == N_FUTURE and not future_block.isna().any().any()


def _download_ts_price_data(
    times_dir: Path,
    action_dates: list[str],
    tickers: list[str],
) -> Path:
    """Fallback for old experiment folders that do not contain enough future prices."""
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

    data_dir = times_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / TS_PRICE_FILE

    start_date = action_dates[0]
    # 60 trading days are longer than 60 calendar days. Use a wide calendar
    # buffer, then verify that at least 60 actual trading rows are available.
    buffer_days = max(180, N_FUTURE * 3)
    end_date = (
        pd.Timestamp(action_dates[-1]) + pd.Timedelta(days=buffer_days)
    ).strftime("%Y-%m-%d")

    print("  Existing future price data is insufficient; downloading TS data...")
    print(f"  Download range: {start_date} ~ {end_date}")
    raw = YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=tickers,
    ).fetch_data()
    if raw is None or raw.empty:
        raise ValueError("TS future price download returned no data.")

    actual_price_col = PRICE_COL if PRICE_COL in raw.columns else "close"
    raw[["date", "tic", actual_price_col]].to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
    return output_path


def _resolve_price_source(
    times_dir: Path,
    actions: pd.DataFrame,
) -> tuple[Path, pd.DataFrame]:
    tickers = [column for column in actions.columns if column != "date"]
    action_dates = actions["date"].tolist()
    data_dir = times_dir / "data"

    for filename in [TS_PRICE_FILE, LEGACY_ATDS_PRICE_FILE, FUTURE_PRICE_FILE]:
        candidate = data_dir / filename
        if not candidate.exists():
            continue
        price_table = _load_price_table(candidate, tickers)
        if _has_required_future_data(price_table, action_dates, tickers):
            return candidate, price_table

    downloaded = _download_ts_price_data(times_dir, action_dates, tickers)
    price_table = _load_price_table(downloaded, tickers)
    if not _has_required_future_data(price_table, action_dates, tickers):
        raise ValueError(
            "TS still does not have a complete 60-trading-day future window "
            f"after downloading data for {times_dir}."
        )
    return downloaded, price_table


# =====================================================
# 4. Unified TS calculation
# =====================================================
def calculate_one_times_annual_ts(
    actions: pd.DataFrame,
    price_table: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    tickers = [column for column in actions.columns if column != "date"]
    action_values = actions.set_index("date")[tickers]
    return calculate_annual_ts_versions(
        action_values=action_values,
        price_pivot=price_table.reindex(columns=tickers),
        n_future=N_FUTURE,
    )


def _continuous_action_files(times_dir: Path) -> list[tuple[int, Path]]:
    pattern = re.compile(r"^actions_window_(?P<id>\d+)_.*\.csv$", re.IGNORECASE)
    rows: list[tuple[int, Path]] = []
    for path in times_dir.glob("actions_window_*.csv"):
        match = pattern.fullmatch(path.name)
        if match is not None:
            rows.append((int(match.group("id")), path))
    return sorted(rows, key=lambda item: item[0])


def _merge_price_tables(price_tables: list[pd.DataFrame]) -> pd.DataFrame:
    if not price_tables:
        raise ValueError("No price tables were provided for Cumulative TS.")

    combined = price_tables[0].copy()
    for table in price_tables[1:]:
        all_columns = list(dict.fromkeys([*combined.columns, *table.columns]))
        combined = combined.reindex(columns=all_columns)
        table = table.reindex(columns=all_columns)

        overlap = combined.index.intersection(table.index)
        if len(overlap) > 0:
            left = combined.loc[overlap]
            right = table.loc[overlap]
            both = left.notna() & right.notna()
            mismatch = both & ~np.isclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=1e-5,
                atol=1e-5,
                equal_nan=True,
            )
            mismatch = pd.DataFrame(mismatch, index=left.index, columns=left.columns)
            if mismatch.any().any():
                raise ValueError(
                    "Overlapping continuous TS price sources contain different prices."
                )

        combined = combined.combine_first(table)

    return combined.sort_index()


def calculate_one_times_cumulative_ts(
    continuous_times_dir: Path,
    windows_by_id: dict[int, Path],
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, list[str]]:
    action_files = _continuous_action_files(continuous_times_dir)
    if not action_files:
        raise FileNotFoundError(
            f"No continuous actions_window_*.csv found in {continuous_times_dir}"
        )

    action_frames: list[pd.DataFrame] = []
    price_tables: list[pd.DataFrame] = []
    price_sources: list[str] = []
    expected_tickers: list[str] | None = None

    for window_id, action_path in action_files:
        window_dir = windows_by_id.get(window_id)
        if window_dir is None:
            raise ValueError(
                f"Continuous action file refers to missing window_{window_id:02d}."
            )

        original_times_dir = window_dir / continuous_times_dir.name
        if not original_times_dir.is_dir():
            raise FileNotFoundError(
                f"Cannot find matching original times folder: {original_times_dir}"
            )

        actions = _load_actions(action_path)
        tickers = [column for column in actions.columns if column != "date"]
        if expected_tickers is None:
            expected_tickers = tickers
        elif set(tickers) != set(expected_tickers):
            raise ValueError(
                "Continuous five-year windows use different stock universes; "
                "Cumulative TS requires a consistent stock universe."
            )
        actions = actions[["date", *expected_tickers]]

        price_path, price_table = _resolve_price_source(original_times_dir, actions)
        action_frames.append(actions)
        price_tables.append(price_table.reindex(columns=expected_tickers))
        price_sources.append(str(price_path))

    combined_actions = pd.concat(action_frames, ignore_index=True)
    combined_actions = combined_actions.drop_duplicates(subset=["date"], keep="last")
    combined_actions = combined_actions.sort_values("date").reset_index(drop=True)

    combined_prices = _merge_price_tables(price_tables)
    action_values = combined_actions.set_index("date")[expected_tickers]
    scores, daily, stock_detail = calculate_cumulative_ts_versions(
        action_values=action_values,
        price_pivot=combined_prices.reindex(columns=expected_tickers),
        n_future=N_FUTURE,
    )
    return scores, daily, stock_detail, price_sources


# =====================================================
# 5. Output
# =====================================================
def main() -> None:
    print("Trading Score Evaluation")
    print(f"\nTarget Folder:\n{TARGET_FOLDER}")
    print(f"\nSearching under:\n{EXPERIMENT_ROOT}")
    print(f"\nTS n_future = {N_FUTURE} trading days")
    print("TS versions: v1 = runlen, v2 = 1/runlen")

    rolling_root = find_target_folder(EXPERIMENT_ROOT, TARGET_FOLDER)
    if rolling_root is None:
        return

    windows = find_windows(rolling_root)
    if not windows:
        print(f"ERROR: No window folders found in {rolling_root}")
        return

    output_root = rolling_root / "ts_evaluation"
    output_root.mkdir(parents=True, exist_ok=True)

    all_times_rows: list[dict[str, object]] = []
    window_average_rows: list[dict[str, object]] = []
    windows_by_id: dict[int, Path] = {}

    for window_dir in windows:
        match = WINDOW_PATTERN.match(window_dir.name)
        if match is not None:
            windows_by_id[int(match.group("id"))] = window_dir

        window_label = short_window_name(window_dir)
        times_dirs = find_times(window_dir)
        print(f"\nProcessing {window_label}: {len(times_dirs)} times")
        if len(times_dirs) != EXPECTED_TIMES_PER_WINDOW:
            print(
                f"WARNING: expected {EXPECTED_TIMES_PER_WINDOW} times, "
                f"found {len(times_dirs)}."
            )

        window_output = output_root / "annual" / window_label
        window_output.mkdir(parents=True, exist_ok=True)
        window_rows: list[dict[str, object]] = []

        for times_dir in times_dirs:
            times_label, seed = times_info(times_dir)
            print(f"  {times_label} (seed={seed})")

            try:
                action_path = _latest_action_csv(times_dir)
                actions = _load_actions(action_path)
                price_path, price_table = _resolve_price_source(times_dir, actions)
                scores, daily, stock_detail = calculate_one_times_annual_ts(
                    actions,
                    price_table,
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"  ERROR: {exc}")
                continue

            daily_path = window_output / f"{times_label}_annual_ts_daily.csv"
            stock_path = window_output / f"{times_label}_ts_stock_detail.csv"
            daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
            stock_detail.to_csv(stock_path, index=False, encoding="utf-8-sig")

            row = {
                "window": window_label,
                "times": times_label,
                "seed": seed,
                "Annual_TS": scores["Annual_TS"],
                "Annual_TS_v2": scores["Annual_TS_v2"],
                "total_Daily_TS": float(daily["Daily_TS"].sum()),
                "total_Max_TS": float(daily["Max_TS"].sum()),
                "total_Daily_TS_v2": float(daily["Daily_TS_v2"].sum()),
                "total_Max_TS_v2": float(daily["Max_TS_v2"].sum()),
                "trading_days": len(daily),
                "n_future": N_FUTURE,
                "action_source": str(action_path),
                "price_source": str(price_path),
            }
            window_rows.append(row)
            all_times_rows.append(row)
            print(
                f"  Annual TS = {scores['Annual_TS']:.6f} | "
                f"Annual TS_v2 = {scores['Annual_TS_v2']:.6f}"
            )

        if not window_rows:
            print(f"WARNING: no valid Annual TS result for {window_label}")
            continue

        window_df = pd.DataFrame(window_rows)
        average_ts = float(pd.to_numeric(window_df["Annual_TS"], errors="coerce").mean())
        average_ts_v2 = float(
            pd.to_numeric(window_df["Annual_TS_v2"], errors="coerce").mean()
        )
        summary_df = window_df[
            [
                "times",
                "seed",
                "Annual_TS",
                "Annual_TS_v2",
                "total_Daily_TS",
                "total_Max_TS",
                "total_Daily_TS_v2",
                "total_Max_TS_v2",
                "trading_days",
            ]
        ].copy()
        summary_df.loc[len(summary_df)] = {
            "times": "Average",
            "seed": np.nan,
            "Annual_TS": average_ts,
            "Annual_TS_v2": average_ts_v2,
            "total_Daily_TS": np.nan,
            "total_Max_TS": np.nan,
            "total_Daily_TS_v2": np.nan,
            "total_Max_TS_v2": np.nan,
            "trading_days": np.nan,
        }
        summary_df.to_csv(
            window_output / "annual_ts_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        window_average_rows.append(
            {
                "window": window_label,
                "average_Annual_TS": average_ts,
                "average_Annual_TS_v2": average_ts_v2,
                "times_count": len(window_rows),
            }
        )

    if all_times_rows:
        pd.DataFrame(all_times_rows).to_csv(
            output_root / "annual_ts_all_times.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if window_average_rows:
        pd.DataFrame(window_average_rows).to_csv(
            output_root / "annual_ts_window_average.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # Continuous five-year experiment: calculate one Cumulative TS per times run.
    continuous_root = rolling_root / "continuous_5y_test"
    cumulative_rows: list[dict[str, object]] = []
    if continuous_root.is_dir():
        cumulative_output = output_root / "cumulative_5y"
        cumulative_output.mkdir(parents=True, exist_ok=True)
        continuous_times_dirs = find_times(continuous_root)
        print(f"\nProcessing continuous 5Y: {len(continuous_times_dirs)} times")

        for continuous_times_dir in continuous_times_dirs:
            times_label, seed = times_info(continuous_times_dir)
            print(f"  {times_label} (seed={seed})")
            try:
                scores, daily, stock_detail, price_sources = (
                    calculate_one_times_cumulative_ts(
                        continuous_times_dir=continuous_times_dir,
                        windows_by_id=windows_by_id,
                    )
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"  ERROR: {exc}")
                continue

            times_output = cumulative_output / times_label
            times_output.mkdir(parents=True, exist_ok=True)
            daily.to_csv(
                times_output / "cumulative_ts_daily.csv",
                index=False,
                encoding="utf-8-sig",
            )
            stock_detail.to_csv(
                times_output / "cumulative_ts_stock_detail.csv",
                index=False,
                encoding="utf-8-sig",
            )

            row = {
                "times": times_label,
                "seed": seed,
                "Cumulative_TS": scores["Cumulative_TS"],
                "Cumulative_TS_v2": scores["Cumulative_TS_v2"],
                "total_Daily_TS": float(daily["Daily_TS"].sum()),
                "total_Max_TS": float(daily["Max_TS"].sum()),
                "total_Daily_TS_v2": float(daily["Daily_TS_v2"].sum()),
                "total_Max_TS_v2": float(daily["Max_TS_v2"].sum()),
                "trading_days": len(daily),
                "n_future": N_FUTURE,
                "continuous_action_dir": str(continuous_times_dir),
                "price_sources": " | ".join(price_sources),
            }
            cumulative_rows.append(row)
            print(
                f"  Cumulative TS = {scores['Cumulative_TS']:.6f} | "
                f"Cumulative TS_v2 = {scores['Cumulative_TS_v2']:.6f}"
            )

        if cumulative_rows:
            cumulative_df = pd.DataFrame(cumulative_rows)
            cumulative_df.to_csv(
                cumulative_output / "cumulative_ts_all_times.csv",
                index=False,
                encoding="utf-8-sig",
            )
            summary = pd.DataFrame(
                [
                    {
                        "times": "Average",
                        "Cumulative_TS": float(cumulative_df["Cumulative_TS"].mean()),
                        "Cumulative_TS_v2": float(
                            cumulative_df["Cumulative_TS_v2"].mean()
                        ),
                        "times_count": len(cumulative_df),
                    }
                ]
            )
            summary.to_csv(
                cumulative_output / "cumulative_ts_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )
    else:
        print("\nContinuous 5Y folder not found; Cumulative TS skipped.")

    print("\nTS evaluation completed.")
    print(f"Output: {output_root}")
    print(f"Valid Annual TS times count: {len(all_times_rows)}")
    print(f"Valid Cumulative TS times count: {len(cumulative_rows)}")


if __name__ == "__main__":
    main()
