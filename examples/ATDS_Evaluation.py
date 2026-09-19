from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# 計算ADTS, output atds_evaluation folder

# =====================================================
# 1. User setting
# =====================================================
# 平常只需要修改 TARGET_FOLDER。
EXPERIMENT_ROOT = Path(
    r"C:\Users\User\Desktop\GraduationProject\Experence_result"
)

TARGET_FOLDER = (
    "rolling_2018-2025_train3y_test1y_shift1y_20260917_010528_reinforce"
    # "rolling_2018-2025_train3y_test1y_shift1y_20260903_000326_reinforce" #v0
)

# ATDS 定義固定使用未來 60 個交易日。
N_FUTURE = 60
PRICE_COL = "close"
EXPECTED_TIMES_PER_WINDOW = 5

WINDOW_PATTERN = re.compile(r"^window_(?P<id>\d+)(?:_|$)")
TIMES_PATTERN = re.compile(r"^times_(?P<id>\d+)_seed_(?P<seed>\d+)$")
FUTURE_PRICE_FILE = "future_ranking_price_data.csv"
ATDS_PRICE_FILE = "atds_price_data.csv"


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


def _download_atds_price_data(
    times_dir: Path,
    action_dates: list[str],
    tickers: list[str],
) -> Path:
    """Fallback for old experiment folders that do not contain enough future prices."""
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

    data_dir = times_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / ATDS_PRICE_FILE

    start_date = action_dates[0]
    # 60 trading days are longer than 60 calendar days. Use a wide calendar
    # buffer, then verify that at least 60 actual trading rows are available.
    buffer_days = max(180, N_FUTURE * 3)
    end_date = (
        pd.Timestamp(action_dates[-1]) + pd.Timedelta(days=buffer_days)
    ).strftime("%Y-%m-%d")

    print("  Existing future price data is insufficient; downloading ATDS data...")
    print(f"  Download range: {start_date} ~ {end_date}")
    raw = YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=tickers,
    ).fetch_data()
    if raw is None or raw.empty:
        raise ValueError("ATDS future price download returned no data.")

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

    for filename in [ATDS_PRICE_FILE, FUTURE_PRICE_FILE]:
        candidate = data_dir / filename
        if not candidate.exists():
            continue
        price_table = _load_price_table(candidate, tickers)
        if _has_required_future_data(price_table, action_dates, tickers):
            return candidate, price_table

    downloaded = _download_atds_price_data(times_dir, action_dates, tickers)
    price_table = _load_price_table(downloaded, tickers)
    if not _has_required_future_data(price_table, action_dates, tickers):
        raise ValueError(
            "ATDS still does not have a complete 60-trading-day future window "
            f"after downloading data for {times_dir}."
        )
    return downloaded, price_table


# =====================================================
# 4. ATDS calculation
# =====================================================
def calculate_runlen(test_prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate same-direction up/down streak length for each stock.

    The first day is runlen = 1. A flat day keeps the previous runlen and
    previous non-flat direction, so runlen never becomes 0.
    """
    values = test_prices.to_numpy(dtype=float)
    if values.size == 0:
        return pd.DataFrame(index=test_prices.index, columns=test_prices.columns)
    if not np.all(np.isfinite(values)):
        raise ValueError("Test price table contains missing/non-finite values.")

    runlen = np.zeros_like(values, dtype=int)
    runlen[0, :] = 1

    for stock_index in range(values.shape[1]):
        previous_direction: int | None = None
        current_run = 1

        for row_index in range(1, values.shape[0]):
            diff = values[row_index, stock_index] - values[row_index - 1, stock_index]
            if diff > 0:
                direction = 1
            elif diff < 0:
                direction = -1
            else:
                direction = 0

            if direction == 0:
                # Flat: keep both the streak length and previous direction.
                pass
            elif previous_direction is None:
                # Match the slide example: the first observed price is runlen=1,
                # and the next same-direction move becomes runlen=2.
                current_run = 2
                previous_direction = direction
            elif direction == previous_direction:
                current_run += 1
            else:
                current_run = 1
                previous_direction = direction

            runlen[row_index, stock_index] = current_run

    return pd.DataFrame(
        runlen,
        index=test_prices.index,
        columns=test_prices.columns,
    )


def calculate_one_times_atds(
    actions: pd.DataFrame,
    price_table: pd.DataFrame,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    tickers = [column for column in actions.columns if column != "date"]
    action_dates = actions["date"].tolist()
    source_dates = list(price_table.index)
    source_date_to_position = {
        date: index for index, date in enumerate(source_dates)
    }

    test_prices = price_table.reindex(index=action_dates, columns=tickers)
    if test_prices.isna().any().any():
        raise ValueError("ATDS test price table contains missing values.")
    runlen = calculate_runlen(test_prices)

    price_values = price_table.reindex(columns=tickers).to_numpy(dtype=float)
    action_values = actions[tickers].to_numpy(dtype=float)
    stock_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, float | str]] = []

    for row_index, date in enumerate(action_dates):
        current_position = source_date_to_position.get(date)
        if current_position is None:
            raise ValueError(f"ATDS price data missing action date: {date}")

        future_start = current_position + 1
        future_end = current_position + N_FUTURE + 1
        future_window = price_values[future_start:future_end]
        if len(future_window) != N_FUTURE:
            raise ValueError(
                f"ATDS needs {N_FUTURE} future trading days after {date}, "
                f"but only {len(future_window)} are available."
            )

        current_prices = price_values[current_position]
        if (
            not np.all(np.isfinite(current_prices))
            or not np.all(np.isfinite(future_window))
            or np.any(current_prices <= 0)
        ):
            raise ValueError(f"Invalid ATDS price values around {date}.")

        avg_future_prices = np.mean(future_window, axis=0)
        d_values = (avg_future_prices - current_prices) / current_prices
        runlen_values = runlen.loc[date, tickers].to_numpy(dtype=float)
        stock_scores = np.sign(d_values) * np.abs(d_values) * 100.0 * runlen_values
        act_values = np.sign(action_values[row_index])
        tds_values = act_values * stock_scores

        daily_tds = float(np.sum(tds_values))
        max_tds = float(np.sum(np.abs(stock_scores)))
        daily_rows.append(
            {
                "date": date,
                "DailyTDS": daily_tds,
                "MaxTDS": max_tds,
            }
        )

        for stock_index, ticker in enumerate(tickers):
            stock_rows.append(
                {
                    "date": date,
                    "tic": ticker,
                    "close": float(current_prices[stock_index]),
                    "d": float(d_values[stock_index]),
                    "runlen": int(runlen_values[stock_index]),
                    "s": float(stock_scores[stock_index]),
                    "executed_action": float(action_values[row_index, stock_index]),
                    "act": int(act_values[stock_index]),
                    "tds": float(tds_values[stock_index]),
                }
            )

    daily = pd.DataFrame(daily_rows)
    denominator = float(daily["MaxTDS"].sum())
    numerator = float(daily["DailyTDS"].sum())
    if denominator <= 0:
        atds = np.nan
    else:
        atds = numerator / denominator * 100.0

    return atds, daily, pd.DataFrame(stock_rows)


# =====================================================
# 5. Output
# =====================================================
def main() -> None:
    print("ATDS Evaluation")
    print(f"\nTarget Folder:\n{TARGET_FOLDER}")
    print(f"\nSearching under:\n{EXPERIMENT_ROOT}")
    print(f"\nATDS n_future = {N_FUTURE} trading days")

    rolling_root = find_target_folder(EXPERIMENT_ROOT, TARGET_FOLDER)
    if rolling_root is None:
        return

    windows = find_windows(rolling_root)
    if not windows:
        print(f"ERROR: No window folders found in {rolling_root}")
        return

    output_root = rolling_root / "atds_evaluation"
    output_root.mkdir(parents=True, exist_ok=True)

    all_times_rows: list[dict[str, object]] = []
    window_average_rows: list[dict[str, object]] = []

    for window_dir in windows:
        window_label = short_window_name(window_dir)
        times_dirs = find_times(window_dir)
        print(f"\nProcessing {window_label}: {len(times_dirs)} times")
        if len(times_dirs) != EXPECTED_TIMES_PER_WINDOW:
            print(
                f"WARNING: expected {EXPECTED_TIMES_PER_WINDOW} times, "
                f"found {len(times_dirs)}."
            )

        window_output = output_root / window_label
        window_output.mkdir(parents=True, exist_ok=True)
        window_rows: list[dict[str, object]] = []

        for times_dir in times_dirs:
            times_label, seed = times_info(times_dir)
            print(f"  {times_label} (seed={seed})")

            try:
                action_path = _latest_action_csv(times_dir)
                actions = _load_actions(action_path)
                price_path, price_table = _resolve_price_source(times_dir, actions)
                atds, daily, stock_detail = calculate_one_times_atds(
                    actions,
                    price_table,
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"  ERROR: {exc}")
                continue

            daily_path = window_output / f"{times_label}_ATDS_daily.csv"
            stock_path = window_output / f"{times_label}_TDS_stock_detail.csv"
            daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
            stock_detail.to_csv(stock_path, index=False, encoding="utf-8-sig")

            row = {
                "window": window_label,
                "times": times_label,
                "seed": seed,
                "ATDS": atds,
                "total_DailyTDS": float(daily["DailyTDS"].sum()),
                "total_MaxTDS": float(daily["MaxTDS"].sum()),
                "trading_days": len(daily),
                "n_future": N_FUTURE,
                "action_source": str(action_path),
                "price_source": str(price_path),
            }
            window_rows.append(row)
            all_times_rows.append(row)
            print(f"  ATDS = {atds:.6f}")

        if not window_rows:
            print(f"WARNING: no valid ATDS result for {window_label}")
            continue

        window_df = pd.DataFrame(window_rows)
        average_atds = float(pd.to_numeric(window_df["ATDS"], errors="coerce").mean())
        summary_df = window_df[
            ["times", "seed", "ATDS", "total_DailyTDS", "total_MaxTDS", "trading_days"]
        ].copy()
        summary_df.loc[len(summary_df)] = {
            "times": "Average",
            "seed": np.nan,
            "ATDS": average_atds,
            "total_DailyTDS": np.nan,
            "total_MaxTDS": np.nan,
            "trading_days": np.nan,
        }
        summary_df.to_csv(
            window_output / "ATDS_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        window_average_rows.append(
            {
                "window": window_label,
                "average_ATDS": average_atds,
                "times_count": len(window_rows),
            }
        )

    if all_times_rows:
        pd.DataFrame(all_times_rows).to_csv(
            output_root / "ATDS_all_times.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if window_average_rows:
        pd.DataFrame(window_average_rows).to_csv(
            output_root / "ATDS_window_average.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\nATDS evaluation completed.")
    print(f"Output: {output_root}")
    print(f"Valid times ATDS count: {len(all_times_rows)}")


if __name__ == "__main__":
    main()
