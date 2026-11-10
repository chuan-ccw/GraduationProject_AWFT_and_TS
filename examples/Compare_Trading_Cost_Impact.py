#!/usr/bin/env python3
"""Compare actual net performance with fees added back for the same executed trades.

The estimated no-fee curve is NOT a re-simulation of a zero-fee agent.
It preserves the already executed quantities and adds back their actual fees.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 計算手續費 交易次數 (!注意! 路徑會有問題)

# =====================================================
# 1. User setting
# =====================================================
# 平常只需要修改 TARGET_FOLDER（與 TS_Evaluation.py 相同）。
# 可填 rolling_* 資料夾名稱、完整路徑，或 ZIP 檔案路徑。
EXPERIMENT_ROOT = Path(
    os.environ.get(
        "EXPERIMENT_ROOT",
        r"C:\Users\User\Desktop\GraduationProject\Experence_result",
    )
)
TARGET_FOLDER = (
    "rolling_2018-2025_train3y_test1y_shift1y_20260912_000936_reinforce"
)

# 必須與該批實驗回測時的 buy_cost_list / sell_cost_list 一致。
FEE_RATE = 0.001

WINDOW_PATTERN = re.compile(
    r"(?P<folder>window_(?P<window>\d+)_train_[^/]+)"
    r"/(?P<times_dir>times_(?P<times>\d+)_seed_(?P<seed>\d+))"
    r"/(?P<filename>actions_(?!window_)[^/]+\.csv)$"
)
TEST_YEAR_PATTERN = re.compile(r"_test_(\d{4})-(\d{4})$")


def resolve_source(value: str) -> Path:
    """Accept a folder name under EXPERIMENT_ROOT, a full path, or a ZIP."""
    name = str(value).strip().strip('"').strip("'")
    if not name:
        raise ValueError("TARGET_FOLDER is empty")
    direct = Path(name).expanduser()
    if direct.is_dir() or (direct.is_file() and direct.suffix.lower() == ".zip"):
        return direct.resolve()
    choices = [EXPERIMENT_ROOT / name]
    if not name.lower().endswith(".zip"):
        choices.append(EXPERIMENT_ROOT / f"{name}.zip")
    for path in choices:
        if path.is_dir() or (path.is_file() and path.suffix.lower() == ".zip"):
            return path.resolve()
    if not EXPERIMENT_ROOT.is_dir():
        raise FileNotFoundError(
            f"Experiment root not found: {EXPERIMENT_ROOT}\n"
            "Set EXPERIMENT_ROOT or provide a full folder/ZIP path."
        )
    matches = [
        p.resolve() for p in EXPERIMENT_ROOT.rglob(name)
        if p.is_dir() or (p.is_file() and p.suffix.lower() == ".zip")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one matching experiment source, found {len(matches)}: "
            f"{name} under {EXPERIMENT_ROOT}\n{matches}"
        )
    return matches[0]


def get_year(folder: str, dates: pd.Series) -> str:
    matched = TEST_YEAR_PATTERN.search(folder)
    if matched:
        start, end = matched.groups()
        return start if start == end else f"{start}-{end}"
    years = dates.astype(str).str[:4].unique()
    return "-".join([years[0], years[-1]]) if len(years) > 1 else years[0]



class ResultSource:
    """Read experiment CSVs from either an extracted directory or its ZIP."""

    def __init__(self, source: Path):
        self.source = source
        self.zip_file = zipfile.ZipFile(source) if source.is_file() else None
        if self.zip_file:
            names = [n for n in self.zip_file.namelist() if not n.endswith("/")]
            # Uploaded rolling experiment ZIPs normally have one wrapping folder.
            roots = {n.split("/", 1)[0] for n in names}
            root = next(iter(roots)) if len(roots) == 1 else None
            self.entries = {
                (n[len(root) + 1:] if root and n.startswith(root + "/") else n): n
                for n in names
            }
        else:
            if not source.is_dir():
                raise FileNotFoundError(source)
            self.entries = {
                p.relative_to(source).as_posix(): p
                for p in source.rglob("*")
                if p.is_file()
            }

    def csv(self, relpath: str, **kwargs) -> pd.DataFrame:
        entry = self.entries[relpath]
        if self.zip_file:
            return pd.read_csv(io.BytesIO(self.zip_file.read(entry)), **kwargs)
        return pd.read_csv(entry, **kwargs)

    def match(self, prefix: str, filename_prefix: str) -> str:
        matches = [
            name for name in self.entries
            if name.startswith(prefix + "/")
            and name.rsplit("/", 1)[-1].startswith(filename_prefix)
            and name.endswith(".csv")
            and name.count("/") == prefix.count("/") + 1
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {filename_prefix}*.csv under {prefix}, got {matches}"
            )
        return matches[0]

    def close(self):
        if self.zip_file:
            self.zip_file.close()


def price_table(src: ResultSource, folder: str, times_folder: str) -> pd.DataFrame:
    path = f"{folder}/{times_folder}/data/trade_data.csv"
    raw = src.csv(path, usecols=["date", "tic", "close"])
    if raw.duplicated(["date", "tic"]).any():
        raise ValueError(f"Duplicate date/ticker prices in {path}")
    return raw.pivot(index="date", columns="tic", values="close")


def compute_fees(actions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    stocks = [col for col in actions.columns if col != "date" and not str(col).startswith("Unnamed:")]
    if not stocks:
        raise ValueError("No stock columns in actions CSV")
    dates = actions["date"].astype(str).str[:10]
    closes = prices.reindex(index=dates.to_numpy(), columns=stocks)
    if closes.isna().any().any():
        raise ValueError("A trade date/ticker is missing from trade_data.csv")
    qty = actions[stocks].to_numpy(dtype=float)
    if not np.isfinite(qty).all():
        raise ValueError("Missing/nonfinite executed action")
    fees = (np.abs(qty) * closes.to_numpy(dtype=float) * FEE_RATE).sum(axis=1)
    return pd.DataFrame({
        "date": dates.to_numpy(),
        "daily_fee": fees,
        "buy_count": (qty > 0).sum(axis=1),
        "sell_count": (qty < 0).sum(axis=1),
        "trade_count": (qty != 0).sum(axis=1),
    })


def create_daily(assets: pd.DataFrame, fees: pd.DataFrame, asset_col: str) -> pd.DataFrame:
    # Trade on date[t] affects the recorded closing asset on date[t+1].
    dates = assets["date"].astype(str).str[:10].to_numpy()
    if len(dates) != len(fees) + 1 or not np.array_equal(
        dates[:-1], fees["date"].to_numpy()
    ):
        raise ValueError("Asset/action dates do not align; stop rather than shift fees incorrectly")
    incremental_fees = np.r_[0.0, fees["daily_fee"].to_numpy()]
    incremental_trades = np.r_[0, fees["trade_count"].to_numpy()]
    incremental_buy = np.r_[0, fees["buy_count"].to_numpy()]
    incremental_sell = np.r_[0, fees["sell_count"].to_numpy()]
    assets_net = assets[asset_col].to_numpy(dtype=float)
    cumulative_fee = np.cumsum(incremental_fees)
    return pd.DataFrame({
        "date": dates,
        "with_fee": assets_net,
        "without_fee_same_actions": assets_net + cumulative_fee,
        "cumulative_fee": cumulative_fee,
        "daily_fee": incremental_fees,
        "daily_trade_count": incremental_trades,
        "daily_buy_count": incremental_buy,
        "daily_sell_count": incremental_sell,
    })


def stats(daily: pd.DataFrame, **context) -> dict:
    start = float(daily["with_fee"].iloc[0])
    end = float(daily["with_fee"].iloc[-1])
    no_fee = float(daily["without_fee_same_actions"].iloc[-1])
    total_fee = float(daily["daily_fee"].sum())
    if not np.isclose(no_fee - end, total_fee, atol=1e-6):
        raise AssertionError("Add-back fees do not match final portfolio difference")
    trades = int(daily["daily_trade_count"].sum())
    return {
        **context,
        "initial_asset": start,
        "final_with_fee": end,
        "final_without_fee_same_actions": no_fee,
        "total_fee": total_fee,
        "buy_count": int(daily["daily_buy_count"].sum()),
        "sell_count": int(daily["daily_sell_count"].sum()),
        "trade_count": trades,
        "avg_fee_per_trade": total_fee / trades if trades else 0.0,
        "return_with_fee_pct": (end / start - 1) * 100,
        "return_without_fee_pct": (no_fee / start - 1) * 100,
        "fee_impact_pp": (no_fee - end) / start * 100,
    }


def check_annual_fees(
    src: ResultSource, prefix: str, actions: pd.DataFrame,
    prices: pd.DataFrame, fees: pd.DataFrame
):
    """Verify calculated fees against the saved post-trade asset (if available)."""
    try:
        holdings_path = src.match(prefix, "holdings_")
        weights_path = src.match(prefix, "post_trade_weights_")
    except ValueError:
        print(f"  Warning: no holdings/weights for {prefix}; fee validation skipped")
        return
    holdings = src.csv(holdings_path)
    weights = src.csv(weights_path)
    stocks = [col for col in actions.columns if col != "date" and not str(col).startswith("Unnamed:")]
    if not np.array_equal(
        holdings["date"].astype(str).str[:10].to_numpy()[:-1],
        actions["date"].astype(str).str[:10].to_numpy()
    ):
        raise ValueError(f"Holdings/action dates disagree: {prefix}")
    if not np.array_equal(
        weights["date"].astype(str).str[:10].to_numpy(),
        actions["date"].astype(str).str[:10].to_numpy()
    ):
        raise ValueError(f"Weight/action dates disagree: {prefix}")
    pre_trade_assets = (
        holdings["cash"].to_numpy()[:-1]
        + np.sum(
            holdings[stocks].to_numpy(dtype=float)[:-1]
            * prices.reindex(index=actions["date"].to_numpy(), columns=stocks).to_numpy(),
            axis=1
        )
    )
    implied_fees = pre_trade_assets - weights["total_asset_after_trade"].to_numpy()
    if not np.allclose(implied_fees, fees["daily_fee"].to_numpy(), atol=1e-5, rtol=0):
        raise ValueError(
            f"Calculated fees do not match environment accounting: {prefix}. "
            "Check FEE_RATE and the input price data."
        )


def asset_column(assets: pd.DataFrame) -> str:
    candidates = [
        col for col in assets.columns
        if col.lower() not in {"date", "dji", "^dji", "unnamed: 0"}
    ]
    if len(candidates) != 1:
        raise ValueError(f"Cannot identify the strategy asset column: {candidates}")
    return candidates[0]


def plot_curves(average: pd.DataFrame, output: Path, heading: str):
    x = pd.to_datetime(average["date"])
    plt.figure(figsize=(11, 5.2))
    plt.plot(x, average["with_fee"], label="With fees (actual)", linewidth=2.1)
    plt.plot(
        x, average["without_fee_same_actions"],
        label="Without fees (same executed trades; estimated)",
        linewidth=2, linestyle="--"
    )
    plt.title(heading)
    plt.xlabel("Date")
    plt.ylabel("Portfolio value (USD)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", nargs="?", default=TARGET_FOLDER,
        help="Rolling experiment folder name/path or ZIP (default: TARGET_FOLDER)",
    )
    parser.add_argument("--output", type=Path, help="Optional output directory")
    args = parser.parse_args()
    source = resolve_source(args.results)
    # Keep analysis files separate from the original backtest results.
    out = args.output or source.parent / "trading_cost_impact" / source.stem
    out.mkdir(parents=True, exist_ok=True)
    annual_dir = out / "annual_daily"
    cont_dir = out / "continuous_daily"
    annual_dir.mkdir(exist_ok=True)

    src = ResultSource(source)
    try:
        annual_stats, annual_daily = [], {}
        price_cache = {}
        meta = {}
        for name in src.entries:
            found = WINDOW_PATTERN.fullmatch(name)
            if found:
                folder = found.group("folder")
                wi = int(found.group("window"))
                times = int(found.group("times"))
                seed = found.group("seed")
                key = (wi, times)
                if key in meta:
                    raise ValueError(f"Multiple annual action files for window {wi}, times {times}")
                meta[key] = (folder, seed, name)
        if not meta:
            raise ValueError("Could not find annual window_*/times_*/actions_*.csv")

        for (wi, times), (folder, seed, actions_path) in sorted(meta.items()):
            times_dir = f"times_{times:02d}_seed_{seed}"
            prefix = f"{folder}/{times_dir}"
            prices = price_table(src, folder, times_dir)
            price_cache[(wi, times)] = prices
            actions = src.csv(actions_path)
            fees = compute_fees(actions, prices)
            check_annual_fees(src, prefix, actions, prices, fees)
            assets = src.csv(src.match(prefix, "backtest_result_"))
            year = get_year(folder, assets["date"])
            daily = create_daily(assets, fees, asset_column(assets))
            daily.to_csv(annual_dir / f"year_{year}_times_{times:02d}.csv", index=False)
            annual_stats.append(stats(daily, year=year, window=wi, times=times, seed=int(seed)))
            annual_daily.setdefault(year, []).append(daily)

        annual = pd.DataFrame(annual_stats)
        annual.to_csv(out / "annual_runs_summary.csv", index=False)
        annual_mean = annual.groupby("year", as_index=False).mean(numeric_only=True).drop(
            columns=["window", "times", "seed"]
        )
        annual_mean.to_csv(out / "annual_mean_summary.csv", index=False)
        for year, run_curves in sorted(annual_daily.items()):
            average = pd.concat(run_curves, ignore_index=True).groupby(
                "date", as_index=False
            )[["with_fee", "without_fee_same_actions", "cumulative_fee"]].mean()
            average.to_csv(out / f"annual_{year}_mean_curve.csv", index=False)
            plot_curves(
                average, out / f"annual_{year}_fee_comparison.png",
                f"Annual reset {year} | Average of {len(run_curves)} runs"
            )

        continuous_stats, continuous_daily = [], []
        continuous_available = any(
            name.startswith("continuous_5y_test/times_")
            and name.endswith("/continuous_portfolio_value.csv")
            for name in src.entries
        )
        if continuous_available:
            cont_dir.mkdir(exist_ok=True)
            window_ids = sorted({wi for wi, _ in meta})
            times_ids = sorted({times for _, times in meta})
            for times in times_ids:
                missing = [wi for wi in window_ids if (wi, times) not in meta]
                if missing:
                    raise ValueError(f"Missing annual times_{times:02d} for windows {missing}")
                seed = meta[(window_ids[0], times)][1]
                prefix = f"continuous_5y_test/times_{times:02d}_seed_{seed}"
                account = src.csv(src.match(prefix, "continuous_portfolio_value.csv"))
                windows = []
                cumulative_prior_fee = 0.0
                for wi in window_ids:
                    actions = src.csv(src.match(prefix, f"actions_window_{wi:02d}_"))
                    fees = compute_fees(actions, price_cache[(wi, times)])
                    part = account.loc[account["window_id"] == wi].copy()
                    daily = create_daily(part, fees, "strategy_continuous")
                    daily["cumulative_fee"] += cumulative_prior_fee
                    daily["without_fee_same_actions"] += cumulative_prior_fee
                    daily["window"] = wi
                    cumulative_prior_fee = float(daily["cumulative_fee"].iloc[-1])
                    windows.append(daily)
                combined = pd.concat(windows, ignore_index=True)
                if not combined["date"].is_unique:
                    raise ValueError(f"Duplicate dates in continuous curve for times_{times:02d}")
                combined.to_csv(cont_dir / f"times_{times:02d}_seed_{seed}.csv", index=False)
                continuous_stats.append(stats(combined, times=times, seed=int(seed)))
                continuous_daily.append(combined)
            cont = pd.DataFrame(continuous_stats)
            cont.to_csv(out / "continuous_5y_runs_summary.csv", index=False)
            cont.drop(columns=["times", "seed"]).mean().to_frame().T.to_csv(
                out / "continuous_5y_mean_summary.csv", index=False
            )
            avg_5y = pd.concat(continuous_daily, ignore_index=True).groupby(
                "date", as_index=False
            )[["with_fee", "without_fee_same_actions", "cumulative_fee"]].mean()
            avg_5y.to_csv(out / "continuous_5y_mean_curve.csv", index=False)
            plot_curves(
                avg_5y, out / "continuous_5y_fee_comparison.png",
                f"Continuous test | Average of {len(continuous_daily)} runs"
            )
            plt.figure(figsize=(11, 4.6))
            plt.plot(pd.to_datetime(avg_5y["date"]), avg_5y["cumulative_fee"], linewidth=2)
            plt.title("Continuous test | Cumulative fees (mean)")
            plt.xlabel("Date")
            plt.ylabel("Fees paid / gap between curves (USD)")
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(out / "continuous_5y_cumulative_fee.png", dpi=180)
            plt.close()
            print("Processed annual runs:", len(annual), "continuous runs:", len(cont))
            print("Continuous test averages:")
            print(cont.drop(columns=["times", "seed"]).mean().round(3).to_string())
        else:
            print("No continuous_5y_test results found; only annual summaries generated.")
        print("Annual mean summary:")
        print(annual_mean[[
            "year", "trade_count", "total_fee", "return_with_fee_pct",
            "return_without_fee_pct", "fee_impact_pp"
        ]].round(3).to_string(index=False))
        print("Output:", out.resolve())
    finally:
        src.close()


if __name__ == "__main__":
    main()
