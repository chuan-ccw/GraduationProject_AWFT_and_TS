from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================
# Buy & Hold settings
# =====================================================

DOW_30_TICKER = [
    "AXP",
    "AMGN",
    "AMZN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NVDA",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WMT",
    "DIS",
    "SHW",
]

YEARS = [2021, 2022, 2023, 2024, 2025]

# 每次 Buy & Hold 實驗的初始資產。
INITIAL_AMOUNT = 1_000_000.0

# 交易手續費率。
# 目前設為買入 0.1%、賣出 0.1%。
# 若 FinRL 實驗使用不同費率，只需要修改這兩個參數。
BUY_COST_PCT = 0.001
SELL_COST_PCT = 0.001

# 使用與 FinRL / AWFT 回測相同的 trade_data.csv。
EXPERIMENT_ROOT = Path(
    r"C:\Users\User\Desktop\GraduationProject\Experence_result"
)

# 只需要填 rolling_* 資料夾名稱。
# 程式會自動從 EXPERIMENT_ROOT 底下搜尋，不限定 reward_v0 / reward_v7。
SOURCE_ROLLING_FOLDER = (
    r"rolling_2018-2025_train3y_test1y_shift1y_20260904_234026_reinforce"
)

OUTPUT_DIR = EXPERIMENT_ROOT / "buy_and_hold_dow30"


# =====================================================
# Data loading
# =====================================================

def resolve_rolling_root() -> Path:
    """
    Search EXPERIMENT_ROOT recursively for SOURCE_ROLLING_FOLDER.

    The rolling folder may be inside reward_v0, reward_v7, or another
    experiment subfolder. The folder name must match exactly.
    """
    if not EXPERIMENT_ROOT.exists():
        raise FileNotFoundError(
            "EXPERIMENT_ROOT 不存在：\n"
            f"{EXPERIMENT_ROOT}"
        )

    matches = sorted(
        path
        for path in EXPERIMENT_ROOT.rglob(SOURCE_ROLLING_FOLDER)
        if path.is_dir() and path.name == SOURCE_ROLLING_FOLDER
    )

    if not matches:
        raise FileNotFoundError(
            "找不到指定的 rolling 資料夾：\n"
            f"{SOURCE_ROLLING_FOLDER}\n"
            f"搜尋範圍：{EXPERIMENT_ROOT}"
        )

    if len(matches) > 1:
        matched_paths = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            "找到多個同名 rolling 資料夾，無法判斷要使用哪一個：\n"
            f"{matched_paths}"
        )

    return matches[0]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the columns needed by Buy & Hold."""
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]

    required = {"date", "tic", "close"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            f"trade_data.csv 缺少必要欄位: {sorted(missing)}"
        )

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["tic"] = out["tic"].astype(str).str.strip()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    out = out.dropna(subset=["date", "tic", "close"]).copy()
    out = out[out["tic"].isin(DOW_30_TICKER)].copy()

    return out[["date", "tic", "close"]]


def _find_trade_data_for_year(
    rolling_root: Path,
    year: int,
) -> Path:
    """Find one trade_data.csv for the requested test year."""
    candidates = sorted(
        rolling_root.glob(
            f"window_*_test_{year}-{year}/times_*/data/trade_data.csv"
        )
    )

    # 相容舊版 repeat_* 資料夾命名。
    if not candidates:
        candidates = sorted(
            rolling_root.glob(
                f"window_*_test_{year}-{year}/repeat_*/data/trade_data.csv"
            )
        )

    if not candidates:
        raise FileNotFoundError(
            f"找不到 {year} 年的 trade_data.csv。\n"
            f"搜尋位置: {rolling_root}"
        )

    # 同一個 window 的不同 times / repeat 使用相同市場資料，
    # 因此固定取排序後第一份，避免重複讀取。
    return candidates[0]


def load_all_prices(rolling_root: Path) -> pd.DataFrame:
    """Load 2021-2025 close prices from the rolling experiment."""
    yearly_frames: list[pd.DataFrame] = []

    for year in YEARS:
        csv_path = _find_trade_data_for_year(rolling_root, year)
        df = pd.read_csv(csv_path)
        df = _normalize_columns(df)

        df = df[df["date"].dt.year == year].copy()
        if df.empty:
            raise ValueError(
                f"{csv_path} 中沒有 {year} 年的有效資料。"
            )

        tickers = set(df["tic"].unique())
        missing_tickers = [
            ticker
            for ticker in DOW_30_TICKER
            if ticker not in tickers
        ]
        if missing_tickers:
            raise ValueError(
                f"{year} 年缺少股票資料: {missing_tickers}\n"
                f"來源: {csv_path}"
            )

        yearly_frames.append(df)

        print(
            f"{year}: {csv_path}\n"
            f"  rows={len(df)}, "
            f"dates={df['date'].nunique()}, "
            f"tickers={df['tic'].nunique()}"
        )

    all_prices = pd.concat(yearly_frames, ignore_index=True)
    all_prices = all_prices.drop_duplicates(
        subset=["date", "tic"],
        keep="last",
    )
    all_prices = all_prices.sort_values(
        ["date", "tic"]
    ).reset_index(drop=True)

    return all_prices


def build_price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form price data to date x ticker close-price matrix."""
    matrix = prices.pivot(
        index="date",
        columns="tic",
        values="close",
    )
    matrix = matrix.reindex(columns=DOW_30_TICKER)
    matrix = matrix.sort_index()

    # 美國股票交易日通常一致；若單一股票偶爾缺值，
    # 使用前一交易日 close 延續，避免無故改變投資組合股票數量。
    matrix = matrix.ffill()

    # 起始位置仍缺值代表該股票沒有足夠資料，不應直接計算。
    missing = matrix.columns[matrix.isna().any()].tolist()
    if missing:
        raise ValueError(
            "價格矩陣仍存在缺失值，無法安全計算 Buy & Hold: "
            f"{missing}"
        )

    return matrix


# =====================================================
# Buy & Hold calculation
# =====================================================

def equal_weight_buy_and_hold(
    price_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Equal-weight Buy & Hold with transaction costs.

    Initial capital = INITIAL_AMOUNT USD.

    Buy:
    - Pay BUY_COST_PCT once at the beginning.
    - Invest the remaining effective capital equally across all stocks.

    Hold:
    - No rebalancing and no transaction cost during the holding period.

    Sell:
    - Sell all holdings on the final day.
    - Pay SELL_COST_PCT once on the final portfolio value.

    Fractional shares are allowed.

    Invested capital after the initial buy:
        C_invest = INITIAL_AMOUNT / (1 + BUY_COST_PCT)

    Portfolio value before the final sell:
        A_t = C_invest * (1 / N) * sum(P_i,t / P_i,0)

    Final portfolio value after selling all holdings:
        A_T_after_sell = A_T_before_sell * (1 - SELL_COST_PCT)

    The returned series is normalized by INITIAL_AMOUNT so it can be
    imported directly into Compare_Two_Equity:
        V_t = A_t / INITIAL_AMOUNT
    """
    if price_matrix.empty:
        raise ValueError("price_matrix 不可為空。")

    if not 0.0 <= BUY_COST_PCT < 1.0:
        raise ValueError("BUY_COST_PCT 必須介於 0 到 1 之間。")

    if not 0.0 <= SELL_COST_PCT < 1.0:
        raise ValueError("SELL_COST_PCT 必須介於 0 到 1 之間。")

    initial_prices = price_matrix.iloc[0]

    if (initial_prices <= 0).any():
        bad = initial_prices.index[
            initial_prices <= 0
        ].tolist()
        raise ValueError(
            f"起始 close 必須大於 0，異常股票: {bad}"
        )

    relative_prices = price_matrix.divide(
        initial_prices,
        axis="columns",
    )

    # 初始資產為 INITIAL_AMOUNT。買入手續費以成交金額計算，
    # 因此真正能投入股票的資金為 INITIAL_AMOUNT / (1 + BUY_COST_PCT)。
    invested_capital = INITIAL_AMOUNT / (1.0 + BUY_COST_PCT)

    asset_value = relative_prices.mean(axis=1) * invested_capital

    # 最後一天視為全部賣出，因此最後資產扣除一次賣出手續費。
    if not asset_value.empty:
        asset_value.iloc[-1] = asset_value.iloc[-1] * (1.0 - SELL_COST_PCT)

    # Compare_Two_Equity 使用 normalized equity，因此除以初始資產。
    equity = asset_value / INITIAL_AMOUNT
    equity.name = "strategy_mean_norm"

    return equity


def build_compare_output(
    equity: pd.Series,
) -> pd.DataFrame:
    """
    Build columns compatible with Compare_Two_Equity.

    Buy & Hold is deterministic, so min = mean = max.
    """
    out = pd.DataFrame(
        {
            "date": equity.index.strftime("%Y-%m-%d"),
            "strategy_mean_norm": equity.to_numpy(dtype=float),
            "portfolio_value_usd": (
                equity.to_numpy(dtype=float) * INITIAL_AMOUNT
            ),
        }
    )
    out["strategy_min_norm"] = out["strategy_mean_norm"]
    out["strategy_max_norm"] = out["strategy_mean_norm"]

    return out


def calculate_annual_reset(
    price_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """
    Each year is an independent Buy & Hold experiment.

    At the first trading day of every year:
    - reset normalized equity to 1.0
    - allocate equal capital to all 30 stocks
    - hold until that year's final trading day
    """
    output_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for year in YEARS:
        yearly_prices = price_matrix[
            price_matrix.index.year == year
        ].copy()

        if yearly_prices.empty:
            raise ValueError(f"沒有 {year} 年價格資料。")

        equity = equal_weight_buy_and_hold(yearly_prices)
        output_frames.append(build_compare_output(equity))

        final_equity = float(equity.iloc[-1])
        cumulative_return = final_equity - 1.0

        summary_rows.append(
            {
                "period": str(year),
                "mode": "annual_reset",
                "start_date": equity.index[0].strftime("%Y-%m-%d"),
                "end_date": equity.index[-1].strftime("%Y-%m-%d"),
                "initial_amount_usd": INITIAL_AMOUNT,
                "final_asset_usd": final_equity * INITIAL_AMOUNT,
                "final_equity": final_equity,
                "buy_cost_pct": BUY_COST_PCT,
                "sell_cost_pct": SELL_COST_PCT,
                "cumulative_return": cumulative_return,
                "cumulative_return_percent": cumulative_return * 100.0,
            }
        )

    annual_df = pd.concat(
        output_frames,
        ignore_index=True,
    )

    return annual_df, summary_rows


def calculate_continuous_5y(
    price_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Buy once at the first 2021 trading day and hold continuously to 2025.

    No annual reset and no annual rebalancing.
    """
    mask = (
        (price_matrix.index.year >= YEARS[0])
        & (price_matrix.index.year <= YEARS[-1])
    )
    continuous_prices = price_matrix.loc[mask].copy()

    if continuous_prices.empty:
        raise ValueError("沒有 2021-2025 的連續價格資料。")

    equity = equal_weight_buy_and_hold(continuous_prices)
    output_df = build_compare_output(equity)

    final_equity = float(equity.iloc[-1])
    cumulative_return = final_equity - 1.0

    summary = {
        "period": f"{YEARS[0]}-{YEARS[-1]}",
        "mode": "continuous_5y",
        "start_date": equity.index[0].strftime("%Y-%m-%d"),
        "end_date": equity.index[-1].strftime("%Y-%m-%d"),
        "initial_amount_usd": INITIAL_AMOUNT,
        "final_asset_usd": final_equity * INITIAL_AMOUNT,
        "final_equity": final_equity,
        "buy_cost_pct": BUY_COST_PCT,
        "sell_cost_pct": SELL_COST_PCT,
        "cumulative_return": cumulative_return,
        "cumulative_return_percent": cumulative_return * 100.0,
    }

    return output_df, summary


# =====================================================
# Output
# =====================================================

def main() -> None:
    source_rolling_root = resolve_rolling_root()
    print(f"Found rolling folder: {source_rolling_root}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_all_prices(source_rolling_root)
    price_matrix = build_price_matrix(prices)

    annual_df, annual_summary = calculate_annual_reset(
        price_matrix
    )
    continuous_df, continuous_summary = calculate_continuous_5y(
        price_matrix
    )

    annual_path = OUTPUT_DIR / "buy_and_hold_annual_reset.csv"
    continuous_path = (
        OUTPUT_DIR / "buy_and_hold_continuous_5y.csv"
    )
    summary_path = OUTPUT_DIR / "buy_and_hold_summary.csv"

    annual_df.to_csv(
        annual_path,
        index=False,
        encoding="utf-8-sig",
    )
    continuous_df.to_csv(
        continuous_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = pd.DataFrame(
        annual_summary + [continuous_summary]
    )
    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nInitial amount: ${INITIAL_AMOUNT:,.2f}")
    print(
        "Transaction costs: "
        f"buy={BUY_COST_PCT * 100:.4f}%, "
        f"sell={SELL_COST_PCT * 100:.4f}%"
    )
    print("\n=== Buy & Hold Cumulative Return ===")
    display_df = summary_df[
        [
            "period",
            "mode",
            "start_date",
            "end_date",
            "final_equity",
            "cumulative_return_percent",
        ]
    ].copy()
    display_df["final_equity"] = (
        display_df["final_equity"].map(
            lambda value: f"{value:.6f}"
        )
    )
    display_df["cumulative_return_percent"] = (
        display_df["cumulative_return_percent"].map(
            lambda value: f"{value:.2f}%"
        )
    )
    print(display_df.to_string(index=False))

    print("\nSaved:")
    print(f"Annual Reset : {annual_path}")
    print(f"Continuous 5Y: {continuous_path}")
    print(f"Summary      : {summary_path}")


if __name__ == "__main__":
    main()
