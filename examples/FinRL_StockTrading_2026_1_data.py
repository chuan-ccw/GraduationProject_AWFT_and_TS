"""
Stock NeurIPS2018 Part 1. Data

This series is a reproduction of paper "Deep reinforcement learning for automated stock trading: An ensemble strategy".

Introduce how to use FinRL to fetch and process data that we need for ML/RL trading.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
import yfinance as yf

from finrl import config_tickers
import finrl.config as config
print("config path:", config.__file__)
from finrl.config import INDICATORS
from experiment_settings import env_json, env_str
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# Runtime settings: Run_StockTrading_Examples.py can override config.py through environment variables.
TRAIN_START_DATE = env_str("TRAIN_START_DATE", config.TRAIN_START_DATE)
TRAIN_END_DATE = env_str("TRAIN_END_DATE", config.TRAIN_END_DATE)
TRADE_START_DATE = env_str("TRADE_START_DATE", config.TRADE_START_DATE)
TRADE_END_DATE = env_str("TRADE_END_DATE", config.TRADE_END_DATE)
FUTURE_RANKING_PARAMS = env_json(
    "FUTURE_RANKING_PARAMS",
    {"n_future": 60, "price_col": "close", "future_metric": "future_avg_return"},
)
TRAIN_DATA_PATH = Path(env_str("TRAIN_DATA_PATH", "train_data.csv"))
TRADE_DATA_PATH = Path(env_str("TRADE_DATA_PATH", "trade_data.csv"))
FUTURE_RANKING_DATA_PATH = TRADE_DATA_PATH.with_name("future_ranking_price_data.csv")
TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
TRADE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Data date setting:")
print(f"  TRAIN: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
print(f"  TRADE: {TRADE_START_DATE} to {TRADE_END_DATE}")
print(f"  TRAIN_DATA_PATH: {TRAIN_DATA_PATH}")
print(f"  TRADE_DATA_PATH: {TRADE_DATA_PATH}")
print(f"  FUTURE_RANKING_DATA_PATH: {FUTURE_RANKING_DATA_PATH}")
config.print_observation_feature_mode()

# # %% Part 1. Fetch data - Single ticker

# # Using yfinance directly
# aapl_df_yf = yf.download(tickers="aapl", start="2020-01-01", end="2020-01-31")
# print("=== yfinance download ===")
# print(aapl_df_yf.head())

# # Using FinRL's YahooDownloader
# aapl_df_finrl = YahooDownloader(
#     start_date="2020-01-01",
#     end_date="2020-01-31",
#     ticker_list=["aapl"],
# ).fetch_data()
# print("\n=== FinRL YahooDownloader ===")
# print(aapl_df_finrl.head())

# %% Part 2. Fetch data - DOW 30 tickers

print("\n=== DOW 30 Tickers ===")
print(config_tickers.DOW_30_TICKER)

# Future Ranking must still have n_future trading days available after the
# Test period. Download a calendar-day buffer only for the objective ranking
# source; the model train/trade datasets below remain clipped to TRADE_END_DATE.
n_future = max(1, int(FUTURE_RANKING_PARAMS.get("n_future", 60)))
future_buffer_days = max(30, n_future * 3)
ranking_download_end = (
    pd.Timestamp(TRADE_END_DATE) + pd.Timedelta(days=future_buffer_days)
).strftime("%Y-%m-%d")

df_raw = YahooDownloader(
    start_date=TRAIN_START_DATE,
    end_date=ranking_download_end,
    ticker_list=config_tickers.DOW_30_TICKER,
).fetch_data()
print("\n=== Raw data ===")
print(df_raw.head())
print(df_raw.tail())

price_col = str(FUTURE_RANKING_PARAMS.get("price_col", "close"))
if price_col not in df_raw.columns:
    price_col = "close"
future_ranking_data = df_raw.loc[
    (df_raw["date"] >= TRADE_START_DATE),
    ["date", "tic", price_col],
].copy()
future_ranking_data.to_csv(FUTURE_RANKING_DATA_PATH, index=False, encoding="utf-8-sig")
print(
    "Future Ranking price data saved to "
    f"{FUTURE_RANKING_DATA_PATH} (through {ranking_download_end})"
)

# Preserve the original model-data boundary exactly; the extra future rows
# above are only for Future Ranking and are never passed into the environment.
df_raw = df_raw[df_raw["date"] < TRADE_END_DATE].copy()

# %% Part 3. Preprocess data

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    use_turbulence=True,
    user_defined_feature=False,
)

processed = fe.preprocess_data(df_raw)

list_ticker = processed["tic"].unique().tolist()
list_date = list(
    pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
)
combination = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
    processed, on=["date", "tic"], how="left"
)
processed_full = processed_full[processed_full["date"].isin(processed["date"])]
processed_full = processed_full.sort_values(["date", "tic"])
processed_full = processed_full.fillna(0)

print("\n=== Processed data ===")
print(processed_full.head())
print(processed_full.tail())

# %% Part 4. Split and save data

train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
print(f"\nTrain data length: {len(train)}")
print(f"Trade data length: {len(trade)}")

train.to_csv(TRAIN_DATA_PATH)
trade.to_csv(TRADE_DATA_PATH)
# Also save to project root for backward compatibility when running scripts manually.
if str(TRAIN_DATA_PATH) != "train_data.csv":
    train.to_csv("train_data.csv")
if str(TRADE_DATA_PATH) != "trade_data.csv":
    trade.to_csv("trade_data.csv")
print(f"Data saved to {TRAIN_DATA_PATH} and {TRADE_DATA_PATH}")
