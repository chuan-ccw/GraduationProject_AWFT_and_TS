# directory
from __future__ import annotations

DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

# Default reward experiment settings
# These are only defaults. Run_StockTrading_Examples.py can override them at runtime.
# For reward v2: use REWARD_TYPE = "v2"
# For reward v3: use REWARD_TYPE = "v3"
# For reward v4: use REWARD_TYPE = "v4"
# For reward v5 (Weighted-Future-Trend, WFT): use REWARD_TYPE = "v5"
# For reward v6 (Delta-Weighted-Future-Trend, DWFT): use REWARD_TYPE = "v6"
# For v6, set REWARD_PARAMS["future_variable"] to "future_direction"
# or "future_avg_return". Set REWARD_PARAMS["delta_weight_variable"] to
# "original_delta_weight" or "same_price_delta_weight".
REWARD_TYPE = "v5"
REWARD_PARAMS = {"n_future": 5, "price_col": "close", "c": 0.05}
EXPERIMENT_ROOT = r"C:\Users\User\Desktop\GraduationProject\Experence_result"

# date format: '%Y-%m-%d'
TRAIN_START_DATE = "2014-01-06"
TRAIN_END_DATE = "2024-12-31"

TEST_START_DATE = "2025-01-01"
TEST_END_DATE = "2026-03-20"

TRADE_START_DATE = "2025-01-01"
TRADE_END_DATE = "2026-03-20"

# Observation feature experiment switch.
# False = keep the original FinRL observation used by this project.
# True  = use the expanded technical indicators plus raw open/high/low/volume.
USE_EXPANDED_INDICATORS = False

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
ORIGINAL_INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

EXPANDED_INDICATORS = [
    "close_5_sma",
    "close_10_sma",
    "close_20_sma",
    "close_60_sma",
    "close_120_sma",
    "close_240_sma",
    "macd",
    "kdjk",
    "kdjd",
    "kdjj",
    "atr",
    "cci",
    "close_-6_d",
    "close_-10_d",
    "close_-12_d",
    "close_-18_d",
    "close_5_roc",
    "close_10_roc",
    "close_12_roc",
    "wr",
    "pvo",
]

EXPANDED_RAW_FEATURES = ["open", "high", "low", "volume"]

# Keep INDICATORS as the active technical-indicator list so existing data
# preprocessing call sites do not need a second switch. Raw market features
# are already present in the downloaded dataframe and are appended only by
# StockTradingEnv when expanded mode is enabled.
INDICATORS = EXPANDED_INDICATORS if USE_EXPANDED_INDICATORS else ORIGINAL_INDICATORS
ADDITIONAL_RAW_FEATURES = EXPANDED_RAW_FEATURES if USE_EXPANDED_INDICATORS else []
OBSERVATION_FEATURE_MODE = "expanded" if USE_EXPANDED_INDICATORS else "original"


def get_stock_trading_state_space(stock_dim: int) -> int:
    """Return the active StockTradingEnv observation dimension."""
    feature_count = len(INDICATORS) + len(ADDITIONAL_RAW_FEATURES)
    return 1 + 2 * int(stock_dim) + feature_count * int(stock_dim)


def print_observation_feature_mode(stock_dim: int | None = None) -> None:
    """Print the active observation feature set for experiment verification."""
    print("========================================")
    print(f"Observation Feature Mode: {OBSERVATION_FEATURE_MODE.upper()}")
    print(f"USE_EXPANDED_INDICATORS = {USE_EXPANDED_INDICATORS}")
    print("Technical Indicators:")
    for indicator in INDICATORS:
        print(f"- {indicator}")
    if ADDITIONAL_RAW_FEATURES:
        print("Additional Raw Features:")
        for feature in ADDITIONAL_RAW_FEATURES:
            print(f"- {feature}")
    if stock_dim is not None:
        print(f"Stock Dimension: {stock_dim}")
        print(f"Technical Indicator Count: {len(INDICATORS)}")
        print(f"Additional Raw Feature Count: {len(ADDITIONAL_RAW_FEATURES)}")
        print(f"Final Observation Dimension: {get_stock_trading_state_space(stock_dim)}")
    print("========================================")


# Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}


# Possible time zones
TIME_ZONE_SHANGHAI = "Asia/Shanghai"  # Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = "Europe/Paris"  # CAC,
TIME_ZONE_BERLIN = "Europe/Berlin"  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = "Asia/Jakarta"  # LQ45
TIME_ZONE_SELFDEFINED = "xxx"  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)

# parameters for data sources
ALPACA_API_KEY = "xxx"  # your ALPACA_API_KEY
ALPACA_API_SECRET = "xxx"  # your ALPACA_API_SECRET
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"  # alpaca url
BINANCE_BASE_URL = "https://data.binance.vision/"  # binance url
