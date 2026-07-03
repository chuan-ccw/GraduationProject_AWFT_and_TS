from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.utils import set_random_seed
from datetime import datetime
from pathlib import Path
from typing import Any

from finrl import config


# =====================================================
# 1. 在這裡統一設定本次實驗參數
# =====================================================

# Reward 選擇：
#   "v0" = FinRL 原版 reward，使用 portfolio value change
#   "v1" = simple return + buy low score + sell high score
#   "v2" = sum(action * future t+n day simple return)
#   "v3" = sum(action * future n-day average simple return)
#   "v4" = sum(delta action * future n-day average simple return)

REWARD_TYPE = "v4"

REWARD_PARAMS_BY_TYPE = {
    "v0": {},
    "v1": {
        "alpha": 0.05,
        "beta": 0.05,
        "n_recent": 14,
    },
    "v2": {
        "n_future": 5,
        "price_col": "close",
    },
    "v3": {
        "n_future": 5,
        "price_col": "close",
    },
    "v4": {
        "n_future": 5,
        "price_col": "close",
    },
}

REWARD_PARAMS = REWARD_PARAMS_BY_TYPE.get(REWARD_TYPE, {})

REWARD_DISPLAY_NAMES = {
    "v0": "original_reward",
    "original": "original_reward",
    "finrl_original": "original_reward",
    "v1": "buy_low_sell_high_reward",
    "our_reward_v1": "buy_low_sell_high_reward",
    "v2": "future_day_reward",
    "reward_v2": "future_day_reward",
    "future_day_return": "future_day_reward",
    "v3": "future_avg_reward",
    "reward_v3": "future_avg_reward",
    "future_avg_return": "future_avg_reward",
    "v4": "future_avg_delta_action_reward",
    "reward_v4": "future_avg_delta_action_reward",
    "future_avg_delta_action_return": "future_avg_delta_action_reward",
    "future_avg_delta_action_reward": "future_avg_delta_action_reward",
}


def reward_display_name(reward_type: str | None) -> str:
    """Return the display name used in plots instead of v0/v1/v2/v3/v4."""
    key = str(reward_type or "").strip().lower()
    if key.startswith("reward_") and key not in REWARD_DISPLAY_NAMES:
        key = key[len("reward_"):]
    return REWARD_DISPLAY_NAMES.get(key, str(reward_type or ""))


def set_axis_labels_horizontal(ax, xlabel: str | None = None, ylabel: str | None = None, ylabel_pad: int = 65) -> None:
    """Keep x-axis label horizontal, but keep y-axis label in the default vertical style."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, rotation=0)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def format_yyyy_mm(value) -> str:
    """Format plot x-axis date labels as YYYY-MM, never YYYY/MM."""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.notna(ts):
        return ts.strftime("%Y-%m")
    text = str(value)
    match = __import__("re").search(r"(\d{4})[-/](\d{1,2})", text)
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"
    return text


def make_month_tick_positions(length: int, max_xticks: int = 12) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=int)
    return np.linspace(0, length - 1, min(max_xticks, length)).astype(int)


def set_month_xticklabels(ax, dates, max_xticks: int = 12) -> None:
    positions = make_month_tick_positions(len(dates), max_xticks=max_xticks)
    if len(positions) == 0:
        return
    ax.set_xticks(positions)
    ax.set_xticklabels([format_yyyy_mm(dates[i]) for i in positions], rotation=0, ha="center")


def normalize_date_column(series: pd.Series) -> pd.Series:
    """Normalize real date strings to YYYY-MM-DD and reject numeric day indexes."""
    text = series.astype(str).str.strip()
    is_date_text = text.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")
    parsed = pd.to_datetime(text.where(is_date_text), errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def keep_valid_date_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose date is an internal numeric trading-day index."""
    if "date" not in df.columns:
        return df
    out = df.copy()
    out["date"] = normalize_date_column(out["date"])
    out = out[out["date"].notna()].copy()
    return out.reset_index(drop=True)


def build_dji_baseline(trade_dates, initial_amount: float = 1_000_000.0) -> pd.Series:
    """Download and align DJI close prices to the provided trade dates."""
    trade_index = pd.Index([str(x)[:10] for x in trade_dates], name="date")
    if len(trade_index) == 0:
        return pd.Series(dtype=float, name="dji")

    try:
        import yfinance as yf

        start_ts = pd.to_datetime(trade_index[0], errors="coerce") - pd.Timedelta(days=7)
        end_ts = pd.to_datetime(trade_index[-1], errors="coerce") + pd.Timedelta(days=7)
        df_dji = yf.download(
            "^DJI",
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )
        if df_dji is None or df_dji.empty:
            raise ValueError("empty yfinance result")

        close = None
        if isinstance(df_dji.columns, pd.MultiIndex):
            for candidate in ("Close", "Adj Close"):
                level0 = [str(x) for x in df_dji.columns.get_level_values(0)]
                if candidate in level0:
                    sub = df_dji.xs(candidate, axis=1, level=0)
                    close = sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
                    break
        else:
            for candidate in ["Close", "Adj Close", "close", "adj close"]:
                if candidate in df_dji.columns:
                    close = df_dji[candidate]
                    break

        if close is None:
            raise ValueError(f"cannot find Close column, columns={list(df_dji.columns)}")

        close = pd.to_numeric(close, errors="coerce").dropna()
        if close.empty:
            raise ValueError("DJI close series is empty after numeric conversion")

        close.index = pd.Index(pd.to_datetime(close.index, errors="coerce").strftime("%Y-%m-%d"), name="date")
        close = close[~close.index.duplicated(keep="last")]

        aligned = close.reindex(trade_index).ffill().bfill()
        valid = aligned.dropna()
        if valid.empty:
            raise ValueError("DJI series is empty after reindex to trade dates")

        normalized = aligned / float(valid.iloc[0]) * float(initial_amount)
        normalized.name = "dji"
        return normalized
    except Exception as exc:
        print(f"DJI baseline failed: {exc}")
        return pd.Series(index=trade_index, data=np.nan, name="dji")


UNUSED_COMPARE_IMAGE_NAMES = [
    "compare_report.png",
    "compare_reward_curve.png",
    "compare_cumulative_return.png",
]


def is_backtest_repeat_dir(path: Path) -> bool:
    """Return True only for real backtest repeat folders, not annual_reset folders."""
    return path.is_dir() and re.fullmatch(r"repeat_\d{2}_seed_\d+", path.name) is not None


def select_backtest_result_csv(repeat_dir: Path, reward_type: str) -> Path | None:
    """Select the backtest CSV produced by one real repeat folder."""
    candidates = sorted(repeat_dir.glob(f"backtest_result_{reward_type}_*.csv"))
    if not candidates:
        candidates = sorted(repeat_dir.glob("backtest_result_*.csv"))
    if not candidates:
        return None
    # If the same repeat folder was rerun, use the newest CSV instead of an older stale file.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def cleanup_unused_compare_images(compare_dir: Path) -> None:
    """Remove old compare image aliases that are no longer generated."""
    for name in UNUSED_COMPARE_IMAGE_NAMES:
        path = compare_dir / name
        if path.exists():
            path.unlink()


# Rolling Window 設定
ROLLING_WINDOW_MODE = True
ROLLING_START_DATE = "2018-01-01"
ROLLING_END_DATE = "2025-12-31"
TRAIN_YEARS = 3
TEST_YEARS = 1
SHIFT_YEARS = 1

# 每個 rolling window 的 repeat 次數
# 注意：每個 repeat 都會 train 一個新的 PPO model，然後只 backtest 一次。
REPEAT_EXPERIMENTS = 5
BASE_RANDOM_SEED = 42

# False = 測試時從 PPO policy 分布抽樣。
# True  = 測試時固定選最可能的 action，讓同一個 model + state 的結果可重現。
TEST_PREDICT_DETERMINISTIC = True

# 非 rolling mode 才會用到這組日期
TRAIN_START_DATE = "2018-01-01"
TRAIN_END_DATE = "2020-12-31"
TEST_START_DATE = "2021-01-01"
TEST_END_DATE = "2021-12-31"
TRADE_START_DATE = TEST_START_DATE
TRADE_END_DATE = TEST_END_DATE

# 實驗結果總資料夾
EXPERIMENT_ROOT = getattr(
    config,
    "EXPERIMENT_ROOT",
    r"C:\Users\User\Desktop\GraduationProject\Experence_result",
)

# 是否執行各階段
RUN_DATA = True
RUN_TRAIN = True
RUN_BACKTEST = True

# 是否在 rolling 實驗結束後自動使用「最新」v0 與目前 REWARD_TYPE 產生比較圖。
# 例如 REWARD_TYPE="v2" 時，會產生 v0 vs v2；REWARD_TYPE="v4" 時，會產生 v0 vs v4。
# 建議正式實驗設為 False，改用 examples/Compare_V0_V1_Manual.py 手動指定資料夾。
AUTO_COMPARE_LATEST = False

# 連續 5 年 test：
#   False = 保持 Annual Reset 流程。
#   True  = rolling 結束後額外跑一次連續 test，上一個 window 的 cash/holdings 會接到下一個 window。
RUN_CONTINUOUS_5Y_TEST = False

# 若 RUN_CONTINUOUS_5Y_TEST=True，且 v0 與目前 REWARD_TYPE 都已有 continuous_5y_test 結果，
# 可自動產生連續 5 年的 compare_portfolio_value.png。
AUTO_COMPARE_CONTINUOUS_LATEST = False

# Agent 開關：train 和 backtest 會共用這裡的設定
IF_USING_A2C = False
IF_USING_DDPG = False
IF_USING_PPO = True
IF_USING_TD3 = False
IF_USING_SAC = False

# 訓練步數
A2C_TOTAL_TIMESTEPS = 20000
DDPG_TOTAL_TIMESTEPS = 20000
PPO_TOTAL_TIMESTEPS = 100000
TD3_TOTAL_TIMESTEPS = 20000
SAC_TOTAL_TIMESTEPS = 20000


# =====================================================
# 2. Rolling window / folder helpers
# =====================================================

def _year(date_str: str) -> int:
    return int(date_str[:4])


def generate_rolling_windows(
    start_date: str,
    end_date: str,
    train_years: int,
    test_years: int,
    shift_years: int,
) -> list[dict[str, Any]]:
    """Generate train/test windows.

    Example with 2018-01-01 ~ 2025-12-31, train=3, test=1, shift=1:
    2018-2020 -> 2021, 2019-2021 -> 2022, ..., 2022-2024 -> 2025.
    """
    windows: list[dict[str, Any]] = []
    train_start_year = _year(start_date)
    final_year = _year(end_date)
    window_id = 1

    while True:
        train_end_year = train_start_year + train_years - 1
        test_start_year = train_end_year + 1
        test_end_year = test_start_year + test_years - 1
        if test_end_year > final_year:
            break

        windows.append(
            {
                "window_id": window_id,
                "train_start_year": train_start_year,
                "train_end_year": train_end_year,
                "test_start_year": test_start_year,
                "test_end_year": test_end_year,
                "train_start_date": f"{train_start_year}-01-01",
                "train_end_date": f"{train_end_year}-12-31",
                "test_start_date": f"{test_start_year}-01-01",
                "test_end_date": f"{test_end_year}-12-31",
                "trade_start_date": f"{test_start_year}-01-01",
                "trade_end_date": f"{test_end_year}-12-31",
            }
        )
        train_start_year += shift_years
        window_id += 1

    return windows


def create_rolling_root() -> Path:
    date_tag = datetime.now().strftime("%Y%m%d")
    start_year = _year(ROLLING_START_DATE)
    end_year = _year(ROLLING_END_DATE)
    run_name = (
        f"rolling_{start_year}-{end_year}_"
        f"train{TRAIN_YEARS}y_test{TEST_YEARS}y_shift{SHIFT_YEARS}y_"
        f"{date_tag}"
    )
    rolling_root = Path(EXPERIMENT_ROOT) / f"reward_{REWARD_TYPE}" / run_name
    if rolling_root.exists():
        rolling_root = Path(EXPERIMENT_ROOT) / f"reward_{REWARD_TYPE}" / f"{run_name}_{datetime.now().strftime('%H%M%S')}"
    rolling_root.mkdir(parents=True, exist_ok=True)
    return rolling_root


def window_folder_name(window: dict[str, Any]) -> str:
    return (
        f"window_{window['window_id']:02d}_"
        f"train_{window['train_start_year']}-{window['train_end_year']}_"
        f"test_{window['test_start_year']}-{window['test_end_year']}"
    )


def repeat_folder_name(repeat_index: int, seed: int) -> str:
    return f"repeat_{repeat_index:02d}_seed_{seed}"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False, default=str)


def base_experiment_config() -> dict[str, Any]:
    return {
        "reward_type": REWARD_TYPE,
        "reward_params": REWARD_PARAMS,
        "rolling_window_mode": ROLLING_WINDOW_MODE,
        "rolling_start_date": ROLLING_START_DATE,
        "rolling_end_date": ROLLING_END_DATE,
        "train_years": TRAIN_YEARS,
        "test_years": TEST_YEARS,
        "shift_years": SHIFT_YEARS,
        "repeat_experiments": REPEAT_EXPERIMENTS,
        "test_predict_deterministic": TEST_PREDICT_DETERMINISTIC,
        "run_continuous_5y_test": RUN_CONTINUOUS_5Y_TEST,
        "auto_compare_continuous_latest": AUTO_COMPARE_CONTINUOUS_LATEST,
        "base_random_seed": BASE_RANDOM_SEED,
        "run_data": RUN_DATA,
        "run_train": RUN_TRAIN,
        "run_backtest": RUN_BACKTEST,
        "if_using_a2c": IF_USING_A2C,
        "if_using_ddpg": IF_USING_DDPG,
        "if_using_ppo": IF_USING_PPO,
        "if_using_td3": IF_USING_TD3,
        "if_using_sac": IF_USING_SAC,
        "a2c_total_timesteps": A2C_TOTAL_TIMESTEPS,
        "ddpg_total_timesteps": DDPG_TOTAL_TIMESTEPS,
        "ppo_total_timesteps": PPO_TOTAL_TIMESTEPS,
        "td3_total_timesteps": TD3_TOTAL_TIMESTEPS,
        "sac_total_timesteps": SAC_TOTAL_TIMESTEPS,
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# =====================================================
# 3. Environment variables for child scripts
# =====================================================

def build_child_environment(
    experiment_dir: Path,
    train_start_date: str,
    train_end_date: str,
    test_start_date: str,
    test_end_date: str,
    trade_start_date: str,
    trade_end_date: str,
    random_seed: int,
    rolling_root_dir: Path | None = None,
    window_dir: Path | None = None,
    repeat_index: int | None = None,
    window_id: int | None = None,
) -> dict[str, str]:
    env = os.environ.copy()

    data_dir = experiment_dir / "data"
    model_dir = experiment_dir / "trained_models"
    log_dir = experiment_dir / "training_logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env["REWARD_TYPE"] = REWARD_TYPE
    env["REWARD_PARAMS"] = json.dumps(REWARD_PARAMS, ensure_ascii=False)

    env["TRAIN_START_DATE"] = train_start_date
    env["TRAIN_END_DATE"] = train_end_date
    env["TEST_START_DATE"] = test_start_date
    env["TEST_END_DATE"] = test_end_date
    env["TRADE_START_DATE"] = trade_start_date
    env["TRADE_END_DATE"] = trade_end_date

    env["EXPERIMENT_ROOT"] = str(EXPERIMENT_ROOT)
    env["EXPERIMENT_DIR"] = str(experiment_dir)
    env["ROLLING_ROOT_DIR"] = str(rolling_root_dir or "")
    env["WINDOW_DIR"] = str(window_dir or "")
    env["WINDOW_ID"] = str(window_id or "")
    env["REPEAT_INDEX"] = str(repeat_index or "")
    env["RANDOM_SEED"] = str(random_seed)
    env["PREDICT_DETERMINISTIC"] = str(TEST_PREDICT_DETERMINISTIC)

    env["TRAIN_DATA_PATH"] = str(data_dir / "train_data.csv")
    env["TRADE_DATA_PATH"] = str(data_dir / "trade_data.csv")
    env["TRAINED_MODEL_DIR"] = str(model_dir)
    env["RESULTS_DIR"] = str(log_dir)

    env["IF_USING_A2C"] = str(IF_USING_A2C)
    env["IF_USING_DDPG"] = str(IF_USING_DDPG)
    env["IF_USING_PPO"] = str(IF_USING_PPO)
    env["IF_USING_TD3"] = str(IF_USING_TD3)
    env["IF_USING_SAC"] = str(IF_USING_SAC)

    env["A2C_TOTAL_TIMESTEPS"] = str(A2C_TOTAL_TIMESTEPS)
    env["DDPG_TOTAL_TIMESTEPS"] = str(DDPG_TOTAL_TIMESTEPS)
    env["PPO_TOTAL_TIMESTEPS"] = str(PPO_TOTAL_TIMESTEPS)
    env["TD3_TOTAL_TIMESTEPS"] = str(TD3_TOTAL_TIMESTEPS)
    env["SAC_TOTAL_TIMESTEPS"] = str(SAC_TOTAL_TIMESTEPS)

    project_root = Path(__file__).resolve().parents[1]
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(project_root) + (os.pathsep + old_pythonpath if old_pythonpath else "")

    return env


# =====================================================
# 4. 執行子程式
# =====================================================

def run_script(script_name: str, env: dict[str, str], project_root: Path) -> None:
    print("========================================")
    print(f"正在執行: {script_name}")
    print("========================================")

    try:
        subprocess.run(
            [sys.executable, script_name],
            check=True,
            text=True,
            env=env,
            cwd=project_root,
        )
        print(f"✓ {script_name} 執行成功！\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} 執行失敗！錯誤碼: {e.returncode}")
        print("停止執行後續檔案。")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ 找不到檔案: {script_name}，請檢查路徑是否正確。")
        print("停止執行後續檔案。")
        sys.exit(1)


def scripts_to_run() -> list[str]:
    scripts: list[str] = []
    if RUN_DATA:
        scripts.append("examples/FinRL_StockTrading_2026_1_data.py")
    if RUN_TRAIN:
        scripts.append("examples/FinRL_StockTrading_2026_2_train.py")
    if RUN_BACKTEST:
        scripts.append("examples/FinRL_StockTrading_2026_3_Backtest.py")
    return scripts


def collect_backtest_summary(experiment_dir: Path) -> dict[str, Any]:
    rows = list(experiment_dir.glob("backtest_summary_*.csv"))
    if not rows:
        return {}
    try:
        import pandas as pd

        df = pd.read_csv(rows[0])
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception as exc:
        return {"summary_read_error": str(exc)}


def write_rolling_summaries(rolling_root: Path, summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        return
    import pandas as pd

    summary_df = pd.DataFrame(summary_rows)
    summary_path = rolling_root / f"summary_{REWARD_TYPE}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    numeric_cols = [
        col
        for col in ["final_ppo", "ppo_return", "ppo_max_drawdown", "final_dji"]
        if col in summary_df.columns
    ]
    if numeric_cols:
        agg = (
            summary_df.groupby(["window_id", "train_period", "test_period"])[numeric_cols]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        agg.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in agg.columns]
        agg_path = rolling_root / f"summary_window_aggregate_{REWARD_TYPE}.csv"
        agg.to_csv(agg_path, index=False, encoding="utf-8-sig")

    print(f"Rolling summary saved to: {summary_path}")


def _read_backtest_csv(csv_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        first_col = df.columns[0]
        if first_col != "date":
            df = df.rename(columns={first_col: "date"})
        df = keep_valid_date_rows(df)
        if df.empty:
            return None
        value_cols = [c for c in df.columns if c not in {"date"}]
        if value_cols and not df[value_cols].apply(pd.to_numeric, errors="coerce").notna().any().any():
            return None
        return df
    except Exception as exc:
        print(f"讀取 backtest csv 失敗: {csv_path} -> {exc}")
        return None


def _normalize_series(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    first_valid = series.dropna()
    if first_valid.empty or first_valid.iloc[0] == 0:
        return series
    return series / first_valid.iloc[0]


def fill_dji_mean_norm_if_missing(df: pd.DataFrame, initial_amount: float = 1_000_000.0) -> pd.DataFrame:
    """Fill dji_mean_norm from real dates when older source CSVs have empty DJI data."""
    if "date" not in df.columns:
        return df
    if "dji_mean_norm" in df.columns and pd.to_numeric(df["dji_mean_norm"], errors="coerce").notna().any():
        return df

    out = df.copy()
    dji_series = build_dji_baseline(out["date"].tolist(), initial_amount=initial_amount)
    if dji_series.notna().any():
        dji_series = pd.to_numeric(
            dji_series.reindex(pd.Index(out["date"].astype(str), name="date")),
            errors="coerce",
        )
        out["dji_mean_norm"] = _normalize_series(dji_series).to_numpy()
    return out


def plot_window_annual_reset_aggregate(window_dir: Path, reward_type: str) -> None:
    """Create the per-window Annual Reset folder used by reward comparison.

    New folder layout:
      repeat_01_seed_43/ ...      -> each repeat contains one trained model and one backtest result
      annual_reset_v?/             -> aggregate output, must NOT be read as input

    Output folder:
      annual_reset_v?/
        account_value.csv        -> ppo mean/min/max + dji mean, normalized columns
        backtest_result.csv      -> same content, convenient for plotting
        backtest_result_annual_reset_*.csv/png
        image_source_manifest.csv -> source CSVs used by the Annual Reset plot
    """
    repeat_dirs = sorted([p for p in window_dir.iterdir() if is_backtest_repeat_dir(p)])
    if not repeat_dirs:
        return

    run_dfs: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    trade_start_date = None
    trade_end_date = None
    date_suffix = datetime.now().strftime("%m%d")

    for repeat_dir in repeat_dirs:
        csv_path = select_backtest_result_csv(repeat_dir, reward_type)
        if csv_path is None:
            continue
        df = _read_backtest_csv(csv_path)
        if df is None or "ppo" not in df.columns:
            continue
        source_rows.append(
            {
                "plot": "backtest_result_annual_reset.png",
                "source_role": "single_backtest_repeat",
                "source_repeat_dir": str(repeat_dir),
                "source_csv": str(csv_path),
                "rows": len(df),
                "columns": ",".join(df.columns),
            }
        )

        if trade_start_date is None or trade_end_date is None:
            name = csv_path.stem
            prefix = f"backtest_result_{reward_type}_"
            if name.startswith(prefix):
                tail = name[len(prefix):]
                if "_to_" in tail:
                    try:
                        left, right = tail.rsplit("_", 1)
                        start_str, end_str = left.split("_to_")
                        trade_start_date = start_str
                        trade_end_date = end_str
                        date_suffix = right
                    except Exception:
                        pass

        keep_cols = [c for c in ["date", "ppo", "dji"] if c in df.columns]
        run_dfs.append(df[keep_cols].copy())

    if not run_dfs:
        print(f"{window_dir} 沒有可用的 backtest_result csv，略過 aggregate plot。")
        return

    base_dates = run_dfs[0]["date"].astype(str)
    ppo_runs = []
    dji_runs = []

    for run_idx, df in enumerate(run_dfs, start=1):
        aligned = df.set_index("date").reindex(base_dates).reset_index()
        if "ppo" in aligned.columns:
            ppo_runs.append(pd.to_numeric(aligned["ppo"], errors="coerce").rename(f"ppo_run_{run_idx:02d}"))
        if "dji" in aligned.columns:
            dji_runs.append(pd.to_numeric(aligned["dji"], errors="coerce").rename(f"dji_run_{run_idx:02d}"))

    ppo_df = pd.concat(ppo_runs, axis=1) if ppo_runs else pd.DataFrame(index=range(len(base_dates)))
    dji_df = pd.concat(dji_runs, axis=1) if dji_runs else pd.DataFrame(index=range(len(base_dates)))

    aggregate_df = pd.DataFrame({"date": base_dates})
    if not ppo_df.empty:
        aggregate_df["ppo_mean"] = ppo_df.mean(axis=1)
        aggregate_df["ppo_min"] = ppo_df.min(axis=1)
        aggregate_df["ppo_max"] = ppo_df.max(axis=1)
        aggregate_df["ppo_mean_norm"] = _normalize_series(aggregate_df["ppo_mean"])
        aggregate_df["ppo_min_norm"] = _normalize_series(aggregate_df["ppo_min"])
        aggregate_df["ppo_max_norm"] = _normalize_series(aggregate_df["ppo_max"])
    if not dji_df.empty:
        aggregate_df["dji_mean"] = dji_df.mean(axis=1)
        aggregate_df["dji_mean_norm"] = _normalize_series(aggregate_df["dji_mean"])

    if "dji_mean_norm" not in aggregate_df.columns or not aggregate_df["dji_mean_norm"].notna().any():
        dji_series = build_dji_baseline(base_dates.tolist(), initial_amount=1_000_000.0)
        if dji_series.notna().any():
            dji_series = pd.to_numeric(
                dji_series.reindex(pd.Index(base_dates.astype(str), name="date")),
                errors="coerce",
            )
            aggregate_df["dji_mean"] = dji_series.to_numpy()
            aggregate_df["dji_mean_norm"] = _normalize_series(aggregate_df["dji_mean"])

    trade_start_date = trade_start_date or str(base_dates.iloc[0])
    trade_end_date = trade_end_date or str(base_dates.iloc[-1])

    mean_dir = window_dir / f"annual_reset_{reward_type}"
    mean_dir.mkdir(parents=True, exist_ok=True)
    account_value_path = mean_dir / "account_value.csv"
    backtest_result_path = mean_dir / "backtest_result.csv"
    aggregate_df.to_csv(account_value_path, index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(backtest_result_path, index=False, encoding="utf-8-sig")

    aggregate_csv_path = mean_dir / f"backtest_result_annual_reset_{reward_type}_{trade_start_date}_to_{trade_end_date}_{date_suffix}.csv"
    aggregate_df.to_csv(aggregate_csv_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(len(aggregate_df))
    reward_name = reward_display_name(reward_type)

    if {"ppo_mean_norm", "ppo_min_norm", "ppo_max_norm"}.issubset(aggregate_df.columns):
        ppo_mean = pd.to_numeric(aggregate_df["ppo_mean_norm"], errors="coerce")
        ppo_min = pd.to_numeric(aggregate_df["ppo_min_norm"], errors="coerce")
        ppo_max = pd.to_numeric(aggregate_df["ppo_max_norm"], errors="coerce")
        ret = (ppo_mean.dropna().iloc[-1] - 1) * 100 if ppo_mean.notna().any() else 0
        ax.fill_between(x, ppo_min, ppo_max, color="blue", alpha=0.16)
        ax.plot(x, ppo_mean, color="blue", linewidth=2.2, label=f"{reward_name} PPO mean ({ret:.2f}%)")

    if "dji_mean_norm" in aggregate_df.columns:
        dji_mean = pd.to_numeric(aggregate_df["dji_mean_norm"], errors="coerce")
        ret = (dji_mean.dropna().iloc[-1] - 1) * 100 if dji_mean.notna().any() else 0
        ax.plot(x, dji_mean, color="tab:green", label=f"DJI ({ret:.2f}%)")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.1)
    set_month_xticklabels(ax, aggregate_df["date"].tolist(), max_xticks=12)

    ax.set_title(f"Equity Curve - {reward_name} (initial = 1.0)")
    set_axis_labels_horizontal(ax, "Date", "Asset Allocation (initial = 1.0)", ylabel_pad=85)
    ax.legend()
    plt.tight_layout()

    aggregate_plot_path = mean_dir / f"backtest_result_annual_reset_{reward_type}_{trade_start_date}_to_{trade_end_date}_{date_suffix}.png"
    canonical_plot_path = mean_dir / "backtest_result_annual_reset.png"
    plt.savefig(aggregate_plot_path, dpi=150, bbox_inches="tight")
    plt.savefig(canonical_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    source_manifest = pd.DataFrame(source_rows)
    if not source_manifest.empty:
        source_manifest["output_csv"] = str(aggregate_csv_path)
        source_manifest["output_png"] = str(aggregate_plot_path)
        source_manifest["canonical_output_png"] = str(canonical_plot_path)
        source_manifest.to_csv(mean_dir / "image_source_manifest.csv", index=False, encoding="utf-8-sig")

    print(f"Window Annual Reset folder saved to: {mean_dir}")
    print(f"Annual Reset source repeats: {len(source_rows)}")


def _find_latest_rolling_root(reward_type: str) -> Path | None:
    reward_root = Path(EXPERIMENT_ROOT) / f"reward_{reward_type}"
    if not reward_root.exists():
        return None
    candidates = [p for p in reward_root.iterdir() if p.is_dir() and p.name.startswith("rolling_")]
    if not candidates:
        # backward compatibility for older output layout: reward_v?/rolling_window/rolling_...
        legacy_root = reward_root / "rolling_window"
        if legacy_root.exists():
            candidates = [p for p in legacy_root.iterdir() if p.is_dir() and p.name.startswith("rolling_")]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _collect_annual_reset_series(rolling_root: Path, reward_type: str) -> pd.DataFrame | None:
    frames = []
    source_csvs: list[str] = []
    for csv_path in sorted(rolling_root.glob(f"window_*/annual_reset_{reward_type}/backtest_result.csv")):
        df = pd.read_csv(csv_path)
        if not df.empty:
            source_csvs.append(str(csv_path))
            frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out = keep_valid_date_rows(out)
    out = out.drop_duplicates(subset=["date"], keep="last")
    out.attrs["source_csvs"] = source_csvs
    return out


def write_compare_source_manifest(
    compare_dir: Path,
    root_v0: Path,
    root_target: Path,
    target_reward_type: str,
    df_v0: pd.DataFrame,
    df_target: pd.DataFrame,
) -> None:
    """Record which Annual Reset CSV files were used by compare_portfolio_value.png."""
    rows: list[dict[str, Any]] = []
    for label, root, df in [
        ("v0", root_v0, df_v0),
        (target_reward_type, root_target, df_target),
    ]:
        for source_csv in df.attrs.get("source_csvs", []):
            rows.append(
                {
                    "plot": "compare_portfolio_value.png",
                    "source_role": f"annual_reset_{label}",
                    "rolling_root": str(root),
                    "source_csv": source_csv,
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(compare_dir / "image_source_manifest.csv", index=False, encoding="utf-8-sig")


def compare_latest_v0_v1() -> None:
    """Backward-compatible wrapper: compare latest v0 with latest current REWARD_TYPE."""
    compare_latest_v0_target(REWARD_TYPE)


def compare_latest_v0_target(target_reward_type: str) -> None:
    """Create Experence_result/compare_v0_<target> using latest Annual Reset outputs.

    The baseline is always v0. The target side follows the current REWARD_TYPE,
    so running this script with REWARD_TYPE="v2" automatically creates v0 vs v2.
    """
    target_reward_type = str(target_reward_type).strip().lower()
    if not target_reward_type or target_reward_type in {"v0", "original", "finrl_original"}:
        print("compare_v0_target skipped: 目前 REWARD_TYPE 是 v0，沒有非 v0 版本可比較。")
        return

    root_v0 = _find_latest_rolling_root("v0")
    root_target = _find_latest_rolling_root(target_reward_type)
    if root_v0 is None or root_target is None:
        print(f"compare_v0_{target_reward_type} skipped: 找不到 v0 或 {target_reward_type} 的 rolling 結果。")
        return

    df_v0 = _collect_annual_reset_series(root_v0, "v0")
    df_target = _collect_annual_reset_series(root_target, target_reward_type)
    if df_v0 is None or df_target is None:
        print(f"compare_v0_{target_reward_type} skipped: v0/{target_reward_type} 尚未產生 annual_reset 資料夾。")
        return

    compare_dir = Path(EXPERIMENT_ROOT) / f"compare_v0_{target_reward_type}"
    compare_dir.mkdir(parents=True, exist_ok=True)

    target_cols = [c for c in ["date", "ppo_mean_norm", "ppo_min_norm", "ppo_max_norm", "dji_mean_norm"] if c in df_target.columns]
    merged = pd.merge(
        df_v0[[c for c in ["date", "ppo_mean_norm", "ppo_min_norm", "ppo_max_norm", "dji_mean_norm"] if c in df_v0.columns]],
        df_target[target_cols],
        on="date",
        how="outer",
        suffixes=("_v0", f"_{target_reward_type}"),
    ).sort_values("date").reset_index(drop=True)
    merged = fill_dji_mean_norm_if_missing(merged)
    merged.to_csv(compare_dir / "compare_metrics.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(merged))
    baseline_name = reward_display_name("v0")
    target_name = reward_display_name(target_reward_type)

    def add_strategy(prefix: str, color: str, label: str) -> None:
        mean_col = f"ppo_mean_norm_{prefix}"
        min_col = f"ppo_min_norm_{prefix}"
        max_col = f"ppo_max_norm_{prefix}"
        # If pandas did not suffix because the column did not overlap, support the plain column too.
        if mean_col not in merged.columns and prefix == target_reward_type and "ppo_mean_norm" in merged.columns:
            mean_col = "ppo_mean_norm"
            min_col = "ppo_min_norm"
            max_col = "ppo_max_norm"
        if mean_col not in merged.columns:
            return
        mean = pd.to_numeric(merged[mean_col], errors="coerce")
        min_v = pd.to_numeric(merged[min_col], errors="coerce") if min_col in merged.columns else mean
        max_v = pd.to_numeric(merged[max_col], errors="coerce") if max_col in merged.columns else mean
        valid = mean.dropna()
        ret = (valid.iloc[-1] - 1) * 100 if not valid.empty else 0
        ax.fill_between(x, min_v, max_v, color=color, alpha=0.14)
        ax.plot(x, mean, color=color, linewidth=2.2, label=f"{label} mean ({ret:.2f}%)")

    add_strategy("v0", "blue", baseline_name)
    add_strategy(target_reward_type, "red", target_name)

    dji_candidates = ["dji_mean_norm", "dji_mean_norm_v0", f"dji_mean_norm_{target_reward_type}"]
    dji = next((pd.to_numeric(merged[c], errors="coerce") for c in dji_candidates if c in merged.columns), pd.Series(dtype=float))
    if not dji.empty and dji.notna().any():
        valid = dji.dropna()
        ret = (valid.iloc[-1] - 1) * 100
        ax.plot(x, dji, color="tab:green", linewidth=1.8, label=f"DJI ({ret:.2f}%)")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.1)
    set_month_xticklabels(ax, merged["date"].tolist(), max_xticks=14)
    ax.set_title(f"{baseline_name} vs {target_name} Equity Curve")
    set_axis_labels_horizontal(ax, "Date", "Asset Allocation (initial = 1.0)", ylabel_pad=85)
    ax.legend()
    plt.tight_layout()
    fig.savefig(compare_dir / "compare_portfolio_value.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    cleanup_unused_compare_images(compare_dir)
    write_compare_source_manifest(compare_dir, root_v0, root_target, target_reward_type, df_v0, df_target)

    config_payload = {
        "compare_type": f"v0_vs_{target_reward_type}",
        "v0_rolling_root": str(root_v0),
        "target_reward_type": target_reward_type,
        "target_rolling_root": str(root_target),
        "compare_dir": str(compare_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(compare_dir / "compare_config.json", config_payload)
    print(f"compare_v0_{target_reward_type} saved to: {compare_dir}")


# =====================================================
# 5. Continuous 5-year test helpers
# =====================================================

CONTINUOUS_TEST_DIR_NAME = "continuous_5y_test"


def _load_trade_data_for_continuous(trade_data_path: Path) -> pd.DataFrame:
    """Load trade_data.csv with the same index format used by the backtest script."""
    trade = pd.read_csv(trade_data_path)
    trade = trade.set_index(trade.columns[0])
    trade.index.names = [""]
    return trade


def _get_first_day_tic_order(trade: pd.DataFrame) -> list[str]:
    """Return ticker order for the first trading day in a FinRL trade dataframe."""
    first_day = trade.index.unique()[0]
    first_day_rows = trade.loc[first_day]
    if isinstance(first_day_rows, pd.Series):
        return [str(first_day_rows["tic"])]
    return [str(x) for x in first_day_rows["tic"].tolist()]


def _align_shares_to_tic_order(shares_by_tic: dict[str, float], tic_order: list[str]) -> list[float]:
    """Align carried shares to the next window's ticker order."""
    return [float(shares_by_tic.get(str(tic), 0.0)) for tic in tic_order]


def _make_previous_state_for_continuous(
    cash: float,
    aligned_shares: list[float],
    stock_dimension: int,
) -> list[float]:
    """Build a previous_state whose holdings can be reused by StockTradingEnv."""
    return (
        [float(cash)]
        + [0.0] * stock_dimension
        + [float(x) for x in aligned_shares]
        + [0.0] * (len(config.INDICATORS) * stock_dimension)
    )


def _run_continuous_window_ppo(
    trade_data_path: Path,
    trained_model_dir: Path,
    carried_cash: float,
    carried_shares_by_tic: dict[str, float],
    reward_type: str,
    reward_params: dict[str, Any],
    use_carried_state: bool,
    random_seed: int,
    predict_deterministic: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict[str, float], list[str]]:
    """Run one test window using carried cash/holdings as the initial portfolio."""
    from stable_baselines3 import PPO
    from finrl.config import INDICATORS
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    set_random_seed(random_seed)

    trade = _load_trade_data_for_continuous(trade_data_path)
    stock_dimension = len(trade.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    tic_order = _get_first_day_tic_order(trade)
    aligned_shares = _align_shares_to_tic_order(carried_shares_by_tic, tic_order)

    model_path = trained_model_dir / "agent_ppo"
    if not model_path.with_suffix(".zip").exists() and not model_path.exists():
        raise FileNotFoundError(f"找不到 PPO model: {model_path}.zip")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    previous_state = _make_previous_state_for_continuous(carried_cash, aligned_shares, stock_dimension)

    env_kwargs = {
        "hmax": 100,
        "initial_amount": float(carried_cash),
        "num_stock_shares": aligned_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4 if normalize_reward_type_for_continuous(reward_type) == "v0" else 1.0,
        "initial": not use_carried_state,
        "previous_state": previous_state,
    }

    env = StockTradingEnv(
        df=trade,
        turbulence_threshold=70,
        risk_indicator_col="vix",
        reward_type=reward_type,
        reward_params=reward_params,
        **env_kwargs,
    )

    model = PPO.load(model_path)
    test_env, test_obs = env.get_sb_env()
    test_env.reset()
    max_steps = len(env.df.index.unique()) - 1

    for i in range(len(env.df.index.unique())):
        action, _states = model.predict(test_obs, deterministic=predict_deterministic)
        test_obs, rewards_step, dones, info = test_env.step(action)
        if dones[0] or i >= max_steps - 1:
            break

    account_value = test_env.env_method(method_name="save_asset_memory")[0]
    holdings = test_env.env_method(method_name="save_holdings_memory")[0]

    if holdings.empty:
        raise ValueError(f"{trade_data_path} 產生的 holdings 記錄是空的。")

    last_holdings = holdings.iloc[-1]
    next_cash = float(last_holdings["cash"])
    next_shares_by_tic = {
        str(tic): float(last_holdings[tic])
        for tic in tic_order
        if tic in holdings.columns
    }
    return account_value, holdings, next_cash, next_shares_by_tic, tic_order


def normalize_reward_type_for_continuous(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("reward_"):
        text = text[len("reward_"):]
    aliases = {
        "original": "v0",
        "finrl_original": "v0",
        "our_reward_v1": "v1",
        "future_day_return": "v2",
        "future_avg_return": "v3",
        "future_avg_delta_action_return": "v4",
        "future_avg_delta_action_reward": "v4",
    }
    return aliases.get(text, text)


def run_continuous_5y_test(
    rolling_root: Path,
    windows: list[dict[str, Any]],
    reward_type: str,
    initial_cash: float = 1_000_000.0,
) -> None:
    """Run repeated continuous tests across all rolling test windows.

    Each repeat uses the model trained in the matching repeat folder of each
    window. The portfolio state is reset at the beginning of that repeat.
    Within one repeat, the final cash and stock holdings from window t become
    the initial portfolio of window t+1.
    """
    reward_type = normalize_reward_type_for_continuous(reward_type)
    out_dir = rolling_root / CONTINUOUS_TEST_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    repeat_results: list[pd.DataFrame] = []
    repeat_summary_rows: list[dict[str, Any]] = []
    window_summary_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    date_suffix = datetime.now().strftime("%m%d")

    for repeat_index in range(1, REPEAT_EXPERIMENTS + 1):
        test_seed = BASE_RANDOM_SEED + repeat_index
        repeat_dir = out_dir / repeat_folder_name(repeat_index, test_seed)
        repeat_dir.mkdir(parents=True, exist_ok=True)

        carried_cash = float(initial_cash)
        carried_shares_by_tic: dict[str, float] = {}
        account_frames: list[pd.DataFrame] = []

        print("-" * 80)
        print(
            f"Continuous 5-year test repeat {repeat_index:02d}, "
            f"test/model seed {test_seed}"
        )
        print(f"Continuous Repeat Dir: {repeat_dir}")
        print("-" * 80)

        for idx, window in enumerate(windows):
            window_dir = rolling_root / window_folder_name(window)
            experiment_dir = window_dir / repeat_folder_name(repeat_index, test_seed)
            trade_data_path = experiment_dir / "data" / "trade_data.csv"
            trained_model_dir = experiment_dir / "trained_models"

            if not trade_data_path.exists():
                raise FileNotFoundError(f"找不到 trade_data.csv: {trade_data_path}")
            if not trained_model_dir.exists():
                raise FileNotFoundError(f"找不到 trained_models: {trained_model_dir}")

            start_cash = carried_cash
            start_shares = dict(carried_shares_by_tic)
            window_seed = test_seed + int(window["window_id"]) * 1000

            account_value, holdings, carried_cash, carried_shares_by_tic, tic_order = _run_continuous_window_ppo(
                trade_data_path=trade_data_path,
                trained_model_dir=trained_model_dir,
                carried_cash=carried_cash,
                carried_shares_by_tic=carried_shares_by_tic,
                reward_type=reward_type,
                reward_params=REWARD_PARAMS_BY_TYPE.get(reward_type, REWARD_PARAMS),
                use_carried_state=idx > 0,
                random_seed=window_seed,
                predict_deterministic=TEST_PREDICT_DETERMINISTIC,
            )

            account_value = account_value.copy()
            account_value["date"] = account_value["date"].astype(str)
            account_value["repeat_index"] = repeat_index
            account_value["random_seed"] = test_seed
            account_value["window_seed"] = window_seed
            account_value["window_id"] = window["window_id"]
            account_value["window_folder"] = window_dir.name
            account_value["source_trade_data"] = str(trade_data_path)
            account_value["source_model_dir"] = str(trained_model_dir)
            account_frames.append(account_value)

            window_holdings_path = repeat_dir / f"holdings_window_{window['window_id']:02d}_{reward_type}.csv"
            holdings.to_csv(window_holdings_path, index=False, encoding="utf-8-sig")

            ppo = pd.to_numeric(account_value["account_value"], errors="coerce").dropna()
            window_summary_rows.append(
                {
                    "repeat_index": repeat_index,
                    "random_seed": test_seed,
                    "window_seed": window_seed,
                    "window_id": window["window_id"],
                    "window_folder": window_dir.name,
                    "test_start_date": window["trade_start_date"],
                    "test_end_date": window["trade_end_date"],
                    "start_cash": start_cash,
                    "end_cash": carried_cash,
                    "start_position_count": sum(1 for v in start_shares.values() if abs(v) > 1e-9),
                    "end_position_count": sum(1 for v in carried_shares_by_tic.values() if abs(v) > 1e-9),
                    "start_account_value": float(ppo.iloc[0]) if not ppo.empty else np.nan,
                    "end_account_value": float(ppo.iloc[-1]) if not ppo.empty else np.nan,
                    "window_return": float(ppo.iloc[-1] / ppo.iloc[0] - 1) if len(ppo) >= 2 and ppo.iloc[0] != 0 else np.nan,
                    "trade_data": str(trade_data_path),
                    "trained_model_dir": str(trained_model_dir),
                    "holdings_csv": str(window_holdings_path),
                    "predict_deterministic": TEST_PREDICT_DETERMINISTIC,
                }
            )
            source_rows.append(
                {
                    "plot": "continuous_portfolio_value.png",
                    "source_role": "continuous_window_test",
                    "repeat_index": repeat_index,
                    "random_seed": test_seed,
                    "window_seed": window_seed,
                    "window_id": window["window_id"],
                    "window_folder": window_dir.name,
                    "trade_data": str(trade_data_path),
                    "trained_model_dir": str(trained_model_dir),
                    "holdings_csv": str(window_holdings_path),
                    "ticker_order": ",".join(tic_order),
                    "predict_deterministic": TEST_PREDICT_DETERMINISTIC,
                }
            )

        if not account_frames:
            continue

        repeat_result = pd.concat(account_frames, ignore_index=True)
        repeat_result = repeat_result.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        repeat_result = repeat_result.rename(columns={"account_value": "ppo_continuous"})
        repeat_result["ppo_continuous_norm"] = _normalize_series(repeat_result["ppo_continuous"])
        repeat_results.append(repeat_result[["date", "ppo_continuous", "ppo_continuous_norm"]].copy())

        start_date = str(repeat_result["date"].iloc[0])
        end_date = str(repeat_result["date"].iloc[-1])
        repeat_csv = repeat_dir / "continuous_portfolio_value.csv"
        repeat_dated_csv = repeat_dir / f"continuous_portfolio_value_{reward_type}_{start_date}_to_{end_date}_{date_suffix}.csv"
        repeat_result.to_csv(repeat_csv, index=False, encoding="utf-8-sig")
        repeat_result.to_csv(repeat_dated_csv, index=False, encoding="utf-8-sig")

        valid = pd.to_numeric(repeat_result["ppo_continuous_norm"], errors="coerce").dropna()
        repeat_summary_rows.append(
            {
                "repeat_index": repeat_index,
                "random_seed": test_seed,
                "start_date": start_date,
                "end_date": end_date,
                "initial_cash": initial_cash,
                "final_normalized_value": float(valid.iloc[-1]) if not valid.empty else np.nan,
                "return_percent": float((valid.iloc[-1] - 1) * 100) if not valid.empty else np.nan,
                "repeat_csv": str(repeat_csv),
                "predict_deterministic": TEST_PREDICT_DETERMINISTIC,
            }
        )

    if not repeat_results:
        print(f"{rolling_root} 沒有可用 window，略過 continuous 5y test。")
        return

    base_dates = repeat_results[0]["date"].astype(str)
    ppo_runs = []
    ppo_norm_runs = []
    for run_idx, df in enumerate(repeat_results, start=1):
        aligned = df.set_index("date").reindex(base_dates).reset_index()
        ppo_runs.append(pd.to_numeric(aligned["ppo_continuous"], errors="coerce").rename(f"ppo_run_{run_idx:02d}"))
        ppo_norm_runs.append(pd.to_numeric(aligned["ppo_continuous_norm"], errors="coerce").rename(f"ppo_norm_run_{run_idx:02d}"))

    ppo_df = pd.concat(ppo_runs, axis=1)
    ppo_norm_df = pd.concat(ppo_norm_runs, axis=1)

    result = pd.DataFrame({"date": base_dates})
    result["ppo_continuous"] = ppo_df.mean(axis=1)
    result["ppo_continuous_mean"] = result["ppo_continuous"]
    result["ppo_continuous_min"] = ppo_df.min(axis=1)
    result["ppo_continuous_max"] = ppo_df.max(axis=1)
    result["ppo_continuous_norm"] = _normalize_series(result["ppo_continuous"])
    result["ppo_continuous_mean_norm"] = result["ppo_continuous_norm"]
    result["ppo_continuous_min_norm"] = ppo_norm_df.min(axis=1)
    result["ppo_continuous_max_norm"] = ppo_norm_df.max(axis=1)

    dji_series = build_dji_baseline(result["date"].tolist(), initial_amount=initial_cash)
    if dji_series.notna().any():
        result["dji"] = pd.to_numeric(dji_series.reindex(pd.Index(result["date"].astype(str), name="date")), errors="coerce").to_numpy()
        result["dji_mean_norm"] = _normalize_series(result["dji"])

    start_date = str(result["date"].iloc[0])
    end_date = str(result["date"].iloc[-1])

    canonical_csv = out_dir / "continuous_portfolio_value.csv"
    dated_csv = out_dir / f"continuous_portfolio_value_{reward_type}_{start_date}_to_{end_date}_{date_suffix}.csv"
    result.to_csv(canonical_csv, index=False, encoding="utf-8-sig")
    result.to_csv(dated_csv, index=False, encoding="utf-8-sig")

    repeat_summary_path = out_dir / "continuous_repeat_summary.csv"
    window_summary_path = out_dir / "continuous_window_summary.csv"
    pd.DataFrame(repeat_summary_rows).to_csv(repeat_summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(window_summary_rows).to_csv(window_summary_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(result))
    y = pd.to_numeric(result["ppo_continuous_norm"], errors="coerce")
    y_min = pd.to_numeric(result["ppo_continuous_min_norm"], errors="coerce")
    y_max = pd.to_numeric(result["ppo_continuous_max_norm"], errors="coerce")
    valid = y.dropna()
    ret = (valid.iloc[-1] - 1) * 100 if not valid.empty else 0
    reward_name = reward_display_name(reward_type)
    ax.fill_between(x, y_min, y_max, color="blue", alpha=0.16)
    ax.plot(x, y, color="blue", linewidth=2.3, label=f"{reward_name} PPO continuous mean ({ret:.2f}%)")
    if "dji_mean_norm" in result.columns:
        dji_y = pd.to_numeric(result["dji_mean_norm"], errors="coerce")
        dji_valid = dji_y.dropna()
        dji_ret = (dji_valid.iloc[-1] - 1) * 100 if not dji_valid.empty else 0
        ax.plot(x, dji_y, color="tab:green", linewidth=1.8, label=f"DJI ({dji_ret:.2f}%)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.1)
    set_month_xticklabels(ax, result["date"].tolist(), max_xticks=14)
    ax.set_title(f"Continuous 5-Year Test Equity Curve - {reward_name}")
    set_axis_labels_horizontal(ax, "Date", "Asset Allocation (initial = 1.0)", ylabel_pad=95)
    ax.legend()
    plt.tight_layout()
    dated_png = out_dir / f"continuous_portfolio_value_{reward_type}_{start_date}_to_{end_date}_{date_suffix}.png"
    canonical_png = out_dir / "continuous_portfolio_value.png"
    fig.savefig(dated_png, dpi=150, bbox_inches="tight")
    fig.savefig(canonical_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    for row in source_rows:
        row["output_csv"] = str(canonical_csv)
        row["output_png"] = str(canonical_png)
        row["repeat_summary_csv"] = str(repeat_summary_path)
        row["window_summary_csv"] = str(window_summary_path)
    pd.DataFrame(source_rows).to_csv(out_dir / "image_source_manifest.csv", index=False, encoding="utf-8-sig")

    config_payload = {
        "reward_type": reward_type,
        "rolling_root": str(rolling_root),
        "output_dir": str(out_dir),
        "initial_cash": initial_cash,
        "repeat_experiments": REPEAT_EXPERIMENTS,
        "test_predict_deterministic": TEST_PREDICT_DETERMINISTIC,
        "start_date": start_date,
        "end_date": end_date,
        "carry_state": "cash_and_stock_holdings_from_previous_window",
        "annual_reset_source": "matching_repeat_models",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(out_dir / "continuous_config.json", config_payload)
    print(f"Continuous 5-year Annual Reset test saved to: {out_dir}")

def _collect_continuous_5y_series(rolling_root: Path, reward_type: str) -> pd.DataFrame | None:
    csv_path = rolling_root / CONTINUOUS_TEST_DIR_NAME / "continuous_portfolio_value.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty or "date" not in df.columns or "ppo_continuous_norm" not in df.columns:
        return None
    keep_cols = ["date", "ppo_continuous_norm"]
    if "dji_mean_norm" in df.columns:
        keep_cols.append("dji_mean_norm")
    out = df[keep_cols].copy()
    out = keep_valid_date_rows(out)
    out = out.rename(columns={"ppo_continuous_norm": f"ppo_continuous_norm_{reward_type}"})
    out.attrs["source_csvs"] = [str(csv_path)]
    return out


def write_continuous_compare_source_manifest(
    compare_dir: Path,
    root_v0: Path,
    root_target: Path,
    target_reward_type: str,
    df_v0: pd.DataFrame,
    df_target: pd.DataFrame,
) -> None:
    rows: list[dict[str, Any]] = []
    for label, root, df in [
        ("v0", root_v0, df_v0),
        (target_reward_type, root_target, df_target),
    ]:
        for source_csv in df.attrs.get("source_csvs", []):
            rows.append(
                {
                    "plot": "compare_portfolio_value.png",
                    "source_role": f"continuous_5y_{label}",
                    "rolling_root": str(root),
                    "source_csv": source_csv,
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(compare_dir / "image_source_manifest.csv", index=False, encoding="utf-8-sig")


def compare_latest_continuous_v0_target(target_reward_type: str) -> None:
    """Compare latest v0 and target continuous 5-year test outputs."""
    target_reward_type = normalize_reward_type_for_continuous(target_reward_type)
    if not target_reward_type or target_reward_type == "v0":
        print("continuous compare skipped: target_reward_type 是 v0，沒有非 v0 版本可比較。")
        return

    root_v0 = _find_latest_rolling_root("v0")
    root_target = _find_latest_rolling_root(target_reward_type)
    if root_v0 is None or root_target is None:
        print(f"continuous compare skipped: 找不到 v0 或 {target_reward_type} 的 rolling 結果。")
        return

    df_v0 = _collect_continuous_5y_series(root_v0, "v0")
    df_target = _collect_continuous_5y_series(root_target, target_reward_type)
    if df_v0 is None or df_target is None:
        print(f"continuous compare skipped: v0/{target_reward_type} 尚未產生 continuous_5y_test 資料。")
        return

    merged = pd.merge(df_v0, df_target, on="date", how="outer").sort_values("date").reset_index(drop=True)
    merged = fill_dji_mean_norm_if_missing(merged)
    compare_root = Path(EXPERIMENT_ROOT) / f"compare_v0_{target_reward_type}_continuous5y"
    compare_dir = compare_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    compare_dir.mkdir(parents=True, exist_ok=False)
    merged.to_csv(compare_dir / "compare_metrics.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(merged))
    baseline_name = reward_display_name("v0")
    target_name = reward_display_name(target_reward_type)
    for col, color, label in [
        ("ppo_continuous_norm_v0", "blue", baseline_name),
        (f"ppo_continuous_norm_{target_reward_type}", "red", target_name),
    ]:
        if col not in merged.columns:
            continue
        y = pd.to_numeric(merged[col], errors="coerce")
        valid = y.dropna()
        ret = (valid.iloc[-1] - 1) * 100 if not valid.empty else 0
        ax.plot(x, y, color=color, linewidth=2.3, label=f"{label} ({ret:.2f}%)")

    dji_candidates = ["dji_mean_norm", "dji_mean_norm_v0", f"dji_mean_norm_{target_reward_type}"]
    dji = next((pd.to_numeric(merged[c], errors="coerce") for c in dji_candidates if c in merged.columns), pd.Series(dtype=float))
    if not dji.empty and dji.notna().any():
        valid = dji.dropna()
        ret = (valid.iloc[-1] - 1) * 100
        ax.plot(x, dji, color="tab:green", linewidth=1.8, label=f"DJI ({ret:.2f}%)")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.1)
    set_month_xticklabels(ax, merged["date"].tolist(), max_xticks=14)
    ax.set_title(f"{baseline_name} vs {target_name} Continuous 5-Year Test Equity Curve")
    set_axis_labels_horizontal(ax, "Date", "Asset Allocation (initial = 1.0)", ylabel_pad=95)
    ax.legend()
    plt.tight_layout()
    fig.savefig(compare_dir / "compare_portfolio_value.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    cleanup_unused_compare_images(compare_dir)
    write_continuous_compare_source_manifest(compare_dir, root_v0, root_target, target_reward_type, df_v0, df_target)

    summary_rows = []
    for label, col in [
        ("v0", "ppo_continuous_norm_v0"),
        (target_reward_type, f"ppo_continuous_norm_{target_reward_type}"),
        ("dji", "dji_mean_norm"),
    ]:
        if col in merged.columns:
            valid = pd.to_numeric(merged[col], errors="coerce").dropna()
            summary_rows.append(
                {
                    "strategy": label,
                    "final_normalized_value": float(valid.iloc[-1]) if not valid.empty else np.nan,
                    "return_percent": float((valid.iloc[-1] - 1) * 100) if not valid.empty else np.nan,
                }
            )
    pd.DataFrame(summary_rows).to_csv(compare_dir / "compare_summary.csv", index=False, encoding="utf-8-sig")

    save_json(
        compare_dir / "compare_config.json",
        {
            "compare_type": f"continuous_5y_v0_vs_{target_reward_type}",
            "v0_rolling_root": str(root_v0),
            "target_reward_type": target_reward_type,
            "target_rolling_root": str(root_target),
            "compare_dir": str(compare_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(f"continuous compare_v0_{target_reward_type} saved to: {compare_dir}")

# =====================================================
# 6. Main
# =====================================================

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    if ROLLING_WINDOW_MODE:
        windows = generate_rolling_windows(
            ROLLING_START_DATE,
            ROLLING_END_DATE,
            TRAIN_YEARS,
            TEST_YEARS,
            SHIFT_YEARS,
        )
        rolling_root = create_rolling_root()
        save_json(
            rolling_root / "rolling_config.json",
            {**base_experiment_config(), "rolling_root": str(rolling_root), "windows": windows},
        )

        print("========================================")
        print("Rolling Window 實驗設定")
        print("========================================")
        print(f"REWARD_TYPE = {REWARD_TYPE}")
        print(f"期間        = {ROLLING_START_DATE} ~ {ROLLING_END_DATE}")
        print(f"Train/Test  = {TRAIN_YEARS}y / {TEST_YEARS}y, shift {SHIFT_YEARS}y")
        print(f"Train/backtest repeats = {REPEAT_EXPERIMENTS}")
        print(f"Root             = {rolling_root}")
        print("流程：每個 window 的每個 repeat 都 train 一個 PPO model，並 backtest 一次")
        print("========================================\n")

        summary_rows: list[dict[str, Any]] = []

        # 正確順序：每個 window 的每個 repeat 都先 train 一個新 model，接著只 backtest 一次。
        for window in windows:
            window_dir = rolling_root / window_folder_name(window)
            window_dir.mkdir(parents=True, exist_ok=True)

            for repeat_index in range(1, REPEAT_EXPERIMENTS + 1):
                random_seed = BASE_RANDOM_SEED + repeat_index
                experiment_dir = window_dir / repeat_folder_name(repeat_index, random_seed)
                experiment_dir.mkdir(parents=True, exist_ok=True)

                run_config = {
                    **base_experiment_config(),
                    **window,
                    "rolling_root": str(rolling_root),
                    "window_dir": str(window_dir),
                    "experiment_dir": str(experiment_dir),
                    "repeat_index": repeat_index,
                    "random_seed": random_seed,
                    "train_period": f"{window['train_start_year']}-{window['train_end_year']}",
                    "test_period": f"{window['test_start_year']}-{window['test_end_year']}",
                    "stage": "train_and_backtest_repeat",
                }
                save_json(experiment_dir / "experiment_config.json", run_config)

                print("=" * 80)
                print(
                    f"Window {window['window_id']:02d}, repeat {repeat_index:02d}, "
                    f"train/backtest seed {random_seed}"
                )
                print(f"Train: {window['train_start_date']} ~ {window['train_end_date']}")
                print(f"Test : {window['test_start_date']} ~ {window['test_end_date']}")
                print(f"Repeat Dir: {experiment_dir}")
                print("=" * 80)

                child_env = build_child_environment(
                    experiment_dir=experiment_dir,
                    train_start_date=window["train_start_date"],
                    train_end_date=window["train_end_date"],
                    test_start_date=window["test_start_date"],
                    test_end_date=window["test_end_date"],
                    trade_start_date=window["trade_start_date"],
                    trade_end_date=window["trade_end_date"],
                    random_seed=random_seed,
                    rolling_root_dir=rolling_root,
                    window_dir=window_dir,
                    repeat_index=repeat_index,
                    window_id=window["window_id"],
                )

                for script in scripts_to_run():
                    run_script(script, child_env, project_root)

                backtest_summary = collect_backtest_summary(experiment_dir)
                summary_rows.append(
                    {
                        "window_id": window["window_id"],
                        "repeat_index": repeat_index,
                        "random_seed": random_seed,
                        "train_period": f"{window['train_start_year']}-{window['train_end_year']}",
                        "test_period": f"{window['test_start_year']}-{window['test_end_year']}",
                        "experiment_dir": str(experiment_dir),
                        **backtest_summary,
                    }
                )
                write_rolling_summaries(rolling_root, summary_rows)

            plot_window_annual_reset_aggregate(window_dir, REWARD_TYPE)

        if RUN_CONTINUOUS_5Y_TEST:
            run_continuous_5y_test(rolling_root, windows, REWARD_TYPE)
            if AUTO_COMPARE_CONTINUOUS_LATEST:
                compare_latest_continuous_v0_target(REWARD_TYPE)
            else:
                print("continuous compare skipped: AUTO_COMPARE_CONTINUOUS_LATEST=False。")
        else:
            print("continuous 5-year test skipped: RUN_CONTINUOUS_5Y_TEST=False。")

        if AUTO_COMPARE_LATEST:
            compare_latest_v0_v1()
        else:
            print("compare_latest_v0_target skipped: AUTO_COMPARE_LATEST=False。")
            print("若要指定 v0 與任一 reward 版本資料夾比較，請執行 examples/Compare_V0_V1_Manual.py。")

        print("🎉 Rolling Window 重複實驗全部執行完畢！")
        print(f"結果總資料夾：{rolling_root}")

    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = (
            Path(EXPERIMENT_ROOT)
            / f"reward_{REWARD_TYPE}"
            / f"{TRADE_START_DATE}_to_{TRADE_END_DATE}"
            / f"{timestamp}_{REWARD_TYPE}"
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        random_seed = BASE_RANDOM_SEED + 1
        save_json(
            experiment_dir / "experiment_config.json",
            {
                **base_experiment_config(),
                "train_start_date": TRAIN_START_DATE,
                "train_end_date": TRAIN_END_DATE,
                "test_start_date": TEST_START_DATE,
                "test_end_date": TEST_END_DATE,
                "trade_start_date": TRADE_START_DATE,
                "trade_end_date": TRADE_END_DATE,
                "experiment_dir": str(experiment_dir),
                "random_seed": random_seed,
            },
        )
        child_env = build_child_environment(
            experiment_dir=experiment_dir,
            train_start_date=TRAIN_START_DATE,
            train_end_date=TRAIN_END_DATE,
            test_start_date=TEST_START_DATE,
            test_end_date=TEST_END_DATE,
            trade_start_date=TRADE_START_DATE,
            trade_end_date=TRADE_END_DATE,
            random_seed=random_seed,
        )
        for script in scripts_to_run():
            run_script(script, child_env, project_root)
        print("🎉 單次實驗執行完畢！")
        print(f"結果資料夾：{experiment_dir}")
