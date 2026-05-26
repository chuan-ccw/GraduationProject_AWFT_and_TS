from __future__ import annotations

import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any

from finrl import config


# =====================================================
# 1. 在這裡統一設定本次實驗參數
# =====================================================

# Reward 選擇："v0" = FinRL 原版 reward；"v1" = 我們設計的 reward
REWARD_TYPE = "vㄢ"
REWARD_PARAMS = {
    "alpha": 0.05,
    "beta": 0.05,
    # gamma 若設為 0，代表 portfolio value 已扣手續費，reward 不額外懲罰交易成本
    # "gamma": 1.0,
    "n_recent": 14,
}

# Rolling Window 設定
ROLLING_WINDOW_MODE = True
ROLLING_START_DATE = "2018-01-01"
ROLLING_END_DATE = "2025-12-31"
TRAIN_YEARS = 3
TEST_YEARS = 1
SHIFT_YEARS = 1

# 每個 rolling window 重複實驗次數
# 注意：repeat_01 在 window_01 ~ window_05 使用相同 seed；repeat_02 才換下一個 seed。
REPEAT_EXPERIMENTS = 5
BASE_RANDOM_SEED = 42

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

# Agent 開關：train 和 backtest 會共用這裡的設定
IF_USING_A2C = False
IF_USING_DDPG = False
IF_USING_PPO = True
IF_USING_TD3 = False
IF_USING_SAC = False

# 訓練步數
A2C_TOTAL_TIMESTEPS = 20000
DDPG_TOTAL_TIMESTEPS = 20000
PPO_TOTAL_TIMESTEPS = 10000
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_year = _year(ROLLING_START_DATE)
    end_year = _year(ROLLING_END_DATE)
    run_name = (
        f"rolling_{start_year}-{end_year}_"
        f"train{TRAIN_YEARS}y_test{TEST_YEARS}y_shift{SHIFT_YEARS}y_"
        f"{timestamp}_{REWARD_TYPE}"
    )
    rolling_root = (
        Path(EXPERIMENT_ROOT)
        / f"reward_{REWARD_TYPE}"
        / "rolling_window"
        / run_name
    )
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
    summary_path = rolling_root / "rolling_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    numeric_cols = [
        col
        for col in ["final_ppo", "ppo_return", "ppo_max_drawdown", "final_mvo", "final_dji"]
        if col in summary_df.columns
    ]
    if numeric_cols:
        agg = (
            summary_df.groupby(["window_id", "train_period", "test_period"])[numeric_cols]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        agg.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in agg.columns]
        agg_path = rolling_root / "rolling_window_aggregate_summary.csv"
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
        df["date"] = df["date"].astype(str)
        return df
    except Exception as exc:
        print(f"讀取 backtest csv 失敗: {csv_path} -> {exc}")
        return None


def plot_window_repeat_aggregate(window_dir: Path, reward_type: str) -> None:
    """For one rolling window, aggregate repeated backtest results across seeds.

    Output:
      1. CSV containing ppo mean/min/max and mean mvo/dji
      2. PNG where PPO is a blue mean line with a light blue min-max band
    """
    repeat_dirs = sorted([p for p in window_dir.iterdir() if p.is_dir() and p.name.startswith("repeat_")])
    if not repeat_dirs:
        return

    run_dfs: list[pd.DataFrame] = []
    trade_start_date = None
    trade_end_date = None
    date_suffix = datetime.now().strftime("%m%d")

    for repeat_dir in repeat_dirs:
        candidates = sorted(repeat_dir.glob("backtest_result_*.csv"))
        if not candidates:
            continue
        csv_path = candidates[0]
        df = _read_backtest_csv(csv_path)
        if df is None or "ppo" not in df.columns:
            continue

        if trade_start_date is None or trade_end_date is None:
            name = csv_path.stem  # backtest_result_v0_2021-01-01_to_2021-12-31_0526
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

        keep_cols = [c for c in ["date", "ppo", "mvo", "dji"] if c in df.columns]
        run_dfs.append(df[keep_cols].copy())

    if not run_dfs:
        print(f"{window_dir} 沒有可用的 backtest_result csv，略過 aggregate plot。")
        return

    base_dates = run_dfs[0]["date"].astype(str)
    ppo_runs = []
    mvo_runs = []
    dji_runs = []

    for run_idx, df in enumerate(run_dfs, start=1):
        aligned = df.set_index("date").reindex(base_dates).reset_index()
        if "ppo" in aligned.columns:
            ppo_runs.append(pd.to_numeric(aligned["ppo"], errors="coerce").rename(f"ppo_run_{run_idx:02d}"))
        if "mvo" in aligned.columns:
            mvo_runs.append(pd.to_numeric(aligned["mvo"], errors="coerce").rename(f"mvo_run_{run_idx:02d}"))
        if "dji" in aligned.columns:
            dji_runs.append(pd.to_numeric(aligned["dji"], errors="coerce").rename(f"dji_run_{run_idx:02d}"))

    ppo_df = pd.concat(ppo_runs, axis=1) if ppo_runs else pd.DataFrame(index=range(len(base_dates)))
    mvo_df = pd.concat(mvo_runs, axis=1) if mvo_runs else pd.DataFrame(index=range(len(base_dates)))
    dji_df = pd.concat(dji_runs, axis=1) if dji_runs else pd.DataFrame(index=range(len(base_dates)))

    aggregate_df = pd.DataFrame({"date": base_dates})
    if not ppo_df.empty:
        aggregate_df["ppo_mean"] = ppo_df.mean(axis=1)
        aggregate_df["ppo_min"] = ppo_df.min(axis=1)
        aggregate_df["ppo_max"] = ppo_df.max(axis=1)
    if not mvo_df.empty:
        aggregate_df["mvo_mean"] = mvo_df.mean(axis=1)
    if not dji_df.empty:
        aggregate_df["dji_mean"] = dji_df.mean(axis=1)

    trade_start_date = trade_start_date or str(base_dates.iloc[0])
    trade_end_date = trade_end_date or str(base_dates.iloc[-1])

    aggregate_csv_path = window_dir / f"backtest_result_repeat_mean_{reward_type}_{trade_start_date}_to_{trade_end_date}_{date_suffix}.csv"
    aggregate_df.to_csv(aggregate_csv_path, index=False, encoding="utf-8-sig")

    plt.rcParams["figure.figsize"] = (15, 5)
    fig, ax = plt.subplots()
    x = np.arange(len(aggregate_df))

    if {"ppo_mean", "ppo_min", "ppo_max"}.issubset(aggregate_df.columns):
        ppo_mean = pd.to_numeric(aggregate_df["ppo_mean"], errors="coerce")
        ppo_min = pd.to_numeric(aggregate_df["ppo_min"], errors="coerce")
        ppo_max = pd.to_numeric(aggregate_df["ppo_max"], errors="coerce")
        ax.fill_between(x, ppo_min, ppo_max, color="blue", alpha=0.18, label="ppo range (min-max)")
        ax.plot(x, ppo_mean, color="blue", linewidth=2.2, label="ppo mean")

    if "mvo_mean" in aggregate_df.columns:
        ax.plot(x, aggregate_df["mvo_mean"], color="tab:orange", label="mvo")
    if "dji_mean" in aggregate_df.columns:
        ax.plot(x, aggregate_df["dji_mean"], color="tab:green", label="dji")

    max_xticks = 12
    if len(aggregate_df) > 0:
        tick_positions = np.linspace(0, len(aggregate_df) - 1, min(max_xticks, len(aggregate_df))).astype(int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([aggregate_df["date"].iloc[i] for i in tick_positions])

    ax.set_title("Portfolio Value Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()

    aggregate_plot_path = window_dir / f"backtest_result_repeat_mean_{reward_type}_{trade_start_date}_to_{trade_end_date}_{date_suffix}.png"
    plt.savefig(aggregate_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Window aggregate csv saved to: {aggregate_csv_path}")
    print(f"Window aggregate plot saved to: {aggregate_plot_path}")


# =====================================================
# 5. Main
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
        print(f"Repeats     = {REPEAT_EXPERIMENTS}")
        print(f"Root        = {rolling_root}")
        print("Seed 規則：同一個 repeat 在所有 windows 使用同一個 seed")
        print("========================================\n")

        summary_rows: list[dict[str, Any]] = []

        # 正確順序：repeat_01 跑完 window_01~window_05，再換 repeat_02。
        for repeat_index in range(1, REPEAT_EXPERIMENTS + 1):
            random_seed = BASE_RANDOM_SEED + repeat_index
            print("#" * 80)
            print(f"開始 repeat_{repeat_index:02d}, seed = {random_seed}")
            print("#" * 80)

            for window in windows:
                window_dir = rolling_root / window_folder_name(window)
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
                }
                save_json(experiment_dir / "experiment_config.json", run_config)

                print("=" * 80)
                print(
                    f"Window {window['window_id']:02d}, repeat {repeat_index:02d}, seed {random_seed}"
                )
                print(f"Train: {window['train_start_date']} ~ {window['train_end_date']}")
                print(f"Test : {window['test_start_date']} ~ {window['test_end_date']}")
                print(f"Dir  : {experiment_dir}")
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

        for window in windows:
            window_dir = rolling_root / window_folder_name(window)
            plot_window_repeat_aggregate(window_dir, REWARD_TYPE)

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
