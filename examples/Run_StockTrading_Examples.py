from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from finrl import config


# =====================================================
# 1. 在這裡統一設定本次實驗參數
# =====================================================

# Reward 選擇："v0" = FinRL 原版 reward；"v1" = 我們設計的 reward
REWARD_TYPE = "v1"
REWARD_PARAMS = {
    "alpha": 0.05,
    "beta": 0.05,
    "gamma": 1.0,
    "n_recent": 14,
}

# 日期設定：這裡會覆蓋 finrl/config.py 的預設值
TRAIN_START_DATE = "2014-01-06"
TRAIN_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"
TEST_END_DATE = "2026-03-20"
TRADE_START_DATE = "2025-01-01"
TRADE_END_DATE = "2026-03-20"

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
# 2. 自動建立本次實驗資料夾
# =====================================================

def create_experiment_folder() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_range = f"{TRADE_START_DATE}_to_{TRADE_END_DATE}"
    experiment_dir = (
        Path(EXPERIMENT_ROOT)
        / f"reward_{REWARD_TYPE}"
        / date_range
        / f"{timestamp}_{REWARD_TYPE}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def save_experiment_config(experiment_dir: Path) -> None:
    experiment_config = {
        "reward_type": REWARD_TYPE,
        "reward_params": REWARD_PARAMS,
        "train_start_date": TRAIN_START_DATE,
        "train_end_date": TRAIN_END_DATE,
        "test_start_date": TEST_START_DATE,
        "test_end_date": TEST_END_DATE,
        "trade_start_date": TRADE_START_DATE,
        "trade_end_date": TRADE_END_DATE,
        "experiment_root": str(EXPERIMENT_ROOT),
        "experiment_dir": str(experiment_dir),
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

    config_path = experiment_dir / "experiment_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=4, ensure_ascii=False)

    print(f"已儲存實驗設定：{config_path}")


def build_child_environment(experiment_dir: Path) -> dict[str, str]:
    env = os.environ.copy()

    env["REWARD_TYPE"] = REWARD_TYPE
    env["REWARD_PARAMS"] = json.dumps(REWARD_PARAMS, ensure_ascii=False)

    env["TRAIN_START_DATE"] = TRAIN_START_DATE
    env["TRAIN_END_DATE"] = TRAIN_END_DATE
    env["TEST_START_DATE"] = TEST_START_DATE
    env["TEST_END_DATE"] = TEST_END_DATE
    env["TRADE_START_DATE"] = TRADE_START_DATE
    env["TRADE_END_DATE"] = TRADE_END_DATE

    env["EXPERIMENT_ROOT"] = str(EXPERIMENT_ROOT)
    env["EXPERIMENT_DIR"] = str(experiment_dir)

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
# 3. 執行子程式
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


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    experiment_dir = create_experiment_folder()
    save_experiment_config(experiment_dir)
    env = build_child_environment(experiment_dir)

    print("========================================")
    print("本次實驗設定")
    print("========================================")
    print(f"REWARD_TYPE    = {REWARD_TYPE}")
    print(f"REWARD_PARAMS  = {REWARD_PARAMS}")
    print(f"TRAIN          = {TRAIN_START_DATE} to {TRAIN_END_DATE}")
    print(f"TRADE          = {TRADE_START_DATE} to {TRADE_END_DATE}")
    print(f"EXPERIMENT_DIR = {experiment_dir}")
    print("========================================\n")

    scripts_to_run: list[str] = []
    if RUN_DATA:
        scripts_to_run.append("examples/FinRL_StockTrading_2026_1_data.py")
    if RUN_TRAIN:
        scripts_to_run.append("examples/FinRL_StockTrading_2026_2_train.py")
    if RUN_BACKTEST:
        scripts_to_run.append("examples/FinRL_StockTrading_2026_3_Backtest.py")

    for script in scripts_to_run:
        run_script(script, env, project_root)

    print("🎉 所有檔案皆已依序執行完畢！")
    print(f"本次實驗結果資料夾：{experiment_dir}")
