import subprocess
import sys


def run_script(script_name):
    print(f"========================================")
    print(f"正在執行: {script_name}...")
    print(f"========================================")

    try:
        # run() 會等待程式執行完畢
        # text=True 和 check=True 讓輸出能正常顯示，並在出錯時拋出異常
        subprocess.run([sys.executable, script_name], check=True, text=True)
        print(f"✓ {script_name} 執行成功！\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} 執行失敗！錯誤碼: {e.returncode}")
        print("停止執行後續檔案。")
        sys.exit(1)  # 中斷主程式
    except FileNotFoundError:
        print(f"❌ 找不到檔案: {script_name}，請檢查路徑是否正確。")
        print("停止執行後續檔案。")
        sys.exit(1)


if __name__ == "__main__":
    # 在這裡設定你想依序執行的檔案清單
    scripts_to_run = ["examples/FinRL_StockTrading_2026_1_data.py",
                      "examples/FinRL_StockTrading_2026_2_train.py",
                      "examples/FinRL_StockTrading_2026_3_Backtest.py"]

    for script in scripts_to_run:
        run_script(script)

    print("🎉 所有檔案皆已依序執行完畢！")
