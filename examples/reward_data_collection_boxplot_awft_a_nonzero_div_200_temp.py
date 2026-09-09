from pathlib import Path
import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from matplotlib.cbook import boxplot_stats

#畫圖AWFT action / 200 的boxplot, csv (暫時)

# ============================================================
# 1. User setting
# ============================================================

# 平常只需要修改這個 rolling 資料夾名稱
FOLDER_INPUT = (
    "rolling_2018-2025_train3y_test1y_shift1y_20260904_234026_reinforce"
)

EXPERIMENT_ROOT = Path(
    os.environ.get(
        "EXPERIMENT_ROOT",
        r"C:\Users\User\Desktop\GraduationProject\Experence_result",
    )
)

# {a | a != 0} 中每筆資料除以 200
ACTION_DIVISOR = 200.0

# Boxplot whisker 規則
WHIS = 1.5

# 圖片大小
FIGSIZE = (7.25, 7)

# 主圖 whisker 上下留白比例
MAIN_MARGIN_RATIO = 0.08


# ============================================================
# 2. 尋找 rolling 實驗資料夾
# ============================================================

def find_experiment_dir(folder_input):
    """
    使用者可輸入：
    1. rolling 資料夾名稱
    2. rolling 資料夾完整路徑

    若只輸入名稱，會從 EXPERIMENT_ROOT 底下尋找。
    """
    folder_input = folder_input.strip().strip('"').strip("'")

    if not folder_input:
        raise ValueError("沒有輸入 rolling 資料夾名稱。")

    direct_path = Path(folder_input)

    if direct_path.is_dir():
        return direct_path.resolve()

    matches = [
        path
        for path in EXPERIMENT_ROOT.glob(f"reward_*/{folder_input}")
        if path.is_dir()
    ]

    if not matches:
        matches = [
            path
            for path in EXPERIMENT_ROOT.rglob(folder_input)
            if path.is_dir() and path.name == folder_input
        ]

    if not matches:
        raise FileNotFoundError(
            "找不到 rolling 資料夾：\n"
            f"  {folder_input}\n"
            "搜尋根目錄：\n"
            f"  {EXPERIMENT_ROOT}"
        )

    if len(matches) > 1:
        match_text = "\n".join(
            f"  - {path}"
            for path in matches
        )

        raise RuntimeError(
            "找到多個同名 rolling 資料夾，請改輸入完整路徑：\n"
            f"{match_text}"
        )

    return matches[0]


# ============================================================
# 3. 資料清理
# ============================================================

def clean_data(data):
    return pd.to_numeric(
        data,
        errors="coerce",
    ).dropna()


# ============================================================
# 4. 讀取 {a | a != 0} / 200
# ============================================================

def load_scaled_action(input_csv):
    df = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
    )

    if "value" not in df.columns:
        raise ValueError(
            f"{input_csv.name} 找不到 value 欄位。"
        )

    a = clean_data(df["value"])

    # awft_a_nonzero.csv 本身已經是 {a | a != 0}，
    # 這裡再檢查一次 a != 0，然後除以 200。
    a_nonzero_scaled = (
        a[a != 0]
        / ACTION_DIVISOR
    )

    if a_nonzero_scaled.empty:
        raise ValueError(
            "{a | a != 0} 為空集合，無法畫 Boxplot。"
        )

    return a_nonzero_scaled


# ============================================================
# 5. 主圖 Y 軸範圍
# ============================================================

def get_whisker_ylim(data):
    stats = boxplot_stats(
        data,
        whis=WHIS,
    )[0]

    lower = float(stats["whislo"])
    upper = float(stats["whishi"])

    span = upper - lower

    if span == 0:
        span = float(
            data.max()
            - data.min()
        )

    if span == 0:
        span = 0.001

    margin = span * MAIN_MARGIN_RATIO

    y_min = min(
        lower - margin,
        0.0,
    )

    y_max = max(
        upper + margin,
        0.0,
    )

    return y_min, y_max


# ============================================================
# 6. 主圖：不顯示離群值
# ============================================================

def draw_main_boxplot(
    data,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.96,
        top=0.88,
        bottom=0.12,
    )

    ax.boxplot(
        data,
        vert=True,
        widths=0.42,
        whis=WHIS,
        showfliers=False,
    )

    ax.set_title(
        "Box Plot of AWFT Action {a | a != 0} / 200",
        fontsize=19,
        pad=18,
        loc="center",
    )

    ax.set_ylabel(
        "Action a / 200",
        fontsize=14,
    )

    ax.set_xticks([])
    ax.set_xlim(
        0.68,
        1.32,
    )

    y_min, y_max = get_whisker_ylim(
        data
    )

    ax.set_ylim(
        y_min,
        y_max,
    )

    ax.yaxis.set_major_locator(
        MaxNLocator(
            nbins=12
        )
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
        alpha=0.55,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 7. Full range：顯示離群值
# ============================================================

def draw_full_range_boxplot(
    data,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.96,
        top=0.88,
        bottom=0.12,
    )

    ax.boxplot(
        data,
        vert=True,
        widths=0.42,
        whis=WHIS,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 2.2,
            "alpha": 0.30,
        },
    )

    ax.set_title(
        "Box Plot of AWFT Action {a | a != 0} / 200 (Full range)",
        fontsize=16,
        pad=18,
        loc="center",
    )

    ax.set_ylabel(
        "Action a / 200",
        fontsize=14,
    )

    ax.set_xticks([])
    ax.set_xlim(
        0.68,
        1.32,
    )

    minimum = float(
        data.min()
    )

    maximum = float(
        data.max()
    )

    full_range = (
        maximum
        - minimum
    )

    if full_range > 0:
        margin = (
            full_range
            * 0.03
        )

        ax.set_ylim(
            minimum - margin,
            maximum + margin,
        )

    ax.yaxis.set_major_locator(
        MaxNLocator(
            nbins=10
        )
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.30,
    )

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
        alpha=0.55,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 8. 處理單一 Window
# ============================================================

def process_window(window_dir):
    input_csv = (
        window_dir
        / "awft_a_nonzero.csv"
    )

    if not input_csv.is_file():
        print(
            "WARNING: 找不到 "
            f"{input_csv}"
        )
        return

    data = load_scaled_action(
        input_csv
    )

    output_csv = (
        window_dir
        / "awft_a_nonzero_div_200.csv"
    )

    output_main = (
        window_dir
        / "awft_a_nonzero_div_200_boxplot.png"
    )

    output_full = (
        window_dir
        / "awft_a_nonzero_div_200_boxplot_full_range.png"
    )

    pd.DataFrame(
        {
            "value":
            data.reset_index(
                drop=True
            )
        }
    ).to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    draw_main_boxplot(
        data,
        output_main,
    )

    draw_full_range_boxplot(
        data,
        output_full,
    )

    print("=" * 70)
    print(
        f"Window：{window_dir.name}"
    )
    print("=" * 70)

    print(
        f"資料筆數 : {len(data)}"
    )
    print(
        f"Minimum : {data.min():.8f}"
    )
    print(
        f"Q1      : {data.quantile(0.25):.8f}"
    )
    print(
        f"Median  : {data.median():.8f}"
    )
    print(
        f"Q3      : {data.quantile(0.75):.8f}"
    )
    print(
        f"Maximum : {data.max():.8f}"
    )

    print()
    print(
        f"CSV        : {output_csv}"
    )
    print(
        f"Main plot  : {output_main}"
    )
    print(
        f"Full range : {output_full}"
    )
    print()


# ============================================================
# 9. 主程式
# ============================================================

def main():
    experiment_dir = find_experiment_dir(
        FOLDER_INPUT
    )

    reward_data_dir = (
        experiment_dir
        / "reward_data_collection"
    )

    if not reward_data_dir.is_dir():
        raise FileNotFoundError(
            "找不到 reward_data_collection：\n"
            f"  {reward_data_dir}"
        )

    window_dirs = sorted(
        path
        for path
        in reward_data_dir.iterdir()
        if (
            path.is_dir()
            and path.name.startswith(
                "window_"
            )
        )
    )

    if not window_dirs:
        raise FileNotFoundError(
            "reward_data_collection 底下找不到 window_xx 資料夾：\n"
            f"  {reward_data_dir}"
        )

    print("=" * 70)
    print(
        "AWFT Action {a | a != 0} / 200 Boxplot"
    )
    print("=" * 70)

    print(
        f"Experiment：{experiment_dir}"
    )
    print(
        f"Input data：{reward_data_dir}"
    )
    print(
        f"Window 數量：{len(window_dirs)}"
    )
    print()

    for window_dir in window_dirs:
        process_window(
            window_dir
        )

    print("=" * 70)
    print(
        "全部 Window 圖片輸出完成"
    )
    print(
        "輸出位置：各個 reward_data_collection/window_xx 資料夾"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
