from pathlib import Path
import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from matplotlib.cbook import boxplot_stats

# 從reward_data_collection抓資料畫box plot和統整csv

FOLDER_INPUT = "rolling_2018-2025_train3y_test1y_shift1y_20260903_104709_reinforce"

# ============================================================
# 1. 路徑設定
# ============================================================

EXPERIMENT_ROOT = Path(
    os.environ.get(
        "EXPERIMENT_ROOT",
        r"C:\Users\User\Desktop\GraduationProject\Experence_result",
    )
)


# ============================================================
# 2. 共用繪圖設定
# ============================================================

# 現在每張 PNG 只放一張圖。
FIGSIZE = (7.25, 7)

# Boxplot whisker 規則
WHIS = 1.5

# 主圖 whisker 上下留白比例
MAIN_MARGIN_RATIO = 0.08

# 主標題大小
MAIN_TITLE_SIZE = 19

# Full range 標題大小
FULL_TITLE_SIZE = 16

# 軸標籤大小
AXIS_LABEL_SIZE = 14

# X 軸文字大小
XTICK_LABEL_SIZE = 11

# 刻度字大小
TICK_LABEL_SIZE = 10


# CSV 檔名對應到圖片顯示名稱與 Y 軸名稱。
CSV_CONFIG = {
    "a": {
        "display_name": "Action a",
        "ylabel": "Action a",
        "zero_comparison": True,
    },
    "reward": {
        "display_name": "Step Reward",
        "ylabel": "Reward",
        "zero_comparison": False,
    },
    "w": {
        "display_name": "Portfolio Weight w",
        "ylabel": "Weight w",
        "zero_comparison": True,
    },
    "delta_w": {
        "display_name": "Delta Weight",
        "ylabel": "Delta w",
        "zero_comparison": True,
    },
    "d": {
        "display_name": "Direction d",
        "ylabel": "Direction d",
        "zero_comparison": False,
    },
}

AWFT_DISPLAY_CONFIG = {
    "a": {
        "display_name": "AWFT Action",
        "ylabel": "Action a",
    },
    "w": {
        "display_name": "AWFT Weight",
        "ylabel": "Weight w",
    },
    "d": {
        "display_name": "AWFT Future Trend",
        "ylabel": "Future Trend d",
    },
    "reward": {
        "display_name": "AWFT Reward",
        "ylabel": "Reward",
    },
}


# boxplot_statistics.csv 欄位順序
# variable / dataset 保留在最前面，用來辨識每一列屬於哪個變數與資料集。
STATISTICS_CSV_COLUMNS = [
    "variable",
    "dataset",
    "minimum",
    "lower_whisker",
    "q1_25_percent",
    "q2_median_50_percent",
    "q3_75_percent",
    "upper_whisker",
    "maximum",
    "count",
    "zero_count",
    "zero_ratio_percent",
    "whisker_range_count",
    "whisker_range_ratio_percent",
    "outlier_count",
    "outlier_ratio_percent",
]


# ============================================================
# 3. 尋找 rolling 實驗資料夾
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
        # 若未來 reward 資料夾不是 reward_v0 / reward_v6 / reward_v7 形式，
        # 再做一次較寬鬆的搜尋。
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
        match_text = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            "找到多個同名 rolling 資料夾，請改輸入完整路徑：\n"
            f"{match_text}"
        )

    return matches[0]


# ============================================================
# 4. 資料清理
# ============================================================

def clean_data(data):
    """
    將資料轉成數值並移除 NaN。
    """
    return pd.to_numeric(data, errors="coerce").dropna()


# ============================================================
# 5. 取得 Boxplot 統計值
# ============================================================

def get_boxplot_values(data):
    """
    取得標準 Boxplot 統計值：
    Q1 / Median(Q2) / Q3 / Lower Whisker / Upper Whisker
    """
    data = clean_data(data)

    if data.empty:
        raise ValueError("沒有可用的數值資料，無法計算 Boxplot。")

    stats = boxplot_stats(data, whis=WHIS)[0]

    return {
        "q1": stats["q1"],
        "median": stats["med"],
        "q3": stats["q3"],
        "lower_whisker": stats["whislo"],
        "upper_whisker": stats["whishi"],
    }


# ============================================================
# 6. 統計資訊：Terminal + CSV row
# ============================================================

def get_statistics(data, variable, dataset):
    """
    計算原本 Terminal 顯示的統計資訊，
    並回傳一列可寫入 CSV 的資料。
    """
    data = clean_data(data)

    if data.empty:
        raise ValueError(f"{variable} - {dataset} 沒有可用資料。")

    box = get_boxplot_values(data)

    count = len(data)
    zero_count = int((data == 0).sum())

    # whisker 範圍包含上下 whisker 本身。
    whisker_mask = (data >= box["lower_whisker"]) & (data <= box["upper_whisker"])
    whisker_range_count = int(whisker_mask.sum())
    outlier_count = count - whisker_range_count

    return {
        "variable": variable,
        "dataset": dataset,
        "count": count,
        "zero_count": zero_count,
        "zero_ratio_percent": zero_count / count * 100,
        "minimum": data.min(),
        "percentile_1": data.quantile(0.01),
        "percentile_5": data.quantile(0.05),
        "q1_25_percent": box["q1"],
        "q2_median_50_percent": box["median"],
        "q3_75_percent": box["q3"],
        "percentile_95": data.quantile(0.95),
        "percentile_99": data.quantile(0.99),
        "maximum": data.max(),
        "lower_whisker": box["lower_whisker"],
        "upper_whisker": box["upper_whisker"],
        "whisker_range_count": whisker_range_count,
        "whisker_range_ratio_percent": whisker_range_count / count * 100,
        "outlier_count": outlier_count,
        "outlier_ratio_percent": outlier_count / count * 100,
    }


def print_statistics(stats):
    """
    保留原本 Terminal 統計資訊輸出。
    """
    print("=" * 70)
    print(f"{stats['variable']} - {stats['dataset']}")
    print("=" * 70)

    print(f"資料筆數       : {stats['count']}")
    print(f"0 的數量       : {stats['zero_count']}")
    print(f"0 的比例       : {stats['zero_ratio_percent']:.2f}%")
    print()

    print(f"Minimum        : {stats['minimum']:.8f}")
    print(f"1%             : {stats['percentile_1']:.8f}")
    print(f"5%             : {stats['percentile_5']:.8f}")
    print(f"Q1 (25%)       : {stats['q1_25_percent']:.8f}")
    print(f"Median (50%)   : {stats['q2_median_50_percent']:.8f}")
    print(f"Q3 (75%)       : {stats['q3_75_percent']:.8f}")
    print(f"95%            : {stats['percentile_95']:.8f}")
    print(f"99%            : {stats['percentile_99']:.8f}")
    print(f"Maximum        : {stats['maximum']:.8f}")
    print()

    print("Boxplot")
    print(f"Lower Whisker  : {stats['lower_whisker']:.8f}")
    print(f"Q1             : {stats['q1_25_percent']:.8f}")
    print(f"Median (Q2)    : {stats['q2_median_50_percent']:.8f}")
    print(f"Q3             : {stats['q3_75_percent']:.8f}")
    print(f"Upper Whisker  : {stats['upper_whisker']:.8f}")
    print()


# ============================================================
# 7. 主圖 Y 軸範圍（依 whisker）
# ============================================================

def get_whisker_range(data, include_zero=True, margin_ratio=MAIN_MARGIN_RATIO):
    """
    主圖使用 whisker 範圍，而不是 min ~ max，
    讓主要分布更清楚。
    """
    data = clean_data(data)
    box = get_boxplot_values(data)

    lower = box["lower_whisker"]
    upper = box["upper_whisker"]

    whisker_range = upper - lower

    if whisker_range == 0:
        data_range = data.max() - data.min()

        if data_range == 0:
            data_range = 0.001

        whisker_range = data_range

    margin = whisker_range * margin_ratio

    y_min = lower - margin
    y_max = upper + margin

    if include_zero:
        y_min = min(y_min, 0)
        y_max = max(y_max, 0)

    return y_min, y_max


# ============================================================
# 8. Y 軸刻度
# ============================================================

def set_dense_y_ticks(ax, number=12):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=number))


# ============================================================
# 9. 建立單張圖
# ============================================================

def create_single_figure():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(
        left=0.14,
        right=0.96,
        top=0.88,
        bottom=0.12,
    )
    return fig, ax


# ============================================================
# 10. 主圖共用格式
# ============================================================

def format_main_axis(ax, display_name, ylabel):
    ax.set_title(
        f"Box Plot of {display_name}",
        fontsize=MAIN_TITLE_SIZE,
        pad=18,
        loc="center",
    )

    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)

    set_dense_y_ticks(ax, number=12)

    ax.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
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


# ============================================================
# 11. Full range 圖共用格式
# ============================================================

def format_full_range_axis(ax, data, display_name, ylabel, show_xlabels=False):
    data = clean_data(data)

    ax.set_title(
        f"Box Plot of {display_name} (Full range)",
        fontsize=FULL_TITLE_SIZE,
        pad=18,
        loc="center",
    )

    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)

    minimum = data.min()
    maximum = data.max()
    full_range = maximum - minimum

    if full_range > 0:
        full_margin = full_range * 0.03
        ax.set_ylim(
            minimum - full_margin,
            maximum + full_margin,
        )

    set_dense_y_ticks(ax, number=10)

    ax.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
    )

    if not show_xlabels:
        ax.set_xticks([])

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


# ============================================================
# 12. 一般 Boxplot
# 用於：
# 1. Step Reward
# 2. Direction d
# ============================================================

def draw_full_boxplot(data, display_name, ylabel, output_dir, file_stem):
    data = clean_data(data)

    stats = get_statistics(data, display_name, "All data")
    print_statistics(stats)

    # -------------------------
    # 主圖：不顯示離群值
    # -------------------------
    fig, ax = create_single_figure()

    ax.boxplot(
        data,
        vert=True,
        widths=0.42,
        whis=WHIS,
        showfliers=False,
    )

    format_main_axis(ax, display_name, ylabel)

    # Step Reward、Direction d 底下不顯示 Main distribution。
    ax.set_xticks([])
    ax.set_xlim(0.68, 1.32)

    y_min, y_max = get_whisker_range(
        data,
        include_zero=True,
        margin_ratio=MAIN_MARGIN_RATIO,
    )
    ax.set_ylim(y_min, y_max)

    save_path = output_dir / f"{file_stem}_boxplot.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------
    # Full range：顯示離群值
    # -------------------------
    fig, ax = create_single_figure()

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

    format_full_range_axis(
        ax,
        data,
        display_name,
        ylabel,
        show_xlabels=False,
    )
    ax.set_xlim(0.68, 1.32)

    full_save_path = output_dir / f"{file_stem}_boxplot_full_range.png"
    fig.savefig(full_save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return [stats]


# ============================================================
# 13. 大量 0 的 Boxplot
# 用於：
# 1. Portfolio Weight w
# 2. Delta Weight
# ============================================================

def draw_zero_comparison_boxplot(data, display_name, ylabel, output_dir, file_stem):
    data = clean_data(data)
    nonzero_data = data[data != 0]

    if nonzero_data.empty:
        raise ValueError(f"{display_name} 全部都是 0，無法建立 Non-zero only Boxplot。")

    all_stats = get_statistics(data, display_name, "All data")
    nonzero_stats = get_statistics(nonzero_data, display_name, "Non-zero only")

    print_statistics(all_stats)
    print_statistics(nonzero_stats)

    # -------------------------
    # 主圖：不顯示離群值
    # -------------------------
    fig, ax = create_single_figure()

    ax.boxplot(
        [data, nonzero_data],
        positions=[1, 2],
        widths=0.48,
        vert=True,
        whis=WHIS,
        showfliers=False,
    )

    format_main_axis(ax, display_name, ylabel)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        ["All data", "Non-zero only"],
        fontsize=XTICK_LABEL_SIZE,
    )
    ax.set_xlim(0.55, 2.45)

    y_min, y_max = get_whisker_range(
        nonzero_data,
        include_zero=True,
        margin_ratio=MAIN_MARGIN_RATIO,
    )
    ax.set_ylim(y_min, y_max)

    zero_ratio = (data == 0).mean() * 100
    ax.text(
        0.03,
        0.97,
        f"Zero values: {zero_ratio:.1f}%",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
    )

    save_path = output_dir / f"{file_stem}_boxplot.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------
    # Full range：顯示離群值
    # -------------------------
    fig, ax = create_single_figure()

    ax.boxplot(
        [data, nonzero_data],
        positions=[1, 2],
        widths=0.48,
        vert=True,
        whis=WHIS,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 2.0,
            "alpha": 0.26,
        },
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        ["All data", "Non-zero only"],
        fontsize=XTICK_LABEL_SIZE,
    )
    ax.set_xlim(0.55, 2.45)

    format_full_range_axis(
        ax,
        data,
        display_name,
        ylabel,
        show_xlabels=True,
    )

    full_save_path = output_dir / f"{file_stem}_boxplot_full_range.png"
    fig.savefig(full_save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return [all_stats, nonzero_stats]


# ============================================================
# 14. 處理單一 Window
# ============================================================

def process_window(window_dir, reward_version=None):
    print("#" * 70)
    print(f"處理 Window：{window_dir.name}")
    print(f"位置：{window_dir}")
    print("#" * 70)

    statistics_rows = []
    processed_count = 0

    for csv_file in sorted(window_dir.glob("*.csv")):
        config = CSV_CONFIG.get(csv_file.stem)

        if config is None:
            print(f"WARNING: 不支援的 CSV，跳過：{csv_file.name}")
            continue

        display_name = config["display_name"]
        ylabel = config["ylabel"]
        if reward_version == "v7" and csv_file.stem in AWFT_DISPLAY_CONFIG:
            display_name = AWFT_DISPLAY_CONFIG[csv_file.stem]["display_name"]
            ylabel = AWFT_DISPLAY_CONFIG[csv_file.stem]["ylabel"]

        df = pd.read_csv(csv_file, encoding="utf-8-sig")

        if "value" not in df.columns:
            print(f"WARNING: {csv_file.name} 找不到 value 欄位，跳過。")
            continue

        if config["zero_comparison"]:
            rows = draw_zero_comparison_boxplot(
                df["value"],
                display_name=display_name,
                ylabel=ylabel,
                output_dir=window_dir,
                file_stem=csv_file.stem,
            )
        else:
            rows = draw_full_boxplot(
                df["value"],
                display_name=display_name,
                ylabel=ylabel,
                output_dir=window_dir,
                file_stem=csv_file.stem,
            )

        statistics_rows.extend(rows)
        processed_count += 1

    if statistics_rows:
        statistics_file = window_dir / "boxplot_statistics.csv"
        pd.DataFrame(statistics_rows, columns=STATISTICS_CSV_COLUMNS).to_csv(
            statistics_file,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"統計 CSV：{statistics_file}")

    if processed_count == 0:
        print(f"WARNING: {window_dir.name} 沒有找到可處理的 CSV。")


# ============================================================
# 15. 主程式
# ============================================================

def main():
    folder_input = FOLDER_INPUT

    experiment_dir = find_experiment_dir(folder_input)
    reward_data_dir = experiment_dir / "reward_data_collection"

    if not reward_data_dir.is_dir():
        raise FileNotFoundError(
            "找不到 reward_data_collection：\n"
            f"  {reward_data_dir}"
        )

    window_dirs = sorted(
        path
        for path in reward_data_dir.iterdir()
        if path.is_dir() and path.name.startswith("window_")
    )

    if not window_dirs:
        raise FileNotFoundError(
            "reward_data_collection 底下找不到 window_xx 資料夾：\n"
            f"  {reward_data_dir}"
        )

    print("=" * 70)
    print("Boxplot 資料整理")
    print("=" * 70)
    print(f"Experiment：{experiment_dir}")
    print(f"Input data：{reward_data_dir}")
    print(f"Window 數量：{len(window_dirs)}")
    print()

    reward_version = None
    parent_name = experiment_dir.parent.name.lower()
    if parent_name.startswith("reward_v"):
        reward_version = parent_name[len("reward_") :]

    for window_dir in window_dirs:
        process_window(window_dir, reward_version=reward_version)

    print("=" * 70)
    print("全部圖片與統計 CSV 輸出完成")
    print("輸出位置：各個 reward_data_collection/window_xx 資料夾")
    print("=" * 70)


if __name__ == "__main__":
    main()
