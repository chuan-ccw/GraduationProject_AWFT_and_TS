from pathlib import Path
import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from matplotlib.cbook import boxplot_stats

# 從 reward_data_collection 抓取 AWFT 的 a / w 資料， 紀錄 畫圖(暫時小功能)

# 篩選指定集合後畫 box plot 並輸出統計資料。

FOLDER_INPUT = "rolling_2018-2025_train3y_test1y_shift1y_20260904_234026_reinforce"


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

# 刻度字大小
TICK_LABEL_SIZE = 10


# ============================================================
# 3. 輸出欄位
# ============================================================

BOXPLOT_STATISTICS_CSV_COLUMNS = [
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

AWFT_SUMMARY_COLUMNS = [
    "a_nonzero_count",
    "w_hold_nonzero_count",
    "mean_a_nonzero",
    "mean_w_hold_nonzero",
    "mean_a_nonzero_div_mean_w_hold_nonzero",
    "mean_a_all",
    "mean_w_all",
    "mean_a_all_div_mean_w_all",
]


# ============================================================
# 4. 尋找 rolling 實驗資料夾
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
        match_text = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            "找到多個同名 rolling 資料夾，請改輸入完整路徑：\n"
            f"{match_text}"
        )

    return matches[0]


# ============================================================
# 5. 資料清理
# ============================================================

def clean_data(data):
    """
    將資料轉成數值並移除 NaN。
    """
    return pd.to_numeric(data, errors="coerce").dropna()


def load_value_csv(csv_path):
    """
    讀取 reward_data_collection 中的單一變數 CSV。

    預期欄位：
        value
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "value" not in df.columns:
        raise ValueError(f"{csv_path.name} 找不到 value 欄位。")

    return pd.to_numeric(df["value"], errors="coerce")


# ============================================================
# 6. 取得 Boxplot 統計值
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
# 7. 統計資訊
# ============================================================

def get_statistics(data, variable, dataset):
    """
    計算 Boxplot 統計資訊，並回傳一列可寫入 CSV 的資料。
    """
    data = clean_data(data)

    if data.empty:
        raise ValueError(f"{variable} - {dataset} 沒有可用資料。")

    box = get_boxplot_values(data)

    count = len(data)
    zero_count = int((data == 0).sum())

    whisker_mask = (
        (data >= box["lower_whisker"])
        & (data <= box["upper_whisker"])
    )
    whisker_range_count = int(whisker_mask.sum())
    outlier_count = count - whisker_range_count

    return {
        "variable": variable,
        "dataset": dataset,
        "minimum": data.min(),
        "lower_whisker": box["lower_whisker"],
        "q1_25_percent": box["q1"],
        "q2_median_50_percent": box["median"],
        "q3_75_percent": box["q3"],
        "upper_whisker": box["upper_whisker"],
        "maximum": data.max(),
        "count": count,
        "zero_count": zero_count,
        "zero_ratio_percent": zero_count / count * 100,
        "whisker_range_count": whisker_range_count,
        "whisker_range_ratio_percent": whisker_range_count / count * 100,
        "outlier_count": outlier_count,
        "outlier_ratio_percent": outlier_count / count * 100,
    }


def print_statistics(stats):
    print("=" * 70)
    print(f"{stats['variable']} - {stats['dataset']}")
    print("=" * 70)

    print(f"資料筆數       : {stats['count']}")
    print(f"Minimum        : {stats['minimum']:.8f}")
    print(f"Lower Whisker  : {stats['lower_whisker']:.8f}")
    print(f"Q1 (25%)       : {stats['q1_25_percent']:.8f}")
    print(f"Median (50%)   : {stats['q2_median_50_percent']:.8f}")
    print(f"Q3 (75%)       : {stats['q3_75_percent']:.8f}")
    print(f"Upper Whisker  : {stats['upper_whisker']:.8f}")
    print(f"Maximum        : {stats['maximum']:.8f}")
    print(f"Whisker 範圍   : {stats['whisker_range_ratio_percent']:.2f}%")
    print(f"離群值比例     : {stats['outlier_ratio_percent']:.2f}%")
    print()


# ============================================================
# 8. 主圖 Y 軸範圍
# ============================================================

def get_whisker_range(data, include_zero=True, margin_ratio=MAIN_MARGIN_RATIO):
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


def set_dense_y_ticks(ax, number=12):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=number))


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
# 9. AWFT Boxplot
# ============================================================

def draw_awft_boxplot(
    data,
    display_name,
    ylabel,
    output_dir,
    file_stem,
    include_zero=True,
):
    """
    每個集合輸出：
    1. 主圖：不顯示離群值，Y 軸聚焦 whisker
    2. Full range：顯示完整資料與離群值
    """
    data = clean_data(data)

    if data.empty:
        raise ValueError(f"{display_name} 沒有可用資料。")

    stats = get_statistics(
        data,
        variable=display_name,
        dataset="Filtered data",
    )
    print_statistics(stats)

    # -------------------------
    # 主圖
    # -------------------------
    fig, ax = create_single_figure()

    ax.boxplot(
        data,
        vert=True,
        widths=0.42,
        whis=WHIS,
        showfliers=False,
    )

    ax.set_title(
        f"Box Plot of {display_name}",
        fontsize=MAIN_TITLE_SIZE,
        pad=18,
        loc="center",
    )
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks([])
    ax.set_xlim(0.68, 1.32)

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

    y_min, y_max = get_whisker_range(
        data,
        include_zero=include_zero,
        margin_ratio=MAIN_MARGIN_RATIO,
    )
    ax.set_ylim(y_min, y_max)

    save_path = output_dir / f"{file_stem}_boxplot.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------
    # Full range
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

    ax.set_title(
        f"Box Plot of {display_name} (Full range)",
        fontsize=FULL_TITLE_SIZE,
        pad=18,
        loc="center",
    )
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks([])
    ax.set_xlim(0.68, 1.32)

    minimum = data.min()
    maximum = data.max()
    full_range = maximum - minimum

    if full_range > 0:
        margin = full_range * 0.03
        ax.set_ylim(minimum - margin, maximum + margin)

    set_dense_y_ticks(ax, number=10)

    ax.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
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

    full_save_path = output_dir / f"{file_stem}_boxplot_full_range.png"
    fig.savefig(full_save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return stats


# ============================================================
# 10. 找出單一 Window 的 a.csv / w.csv
# ============================================================

def find_variable_csv(window_dir, variable):
    """
    reward_data_collection/window_xx 預期有：
        a.csv
        w.csv

    若未來檔名不是完全 a.csv / w.csv，
    則允許唯一一個以 a / w 開頭的 CSV。
    """
    exact_path = window_dir / f"{variable}.csv"

    if exact_path.is_file():
        return exact_path

    matches = sorted(window_dir.glob(f"{variable}*.csv"))

    if not matches:
        raise FileNotFoundError(
            f"{window_dir.name} 找不到 {variable}.csv。"
        )

    if len(matches) > 1:
        match_text = "\n".join(f"  - {path.name}" for path in matches)
        raise RuntimeError(
            f"{window_dir.name} 找到多個可能的 {variable} CSV：\n"
            f"{match_text}\n"
            f"請保留唯一一個 {variable} 資料檔，或改成 {variable}.csv。"
        )

    return matches[0]


# ============================================================
# 11. AWFT 集合與 summary
# ============================================================

def calculate_awft_datasets(a_data, w_data):
    """
    建立使用者指定的兩個集合：

    A = {a | a != 0}
    W = {w | a == 0, w != 0}

    a 與 w 必須逐筆對齊，因此長度必須完全相同。
    """
    if len(a_data) != len(w_data):
        raise ValueError(
            "a 與 w 的資料筆數不同，無法依照同一筆交易狀態篩選：\n"
            f"  len(a) = {len(a_data)}\n"
            f"  len(w) = {len(w_data)}"
        )

    aligned = pd.DataFrame(
        {
            "a": pd.to_numeric(a_data, errors="coerce"),
            "w": pd.to_numeric(w_data, errors="coerce"),
        }
    ).dropna(subset=["a", "w"])

    a_nonzero = aligned.loc[
        aligned["a"] != 0,
        "a",
    ]

    w_hold_nonzero = aligned.loc[
        (aligned["a"] == 0) & (aligned["w"] != 0),
        "w",
    ]

    return aligned, a_nonzero, w_hold_nonzero


def safe_divide(numerator, denominator):
    if pd.isna(denominator) or denominator == 0:
        return float("nan")

    return numerator / denominator


def get_awft_summary(aligned, a_nonzero, w_hold_nonzero):
    """
    計算指定統計：

    1. {a | a != 0}
    2. {w | a == 0, w != 0}
    3. mean({a | a != 0})
    4. mean({w | a == 0, w != 0})
    5. mean({a | a != 0}) / mean({w | a == 0, w != 0})
    6. mean(a) / mean(w)

    注意：這裡的 mean(a) 是 signed arithmetic mean，
    不會自動轉成 abs(a)。
    """
    mean_a_nonzero = a_nonzero.mean()
    mean_w_hold_nonzero = w_hold_nonzero.mean()

    mean_a_all = aligned["a"].mean()
    mean_w_all = aligned["w"].mean()

    return {
        "a_nonzero_count": len(a_nonzero),
        "w_hold_nonzero_count": len(w_hold_nonzero),
        "mean_a_nonzero": mean_a_nonzero,
        "mean_w_hold_nonzero": mean_w_hold_nonzero,
        "mean_a_nonzero_div_mean_w_hold_nonzero": safe_divide(
            mean_a_nonzero,
            mean_w_hold_nonzero,
        ),
        "mean_a_all": mean_a_all,
        "mean_w_all": mean_w_all,
        "mean_a_all_div_mean_w_all": safe_divide(
            mean_a_all,
            mean_w_all,
        ),
    }


def print_awft_summary(summary):
    print("=" * 70)
    print("AWFT Scale Statistics")
    print("=" * 70)
    print(f"|{{a | a != 0}}|                         : {summary['a_nonzero_count']}")
    print(f"|{{w | a == 0, w != 0}}|                : {summary['w_hold_nonzero_count']}")
    print(f"mean({{a | a != 0}})                     : {summary['mean_a_nonzero']:.8f}")
    print(
        "mean({w | a == 0, w != 0})             : "
        f"{summary['mean_w_hold_nonzero']:.8f}"
    )
    print(
        "mean({a | a != 0}) / mean({w | a == 0, w != 0}) : "
        f"{summary['mean_a_nonzero_div_mean_w_hold_nonzero']:.8f}"
    )
    print(f"mean(a)                                  : {summary['mean_a_all']:.8f}")
    print(f"mean(w)                                  : {summary['mean_w_all']:.8f}")
    print(
        "mean(a) / mean(w)                      : "
        f"{summary['mean_a_all_div_mean_w_all']:.8f}"
    )
    print()


# ============================================================
# 12. 處理單一 Window
# ============================================================

def process_window(window_dir):
    print("#" * 70)
    print(f"處理 Window：{window_dir.name}")
    print(f"位置：{window_dir}")
    print("#" * 70)

    a_csv = find_variable_csv(window_dir, "a")
    w_csv = find_variable_csv(window_dir, "w")

    print(f"a CSV：{a_csv.name}")
    print(f"w CSV：{w_csv.name}")
    print()

    a_data = load_value_csv(a_csv)
    w_data = load_value_csv(w_csv)

    aligned, a_nonzero, w_hold_nonzero = calculate_awft_datasets(
        a_data,
        w_data,
    )

    if a_nonzero.empty:
        raise ValueError(
            f"{window_dir.name} 的 {{a | a != 0}} 為空集合，無法畫 Boxplot。"
        )

    if w_hold_nonzero.empty:
        raise ValueError(
            f"{window_dir.name} 的 {{w | a == 0, w != 0}} 為空集合，"
            "無法畫 Boxplot。"
        )

    # -------------------------
    # 1. 儲存篩選後集合
    # -------------------------
    a_nonzero_file = window_dir / "awft_a_nonzero.csv"
    w_hold_nonzero_file = window_dir / "awft_w_a_zero_nonzero.csv"

    pd.DataFrame({"value": a_nonzero.reset_index(drop=True)}).to_csv(
        a_nonzero_file,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({"value": w_hold_nonzero.reset_index(drop=True)}).to_csv(
        w_hold_nonzero_file,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------
    # 2. 畫 Boxplot
    # -------------------------
    a_stats = draw_awft_boxplot(
        a_nonzero,
        display_name="AWFT Action {a | a != 0}",
        ylabel="Action a",
        output_dir=window_dir,
        file_stem="awft_a_nonzero",
        include_zero=True,
    )

    w_stats = draw_awft_boxplot(
        w_hold_nonzero,
        display_name="AWFT Weight {w | a = 0, w != 0}",
        ylabel="Weight w",
        output_dir=window_dir,
        file_stem="awft_w_a_zero_nonzero",
        include_zero=True,
    )

    # -------------------------
    # 3. Boxplot 統計
    # -------------------------
    boxplot_statistics_file = window_dir / "awft_boxplot_statistics.csv"

    pd.DataFrame(
        [a_stats, w_stats],
        columns=BOXPLOT_STATISTICS_CSV_COLUMNS,
    ).to_csv(
        boxplot_statistics_file,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------
    # 4. mean / ratio 統計
    # -------------------------
    summary = get_awft_summary(
        aligned,
        a_nonzero,
        w_hold_nonzero,
    )
    print_awft_summary(summary)

    summary_file = window_dir / "awft_scale_statistics.csv"

    pd.DataFrame(
        [summary],
        columns=AWFT_SUMMARY_COLUMNS,
    ).to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"篩選資料：{a_nonzero_file}")
    print(f"篩選資料：{w_hold_nonzero_file}")
    print(f"Boxplot 統計：{boxplot_statistics_file}")
    print(f"AWFT mean / ratio 統計：{summary_file}")
    print()


# ============================================================
# 13. 主程式
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
    print("AWFT Reward Data Collection Boxplot")
    print("=" * 70)
    print(f"Experiment：{experiment_dir}")
    print(f"Input data：{reward_data_dir}")
    print(f"Window 數量：{len(window_dirs)}")
    print()

    for window_dir in window_dirs:
        process_window(window_dir)

    print("=" * 70)
    print("全部 AWFT 圖片與統計 CSV 輸出完成")
    print("輸出位置：各個 reward_data_collection/window_xx 資料夾")
    print("=" * 70)


if __name__ == "__main__":
    main()
