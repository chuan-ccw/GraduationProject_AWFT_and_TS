from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def make_experiment_dir(
    base_dir: str | Path,
    reward_type: str,
    trade_start_date: str,
    trade_end_date: str,
    run_name: str | None = None,
) -> Path:
    """
    Create one folder for one experiment run.

    Example:
    C:/Users/User/Desktop/GraduationProject/Experence_result/reward_v1/2025-01-01_to_2026-03-20/20260523_132000
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_range = f"{trade_start_date}_to_{trade_end_date}"
    run_name = run_name or timestamp

    out_dir = Path(base_dir) / f"reward_{reward_type}" / date_range / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_config(out_dir: str | Path, config: Mapping[str, Any]) -> None:
    out_dir = Path(out_dir)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dict(config), f, indent=4, ensure_ascii=False, default=str)


def save_dataframe(df: pd.DataFrame, out_dir: str | Path, filename: str) -> Path:
    out_path = Path(out_dir) / filename
    df.to_csv(out_path, index=True)
    return out_path


def save_plot_current_figure(out_dir: str | Path, filename: str, dpi: int = 150) -> Path:
    import matplotlib.pyplot as plt

    out_path = Path(out_dir) / filename
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path


def copy_file_if_exists(src: str | Path, dst_dir: str | Path) -> Path | None:
    src = Path(src)
    if not src.exists():
        return None
    dst = Path(dst_dir) / src.name
    shutil.copy2(src, dst)
    return dst
