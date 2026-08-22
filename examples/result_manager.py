from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


REWARD_EXPERIMENT_HISTORY_FILENAME = "reward_experiment_history.csv"
COMPARE_EXPERIMENT_HISTORY_FILENAME = "compare_experiment_history.csv"


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


def save_config(out_dir: str | Path, config: Mapping[str, Any], filename: str = "config.json") -> None:
    out_dir = Path(out_dir)
    with open(out_dir / filename, "w", encoding="utf-8") as f:
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


def _flatten_mapping(prefix: str, value: Any) -> dict[str, Any]:
    """Flatten nested dictionaries into stable CSV columns."""
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}_{key}" if prefix else str(key)
            flattened.update(_flatten_mapping(nested_prefix, nested_value))
        return flattened
    if isinstance(value, (list, tuple, set)):
        return {prefix: json.dumps(list(value), ensure_ascii=False, default=str)}
    return {prefix: value}


def _append_csv_row(csv_path: str | Path, row: Mapping[str, Any]) -> Path:
    """Append one row while preserving columns created by earlier reward versions."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([dict(row)])

    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(csv_path, encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        columns = list(existing.columns)
        columns.extend(column for column in new_row.columns if column not in columns)
        combined = pd.concat(
            [existing.reindex(columns=columns), new_row.reindex(columns=columns)],
            ignore_index=True,
        )
    else:
        combined = new_row

    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_experiment_metadata(folder: str | Path) -> dict[str, Any]:
    """Read reward/model settings from a rolling, window, times, or curve folder."""
    folder_path = Path(folder).expanduser().resolve()
    start_dir = folder_path if folder_path.is_dir() else folder_path.parent
    config_names = [
        "rolling_config.json",
        "experiment_config.json",
        "continuous_config.json",
    ]

    config_payload: dict[str, Any] = {}
    config_path: Path | None = None
    current = start_dir
    for _ in range(8):
        for filename in config_names:
            candidate = current / filename
            if candidate.exists():
                config_payload = _read_json(candidate)
                if config_payload:
                    config_path = candidate
                    break
        if config_payload or current.parent == current:
            break
        current = current.parent

    reward_type = str(config_payload.get("reward_type", "") or "").strip()
    if not reward_type:
        for part in start_dir.parts[::-1]:
            match = re.fullmatch(r"reward_(.+)", part, flags=re.IGNORECASE)
            if match:
                reward_type = match.group(1)
                break

    reward_params = config_payload.get("reward_params", {})
    if not isinstance(reward_params, Mapping):
        reward_params = {}

    model = str(config_payload.get("primary_agent_name", "") or "").strip().lower()
    if not model:
        enabled = config_payload.get("enabled_agent_names", [])
        if isinstance(enabled, list) and enabled:
            model = ",".join(str(item) for item in enabled)
    if not model:
        for key in ["if_using_reinforce", "if_using_ppo", "if_using_a2c", "if_using_ddpg", "if_using_td3", "if_using_sac"]:
            if config_payload.get(key):
                model = key.replace("if_using_", "")
                break

    return {
        "folder_name": start_dir.name,
        "folder_path": str(start_dir),
        "model": model,
        "reward_type": reward_type,
        "reward_params": dict(reward_params),
        "config_path": str(config_path) if config_path else "",
    }


def append_reward_experiment_record(
    experiment_root: str | Path,
    experiment_dir: str | Path,
    config_payload: Mapping[str, Any],
) -> Path:
    """Append one top-level Reward experiment to the central history CSV."""
    experiment_dir = Path(experiment_dir).resolve()
    payload = dict(config_payload)
    row: dict[str, Any] = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "folder_name": experiment_dir.name,
        "folder_path": str(experiment_dir),
        "model": payload.get("primary_agent_name", ""),
        "reward_type": payload.get("reward_type", ""),
        "indicator_mode": payload.get("indicator_mode", ""),
        "use_expanded_indicators": payload.get("use_expanded_indicators", ""),
        "times_experiments": payload.get("times_experiments", ""),
        "rolling_window_mode": payload.get("rolling_window_mode", ""),
        "rolling_start_date": payload.get("rolling_start_date", ""),
        "rolling_end_date": payload.get("rolling_end_date", ""),
        "train_years": payload.get("train_years", ""),
        "test_years": payload.get("test_years", ""),
        "shift_years": payload.get("shift_years", ""),
    }
    row.update(_flatten_mapping("reward_param", payload.get("reward_params", {})))
    return _append_csv_row(Path(experiment_root) / REWARD_EXPERIMENT_HISTORY_FILENAME, row)


def append_compare_experiment_record(
    experiment_root: str | Path,
    compare_dir: str | Path,
    source_a_folder: str | Path,
    source_b_folder: str | Path,
    compare_type: str,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Append one compare execution with both source experiments and reward parameters."""
    source_a = load_experiment_metadata(source_a_folder)
    source_b = load_experiment_metadata(source_b_folder)
    compare_dir = Path(compare_dir).resolve()

    row: dict[str, Any] = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "compare_type": compare_type,
        "compare_folder_name": compare_dir.name,
        "compare_folder_path": str(compare_dir),
        "source_a_folder_name": source_a["folder_name"],
        "source_a_folder_path": source_a["folder_path"],
        "source_a_model": source_a["model"],
        "source_a_reward_type": source_a["reward_type"],
        "source_a_config_path": source_a["config_path"],
        "source_b_folder_name": source_b["folder_name"],
        "source_b_folder_path": source_b["folder_path"],
        "source_b_model": source_b["model"],
        "source_b_reward_type": source_b["reward_type"],
        "source_b_config_path": source_b["config_path"],
    }
    row.update(_flatten_mapping("source_a_reward_param", source_a["reward_params"]))
    row.update(_flatten_mapping("source_b_reward_param", source_b["reward_params"]))
    if extra:
        row.update(_flatten_mapping("compare", dict(extra)))

    return _append_csv_row(Path(experiment_root) / COMPARE_EXPERIMENT_HISTORY_FILENAME, row)
