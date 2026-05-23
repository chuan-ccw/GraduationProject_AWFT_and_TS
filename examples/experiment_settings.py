from __future__ import annotations

import json
import os
from typing import Any


def env_str(name: str, default: str | None = None) -> str | None:
    """Read a string setting from environment variables."""
    return os.environ.get(name, default)


def env_json(name: str, default: Any) -> Any:
    """Read a JSON setting from environment variables."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Environment variable {name} is not valid JSON: {raw}") from exc


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean setting from environment variables."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    """Read an integer setting from environment variables."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    """Read a float setting from environment variables."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)
