from __future__ import annotations

import os
import runpy
from pathlib import Path

os.environ["TARGET_REWARD_TYPE"] = "v6"
os.environ["COMPARE_DATA_MODE"] = "annual_reset"

runpy.run_path(
    str(Path(__file__).with_name("Compare_V0_V1_Manual.py")),
    run_name="__main__",
)
