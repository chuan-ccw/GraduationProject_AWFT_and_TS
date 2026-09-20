"""Backward-compatible entry point.

Trading Decision Score (TDS/ATDS) was renamed to Trading Score (TS).
Use Compare_TS_Trading_Behavior_Add_Stock_Trend.py for new work.
"""

from Compare_TS_Trading_Behavior_Add_Stock_Trend import *  # noqa: F401,F403
from Compare_TS_Trading_Behavior_Add_Stock_Trend import main


if __name__ == "__main__":
    main()
