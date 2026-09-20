"""Backward-compatible entry point.

Trading Decision Score (TDS/ATDS) was renamed to Trading Score (TS).
Use Compare_TS_Trading_Behavior_Add_Stock_Trend_smaller_trend.py for new work.
"""

from Compare_TS_Trading_Behavior_Add_Stock_Trend_smaller_trend import *  # noqa: F401,F403
from Compare_TS_Trading_Behavior_Add_Stock_Trend_smaller_trend import main


if __name__ == "__main__":
    main()
