"""Backward-compatible entry point.

Trading Decision Score (TDS/ATDS) was renamed to Trading Score (TS).
Use Compare_TS_Trading_Behavior.py for new work.
"""

from Compare_TS_Trading_Behavior import *  # noqa: F401,F403
from Compare_TS_Trading_Behavior import main


if __name__ == "__main__":
    main()
