"""Backward-compatible entry point.

Trading Decision Score (TDS/ATDS) was renamed to Trading Score (TS).
Use TS_turnover_graph.py for new work.
"""

from TS_turnover_graph import *  # noqa: F401,F403
from TS_turnover_graph import main


if __name__ == "__main__":
    main()
