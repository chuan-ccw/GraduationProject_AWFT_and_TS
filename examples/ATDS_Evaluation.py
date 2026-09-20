"""Backward-compatible entry point.

Trading Decision Score (TDS/ATDS) was renamed to Trading Score (TS).
Use TS_Evaluation.py for new work.
"""

from TS_Evaluation import *  # noqa: F401,F403
from TS_Evaluation import main


if __name__ == "__main__":
    main()
