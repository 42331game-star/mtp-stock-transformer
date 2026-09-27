"""Compatibility shim: ``python scripts/backtest.py`` is ``mtp-backtest``.

The implementation lives in :mod:`mtp.cli.backtest`; import from there in new code.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtp.cli.backtest import backtest_symbol, main, spy_stats  # noqa: E402,F401

if __name__ == "__main__":
    main()
