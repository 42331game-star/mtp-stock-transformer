"""Compatibility shim: ``python scripts/evaluate.py`` is ``mtp-evaluate``.

The implementation lives in :mod:`mtp.cli.evaluate`; import from there in new code.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtp.cli.evaluate import (  # noqa: E402,F401
    average_rank,
    collect,
    main,
    momentum_baseline,
    pearson,
    pearson_t,
    report,
    spearman,
)

if __name__ == "__main__":
    main()
