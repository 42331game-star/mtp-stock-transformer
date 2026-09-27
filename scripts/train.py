"""Compatibility shim: ``python scripts/train.py`` is ``mtp-train``.

The implementation lives in :mod:`mtp.cli.train`; import from there in new code.
"""

from __future__ import annotations

import os
import sys

# keep working straight from a source checkout, without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtp.cli.train import (  # noqa: E402,F401
    apply_args,
    build_parser,
    main,
    parse_args,
    seed_everything,
)

if __name__ == "__main__":
    main()
