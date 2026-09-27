"""Compatibility shim: ``python scripts/predict.py`` is ``mtp-predict``.

The implementation lives in :mod:`mtp.cli.predict`; import from there in new code.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtp.cli.predict import build_window, main, report  # noqa: E402,F401

if __name__ == "__main__":
    main()
