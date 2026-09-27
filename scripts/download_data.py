"""Compatibility shim: ``python scripts/download_data.py`` is ``mtp-download``.

The implementation lives in :mod:`mtp.cli.download_data`; import from there in
new code. The ``--universe universe`` ticker list shipped with the package is
:meth:`mtp.cli.download_data.UNIVERSE_FILE`.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtp.cli.download_data import (  # noqa: E402,F401
    LARGE_UNIVERSE,
    TW_UNIVERSE,
    UNIVERSE_FILE,
    UNIVERSES,
    fetch,
    load_universe,
    main,
)

if __name__ == "__main__":
    main()
