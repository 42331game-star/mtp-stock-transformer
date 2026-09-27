"""Command-line entry points, installed as console scripts.

+---------------+--------------------------------+----------------------------------+
| Command       | Module                         | Equivalent                       |
+===============+================================+==================================+
| mtp-download  | :mod:`mtp.cli.download_data`   | python scripts/download_data.py  |
| mtp-train     | :mod:`mtp.cli.train`           | python scripts/train.py          |
| mtp-evaluate  | :mod:`mtp.cli.evaluate`        | python scripts/evaluate.py       |
| mtp-predict   | :mod:`mtp.cli.predict`         | python scripts/predict.py        |
| mtp-backtest  | :mod:`mtp.cli.backtest`        | python scripts/backtest.py       |
+---------------+--------------------------------+----------------------------------+

The ``scripts/*.py`` files are thin shims that import from here, so both spellings
run exactly the same code. Import from :mod:`mtp.cli.*` in new code.
"""
