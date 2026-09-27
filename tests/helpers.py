"""Synthetic OHLCV data for the test-suite.

No network access and no real market files: everything the tests need is
generated here so a fresh clone can run ``pytest`` offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_bars(
    n: int = 700,
    start: str = "2023-01-02",
    seed: int = 0,
    drift: float = 0.0,
    freq: str = "B",
) -> pd.DataFrame:
    """Random-walk OHLCV frame with ``n`` rows.

    ``drift`` is the daily log-drift; a non-zero value gives the series a clear
    trend, which is what exposes any code that scales windows with full-series
    statistics instead of window-local ones.
    """
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift, 0.01, n))
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq=freq),
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.01, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.01, n))),
        "close": close,
        "volume": rng.integers(1e5, 1e7, n).astype(float),
    })


def make_regime_bars(n: int = 800, start: str = "2020-01-01", seed: int = 7) -> pd.DataFrame:
    """Series with two clearly different regimes (quiet -> wild).

    Scale-dependent features (ATR, Bollinger width, relative volume) sit far
    apart in the two halves, so any code that standardises with *whole-series*
    statistics produces windows whose per-window mean/std is not 0/1 — which is
    exactly what ``test_scaling_is_window_local`` asserts against.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    sigma = np.where(np.arange(n) < half, 0.004, 0.03)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, sigma, n))
    volume = rng.integers(1e5, 1e7, n).astype(float)
    volume[half:] *= 6
    spread = sigma * 1.5
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="B"),
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * (1 + np.abs(rng.normal(0, spread, n))),
        "low": close * (1 - np.abs(rng.normal(0, spread, n))),
        "close": close,
        "volume": volume,
    })
