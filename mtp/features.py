"""Shared feature engineering.

Training, back-testing and live inference all import from this module, so the
16 inputs the model sees are guaranteed to be identical in every setting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Multi-step prediction horizons (trading days).
HORIZONS = (1, 3, 5, 10)

#: The 16 model inputs, in order. Order is part of the checkpoint contract.
FEATURE_COLS = [
    # 1. return / microstructure
    "ret_1d", "ret_5d", "ret_20d", "hl_ratio", "vol_change",
    # 2. moving-average bias (5 / 20 / 60 / 120 / 240)
    "bias_5", "bias_20", "bias_60", "bias_120", "bias_240",
    # 3. volatility channels
    "atr_norm", "bb_width", "bb_pct_b",
    # 4. momentum
    "rsi_norm",
    # 5. volume / candle shape
    "vol_ma20_ratio", "upper_shadow",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 16 technical features. Returns a copy; input is untouched.

    Expects columns: open, high, low, close, volume.
    """
    out = df.copy()
    close = out["close"]

    # 1. log returns & range
    out["ret_1d"] = np.log(close / close.shift(1))
    out["ret_5d"] = np.log(close / close.shift(5))
    out["ret_20d"] = np.log(close / close.shift(20))
    out["hl_ratio"] = (out["high"] - out["low"]) / close
    out["vol_change"] = np.log((out["volume"] + 1) / (out["volume"].shift(1) + 1))

    # 2. moving-average bias (mean reversion / trend stretch)
    for w in (5, 20, 60, 120, 240):
        out[f"bias_{w}"] = (close - close.rolling(w).mean()) / close

    # 3. ATR(14) normalised by price + Bollinger band position
    tr = np.maximum(
        out["high"] - out["low"],
        np.maximum(
            (out["high"] - close.shift(1)).abs(),
            (out["low"] - close.shift(1)).abs(),
        ),
    )
    out["atr_raw"] = tr.rolling(14).mean()
    out["atr_norm"] = out["atr_raw"] / close

    mid = close.rolling(20).mean()
    std = close.rolling(20).std() + 1e-8
    out["bb_width"] = (4 * std) / close
    out["bb_pct_b"] = (close - (mid - 2 * std)) / (4 * std)

    # 4. RSI(14), re-centred to [-0.5, 0.5]
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean() + 1e-8
    out["rsi_norm"] = (100 - (100 / (1 + gain / loss))) / 100.0 - 0.5

    # 5. relative volume & upper candle shadow
    out["vol_ma20_ratio"] = out["volume"] / (out["volume"].rolling(20).mean() + 1.0)
    out["upper_shadow"] = (out["high"] - np.maximum(out["close"], out["open"])) / close

    return out


def add_targets(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Forward-looking cumulative log returns for each horizon."""
    out = df.copy()
    for h in horizons:
        out[f"target_h{h}"] = np.log(out["close"].shift(-h) / out["close"])
    return out


def standardize(window: np.ndarray):
    """Window-local z-score for a **single** window ``[time, feature]``.

    mean/std come from that window alone — never from the full series, so a
    window can never be scaled with statistics that lie in its future.
    Used by live inference (``mtp-predict``).

    Returns ``(standardized, mean, std)``.
    """
    window = np.asarray(window)
    mean = window.mean(axis=0)
    std = window.std(axis=0) + 1e-7
    return (window - mean) / std, mean, std


def standardize_windows(windows: np.ndarray) -> np.ndarray:
    """Window-local z-score for a batch ``[n, time, feature]``.

    Each window is normalised with **its own** mean/std, i.e. bit-for-bit the
    same semantics as :func:`standardize` applied to one window. Training and
    back-testing use this, so the three code paths (train / backtest / live)
    all see identically scaled inputs.
    """
    w = np.asarray(windows, dtype=np.float32)
    if w.ndim != 3:
        raise ValueError(f"expected [n, time, feature], got shape {w.shape}")
    mean = w.mean(axis=1, keepdims=True)
    std = w.std(axis=1, keepdims=True) + 1e-7
    return (w - mean) / std
