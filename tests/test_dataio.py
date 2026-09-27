"""yfinance column/date normalisation — the bug that made downloads a no-op."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import make_bars  # noqa: E402
from mtp import REQUIRED_COLS, normalize_ohlcv, read_symbol  # noqa: E402

N = 30


def _flat(n: int = N, index_name: str | None = "Date") -> pd.DataFrame:
    df = make_bars(n).drop(columns=["date"])
    df.index = pd.date_range("2024-01-01", periods=n, freq="B", name=index_name)
    return df.rename(columns=str.title)


def test_flat_columns_with_named_date_index():
    out = normalize_ohlcv(_flat(index_name="Date"))
    assert out is not None
    assert list(out.columns) == list(REQUIRED_COLS)
    assert len(out) == N
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_flat_columns_with_unnamed_index():
    """yfinance sometimes returns an index with no name -> column ``index``."""
    out = normalize_ohlcv(_flat(index_name=None))
    assert out is not None
    assert list(out.columns) == list(REQUIRED_COLS)


def test_multiindex_price_then_ticker():
    flat = make_bars(N).drop(columns=["date"])
    raw = pd.DataFrame(
        flat.to_numpy(),
        index=pd.date_range("2024-01-01", periods=N, freq="B", name="Date"),
        columns=pd.MultiIndex.from_product([flat.columns, ["TEST"]]),
    )
    out = normalize_ohlcv(raw)
    assert out is not None
    assert list(out.columns) == list(REQUIRED_COLS)


def test_multiindex_ticker_then_price():
    flat = make_bars(N).drop(columns=["date"])
    raw = pd.DataFrame(
        flat.to_numpy(),
        index=pd.date_range("2024-01-01", periods=N, freq="B", name="Date"),
        columns=pd.MultiIndex.from_product([["TEST"], flat.columns]),
    )
    out = normalize_ohlcv(raw)
    assert out is not None
    assert list(out.columns) == list(REQUIRED_COLS)


def test_missing_volume_returns_none():
    raw = _flat().drop(columns=["Volume"])
    assert normalize_ohlcv(raw) is None


def test_several_tickers_in_one_frame_returns_none():
    """A multi-ticker download must not be silently flattened into garbage."""
    flat = make_bars(N).drop(columns=["date"])
    cols = pd.MultiIndex.from_product([["AAA", "BBB"], flat.columns])
    raw = pd.DataFrame(
        np.tile(flat.to_numpy(), (1, 2)),   # same series for both tickers
        index=pd.date_range("2024-01-01", periods=N, freq="B", name="Date"),
        columns=cols,
    )
    assert normalize_ohlcv(raw) is None


def test_empty_frame_returns_none():
    assert normalize_ohlcv(pd.DataFrame()) is None
    assert normalize_ohlcv(None) is None


def test_duplicate_dates_are_dropped():
    raw = _flat()
    doubled = pd.concat([raw, raw.iloc[[-1]]])
    out = normalize_ohlcv(doubled)
    assert out is not None
    assert not out["date"].duplicated().any()
    assert len(out) == N


def test_dates_are_sorted():
    raw = _flat().iloc[::-1]
    out = normalize_ohlcv(raw)
    assert out is not None
    assert out["date"].is_monotonic_increasing
    assert np.allclose(
        out["close"].to_numpy(), make_bars(N)["close"].to_numpy()
    ), "reversing the frame must not change the series"


def test_read_symbol_normalises_legacy_parquet(tmp_path):
    df = make_bars(N).rename(columns={"date": "Date"})
    path = tmp_path / "LEGACY.parquet"
    df.to_parquet(path, index=False)
    out = read_symbol(str(path))
    assert list(out.columns)[0] == "date"
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
