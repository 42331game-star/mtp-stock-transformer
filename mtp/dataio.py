"""Normalising raw price data (yfinance, CSV exports) into the canonical frame.

Everything downstream — features, dataset, backtest — expects exactly
``date / open / high / low / close / volume``, lowercase. Fetching lives in
``mtp-download`` / ``mtp-predict``; this module is the shared contract they both validate against.
"""

from __future__ import annotations

import pandas as pd

#: canonical column order
REQUIRED_COLS = ("date", "open", "high", "low", "close", "volume")

#: price field names used to locate the right level of a MultiIndex
PRICE_FIELDS = {"open", "high", "low", "close", "adj close", "adjclose", "volume"}

#: index/column names that may hold the date instead of ``date``
DATE_ALIASES = ("date", "datetime", "time", "index")


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame | None:
    """Flatten a yfinance-style frame into ``date/open/high/low/close/volume``.

    Handles every layout yfinance has shipped over time:

    * flat columns ``Open, High, ...`` with an index named ``Date`` / ``date``
      / nothing at all,
    * MultiIndex ``(Price, Ticker)`` **or** ``(Ticker, Price)``,
    * a date column called ``Date`` / ``index`` / ``datetime``.

    Returns ``None`` when the frame has no usable OHLCV columns — callers must
    treat that as "skip this ticker", never as "fill the gaps".
    """
    if df is None or len(df) == 0:
        return None
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        for level in range(out.columns.nlevels):
            values = {str(v).strip().lower() for v in out.columns.get_level_values(level)}
            if values & PRICE_FIELDS:
                out.columns = out.columns.get_level_values(level)
                break
        else:  # no price-looking level -> give up rather than guess
            return None
    out.columns = [str(c).strip().lower() for c in out.columns]

    if "date" not in out.columns:
        out = out.reset_index()
        # reset_index may yield "date" (named index), "index" (unnamed) or
        # "level_0"; lowercase first so ``Date`` matches too
        out.columns = [
            "index" if str(c) == "level_0" else str(c).strip().lower()
            for c in out.columns
        ]
    if "date" not in out.columns:
        datetime_cols = [c for c in out.columns if pd.api.types.is_datetime64_any_dtype(out[c])]
        if not datetime_cols:
            return None
        out = out.rename(columns={datetime_cols[0]: "date"})

    try:
        out["date"] = pd.to_datetime(out["date"])
    except (TypeError, ValueError):
        return None

    if any(c not in out.columns for c in REQUIRED_COLS):
        return None
    if out.columns.duplicated().any():  # e.g. several tickers in one frame
        return None
    if out["date"].duplicated().any():
        out = out.drop_duplicates(subset="date", keep="last")
    return out.sort_values("date").reset_index(drop=True)[list(REQUIRED_COLS)]
