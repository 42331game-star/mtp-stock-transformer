"""Download OHLCV history from yfinance into local parquet files.

Usage:
    mtp-download --universe large     # 54 core US names/ETFs (training set)
    mtp-download --universe universe  # 200+ US broad pool
    mtp-download --universe tw        # 50 Taiwan large caps

    python scripts/download_data.py --universe large  # same code, no install needed

Columns are written lowercase with an explicit ``date`` column; the layout
normalisation lives in ``mtp.dataio.normalize_ohlcv`` so that live inference
(``mtp-predict``) applies exactly the same rules.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from mtp.dataio import normalize_ohlcv

# --- universes -------------------------------------------------------------

#: Training universe: 50 core US names + major ETFs (data/market_large).
LARGE_UNIVERSE = [
    "AAPL", "AMAT", "AXP", "BA", "BAC", "BLK", "C", "CAT", "COP", "COST",
    "CSCO", "CVX", "DE", "DIA", "F", "FCX", "GE", "GS", "HON", "IBM",
    "INTC", "IWM", "JNJ", "JPM", "KO", "MDT", "MMM", "MRK", "MS", "MSFT",
    "NEM", "NVDA", "ORCL", "PEP", "PG", "QCOM", "QQQ", "SLB", "SPY", "TXN",
    "UNH", "UNP", "WFC", "WMT", "XLE", "XLF", "XLI", "XLK", "XLU", "XLV",
    "XOM",
]

#: Broad US pool (data/market_universe) — see download list in docs.
UNIVERSE_FILE = os.path.join(os.path.dirname(__file__), "universe_us.txt")

#: Taiwan 50 large caps (data/market_tw).
TW_UNIVERSE = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2881.TW", "2882.TW", "2412.TW",
    "2891.TW", "3711.TW", "2886.TW", "2303.TW", "6669.TW", "2603.TW", "1216.TW", "2357.TW",
    "2885.TW", "2345.TW", "2884.TW", "3045.TW", "2892.TW", "5880.TW", "2880.TW", "3008.TW",
    "2207.TW", "6505.TW", "4904.TW", "2002.TW", "3034.TW", "3231.TW", "2327.TW", "2890.TW",
    "2883.TW", "2301.TW", "2395.TW", "1303.TW", "2408.TW", "3037.TW", "3443.TW", "3661.TW",
    "3665.TW", "4958.TW", "1301.TW", "1326.TW", "2379.TW", "2615.TW", "2618.TW", "2912.TW",
    "1519.TW", "8046.TW",
]

UNIVERSES = {
    "large": (LARGE_UNIVERSE, "./data/market_large", "2010-01-01"),
    "universe": (None, "./data/market_universe", "2015-01-01"),
    "tw": (TW_UNIVERSE, "./data/market_tw", "2000-01-01"),
}


def load_universe(name: str):
    tickers, data_dir, start = UNIVERSES[name]
    if tickers is None:
        with open(UNIVERSE_FILE, encoding="utf-8") as fh:
            body = "\n".join(ln for ln in fh if not ln.lstrip().startswith("#"))
        tickers = body.split()
    return tickers, data_dir, start


def fetch(ticker: str, start: str, min_rows: int = 500) -> pd.DataFrame | None:
    """Download one ticker. Returns ``None`` when there is not enough history."""
    raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if raw is None or len(raw) == 0:
        return None
    df = normalize_ohlcv(raw)
    if df is None:
        raise ValueError("unrecognised OHLCV layout from yfinance")
    if len(df) < min_rows:
        return None
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", choices=sorted(UNIVERSES), default="large")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()

    tickers, data_dir, start = load_universe(args.universe)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading {len(tickers)} tickers -> {data_dir} (from {start})")

    written, existing, failed = [], [], []
    for ticker in tqdm(tickers):
        path = os.path.join(data_dir, f"{ticker}.parquet")
        if os.path.exists(path) and not args.force:
            existing.append(ticker)
            continue
        try:
            df = fetch(ticker, start)
        except Exception as exc:  # network hiccups should not kill the batch
            failed.append((ticker, f"{type(exc).__name__}: {exc}"))
            tqdm.write(f"! {ticker}: {failed[-1][1]}")
            continue
        if df is None:
            failed.append((ticker, "no data / shorter than 500 rows"))
            tqdm.write(f"! {ticker}: {failed[-1][1]}")
            continue
        df.to_parquet(path, index=False)
        written.append(ticker)

    print(
        f"Done. wrote={len(written)} | kept existing={len(existing)} | failed={len(failed)}"
    )
    if failed:
        print("Failed tickers:")
        for ticker, why in failed:
            print(f"  - {ticker}: {why}")
        # a partial dataset silently poisons every split downstream -> fail loudly
        raise SystemExit(1)


if __name__ == "__main__":
    main()
