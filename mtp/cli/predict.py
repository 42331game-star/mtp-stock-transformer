"""Interactive single-ticker MTP forecast.

    mtp-predict NVDA
    mtp-predict 2330.TW
    mtp-predict                    # then type a ticker at the prompt
    python scripts/predict.py NVDA  # same code, no install needed
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yfinance as yf

from mtp import (
    MTPConfig,
    FEATURE_COLS,
    add_features,
    load_model,
    normalize_ohlcv,
    standardize,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_window(ticker: str, cfg: MTPConfig):
    """Last ``seq_len`` days of one ticker, scaled exactly like training data."""
    raw = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    df = normalize_ohlcv(raw)
    if df is None:
        raise ValueError(f"no usable OHLCV data for {ticker}")

    df = add_features(df).dropna(subset=list(FEATURE_COLS))
    if len(df) < cfg.seq_len:
        raise ValueError(f"need >= {cfg.seq_len} rows after feature warm-up, got {len(df)}")

    feats = df[FEATURE_COLS].to_numpy(dtype=np.float32)[-cfg.seq_len:]
    feats, _, _ = standardize(feats)   # single window -> window-local z-score
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return x, float(df["close"].iloc[-1]), float(df["atr_raw"].iloc[-1])


def report(ticker: str, preds: np.ndarray, price: float, atr: float, cfg: MTPConfig) -> None:
    pct = (np.exp(preds) - 1.0) * 100
    print("\n" + "=" * 58)
    print(f"MTP forecast | {ticker} | last close {price:.2f}")
    print("-" * 58)
    for h, p in zip(cfg.horizons, pct):
        print(f"  T+{h:<3d} {p:+6.2f}%  ->  target ~ {price * (1 + p / 100):.2f}")
    print(f"  ATR(14) {atr:.3f} | suggested stop {price - atr * cfg.atr_stop_mult:.2f}")

    p5 = pct[list(cfg.horizons).index(5)] if 5 in cfg.horizons else 0.0
    p10 = pct[list(cfg.horizons).index(10)] if 10 in cfg.horizons else 0.0
    # same thresholds the backtest uses for an entry (mtp.config)
    if p10 >= cfg.min_exp_return_10d * 100 and p5 >= cfg.min_exp_return_5d * 100:
        posture = "bullish continuation (entry path)"
    elif p10 < -2.0:
        posture = "bearish pullback"
    else:
        posture = "range-bound"
    print(f"  posture: {posture}")
    print("=" * 58 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ticker", nargs="?", help="e.g. NVDA or 2330.TW")
    ap.add_argument("--checkpoint", default=None, help="path to a .pth weights file")
    args = ap.parse_args()

    cfg = MTPConfig.load(args.checkpoint)
    try:
        model = load_model(cfg)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}\nTrain one first:  mtp-train")

    ticker = args.ticker
    while True:
        if not ticker:
            try:
                ticker = input("ticker> ").strip()
            except EOFError:      # non-interactive shell / Ctrl-D
                break
        if ticker.lower() in {"exit", "quit", "q"}:
            break
        try:
            x, price, atr = build_window(ticker, cfg)
            with torch.no_grad():
                preds = model(x)[0].cpu().numpy()
            report(ticker, preds, price, atr, cfg)
        except Exception as exc:
            print(f"! {ticker}: {exc}")
        ticker = None  # fall back to the interactive prompt


if __name__ == "__main__":
    main()
