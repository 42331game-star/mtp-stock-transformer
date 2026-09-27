"""Walk-forward back-test of the MTP path-trading rules.

Simulates the production logic on the global **test** slice (the dates the
checkpoint never saw a label for — see mtp.dataset.compute_date_boundaries):
  * enter only when the T+5 / T+10 forecast path is bullish *and* the expected
    edge covers the round-trip trading cost several times over,
  * exit on a 2.3x ATR trailing stop or after ``horizon_days`` sessions,
  * charge commission + transfer tax on both legs.

    mtp-backtest
    mtp-backtest --data-dir ./data/market_large --tax-rate 0.0003
    mtp-backtest --no-benchmark                    # offline, skips the SPY fetch
    python scripts/backtest.py                     # same code, no install needed
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from mtp import (
    MTPConfig,
    build_symbol_split,
    load_model,
    predict_windows,
    resolve_boundaries,
)

#: horizons the entry rules need (path resonance across short/medium/long)
REQUIRED_HORIZONS = (1, 3, 5, 10)


def backtest_symbol(path: str, model, cfg: MTPConfig, boundaries: tuple):
    """Return (trades DataFrame, buy&hold return) for one symbol."""
    split = build_symbol_split(
        path,
        mode="test",
        seq_len=cfg.seq_len,
        horizons=cfg.horizons,
        boundaries=boundaries,
        label_span=cfg.horizon_days,
        min_rows=cfg.min_rows,
    )
    if split is None:
        return None, None

    preds = predict_windows(model, split["windows"])
    if len(preds) == 0:
        return None, None

    closes, highs, lows = split["close"], split["high"], split["low"]
    atrs, dates, lo = split["atr"], split["dates"], split["lo"]

    first = lo + cfg.seq_len
    bnh = float(closes[-1] / closes[first] - 1.0)

    i1, i3, i5, i10 = (list(cfg.horizons).index(h) for h in REQUIRED_HORIZONS)
    min_cost = cfg.roundtrip_cost * cfg.cost_cover_ratio

    trades = []
    for k, pred in enumerate(preds):
        # cumulative log-return path -> simple returns
        d1, d3 = np.exp(pred[i1]) - 1, np.exp(pred[i3]) - 1
        d5, d10 = np.exp(pred[i5]) - 1, np.exp(pred[i10]) - 1
        if d10 < min_cost:                                   # edge vs. friction
            continue
        if not (d5 >= cfg.min_exp_return_5d and d10 >= cfg.min_exp_return_10d
                and d3 > cfg.min_exp_return_3d):
            continue                                         # path not aligned
        if d1 < cfg.crash_guard_d1:                          # near-term crash guard
            continue

        idx = k + first
        if idx + cfg.horizon_days >= len(closes):            # not enough runway left
            continue
        entry = closes[idx]
        stop_dist = atrs[idx] * cfg.atr_stop_mult
        highest, exit_price, exit_day = entry, None, 0

        for step in range(1, cfg.horizon_days + 1):
            j = idx + step
            highest = max(highest, highs[j])
            line = highest - stop_dist
            if lows[j] <= line:
                exit_price, exit_day = line, step
                break
        if exit_price is None:
            exit_price, exit_day = closes[idx + cfg.horizon_days], cfg.horizon_days

        gross = exit_price / entry - 1.0
        cost = cfg.commission_rate + (exit_price / entry) * (cfg.commission_rate + cfg.tax_rate)
        trades.append({
            "entry_date": dates[idx],
            "exit_date": dates[idx + exit_day],
            "holding_days": exit_day,
            "pred_d5": d5 * 100,
            "pred_d10": d10 * 100,
            "gross_return": gross,
            "net_return": gross - cost,
            "cost": cost,
            "reason": "trailing_stop" if exit_day < cfg.horizon_days else "time_horizon",
        })

    return pd.DataFrame(trades), bnh


def spy_stats(start: str, end: str):
    spy = yf.download("SPY", start="2018-01-01", end=end, progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0].lower() for c in spy.columns]
    else:
        spy.columns = [c.lower() for c in spy.columns]
    sub = spy.loc[start:end]
    ret = (sub["close"].iloc[-1] / sub["close"].iloc[0] - 1.0) * 100
    peak = sub["close"].cummax()
    mdd = ((sub["close"] - peak) / peak).min() * 100
    daily = sub["close"].pct_change()
    sharpe = daily.mean() / (daily.std() + 1e-8) * np.sqrt(252)
    return ret, mdd, sharpe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--checkpoint", default=None, help="path to a .pth weights file")
    ap.add_argument("--tax-rate", type=float, default=None,
                    help="sell-side tax: 0.003 (TW) or 0.0003 (US)")
    ap.add_argument("--no-benchmark", action="store_true",
                    help="skip the SPY download (offline runs)")
    args = ap.parse_args()

    cfg = MTPConfig.load(args.checkpoint)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.tax_rate is not None:
        cfg.tax_rate = args.tax_rate

    missing = [h for h in REQUIRED_HORIZONS if h not in cfg.horizons]
    if missing:
        raise SystemExit(
            f"The entry rules need horizons {REQUIRED_HORIZONS}, but this "
            f"checkpoint predicts {tuple(cfg.horizons)} (missing {missing})."
        )

    boundaries = resolve_boundaries(cfg)
    train_end, val_end = boundaries
    model = load_model(cfg)
    files = sorted(glob.glob(os.path.join(cfg.data_dir, "*.parquet")))
    print(
        f"Back-testing {len(files)} symbols | round-trip cost {cfg.roundtrip_cost * 100:.2f}% "
        f"| test slice starts {np.datetime64(val_end, 'D')}"
    )

    frames, bnhs, skipped = [], [], 0
    for f in tqdm(files):
        trades, bnh = backtest_symbol(f, model, cfg, boundaries)
        if trades is None:
            skipped += 1
            continue
        if not trades.empty:
            frames.append(trades)
        if bnh is not None:
            bnhs.append(bnh)
    if skipped:
        print(f"Skipped {skipped} symbol(s) without history in the test slice.")

    if not frames:
        print("No trade satisfied the MTP path thresholds.")
        return

    all_t = pd.concat(frames).sort_values("entry_date").reset_index(drop=True)
    wins = all_t[all_t["net_return"] > 0]
    losses = all_t[all_t["net_return"] <= 0]

    # Equal-weight portfolio curve from per-entry mean returns.
    daily = all_t.groupby("entry_date")["net_return"].mean().reset_index()
    hold = max(all_t["holding_days"].mean(), 1.0)
    daily["daily_ret"] = daily["net_return"] / hold
    daily["equity"] = (1 + daily["daily_ret"]).cumprod()
    daily["peak"] = daily["equity"].cummax()
    daily["dd"] = (daily["equity"] - daily["peak"]) / daily["peak"]

    ret = (daily["equity"].iloc[-1] - 1) * 100
    mdd = daily["dd"].min() * 100
    sharpe = daily["daily_ret"].mean() / (daily["daily_ret"].std() + 1e-8) * np.sqrt(252)
    start, end = str(all_t["entry_date"].min())[:10], str(all_t["exit_date"].max())[:10]

    print("\n" + "=" * 66)
    print(f" MTP path-trading back-test  ({start} ~ {end})")
    print("=" * 66)
    print(f" trades                : {len(all_t)}")
    print(f" avg holding days      : {all_t['holding_days'].mean():.1f}")
    print(f" trailing-stop exits   : {(all_t['reason'] == 'trailing_stop').sum()}")
    print(f" avg gross / net return: {all_t['gross_return'].mean() * 100:+.2f}% / {all_t['net_return'].mean() * 100:+.2f}%")
    print(f" avg friction cost     : {all_t['cost'].mean() * 100:.2f}% per round trip")
    pl = (wins["net_return"].mean() / abs(losses["net_return"].mean())) if len(losses) else float("inf")
    print(f" net win rate / P-L    : {len(wins) / len(all_t) * 100:.2f}% / {pl:.2f}")
    print("-" * 66)
    print(f" strategy   : {ret:+.2f}% | MDD {mdd:.2f}% | Sharpe {sharpe:.2f}")
    print(f" buy&hold   : {np.mean(bnhs) * 100:+.2f}% (avg per symbol)")
    if not args.no_benchmark:
        try:
            s_ret, s_mdd, s_sh = spy_stats(start, end)
            print(f" SPY        : {s_ret:+.2f}% | MDD {s_mdd:.2f}% | Sharpe {s_sh:.2f}")
        except Exception as exc:
            print(f" SPY benchmark unavailable: {exc}")
    print("=" * 66)
    print("Note: equal-weight curve, no overlapping-position accounting, "
          "fills at the stop line. See docs/zh/04-backtest.md section 5.")


if __name__ == "__main__":
    main()
