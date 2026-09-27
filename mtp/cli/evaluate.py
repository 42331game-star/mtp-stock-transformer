"""Out-of-sample skill report: does the model predict returns at all?

Scores the untouched split against two trivial baselines and reports
per-horizon IC / rank-IC / directional accuracy with a t-statistic.

    mtp-evaluate                                  # test slice (untouched)
    mtp-evaluate --split val                      # slice used for model selection
    mtp-evaluate --checkpoint ./checkpoints/mtp_small.pth
    python scripts/evaluate.py                    # same code, no install needed

Baselines
    zero      always forecast 0% (a fair floor: returns have ~zero mean)
    momentum  extrapolate the last h-day return inside the 250-day window

Reading the output
    |IC| < ~0.02 or |t| < 2   no statistically detectable signal
    huber(model) >= huber(zero)  the network is *worse* than predicting flat
    avg pred far from avg real   miscalibrated output (magnitude is wrong
                                 even when the sign is right)

Run this before trusting any number from ``mtp-backtest``: a backtest on an
unskilled model is just a backtest of its hand-tuned thresholds.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from mtp import (
    MTPConfig,
    SPLITS,
    build_symbol_split,
    load_model,
    predict_windows,
    resolve_boundaries,
    smooth_l1_np,
)


def momentum_baseline(close: np.ndarray, first: int, n: int, horizons) -> np.ndarray:
    """Persistence baseline: log return of the last ``h`` days *inside* the window.

    ``first`` is the index of the first label row, so the last observed close of
    window ``k`` sits at ``first + k - 1``.
    """
    ends = first + np.arange(n) - 1
    out = np.full((n, len(horizons)), np.nan, dtype=np.float64)
    for j, h in enumerate(horizons):
        starts = ends - h
        ok = starts >= 0
        out[ok, j] = np.log(close[ends[ok]] / close[starts[ok]])
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def average_rank(a: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged — scipy is deliberately not a dependency."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(average_rank(a), average_rank(b))


def pearson_t(ic: float, n: int) -> float:
    if n < 3 or abs(ic) >= 1.0:
        return float("nan")
    return ic * np.sqrt(n - 2) / np.sqrt(max(1.0 - ic ** 2, 1e-12))


def collect(cfg: MTPConfig, boundaries: tuple, split: str, model) -> dict:
    """Predictions + realised returns + momentum baseline, pooled over symbols."""
    files = sorted(glob.glob(os.path.join(cfg.data_dir, "*.parquet")))
    if not files:
        raise SystemExit(
            f"No parquet files under {cfg.data_dir!r}. Run mtp-download first."
        )

    preds, real, mom, symbols, skipped = [], [], [], 0, 0
    first_date, last_date = None, None
    max_horizon = max(cfg.horizons)

    for path in files:
        s = build_symbol_split(
            path,
            mode=split,
            seq_len=cfg.seq_len,
            horizons=cfg.horizons,
            boundaries=boundaries,
            label_span=max_horizon,
            min_rows=cfg.min_rows,
        )
        if s is None:
            skipped += 1
            continue
        n = s["hi"] - s["lo"]
        if n <= 0:
            skipped += 1
            continue

        p = predict_windows(model, s["windows"])
        r = s["targets"].astype(np.float64)
        m = momentum_baseline(s["close"], s["lo"] + cfg.seq_len, n, cfg.horizons)

        # realised targets can be NaN if label_span < max horizon; drop those rows
        mask = np.isfinite(r).all(axis=1) & np.isfinite(m).all(axis=1) & np.isfinite(p).all(axis=1)
        if not mask.any():
            skipped += 1
            continue

        first = s["lo"] + cfg.seq_len
        dates = s["dates"]
        d0, d1 = dates[first], dates[min(first + n - 1 + max_horizon, len(dates) - 1)]
        first_date = d0 if first_date is None else min(first_date, d0)
        last_date = d1 if last_date is None else max(last_date, d1)

        preds.append(p[mask])
        real.append(r[mask])
        mom.append(m[mask])
        symbols += 1

    if not preds:
        raise SystemExit(f"No symbol has data in the {split!r} slice.")

    return {
        "pred": np.concatenate(preds, axis=0),
        "real": np.concatenate(real, axis=0),
        "mom": np.concatenate(mom, axis=0),
        "symbols": symbols,
        "skipped": skipped,
        "start": str(np.datetime64(first_date, "D")),
        "end": str(np.datetime64(last_date, "D")),
    }


def report(cfg: MTPConfig, data: dict, split: str, boundaries: tuple) -> None:
    pred, real, mom = data["pred"], data["real"], data["mom"]
    zero = np.zeros_like(real)
    n = len(pred)
    train_end, val_end = boundaries
    skip_note = f" | skipped {data['skipped']} symbol(s)" if data["skipped"] else ""

    print("\n" + "=" * 92)
    scope = "IN-SAMPLE (not evidence of generalisation)" if split == "train" else "out-of-sample"
    print(f" MTP skill report | split={split} | {scope}")
    print("=" * 92)
    print(
        f" symbols {data['symbols']} | samples {n} | period {data['start']} ~ {data['end']}"
        f"{skip_note}"
    )
    print(
        f" cut-offs: train <= {np.datetime64(train_end, 'D')} < "
        f"val <= {np.datetime64(val_end, 'D')} < test"
    )
    print("-" * 92)
    header = (
        f" {'horizon':>7} {'IC':>7} {'rankIC':>7} {'t(IC)':>6} {'dirAcc':>7} "
        f"{'avgPred':>8} {'avgReal':>8} | {'huber':>9} {'zero':>9} {'momentum':>9}"
    )
    print(header)
    print("-" * 92)

    beats_zero = 0
    ics = []
    for j, h in enumerate(cfg.horizons):
        p, y, m = pred[:, j], real[:, j], mom[:, j]
        ic = pearson(p, y)
        rank_ic = spearman(p, y)
        t = pearson_t(ic, n)
        agree = np.mean(np.sign(p) == np.sign(y)) * 100
        h_model = smooth_l1_np(p, y)
        h_zero = smooth_l1_np(zero[:, j], y)
        h_mom = smooth_l1_np(m, y)
        beats_zero += h_model < h_zero
        ics.append(abs(ic) if np.isfinite(ic) else 0.0)
        print(
            f" {'T+' + str(h):>7} {ic:>7.4f} {rank_ic:>7.4f} {t:>6.2f} {agree:>6.1f}% "
            f"{np.mean(p) * 100:>7.2f}% {np.mean(y) * 100:>7.2f}% | "
            f"{h_model:>9.5f} {h_zero:>9.5f} {h_mom:>9.5f}"
        )

    print("-" * 92)
    verdict = []
    if beats_zero == len(cfg.horizons):
        verdict.append(f"model beats the zero baseline on {beats_zero}/{len(cfg.horizons)} horizons")
    elif beats_zero == 0:
        verdict.append(
            f"model LOSES to the zero baseline on every horizon - it is not learning returns"
        )
    else:
        verdict.append(
            f"model beats the zero baseline on only {beats_zero}/{len(cfg.horizons)} horizons"
        )
    best = int(np.argmax(ics)) if ics else 0
    verdict.append(
        f"mean |IC| {np.mean(ics):.4f}, best T+{cfg.horizons[best]} |IC| {ics[best]:.4f}"
    )
    print(" " + " | ".join(verdict))
    print(
        " Rules of thumb: |IC| < 0.02 or |t| < 2 => no detectable signal;\n"
        " huber(model) >= huber(zero) => predicting flat is better.\n"
        " IC pays nothing on its own - only a backtest with costs does, and only\n"
        " if this table shows real skill first. See docs/zh/04-backtest.md section 5."
    )
    print("=" * 92 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--checkpoint", default=None, help="path to a .pth weights file")
    ap.add_argument("--split", choices=SPLITS, default="test",
                    help="slice to score (default: test, never used for selection)")
    args = ap.parse_args()

    cfg = MTPConfig.load(args.checkpoint)
    if args.data_dir:
        cfg.data_dir = args.data_dir

    boundaries = resolve_boundaries(cfg)
    model = load_model(cfg)
    data = collect(cfg, boundaries, args.split, model)
    report(cfg, data, args.split, boundaries)


if __name__ == "__main__":
    main()
