"""Dataset that turns raw OHLCV parquet files into sliding MTP windows.

Split policy
------------
Every symbol shares **one calendar**: the train / validation / test cut-off
dates are computed from the pooled trading dates of all files in ``data_dir``
(see :func:`compute_date_boundaries`) instead of slicing each symbol at its own
80% mark. That way a training label from one symbol can never reach into the
evaluation period of another.

A sample whose window starts at index ``i`` reads rows ``[i, i + seq_len)`` and
is labelled on rows ``i + seq_len … i + seq_len + label_span``. It is assigned
to a split by that *label* range:

* ``train`` – the whole label ends at least ``embargo`` sessions *before* the
  train cut-off,
* ``val``   – the label sits inside ``(train_end, val_end]``, ``embargo`` away
  from both,
* ``test``  – the label starts at least ``embargo`` sessions *after* the val
  cut-off.

Each boundary is padded with an **embargo** of ``max(horizons)`` sessions (10
days by default) on *both* sides, so no price path is shared between two
splits: adjacent 10-day returns either side of a cut-off are strongly
autocorrelated, and the embargo keeps them out of both slices. It is pinned to
``max(horizons)`` rather than the caller's ``label_span``, because
back-testing reserves ``horizon_days`` rows while evaluation reserves
``max(horizons)`` — both must start on the same session.

All three consumers of this module (``MTPStockDataset``,
``mtp-backtest``, ``mtp-evaluate``) go through
:func:`build_symbol_split`, so they can never disagree on which rows belong to
which split nor on how a window is scaled.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dataio import DATE_ALIASES
from .features import (
    FEATURE_COLS,
    HORIZONS,
    add_features,
    add_targets,
    standardize_windows,
)

#: split names accepted by :func:`window_index_range` / ``MTPStockDataset``
SPLITS = ("train", "val", "test")


def read_symbol(path: str) -> pd.DataFrame:
    """Read one symbol parquet file and normalise the date column to ``date``.

    Files written by older versions of the downloader (``mtp/cli/download_data.py``) may carry
    ``Date`` / ``index`` instead of ``date``; both are accepted here so that an
    already downloaded dataset keeps working.
    """
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        candidates = [c for c in df.columns if str(c).lower() in DATE_ALIASES]
        # prefer a column that actually holds datetimes (``index`` may be int)
        dated = [c for c in candidates if pd.api.types.is_datetime64_any_dtype(df[c])]
        pick = (dated or candidates or [None])[0]
        if pick is None:
            raise ValueError(
                f"{os.path.basename(path)}: no date column "
                f"(found {list(df.columns)}). Re-run mtp-download."
            )
        df = df.rename(columns={pick: "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_date_boundaries(
    data_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> tuple[np.datetime64, np.datetime64]:
    """Pooled ``(train_end, val_end)`` cut-offs shared by every symbol.

    Both dates come from the *sorted union of all trading dates* in
    ``data_dir``, so they are real calendar dates and strictly increasing.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No parquet files under {data_dir!r}. Run mtp-download first."
        )
    parts = []
    for path in files:
        try:  # cheap path: the column is named "date"
            parts.append(pd.read_parquet(path, columns=["date"])["date"].to_numpy())
        except Exception:  # older layout (``Date`` / ``index``) -> full read
            df = read_symbol(path)
            if "date" in df.columns:
                parts.append(df["date"].to_numpy())
    if not parts:
        raise ValueError(f"No usable date column under {data_dir!r}.")
    dates = np.unique(np.concatenate(parts))
    if len(dates) < 3:
        raise ValueError(f"Only {len(dates)} distinct dates under {data_dir!r}.")

    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not 0 < val_ratio < 1 - train_ratio:
        raise ValueError(f"val_ratio must be in (0, 1 - train_ratio), got {val_ratio}")
    n = len(dates)
    i_train = min(max(int(n * train_ratio), 1), n - 2)
    i_val = min(max(int(n * (train_ratio + val_ratio)), i_train + 1), n - 1)
    return np.datetime64(dates[i_train]), np.datetime64(dates[i_val])


def window_index_range(
    dates: np.ndarray,
    mode: str,
    seq_len: int,
    label_span: int,
    train_end: np.datetime64,
    val_end: np.datetime64,
    embargo: int | None = None,
) -> tuple[int, int]:
    """Half-open ``[lo, hi)`` range of window-start indices for one split.

    ``dates`` must be sorted ascending. A sample starting at ``i`` uses rows
    ``[i, i + seq_len)`` as input and rows ``i + seq_len … i + seq_len +
    label_span`` as its label, so with ``embargo = label_span`` (the default):

    * ``train`` → label rows ``[0, train_end - embargo]``
    * ``val``   → label rows ``[train_end + embargo, val_end - embargo]``
    * ``test``  → label rows ``[val_end + embargo, end]``

    i.e. the label rows of two different splits are separated by at least
    ``2 * embargo + 1`` sessions and never overlap. The upper bound also keeps
    ``label_span`` rows in reserve, so every emitted label is fully realised
    inside the symbol's own history.
    """
    if mode not in SPLITS:
        raise ValueError(f"mode must be one of {SPLITS}, got {mode!r}")
    if seq_len < 1 or label_span < 1:
        raise ValueError("seq_len and label_span must be >= 1")
    if embargo is None:
        embargo = label_span
    if embargo < 0:
        raise ValueError("embargo must be >= 0")

    t = len(dates)
    k_train = int(np.searchsorted(dates, train_end, side="right"))
    k_val = int(np.searchsorted(dates, val_end, side="right"))

    if mode == "train":
        lo_j, hi_j = 0, k_train - embargo
    elif mode == "val":
        lo_j, hi_j = k_train + embargo, k_val - embargo
    else:
        lo_j, hi_j = k_val + embargo, t

    lo = max(0, lo_j - seq_len)
    hi = min(hi_j - label_span, t - label_span) - seq_len
    return lo, max(hi, lo)


def build_symbol_split(
    path: str,
    mode: str,
    seq_len: int,
    horizons=HORIZONS,
    boundaries: tuple | None = None,
    label_span: int | None = None,
    min_rows: int = 0,
    embargo: int | None = None,
) -> dict | None:
    """Features, labels and OHLC arrays for one symbol on one split.

    Returns ``None`` when the file is too short for the requested split.
    Otherwise a dict with:

    ==============  =========================================================
    ``windows``     ``[n, seq_len, n_feat]`` z-scored window by window
    ``targets``     ``[n, len(horizons)]`` forward log returns (may contain
                    NaN only if ``label_span`` < ``max(horizons)``)
    ``dates``       full symbol date array (post warm-up)
    ``close/high/low/atr``  full-symbol arrays, for execution simulation
    ``lo``          window-start index of ``windows[0]``
    ==============  =========================================================
    """
    horizons = tuple(horizons)
    label_span = max(horizons) if label_span is None else int(label_span)
    # The embargo must NOT follow label_span: back-testing reserves
    # ``horizon_days`` rows while evaluation reserves ``max(horizons)``, and the
    # two still have to start on the same session. Pin it to the longest horizon.
    if embargo is None:
        embargo = max(horizons)
    if mode not in SPLITS:
        raise ValueError(f"mode must be one of {SPLITS}, got {mode!r}")

    df = read_symbol(path)
    if min_rows and len(df) < min_rows:
        return None

    df = add_targets(add_features(df), horizons)
    # drop only feature warm-up rows: the target-NaN tail must stay so that
    # row indices line up across dataset / backtest / evaluate
    df = df.dropna(subset=list(FEATURE_COLS)).reset_index(drop=True)
    if len(df) <= seq_len + label_span:
        return None

    dates = df["date"].to_numpy()
    if boundaries is None:
        raise ValueError(
            "boundaries=(train_end, val_end) is required - use "
            "mtp.compute_date_boundaries(data_dir, ...) or MTPConfig.boundaries"
        )
    train_end, val_end = (np.datetime64(b) for b in boundaries)
    lo, hi = window_index_range(
        dates, mode, seq_len, label_span, train_end, val_end, embargo=embargo
    )
    if hi <= lo:
        return None

    feats = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    # [t - seq_len + 1, n_feat, seq_len] -> [n, seq_len, n_feat]
    win = np.lib.stride_tricks.sliding_window_view(feats, seq_len, axis=0)
    win = np.moveaxis(win, -1, 1)[lo:hi]
    target_cols = [f"target_h{h}" for h in horizons]
    return {
        "windows": standardize_windows(np.ascontiguousarray(win)),
        "targets": df[target_cols].to_numpy(dtype=np.float32)[seq_len + lo : seq_len + hi],
        "dates": dates,
        "close": df["close"].to_numpy(dtype=np.float64),
        "high": df["high"].to_numpy(dtype=np.float64),
        "low": df["low"].to_numpy(dtype=np.float64),
        "atr": df["atr_raw"].to_numpy(dtype=np.float64),
        "lo": lo,
        "hi": hi,
    }


class MTPStockDataset(Dataset):
    """Materialised (X, y) pairs for training / validation / evaluation.

    Wraps :func:`build_symbol_split`: for every symbol in ``data_dir`` it keeps
    the rows whose label belongs to ``mode``, z-scored window by window.
    """

    def __init__(
        self,
        data_dir: str = "./data/market_large",
        seq_len: int = 250,
        mode: str = "train",
        horizons=HORIZONS,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        min_rows: int = 600,
        boundaries: tuple | None = None,
        verbose: bool = True,
    ):
        if mode not in SPLITS:
            raise ValueError(f"mode must be one of {SPLITS}, got {mode!r}")
        self.seq_len = seq_len
        self.horizons = tuple(horizons)
        self.max_horizon = max(self.horizons)
        self.mode = mode
        self.boundaries = (
            tuple(np.datetime64(b) for b in boundaries)
            if boundaries is not None
            else compute_date_boundaries(data_dir, train_ratio, val_ratio)
        )

        xs, ys = [], []
        self._build(data_dir, min_rows, xs, ys)

        arr_x = (
            np.concatenate(xs, axis=0)
            if xs
            else np.zeros((0, seq_len, len(FEATURE_COLS)), dtype=np.float32)
        )
        arr_y = (
            np.concatenate(ys, axis=0)
            if ys
            else np.zeros((0, len(self.horizons)), dtype=np.float32)
        )
        xs.clear()
        ys.clear()
        self.X = torch.tensor(arr_x, dtype=torch.float32)
        self.Y = torch.tensor(arr_y, dtype=torch.float32)

        if verbose:
            train_end, val_end = self.boundaries
            print(
                f"[{mode.upper()}] seq_len={self.seq_len} | samples={len(self.X)} "
                f"| features={self.X.shape[-1] if len(self.X) else 0} "
                f"| targets={self.Y.shape[-1] if len(self.Y) else 0} "
                f"| cut-offs train<={np.datetime64(train_end, 'D')} "
                f"val<={np.datetime64(val_end, 'D')}"
            )

    def _build(self, data_dir, min_rows, xs, ys):
        files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        if not files:
            raise FileNotFoundError(
                f"No parquet files under {data_dir!r}. Run mtp-download first."
            )
        for path in files:
            split = build_symbol_split(
                path,
                mode=self.mode,
                seq_len=self.seq_len,
                horizons=self.horizons,
                boundaries=self.boundaries,
                label_span=self.max_horizon,
                min_rows=min_rows,
            )
            if split is None:
                continue
            xs.append(split["windows"])
            ys.append(split["targets"])

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
