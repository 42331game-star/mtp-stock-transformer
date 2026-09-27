"""Split guarantees: no label crosses a cut-off, scaling is window-local.

These are the regressions this project actually shipped with, so they get
their own file instead of hiding inside the smoke test.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import make_bars, make_regime_bars  # noqa: E402
from mtp import (  # noqa: E402
    MTPStockDataset,
    build_symbol_split,
    compute_date_boundaries,
    read_symbol,
    window_index_range,
)

SEQ, SPAN = 60, 10


def _used_rows(lo: int, hi: int, seq_len: int = SEQ, span: int = SPAN) -> tuple[int, int]:
    """(first, last) row index a split's *labels* touch."""
    return lo + seq_len, (hi - 1) + seq_len + span


def _with_files(frames: dict) -> str:
    tmp = tempfile.mkdtemp()
    for name, df in frames.items():
        df.to_parquet(os.path.join(tmp, f"{name}.parquet"), index=False)
    return tmp


def test_label_rows_are_disjoint_across_splits():
    dates = pd.date_range("2020-01-01", periods=600, freq="B").to_numpy()
    train_end, val_end = dates[400], dates[500]

    used = {}
    for mode in ("train", "val", "test"):
        lo, hi = window_index_range(dates, mode, SEQ, SPAN, train_end, val_end)
        assert hi > lo, f"{mode} slice is empty"
        used[mode] = _used_rows(lo, hi)

    # disjoint and chronologically ordered
    assert used["train"][1] < used["val"][0]
    assert used["val"][1] < used["test"][0]

    # embargo: at least 2 * span + 1 sessions separate two splits' labels
    assert used["val"][0] - used["train"][1] >= 2 * SPAN + 1
    assert used["test"][0] - used["val"][1] >= 2 * SPAN + 1

    # the cut-offs themselves are respected
    assert dates[used["train"][1]] <= train_end
    assert dates[used["val"][0]] > train_end
    assert dates[used["val"][1]] <= val_end
    assert dates[used["test"][0]] > val_end


def test_unknown_mode_is_rejected():
    dates = pd.date_range("2020-01-01", periods=50, freq="B").to_numpy()
    try:
        window_index_range(dates, "holdout", SEQ, SPAN, dates[20], dates[30])
    except ValueError:
        return
    raise AssertionError("unknown mode must raise")


def test_boundaries_are_pooled_over_all_symbols():
    df = make_bars(700)
    tmp = _with_files({"SYM0": df, "SYM1": df})
    try:
        train_end, val_end = compute_date_boundaries(tmp, train_ratio=0.7, val_ratio=0.15)
        all_dates = np.unique(df["date"].to_numpy())
        assert train_end == all_dates[int(len(all_dates) * 0.7)]
        assert val_end == all_dates[int(len(all_dates) * 0.85)]
        assert train_end < val_end
    finally:
        shutil.rmtree(tmp)


def test_file_pipeline_respects_cut_offs():
    """Same guarantee, but through read_symbol + feature warm-up handling."""
    df = make_bars(700)
    tmp = _with_files({"SYM0": df})
    try:
        boundaries = compute_date_boundaries(tmp, train_ratio=0.7, val_ratio=0.15)
        train_end, val_end = boundaries
        common = dict(
            seq_len=SEQ, boundaries=boundaries, label_span=SPAN, min_rows=300,
            horizons=(1, 3, 5, 10),
        )
        train = build_symbol_split(
            os.path.join(tmp, "SYM0.parquet"), mode="train", **common
        )
        val = build_symbol_split(os.path.join(tmp, "SYM0.parquet"), mode="val", **common)
        test = build_symbol_split(os.path.join(tmp, "SYM0.parquet"), mode="test", **common)
        assert train and val and test, "every slice must produce samples"

        t_first, t_last = _used_rows(train["lo"], train["hi"])
        v_first, v_last = _used_rows(val["lo"], val["hi"])
        s_first, _ = _used_rows(test["lo"], test["hi"])

        assert train["dates"][t_last] <= train_end
        assert val["dates"][v_first] > train_end
        assert val["dates"][v_last] <= val_end
        assert test["dates"][s_first] > val_end
        assert train["dates"][t_last] < val["dates"][v_first], "labels overlap"
    finally:
        shutil.rmtree(tmp)


def test_split_start_is_independent_of_label_span():
    """backtest reserves `horizon_days`, evaluate reserves `max(horizons)`.

    Both must still begin on the same session, otherwise the two scripts would
    score different periods.
    """
    df = make_bars(700)
    tmp = _with_files({"SYM0": df})
    try:
        boundaries = compute_date_boundaries(tmp, train_ratio=0.7, val_ratio=0.15)
        path = os.path.join(tmp, "SYM0.parquet")
        starts = []
        for span in (1, 5, 10):
            split = build_symbol_split(
                path, mode="test", seq_len=SEQ, boundaries=boundaries,
                label_span=span, min_rows=300,
            )
            assert split is not None
            starts.append(split["lo"])
        assert len(set(starts)) == 1, f"split start varies with label_span: {starts}"
    finally:
        shutil.rmtree(tmp)


def test_dataset_and_backtest_use_identical_rows():
    """MTPStockDataset must be the same data scripts/backtest.py scores."""
    df = make_bars(700)
    tmp = _with_files({"SYM0": df})
    try:
        boundaries = compute_date_boundaries(tmp, train_ratio=0.7, val_ratio=0.15)
        path = os.path.join(tmp, "SYM0.parquet")
        split = build_symbol_split(
            path, mode="test", seq_len=SEQ, boundaries=boundaries,
            label_span=SPAN, min_rows=300,
        )
        ds = MTPStockDataset(
            data_dir=tmp, seq_len=SEQ, mode="test", boundaries=boundaries,
            min_rows=300, verbose=False,
        )
        assert split is not None and len(ds) == len(split["windows"])
        assert np.allclose(ds.X.numpy(), split["windows"], atol=1e-6)
        assert np.allclose(ds.Y.numpy(), split["targets"], atol=1e-6)
    finally:
        shutil.rmtree(tmp)


def test_scaling_is_window_local_not_series_local():
    """A two-regime series breaks any implementation that uses full-series stats."""
    df = make_regime_bars(800)
    tmp = _with_files({"REGIME": df})
    try:
        boundaries = compute_date_boundaries(tmp, train_ratio=0.7, val_ratio=0.15)
        ds = MTPStockDataset(
            data_dir=tmp, seq_len=120, mode="train", boundaries=boundaries,
            min_rows=400, verbose=False,
        )
        assert len(ds) > 10, "regime dataset came out empty"
        x = ds.X.numpy()
        # every window must be centred/scaled on its own 120 days
        assert np.allclose(x.mean(axis=1), 0.0, atol=1e-4), "not window-centred"
        assert np.allclose(x.std(axis=1), 1.0, atol=1e-3), "not window-scaled"
    finally:
        shutil.rmtree(tmp)


def test_read_symbol_accepts_legacy_date_column():
    """Old downloads wrote ``Date``; reading must not blow up on them."""
    df = make_bars(50).rename(columns={"date": "Date"})
    tmp = _with_files({"LEGACY": df})
    try:
        out = read_symbol(os.path.join(tmp, "LEGACY.parquet"))
        assert "date" in out.columns
        assert pd.api.types.is_datetime64_any_dtype(out["date"])
        assert len(out) == 50
    finally:
        shutil.rmtree(tmp)
