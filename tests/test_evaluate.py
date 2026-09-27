"""Metric helpers used by ``mtp-evaluate`` (mtp/cli/evaluate.py)."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtp.cli.evaluate import average_rank, momentum_baseline, pearson, spearman  # noqa: E402
from mtp import smooth_l1_np  # noqa: E402


def test_average_rank_averages_ties():
    assert np.allclose(average_rank(np.array([10.0, 20.0, 20.0, 30.0])), [1, 2.5, 2.5, 4])
    assert np.allclose(average_rank(np.array([5.0, 4.0, 3.0, 2.0, 1.0])), [5, 4, 3, 2, 1])


def test_spearman_is_pearson_of_ranks():
    x = np.array([1.0, 2, 3, 4, 5])
    y = np.array([2.0, 1, 4, 3, 5])
    # textbook value: 1 - 6 * sum(d^2) / (n * (n^2 - 1)) = 1 - 24/120
    assert abs(spearman(x, y) - 0.8) < 1e-9
    assert abs(spearman(x, x) - 1.0) < 1e-12
    assert np.isnan(pearson(np.zeros(5), y)), "constant input has no correlation"


def test_smooth_l1_np_matches_torch():
    rng = np.random.default_rng(0)
    pred = rng.normal(0, 0.05, size=(200, 4))
    target = rng.normal(0, 0.05, size=(200, 4))
    expected = torch.nn.SmoothL1Loss()(
        torch.tensor(pred, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    ).item()
    assert abs(smooth_l1_np(pred, target) - expected) < 1e-5


def test_momentum_baseline_is_the_window_s_past_return():
    close = np.array([100.0, 101, 102, 104, 105, 107])
    # first label row is index 4 -> window is rows [0, 4), last observed close=104
    out = momentum_baseline(close, first=4, n=2, horizons=(1, 2, 3))
    assert out.shape == (2, 3)
    assert np.isclose(out[0, 0], np.log(104 / 102))   # last 1 day of the window
    assert np.isclose(out[0, 1], np.log(104 / 101))   # last 2 days
    assert np.isclose(out[0, 2], np.log(104 / 100))   # last 3 days
    # second window ends one row later (close = 105)
    assert np.isclose(out[1, 0], np.log(105 / 104))
    # a horizon longer than the available history must be NaN, not garbage
    out2 = momentum_baseline(close, first=4, n=1, horizons=(99,))
    assert np.isnan(out2[0, 0])
