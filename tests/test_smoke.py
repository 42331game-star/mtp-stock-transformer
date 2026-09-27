"""Smoke tests: features -> dataset -> model -> one optimizer step -> checkpoint."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import make_bars  # noqa: E402
from mtp import (  # noqa: E402
    MTPStockDataset,
    MTPStockTransformer,
    add_features,
    add_targets,
    compute_date_boundaries,
    standardize,
    standardize_windows,
    FEATURE_COLS,
)


def test_features_are_complete_and_scaled():
    df = add_targets(add_features(make_bars()))
    clean = df.dropna(subset=list(FEATURE_COLS))
    assert len(clean) > 300, "240-day warm-up is eating the sample"
    assert clean[FEATURE_COLS].isna().sum().sum() == 0
    assert len(FEATURE_COLS) == 16

    # a single window is standardised with its own statistics
    z, mean, std = standardize(clean[FEATURE_COLS].to_numpy(np.float32))
    assert np.allclose(z.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(z.std(axis=0), 1, atol=1e-3)

    # ... and the batch version must agree with it, window by window
    batch = np.stack([
        clean[FEATURE_COLS].to_numpy(np.float32)[i : i + 40] for i in (0, 50, 100)
    ])
    z_batch = standardize_windows(batch)
    for i in range(len(batch)):
        z_single, _, _ = standardize(batch[i])
        assert np.allclose(z_batch[i], z_single, atol=1e-6), "batch/single mismatch"


def test_dataset_shapes_and_three_way_split():
    tmp = tempfile.mkdtemp()
    try:
        df = make_bars()
        for i in range(2):
            df.to_parquet(os.path.join(tmp, f"SYM{i}.parquet"), index=False)

        boundaries = compute_date_boundaries(tmp, train_ratio=0.7, val_ratio=0.15)
        common = dict(data_dir=tmp, seq_len=60, min_rows=300, boundaries=boundaries,
                      verbose=False)
        train = MTPStockDataset(mode="train", **common)
        val = MTPStockDataset(mode="val", **common)
        test = MTPStockDataset(mode="test", **common)

        for split in (train, val, test):
            assert len(split) > 0, "every slice must contain samples"
            assert split[0][0].shape == (60, 16)
            assert split[0][1].shape == (4,)

        # chronological: train is the biggest slice, test the last one
        assert len(train) > len(val)
        assert len(train) > len(test)
    finally:
        shutil.rmtree(tmp)


def test_model_forward_and_backward():
    x = torch.randn(8, 60, 16)
    y = torch.randn(8, 4)
    model = MTPStockTransformer(d_model=64, nhead=4, num_layers=2, dropout=0.1)

    out = model(x)
    assert out.shape == (8, 4)

    loss = torch.nn.SmoothL1Loss()(out, y)
    before = model.head_d10[0].weight.detach().clone()
    loss.backward()
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    assert not torch.allclose(before, model.head_d10[0].weight), "no parameter update"


def test_checkpoint_roundtrip_disables_dropout():
    tmp = tempfile.mkdtemp()
    try:
        ckpt = os.path.join(tmp, "w.pth")
        model = MTPStockTransformer(d_model=64, nhead=4, num_layers=2, dropout=0.1)
        torch.save(model.state_dict(), ckpt)

        restored = MTPStockTransformer(d_model=64, nhead=4, num_layers=2, dropout=0.0)
        restored.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        restored.eval()
        model.eval()

        x = torch.randn(4, 60, 16)
        with torch.no_grad():
            assert torch.allclose(restored(x), model(x), atol=1e-6)
    finally:
        shutil.rmtree(tmp)
