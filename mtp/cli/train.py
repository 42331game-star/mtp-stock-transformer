"""Train the MTP Transformer.

    mtp-train                                     # defaults: 15 epochs, 768-d / 10 layers
    mtp-train --epochs 30 --data-dir ./data/market_large
    mtp-train --d-model 384 --layers 8 --batch-size 64   # lighter model
    python scripts/train.py --epochs 30           # same code, no install needed

Device selection: uses ``cuda:0`` automatically when CUDA is visible (AMP +
tf32), otherwise CPU. Restrict to one card with ``CUDA_VISIBLE_DEVICES=0``;
without that, all visible cards are used through ``nn.DataParallel``.

Checkpoints are selected on the **validation** slice only; the test slice is
never touched here (score it afterwards with ``mtp-evaluate``).
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import fields

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mtp import (
    MTPConfig,
    MTPStockDataset,
    MTPStockTransformer,
    compute_date_boundaries,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    # NOTE: the config field is ``num_layers``. Without this dest the flag used
    # to set a throwaway ``cfg.layers`` attribute and silently do nothing.
    ap.add_argument("--layers", dest="num_layers", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--train-ratio", type=float, default=None,
                    help="share of pooled dates used for training (default 0.7)")
    ap.add_argument("--val-ratio", type=float, default=None,
                    help="share of pooled dates used for validation (default 0.15)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (default 42)")
    ap.add_argument("--patience", type=int, default=None,
                    help="stop after N epochs without val improvement (0 = off)")
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="max gradient norm (0 = off)")
    ap.add_argument("--checkpoint", default=None,
                    help="output path (a <path>.json sidecar records the architecture)")
    return ap


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def apply_args(cfg: MTPConfig, args: argparse.Namespace) -> MTPConfig:
    """Copy parsed flags onto the config, refusing any flag that would be a no-op."""
    known = {f.name for f in fields(cfg)}
    ignored = [k for k, v in vars(args).items() if v is not None and k not in known]
    if ignored:
        raise SystemExit(
            f"flag(s) {ignored} do not match any MTPConfig field - "
            "fix the parser instead of silently ignoring them"
        )
    for key, val in vars(args).items():
        if val is not None:
            setattr(cfg, key, val)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    """Keep dataloader workers reproducible (numpy / random are per-process)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main() -> None:
    args = parse_args()
    cfg = apply_args(MTPConfig(), args)

    seed_everything(cfg.seed)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    # One calendar for every symbol: training labels can never reach into the
    # validation period of *any* symbol. The cut-offs are written into the
    # checkpoint sidecar so evaluate/backtest score exactly the same window.
    boundaries = compute_date_boundaries(cfg.data_dir, cfg.train_ratio, cfg.val_ratio)
    cfg.train_end, cfg.val_end = (str(np.datetime64(b)) for b in boundaries)
    train_end, val_end = boundaries
    print(
        f"Calendar split (pooled over all symbols): "
        f"train <= {np.datetime64(train_end, 'D')} < "
        f"val <= {np.datetime64(val_end, 'D')} < test"
    )

    common = dict(
        data_dir=cfg.data_dir, seq_len=cfg.seq_len, horizons=cfg.horizons,
        min_rows=cfg.min_rows, boundaries=boundaries,
    )
    train_ds = MTPStockDataset(mode="train", **common)
    val_ds = MTPStockDataset(mode="val", **common)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(
            f"Empty dataset (train={len(train_ds)}, val={len(val_ds)}). "
            "Download more history or adjust --train-ratio / --val-ratio."
        )

    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        generator=generator, worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=max(cfg.num_workers // 2, 0), pin_memory=True,
    )

    model = MTPStockTransformer(
        input_dim=cfg.input_dim, d_model=cfg.d_model, nhead=cfg.nhead,
        num_layers=cfg.num_layers, dropout=cfg.dropout, horizons=cfg.horizons,
    )
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"DataParallel over {n_gpus} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Huber loss: robust to the fat tails of return distributions.
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    t_max = cfg.t_max or cfg.epochs        # keep LR schedule tied to --epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    use_amp = bool(cfg.amp and torch.cuda.is_available())  # autocast is CUDA-only
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(
        f"\nTraining: {cfg.epochs} epochs | horizons={cfg.horizons} | "
        f"seq_len={cfg.seq_len} | seed={cfg.seed} | lr={cfg.lr}"
    )
    best_val, wait, best_epoch = float("inf"), 0, 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        total_loss, total_samples = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for i, (x_batch, y_batch) in enumerate(pbar):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x_batch), y_batch) / cfg.accum_steps

            scaler.scale(loss).backward()
            if (i + 1) % cfg.accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * cfg.accum_steps * len(y_batch)
            total_samples += len(y_batch)
            pbar.set_postfix(huber=f"{total_loss / max(total_samples, 1):.5f}")

        scheduler.step()

        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_loss += criterion(model(x_val), y_val).item() * len(y_val)
                val_n += len(y_val)
        val_loss /= max(val_n, 1)

        improved = val_loss < best_val
        print(
            f"Epoch {epoch:2d} | train {total_loss / max(total_samples, 1):.5f} "
            f"| val {val_loss:.5f}{' *' if improved else ''}"
        )

        if improved:
            best_val, best_epoch, wait = val_loss, epoch, 0
            os.makedirs(os.path.dirname(cfg.checkpoint) or ".", exist_ok=True)
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state, cfg.checkpoint)
            cfg.save()  # sidecar: architecture + calendar cut-offs
        else:
            wait += 1
            if cfg.patience and wait >= cfg.patience:
                print(f"Early stop: no val improvement for {cfg.patience} epochs.")
                break

    print(
        f"Saved checkpoint -> {cfg.checkpoint} "
        f"(best val {best_val:.5f} @ epoch {best_epoch}/{cfg.epochs})"
    )
    print("Score the untouched test slice with: mtp-evaluate")


if __name__ == "__main__":
    main()
