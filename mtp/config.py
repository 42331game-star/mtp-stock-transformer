"""Single source of truth for every hyper-parameter used by the pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import Tuple

import numpy as np

from .features import FEATURE_COLS, HORIZONS


@dataclass
class MTPConfig:
    # ---- data ----
    data_dir: str = "./data/market_large"
    seq_len: int = 250              # look-back window (trading days)
    horizons: Tuple[int, ...] = HORIZONS
    # Global calendar split: train <= train_end < val <= val_end < test.
    train_ratio: float = 0.7        # fraction of pooled dates used for training
    val_ratio: float = 0.15         # fraction used for validation (test = rest)
    train_end: str | None = None    # ISO date of the cut-offs actually trained on
    val_end: str | None = None      # (filled by mtp-train, reused by evaluate/backtest)
    min_rows: int = 600             # skip symbols with too little history
    num_workers: int = 4

    # ---- model ----
    input_dim: int = len(FEATURE_COLS)
    d_model: int = 768
    nhead: int = 16
    num_layers: int = 10
    dropout: float = 0.2            # set to 0.0 at inference

    # ---- training ----
    batch_size: int = 256
    accum_steps: int = 2            # effective batch = batch_size * accum_steps * n_gpus
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-3
    t_max: int = 0                  # CosineAnnealingLR T_max; 0 -> use epochs
    amp: bool = True
    grad_clip: float = 1.0          # max grad norm; 0 disables
    patience: int = 5               # early-stop after N epochs without val gain; 0 disables
    seed: int = 42

    # ---- artifacts ----
    checkpoint: str = "./checkpoints/stock_mtp_transformer.pth"

    # ---- trading rules (shared by scan / backtest) ----
    min_exp_return_5d: float = 0.025    # T+5 forecast >= +2.5%
    min_exp_return_10d: float = 0.045   # T+10 forecast >= +4.5%
    min_exp_return_3d: float = -0.01    # T+3 must not be already broken
    crash_guard_d1: float = -0.15       # skip when T+1 forecasts a crash
    extension_thresh: float = 0.020     # reserved: hold-extension rule (not implemented)
    atr_stop_mult: float = 2.3          # 2.3x ATR trailing stop
    horizon_days: int = 10              # base holding period
    commission_rate: float = 0.0005     # per side
    tax_rate: float = 0.003             # sell-side transfer tax (TW; use 0.0003 for US)
    cost_cover_ratio: float = 5.0       # required edge / round-trip cost

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    @property
    def target_cols(self) -> list:
        return [f"target_h{h}" for h in self.horizons]

    @property
    def roundtrip_cost(self) -> float:
        return self.commission_rate * 2 + self.tax_rate

    @property
    def boundaries(self) -> "tuple | None":
        """(train_end, val_end) cut-offs this config was trained with, if any.

        Evaluation scripts reuse these instead of recomputing them, so the
        scored period always matches the period the weights never saw.
        """
        if self.train_end and self.val_end:
            return np.datetime64(self.train_end), np.datetime64(self.val_end)
        return None

    # ---- sidecar serialisation -------------------------------------------
    # mtp-train writes "<checkpoint>.json" next to the weights so that
    # predict / backtest can rebuild a non-default architecture automatically.
    def save(self, checkpoint: str | None = None) -> str:
        path = (checkpoint or self.checkpoint) + ".json"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, checkpoint: str | None = None) -> "MTPConfig":
        """Default config, overridden by a sidecar file when one exists."""
        cfg = cls()
        if checkpoint:
            cfg.checkpoint = checkpoint
            path = checkpoint + ".json"
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    cfg = cls.from_dict(json.load(fh))
                cfg.checkpoint = checkpoint
        return cfg

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "MTPConfig":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "horizons" in kwargs:
            kwargs["horizons"] = tuple(int(h) for h in kwargs["horizons"])
        return cls(**kwargs)
