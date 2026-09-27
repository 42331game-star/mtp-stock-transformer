"""Loading trained artifacts: weights, architecture sidecar, calendar cut-offs.

``mtp-train`` writes ``<checkpoint>`` plus a ``<checkpoint>.json``
sidecar recording the architecture *and* the train/val cut-offs. Everything
that scores the model goes through :func:`load_model` + :func:`resolve_boundaries`
so a run can never silently mix weights from one split with data from another.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .config import MTPConfig
from .dataset import compute_date_boundaries
from .model import MTPStockTransformer


def default_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(cfg: MTPConfig, device: torch.device | None = None) -> MTPStockTransformer:
    """Rebuild the architecture from the sidecar and load the weights.

    Dropout is forced to 0 (inference), and a ``module.`` prefix coming from
    ``nn.DataParallel`` is stripped.
    """
    if not os.path.exists(cfg.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {cfg.checkpoint}. Run mtp-train first."
        )
    device = device or default_device()
    model = MTPStockTransformer(
        input_dim=cfg.input_dim, d_model=cfg.d_model, nhead=cfg.nhead,
        num_layers=cfg.num_layers, dropout=0.0, horizons=cfg.horizons,
    )
    state = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.to(device).eval()
    return model


def predict_windows(
    model: MTPStockTransformer, feats: np.ndarray, batch: int = 256,
    device: torch.device | None = None,
) -> np.ndarray:
    """Batched forward pass on ``[n, seq_len, n_feat]``, returned as numpy."""
    device = device or next(model.parameters()).device
    if len(feats) == 0:
        return np.empty((0, len(model.horizons)), dtype=np.float32)
    x = torch.as_tensor(feats, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(x[i : i + batch].to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def resolve_boundaries(cfg: MTPConfig) -> tuple:
    """Cut-offs the weights were trained with.

    Falls back to recomputing them from ``cfg.data_dir`` for checkpoints that
    pre-date the sidecar, and says so — an unannounced recompute would score a
    different period than the one that was held out during training.
    """
    if cfg.boundaries is not None:
        return cfg.boundaries
    print(
        "! checkpoint sidecar has no calendar cut-offs - recomputing from "
        f"{cfg.data_dir}. Make sure this is the same data snapshot as training."
    )
    return compute_date_boundaries(cfg.data_dir, cfg.train_ratio, cfg.val_ratio)


def smooth_l1_np(pred: np.ndarray, target: np.ndarray, beta: float = 1.0) -> float:
    """NumPy twin of ``torch.nn.SmoothL1Loss(reduction='mean', beta=beta)``."""
    err = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    quad = 0.5 * err ** 2 / beta
    lin = err - 0.5 * beta
    return float(np.mean(np.where(err < beta, quad, lin)))
