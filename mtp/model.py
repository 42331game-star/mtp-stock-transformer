"""MTP (Multi-Token Prediction) Transformer for multi-horizon price paths."""

from __future__ import annotations

import torch
import torch.nn as nn

from .features import HORIZONS


class LearnablePositionalEncoding(nn.Module):
    """Learned absolute positional embeddings (max 1000 steps by default)."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


def _head(d_model: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1))


class MTPStockTransformer(nn.Module):
    """Encoder-only Transformer that emits *several* future returns at once.

    Input  : ``[B, seq_len, input_dim]``  — a window of technical features.
    Output : ``[B, len(horizons)]``       — cumulative log-return forecast for
             each horizon (e.g. T+1, T+3, T+5, T+10).

    Instead of a single next-day head, every horizon gets its own head so the
    network learns a *path* (short, medium and long term must agree before a
    trade signal fires — see ``mtp.config`` trading thresholds).
    """

    def __init__(
        self,
        input_dim: int = 16,
        d_model: int = 768,
        nhead: int = 16,
        num_layers: int = 10,
        dropout: float = 0.2,
        horizons=HORIZONS,
    ):
        super().__init__()
        self.horizons = tuple(horizons)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_encoder = LearnablePositionalEncoding(d_model, max_len=1000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # One regression head per horizon. Attribute names (``head_d1`` ...) are
        # part of the checkpoint format — do not rename them if you want to keep
        # loading previously trained weights.
        for h in self.horizons:
            setattr(self, f"head_d{h}", _head(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos_encoder(self.input_proj(x))
        h = self.transformer(h)
        # Fuse the global context with the most recent step.
        return torch.mean(h, dim=1) + h[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encode(x)
        preds = [getattr(self, f"head_d{h}")(feat) for h in self.horizons]
        return torch.cat(preds, dim=-1)  # [B, len(horizons)]
