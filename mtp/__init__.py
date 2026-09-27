"""MTP (Multi-Token Prediction) stock forecasting package.

Public API:
    from mtp import MTPConfig, MTPStockTransformer, MTPStockDataset, add_features
"""

from .config import MTPConfig
from .dataio import REQUIRED_COLS, normalize_ohlcv
from .features import (
    FEATURE_COLS,
    HORIZONS,
    add_features,
    add_targets,
    standardize,
    standardize_windows,
)
from .model import MTPStockTransformer
from .dataset import (
    SPLITS,
    MTPStockDataset,
    build_symbol_split,
    compute_date_boundaries,
    read_symbol,
    window_index_range,
)
from .checkpoint import load_model, predict_windows, resolve_boundaries, smooth_l1_np

__all__ = [
    "MTPConfig",
    "MTPStockTransformer",
    "MTPStockDataset",
    "SPLITS",
    "FEATURE_COLS",
    "HORIZONS",
    "REQUIRED_COLS",
    "add_features",
    "add_targets",
    "build_symbol_split",
    "compute_date_boundaries",
    "load_model",
    "normalize_ohlcv",
    "predict_windows",
    "read_symbol",
    "resolve_boundaries",
    "smooth_l1_np",
    "standardize",
    "standardize_windows",
    "window_index_range",
]

__version__ = "0.1.0"
