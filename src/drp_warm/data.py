"""Data loading, feature/target separation, and PyTorch Dataset for DRP."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

log = logging.getLogger("drp_warm.data")


@dataclass(frozen=True)
class DRPColumns:
    target: str
    ge_features: list[str]
    dd_features: list[str]

    @property
    def feature_columns(self) -> list[str]:
        return self.ge_features + self.dd_features

    @property
    def n_features(self) -> int:
        return len(self.ge_features) + len(self.dd_features)


def detect_columns(df: pd.DataFrame, target_candidates: tuple[str, ...] = ("AUC1", "AUC")) -> DRPColumns:
    target = next((c for c in target_candidates if c in df.columns), None)
    if target is None:
        raise ValueError(f"No target column found. Looked for: {target_candidates}")
    ge = [c for c in df.columns if c.startswith("GE_")]
    dd = [c for c in df.columns if c.startswith("DD_")]
    if not ge:
        raise ValueError("No GE_* (gene-expression) columns found.")
    if not dd:
        raise ValueError("No DD_* (drug descriptor) columns found.")
    return DRPColumns(target=target, ge_features=ge, dd_features=dd)


def load_drp_parquet(path: str | Path) -> tuple[pd.DataFrame, DRPColumns]:
    """Load a DRP parquet and drop rows with NaN in target or features.

    Some upstream parquets (e.g., `topN_generator` output for certain
    targets) leak NaN target values into a handful of rows. Those would
    produce NaN losses during training and silently corrupt the model, so
    we filter them at load time and log how many were removed.
    """
    df = pd.read_parquet(path)
    cols = detect_columns(df)
    keep = [cols.target] + cols.feature_columns
    df = df[keep].reset_index(drop=True)

    n_before = len(df)
    df = df.dropna(subset=keep).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        log.warning(
            "dropped %d rows with NaN in target or features (%.4f%% of %d)",
            n_dropped, 100.0 * n_dropped / n_before, n_before,
        )
    return df, cols


class DRPDataset(Dataset):
    """PyTorch Dataset wrapping a (target, features) numpy array pair."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                f"features and targets length mismatch: {features.shape[0]} vs {targets.shape[0]}"
            )
        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        if np.isnan(targets).any() or np.isnan(features).any():
            raise ValueError(
                "DRPDataset received NaN values; load via load_drp_parquet() to filter them."
            )
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]
