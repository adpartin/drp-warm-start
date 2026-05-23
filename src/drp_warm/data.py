"""Data loading, feature/target separation, and PyTorch Dataset for DRP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
    df = pd.read_parquet(path)
    cols = detect_columns(df)
    keep = [cols.target] + cols.feature_columns
    return df[keep].reset_index(drop=True), cols


class DRPDataset(Dataset):
    """PyTorch Dataset wrapping a (target, features) numpy array pair."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                f"features and targets length mismatch: {features.shape[0]} vs {targets.shape[0]}"
            )
        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.targets = torch.from_numpy(np.asarray(targets, dtype=np.float32))

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]
