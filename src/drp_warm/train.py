"""Warm-phase training loop with selectable LR scheduling.

The LR mode is configurable so the same training code drives both the
production CLR experiment and a fixed-LR SGD baseline (used to validate the
optimizer-choice claim).

- ``lr_mode="clr"``: triangular `torch.optim.lr_scheduler.CyclicLR`
  oscillating between ``base_lr`` and ``max_lr`` over ``cycle_steps`` batches
  per half-cycle. ``cycle_momentum=False`` so momentum stays at its SGD
  default.
- ``lr_mode="fixed"``: no scheduler. ``base_lr`` is the constant LR.

Every epoch we (a) compute val MAE and R² on the held-out fraction of the
input partition, (b) checkpoint the model state, and (c) append a row to
``history.csv``. The checkpoint frequency matches the original PoC (every
epoch) so the continue-phase can be initialized from any warm epoch.
"""

from __future__ import annotations

import csv
import logging
import math
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader

from drp_warm.data import DRPColumns, DRPDataset
from drp_warm.model import DRPRegressor

log = logging.getLogger("drp_warm.train")

LRMode = Literal["clr", "fixed"]


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    dropout: float = 0.2
    lr_mode: LRMode = "clr"
    base_lr: float = 1e-4
    max_lr: float = 1e-3
    cycle_steps: int = 2000  # batches per CLR half-cycle (Smith 2017 default)
    momentum: float = 0.9
    val_fraction: float = 0.2
    seed: int = 42
    device: str | None = None  # None -> auto-detect cuda

    def resolved_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EpochMetrics:
    epoch: int
    train_mse: float
    train_mae: float
    val_mse: float
    val_mae: float
    val_r2: float
    lr_last: float
    seconds: float


def train_warm(
    partition: pd.DataFrame,
    columns: DRPColumns,
    output_dir: Path,
    config: TrainConfig | None = None,
) -> list[EpochMetrics]:
    """Train an MLP regressor on `partition` with warm-phase checkpointing.

    Splits `partition` internally into train/val (`config.val_fraction`),
    fits a `StandardScaler` on the training features only, then runs the
    training loop. Writes per-epoch checkpoints to ``output_dir/models/``
    and a `history.csv` plus `config.json` and `scaler.pkl` to ``output_dir``.
    """
    cfg = config or TrainConfig()
    output_dir = Path(output_dir)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # Internal train/val split (row-level; partition is already disjoint w.r.t. cell/drug)
    n = len(partition)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * cfg.val_fraction))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    y_all = partition[columns.target].to_numpy(dtype=np.float32)
    x_all = partition[columns.feature_columns].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_all[tr_idx]).astype(np.float32)
    x_vl = scaler.transform(x_all[val_idx]).astype(np.float32)
    y_tr = y_all[tr_idx]
    y_vl = y_all[val_idx]
    with (output_dir / "scaler.pkl").open("wb") as fh:
        pickle.dump(scaler, fh)

    train_ds = DRPDataset(x_tr, y_tr)
    val_ds = DRPDataset(x_vl, y_vl)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=max(cfg.batch_size * 8, 256), shuffle=False)

    device = cfg.resolved_device()
    log.info("device=%s  train=%d  val=%d  features=%d", device, len(train_ds), len(val_ds), columns.n_features)

    model = DRPRegressor(input_dim=columns.n_features, dropout=cfg.dropout).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.base_lr, momentum=cfg.momentum)
    scheduler = _build_scheduler(optimizer, cfg)

    mse = nn.MSELoss()
    history: list[EpochMetrics] = []
    history_path = output_dir / "history.csv"
    fieldnames = list(EpochMetrics.__dataclass_fields__.keys())
    with history_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

    _write_json(output_dir / "config.json", asdict(cfg))

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_mse, tr_mae = _train_one_epoch(model, train_loader, optimizer, scheduler, mse, device)
        vl_mse, vl_mae, vl_r2 = _evaluate(model, val_loader, mse, device)
        lr_last = optimizer.param_groups[0]["lr"]
        seconds = time.time() - t0
        metrics = EpochMetrics(
            epoch=epoch,
            train_mse=tr_mse,
            train_mae=tr_mae,
            val_mse=vl_mse,
            val_mae=vl_mae,
            val_r2=vl_r2,
            lr_last=lr_last,
            seconds=seconds,
        )
        history.append(metrics)
        with history_path.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writerow(asdict(metrics))
        ckpt_path = output_dir / "models" / f"epoch_{epoch:03d}.pt"
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "config": asdict(cfg)}, ckpt_path)
        log.info(
            "epoch %3d/%d  train_mae=%.4f  val_mae=%.4f  val_r2=%.4f  lr=%.2e  (%.1fs)",
            epoch, cfg.epochs, tr_mae, vl_mae, vl_r2, lr_last, seconds,
        )

    return history


def _build_scheduler(optimizer: optim.Optimizer, cfg: TrainConfig):
    if cfg.lr_mode == "fixed":
        return None
    if cfg.lr_mode == "clr":
        return optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=cfg.base_lr,
            max_lr=cfg.max_lr,
            step_size_up=cfg.cycle_steps,
            mode="triangular",
            cycle_momentum=False,
        )
    raise ValueError(f"Unknown lr_mode={cfg.lr_mode!r}")


def _train_one_epoch(model, loader, optimizer, scheduler, mse, device) -> tuple[float, float]:
    model.train()
    total_se = 0.0
    total_ae = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = mse(pred, yb)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            total_se += float(((pred - yb) ** 2).sum())
            total_ae += float((pred - yb).abs().sum())
            n += yb.shape[0]
    return total_se / n, total_ae / n


@torch.no_grad()
def _evaluate(model, loader, mse, device) -> tuple[float, float, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        preds.append(model(xb).cpu())
        targets.append(yb.cpu())
    p = torch.cat(preds)
    t = torch.cat(targets)
    vl_mse = float(((p - t) ** 2).mean())
    vl_mae = float((p - t).abs().mean())
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    vl_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if math.isnan(vl_r2):
        vl_r2 = 0.0
    return vl_mse, vl_mae, vl_r2


def _write_json(path: Path, data: dict) -> None:
    import json

    with path.open("w") as fh:
        json.dump(data, fh, indent=2, default=str)
