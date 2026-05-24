"""Continue-phase training: fine-tune a warm checkpoint on disjoint partition B.

Loads the model state and feature scaler produced by the warm phase, then
fine-tunes on partition B until validation MAE reaches a target threshold
(typically the from-scratch reference's ``ref_min × 1.02``) or the epoch
cap is reached. The number of epochs needed to hit the target — ``ceps`` —
is the headline output and the quantity behind the ≥50× speedup claim.

Sharing with the warm phase is kept minimal: this module imports the
per-epoch train and evaluate helpers from ``train`` and adds its own outer
loop with early-stopping logic.
"""

from __future__ import annotations

import csv
import json
import logging
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from drp_warm.callbacks import EarlyStoppingByMetric
from drp_warm.data import DRPColumns, DRPDataset
from drp_warm.model import DRPRegressor
from drp_warm.train import EpochMetrics, TrainConfig, _build_scheduler, _evaluate, _train_one_epoch

log = logging.getLogger("drp_warm.continue_train")


def read_ref_min(ref_dir: Path, metric: str = "val_mae") -> float:
    """Return the minimum value of ``metric`` from a reference run's history.csv."""
    history = pd.read_csv(Path(ref_dir) / "history.csv")
    if metric not in history.columns:
        raise KeyError(f"metric {metric!r} not in {history.columns.tolist()}")
    return float(history[metric].min())


def load_warm_checkpoint(warm_dir: Path, weps: int, input_dim: int, dropout: float, device: torch.device) -> DRPRegressor:
    """Build a DRPRegressor and load the warm-phase state dict from epoch ``weps``."""
    ckpt_path = Path(warm_dir) / "models" / f"epoch_{weps:03d}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"warm checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DRPRegressor(input_dim=input_dim, dropout=dropout).to(device)
    model.load_state_dict(state["state_dict"])
    return model


def continue_train(
    warm_dir: Path,
    weps: int,
    partition: pd.DataFrame,
    columns: DRPColumns,
    target_val_mae: float,
    output_dir: Path,
    config: TrainConfig | None = None,
    ref_margin: float = 0.02,
) -> tuple[int, list[EpochMetrics]]:
    """Fine-tune a warm checkpoint on partition B with target-driven early stop.

    Returns ``(ceps, history)``. ``ceps`` is the epoch at which validation
    MAE first reached ``target_val_mae`` (or ``len(history)`` if the cap was
    reached without convergence).

    The feature scaler from ``warm_dir/scaler.pkl`` is reused — partition B
    features are transformed (not refitted) so the loaded model's weights
    operate in the same input space they were trained in.
    """
    cfg = config or TrainConfig()
    warm_dir = Path(warm_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    n = len(partition)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * cfg.val_fraction))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    y_all = partition[columns.target].to_numpy(dtype=np.float32)
    x_all = partition[columns.feature_columns].to_numpy(dtype=np.float32)

    # Reuse the warm-phase scaler — DO NOT refit on partition B.
    with (warm_dir / "scaler.pkl").open("rb") as fh:
        scaler = pickle.load(fh)
    x_tr = scaler.transform(x_all[tr_idx]).astype(np.float32)
    x_vl = scaler.transform(x_all[val_idx]).astype(np.float32)
    y_tr, y_vl = y_all[tr_idx], y_all[val_idx]
    with (output_dir / "scaler.pkl").open("wb") as fh:
        pickle.dump(scaler, fh)

    train_ds = DRPDataset(x_tr, y_tr)
    val_ds = DRPDataset(x_vl, y_vl)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(cfg.batch_size * 8, 256), shuffle=False)

    device = cfg.resolved_device()
    log.info(
        "device=%s  warm_dir=%s  weps=%d  train=%d  val=%d  target_val_mae=%.5f",
        device, warm_dir, weps, len(train_ds), len(val_ds), target_val_mae,
    )

    model = load_warm_checkpoint(warm_dir, weps, columns.n_features, cfg.dropout, device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.base_lr, momentum=cfg.momentum)
    scheduler = _build_scheduler(optimizer, cfg)
    early_stop = EarlyStoppingByMetric(target=target_val_mae, mode="below")
    mse = nn.MSELoss()

    history: list[EpochMetrics] = []
    history_path = output_dir / "history.csv"
    fieldnames = list(EpochMetrics.__dataclass_fields__.keys())
    with history_path.open("w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writeheader()
    _write_json(output_dir / "config.json", {**asdict(cfg), "weps": weps, "target_val_mae": target_val_mae,
                                              "warm_dir": str(warm_dir), "ref_margin": ref_margin})

    converged = False
    ceps = cfg.epochs
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_mse, tr_mae = _train_one_epoch(model, train_loader, optimizer, scheduler, mse, device)
        vl_mse, vl_mae, vl_r2 = _evaluate(model, val_loader, mse, device)
        seconds = time.time() - t0
        metrics = EpochMetrics(
            epoch=epoch,
            train_mse=tr_mse,
            train_mae=tr_mae,
            val_mse=vl_mse,
            val_mae=vl_mae,
            val_r2=vl_r2,
            lr_last=optimizer.param_groups[0]["lr"],
            seconds=seconds,
        )
        history.append(metrics)
        with history_path.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writerow(asdict(metrics))
        log.info(
            "continue epoch %3d  val_mae=%.5f  val_r2=%.4f  lr=%.2e  (%.1fs)",
            epoch, vl_mae, vl_r2, metrics.lr_last, seconds,
        )
        if early_stop.should_stop(vl_mae):
            converged = True
            ceps = epoch
            log.info("converged: val_mae %.5f ≤ target %.5f at epoch %d", vl_mae, target_val_mae, epoch)
            break

    torch.save({"epoch": ceps, "state_dict": model.state_dict(), "config": asdict(cfg)},
               output_dir / "model_final.pt")
    final = history[-1]
    _write_json(output_dir / "result.json", {
        "weps": weps,
        "ceps": ceps,
        "converged": converged,
        "target_val_mae": target_val_mae,
        "final_val_mae": final.val_mae,
        "final_val_r2": final.val_r2,
    })
    log.info("ceps=%d  converged=%s  final_val_mae=%.5f", ceps, converged, final.val_mae)
    return ceps, history


def _write_json(path: Path, data: dict) -> None:
    with path.open("w") as fh:
        json.dump(data, fh, indent=2, default=str)
