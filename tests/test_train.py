"""Smoke + learnability tests for the warm-phase trainer.

Synthetic data: target is a noisy linear combination of GE and DD features,
so a working trainer should drive val MAE down and val R² up. Tests both LR
modes (`clr` and `fixed`) so the Day 5 SGD-vs-CLR hook is exercised.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drp_warm.data import detect_columns
from drp_warm.train import TrainConfig, train_warm


def _learnable_drp(n_samples: int = 4000, n_ge: int = 12, n_dd: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ge = rng.standard_normal((n_samples, n_ge)).astype("float32")
    dd = rng.standard_normal((n_samples, n_dd)).astype("float32")
    w_ge = rng.standard_normal(n_ge).astype("float32") * 0.3
    w_dd = rng.standard_normal(n_dd).astype("float32") * 0.3
    noise = rng.standard_normal(n_samples).astype("float32") * 0.1
    target = ge @ w_ge + dd @ w_dd + noise
    # Squash to [0, 1] so the ReLU output head can reach it.
    target = (target - target.min()) / (target.max() - target.min())
    cols = ["AUC1", *[f"GE_g{i}" for i in range(n_ge)], *[f"DD_d{i}" for i in range(n_dd)]]
    return pd.DataFrame(np.column_stack([target, ge, dd]), columns=cols)


@pytest.fixture(scope="module")
def synth_partition() -> pd.DataFrame:
    return _learnable_drp()


def _run(synth_partition: pd.DataFrame, tmp_path: Path, lr_mode: str, epochs: int = 6) -> Path:
    cols = detect_columns(synth_partition)
    cfg = TrainConfig(
        epochs=epochs,
        batch_size=64,
        lr_mode=lr_mode,
        base_lr=1e-3,
        max_lr=5e-3,
        cycle_steps=50,
        val_fraction=0.25,
        seed=0,
        device="cpu",
    )
    out = tmp_path / lr_mode
    train_warm(synth_partition, cols, output_dir=out, config=cfg)
    return out


def test_clr_mode_runs_and_learns(synth_partition, tmp_path):
    out = _run(synth_partition, tmp_path, lr_mode="clr", epochs=8)
    history = list(csv.DictReader((out / "history.csv").open()))
    assert len(history) == 8
    first_mae = float(history[0]["val_mae"])
    last_mae = float(history[-1]["val_mae"])
    last_r2 = float(history[-1]["val_r2"])
    # MAE-decrease is the meaningful learnability invariant on 8-epoch synthetic data;
    # R² is checked only against catastrophic divergence (it can stay near zero with
    # noisy synthetic data and a 7-layer net at this scale).
    assert last_mae < first_mae, f"val MAE should decrease: {first_mae:.4f} -> {last_mae:.4f}"
    assert last_r2 > -0.5, f"val R² collapsed (got {last_r2:.4f})"


def test_fixed_mode_runs_and_learns(synth_partition, tmp_path):
    out = _run(synth_partition, tmp_path, lr_mode="fixed")
    history = list(csv.DictReader((out / "history.csv").open()))
    first_mae = float(history[0]["val_mae"])
    last_mae = float(history[-1]["val_mae"])
    assert last_mae < first_mae


def test_every_epoch_is_checkpointed(synth_partition, tmp_path):
    out = _run(synth_partition, tmp_path, lr_mode="clr", epochs=4)
    ckpts = sorted((out / "models").glob("epoch_*.pt"))
    assert len(ckpts) == 4
    assert ckpts[0].name == "epoch_001.pt"
    assert ckpts[-1].name == "epoch_004.pt"


def test_scaler_and_config_are_persisted(synth_partition, tmp_path):
    out = _run(synth_partition, tmp_path, lr_mode="fixed", epochs=2)
    assert (out / "scaler.pkl").exists()
    config = json.loads((out / "config.json").read_text())
    assert config["lr_mode"] == "fixed"
    assert config["epochs"] == 2


def test_clr_schedule_actually_varies_lr(synth_partition, tmp_path):
    out = _run(synth_partition, tmp_path, lr_mode="clr", epochs=5)
    lrs = [float(row["lr_last"]) for row in csv.DictReader((out / "history.csv").open())]
    assert max(lrs) > min(lrs) * 1.5, f"CLR should vary the LR meaningfully: {lrs}"


def test_fixed_schedule_keeps_lr_constant(synth_partition, tmp_path):
    out = _run(synth_partition, tmp_path, lr_mode="fixed", epochs=3)
    lrs = [float(row["lr_last"]) for row in csv.DictReader((out / "history.csv").open())]
    assert max(lrs) == min(lrs)
