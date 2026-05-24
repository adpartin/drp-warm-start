"""Smoke + behavior tests for continue-phase training.

Train a brief warm phase on synthetic data, then exercise the continue
trainer on a different sample of the same distribution. The synthetic data
is the same learnable shape used in ``test_train.py``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drp_warm.continue_train import continue_train, load_warm_checkpoint, read_ref_min
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
    target = (target - target.min()) / (target.max() - target.min())
    cols = ["AUC1", *[f"GE_g{i}" for i in range(n_ge)], *[f"DD_d{i}" for i in range(n_dd)]]
    return pd.DataFrame(np.column_stack([target, ge, dd]), columns=cols)


@pytest.fixture(scope="module")
def warm_run(tmp_path_factory) -> tuple[Path, pd.DataFrame]:
    """Train a brief warm phase; return (warm_dir, second-sample dataframe)."""
    df_a = _learnable_drp(seed=1)
    cols = detect_columns(df_a)
    warm_dir = tmp_path_factory.mktemp("warm")
    cfg = TrainConfig(
        epochs=5, batch_size=64, lr_mode="clr",
        base_lr=1e-3, max_lr=5e-3, cycle_steps=50,
        val_fraction=0.25, seed=0, device="cpu",
    )
    train_warm(df_a, cols, output_dir=warm_dir, config=cfg)
    df_b = _learnable_drp(seed=2)  # same distribution, different sample
    return warm_dir, df_b


def test_load_warm_checkpoint_returns_runnable_model(warm_run):
    warm_dir, df_b = warm_run
    cols = detect_columns(df_b)
    import torch
    model = load_warm_checkpoint(warm_dir, weps=3, input_dim=cols.n_features, dropout=0.2, device=torch.device("cpu"))
    model.eval()
    x = torch.from_numpy(df_b[cols.feature_columns].to_numpy(dtype=np.float32)[:8])
    with torch.no_grad():
        y = model(x)
    assert y.shape == (8,)


def test_continue_train_stops_early_when_target_is_reachable(warm_run, tmp_path):
    warm_dir, df_b = warm_run
    cols = detect_columns(df_b)
    cfg = TrainConfig(epochs=20, batch_size=64, lr_mode="clr",
                      base_lr=1e-3, max_lr=5e-3, cycle_steps=50,
                      val_fraction=0.25, seed=0, device="cpu")
    # Generous target: any non-broken model should hit val_mae ≤ 0.5 quickly on this synthetic data.
    ceps, history = continue_train(
        warm_dir=warm_dir, weps=5, partition=df_b, columns=cols,
        target_val_mae=0.5, output_dir=tmp_path / "continue", config=cfg,
    )
    assert ceps < cfg.epochs, f"should have early-stopped before epoch cap (ceps={ceps})"
    assert ceps == len(history)
    assert history[-1].val_mae <= 0.5


def test_continue_train_runs_to_cap_when_target_unreachable(warm_run, tmp_path):
    warm_dir, df_b = warm_run
    cols = detect_columns(df_b)
    cfg = TrainConfig(epochs=3, batch_size=64, lr_mode="fixed",
                      base_lr=1e-3, val_fraction=0.25, seed=0, device="cpu")
    ceps, history = continue_train(
        warm_dir=warm_dir, weps=5, partition=df_b, columns=cols,
        target_val_mae=1e-6, output_dir=tmp_path / "continue_cap", config=cfg,
    )
    assert ceps == cfg.epochs
    assert len(history) == cfg.epochs


def test_continue_train_persists_result_json(warm_run, tmp_path):
    warm_dir, df_b = warm_run
    cols = detect_columns(df_b)
    cfg = TrainConfig(epochs=5, batch_size=64, lr_mode="clr",
                      base_lr=1e-3, max_lr=5e-3, cycle_steps=50,
                      val_fraction=0.25, seed=0, device="cpu")
    out = tmp_path / "continue_persist"
    ceps, _ = continue_train(
        warm_dir=warm_dir, weps=5, partition=df_b, columns=cols,
        target_val_mae=0.5, output_dir=out, config=cfg,
    )
    result = json.loads((out / "result.json").read_text())
    assert result["weps"] == 5
    assert result["ceps"] == ceps
    assert result["target_val_mae"] == 0.5
    assert (out / "model_final.pt").exists()
    assert (out / "scaler.pkl").exists()
    rows = list(csv.DictReader((out / "history.csv").open()))
    assert len(rows) == ceps


def test_continue_train_reuses_warm_scaler(warm_run, tmp_path):
    """Continue phase must transform partition B with the *warm* scaler, not refit."""
    warm_dir, df_b = warm_run
    cols = detect_columns(df_b)
    out = tmp_path / "continue_scaler"
    cfg = TrainConfig(epochs=2, batch_size=64, lr_mode="fixed", base_lr=1e-3,
                      val_fraction=0.25, seed=0, device="cpu")
    continue_train(warm_dir=warm_dir, weps=3, partition=df_b, columns=cols,
                   target_val_mae=0.01, output_dir=out, config=cfg)
    import pickle
    warm_scaler = pickle.loads((warm_dir / "scaler.pkl").read_bytes())
    cont_scaler = pickle.loads((out / "scaler.pkl").read_bytes())
    assert np.allclose(warm_scaler.mean_, cont_scaler.mean_)
    assert np.allclose(warm_scaler.scale_, cont_scaler.scale_)


def test_read_ref_min_from_history(warm_run):
    warm_dir, _ = warm_run
    ref_min = read_ref_min(warm_dir, metric="val_mae")
    history = pd.read_csv(warm_dir / "history.csv")
    assert ref_min == pytest.approx(history["val_mae"].min())
