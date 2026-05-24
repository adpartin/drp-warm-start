"""Tests for data loading + NaN filtering."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from drp_warm.data import DRPDataset, detect_columns, load_drp_parquet


def _synth_df(n: int = 50, n_ge: int = 4, n_dd: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ge = rng.standard_normal((n, n_ge)).astype("float32")
    dd = rng.standard_normal((n, n_dd)).astype("float32")
    target = rng.uniform(0, 1, size=n).astype("float32")
    cols = ["AUC1", *[f"GE_g{i}" for i in range(n_ge)], *[f"DD_d{i}" for i in range(n_dd)]]
    return pd.DataFrame(np.column_stack([target, ge, dd]), columns=cols)


def test_load_drp_parquet_drops_nan_target_rows(tmp_path, caplog):
    df = _synth_df()
    df.loc[[5, 6, 7], "AUC1"] = float("nan")
    path = tmp_path / "synth.parquet"
    df.to_parquet(path, index=False)

    with caplog.at_level(logging.WARNING, logger="drp_warm.data"):
        loaded, cols = load_drp_parquet(path)

    assert len(loaded) == len(df) - 3
    assert not loaded[cols.target].isna().any()
    assert any("dropped 3 rows" in rec.message for rec in caplog.records)


def test_load_drp_parquet_drops_nan_feature_rows(tmp_path):
    df = _synth_df()
    df.loc[10, "GE_g0"] = float("nan")
    df.loc[15, "DD_d1"] = float("nan")
    path = tmp_path / "synth.parquet"
    df.to_parquet(path, index=False)

    loaded, _ = load_drp_parquet(path)
    assert len(loaded) == len(df) - 2
    assert not loaded.isna().any().any()


def test_load_drp_parquet_no_warning_when_clean(tmp_path, caplog):
    df = _synth_df()
    path = tmp_path / "clean.parquet"
    df.to_parquet(path, index=False)

    with caplog.at_level(logging.WARNING, logger="drp_warm.data"):
        loaded, _ = load_drp_parquet(path)

    assert len(loaded) == len(df)
    assert not any("dropped" in rec.message for rec in caplog.records)


def test_drp_dataset_rejects_nan_input():
    n, d = 10, 4
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((n, d)).astype("float32")
    feats[3, 1] = float("nan")
    targets = rng.uniform(0, 1, size=n).astype("float32")
    with pytest.raises(ValueError, match="NaN"):
        DRPDataset(feats, targets)


def test_detect_columns_prefers_auc1_over_auc():
    df = _synth_df()
    df.insert(1, "AUC", df["AUC1"] * 0.5)
    cols = detect_columns(df)
    assert cols.target == "AUC1"
