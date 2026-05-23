"""Tests for the disjoint cell/drug splitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drp_warm.data import detect_columns
from drp_warm.split import partition


def _synthetic_drp(n_cells: int = 20, n_drugs: int = 12, n_ge: int = 5, n_dd: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    cell_vecs = rng.standard_normal((n_cells, n_ge)).astype("float32")
    drug_vecs = rng.standard_normal((n_drugs, n_dd)).astype("float32")
    rows = []
    for ci in range(n_cells):
        for di in range(n_drugs):
            target = float(rng.uniform(0, 1))
            rows.append(np.concatenate([[target], cell_vecs[ci], drug_vecs[di]]))
    ge_cols = [f"GE_g{i}" for i in range(n_ge)]
    dd_cols = [f"DD_d{i}" for i in range(n_dd)]
    return pd.DataFrame(rows, columns=["AUC1", *ge_cols, *dd_cols])


def test_detect_columns_synthetic():
    df = _synthetic_drp()
    cols = detect_columns(df)
    assert cols.target == "AUC1"
    assert len(cols.ge_features) == 5
    assert len(cols.dd_features) == 4


def test_joint_disjoint_has_no_shared_cells_or_drugs():
    df = _synthetic_drp()
    cols = detect_columns(df)
    parts = partition(df, cols, split_by="both", seed=0)
    assert parts.shared_cells == 0, "partitions A and B must share zero cells"
    assert parts.shared_drugs == 0, "partitions A and B must share zero drugs"
    assert len(parts.a) > 0 and len(parts.b) > 0


def test_cell_disjoint_splits_cells_not_drugs():
    df = _synthetic_drp()
    cols = detect_columns(df)
    parts = partition(df, cols, split_by="cell", seed=0)
    assert parts.shared_cells == 0
    # Cell-only mode allows drug overlap; with 12 drugs both halves will see all.
    assert parts.shared_drugs > 0


def test_drug_disjoint_splits_drugs_not_cells():
    df = _synthetic_drp()
    cols = detect_columns(df)
    parts = partition(df, cols, split_by="drug", seed=0)
    assert parts.shared_drugs == 0
    assert parts.shared_cells > 0


def test_none_random_split_covers_all_rows():
    df = _synthetic_drp()
    cols = detect_columns(df)
    parts = partition(df, cols, split_by="none", test_size=0.3, seed=0)
    assert len(parts.a) + len(parts.b) == len(df)


def test_invalid_split_by_raises():
    df = _synthetic_drp()
    cols = detect_columns(df)
    with pytest.raises(ValueError):
        partition(df, cols, split_by="banana")  # type: ignore[arg-type]
