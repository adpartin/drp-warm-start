"""Disjoint cell/drug partitioning for DRP datasets.

The Top-N aggregate parquet does not always carry explicit CELL / DRUG
identifier columns; cell identity is recoverable from the GE_* feature
columns (one unique cell line per unique GE vector) and drug identity from
the DD_* columns. This module derives integer IDs from those features and
implements four partition modes:

- "none"  : random row-level split (sanity baseline; not disjoint).
- "cell"  : disjoint by cell line; drugs may overlap.
- "drug"  : disjoint by drug; cell lines may overlap.
- "both"  : strict joint disjoint — no cell AND no drug shared between the
            two partitions.

The "both" mode follows the pivot-table construction from the original 2018
PoC: form a (cell × drug) cross-tab, shuffle rows and columns, then assign
the top-left and bottom-right quadrants to partitions A and B respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from drp_warm.data import DRPColumns

SplitBy = Literal["none", "cell", "drug", "both"]


@dataclass(frozen=True)
class Partitions:
    a: pd.DataFrame
    b: pd.DataFrame
    n_cells_a: int
    n_cells_b: int
    n_drugs_a: int
    n_drugs_b: int
    shared_cells: int
    shared_drugs: int


def _feature_ids(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Assign an integer ID to each unique value of the given feature subset.

    Two rows with identical values across `feature_cols` share an ID. Uses
    `pd.util.hash_pandas_object` for a vectorized hash, then `factorize`
    to densify into 0..k-1.
    """
    keys = pd.util.hash_pandas_object(df[feature_cols], index=False)
    ids, _ = pd.factorize(keys, sort=False)
    return ids


def partition(
    df: pd.DataFrame,
    columns: DRPColumns,
    split_by: SplitBy = "both",
    test_size: float = 0.5,
    seed: int | None = 0,
) -> Partitions:
    """Split `df` into two partitions A and B according to `split_by`.

    Returns a `Partitions` bundle with row counts and (cell, drug) overlap
    diagnostics. For `split_by="both"`, `test_size` is ignored — the cross-tab
    is divided into halves along each axis.
    """
    rng = np.random.default_rng(seed)
    cell_ids = _feature_ids(df, columns.ge_features)
    drug_ids = _feature_ids(df, columns.dd_features)

    if split_by == "none":
        idx = np.arange(len(df))
        rng.shuffle(idx)
        cut = int(len(df) * (1 - test_size))
        mask_a = np.zeros(len(df), dtype=bool)
        mask_a[idx[:cut]] = True
    elif split_by == "cell":
        mask_a = _disjoint_mask(cell_ids, test_size, rng)
    elif split_by == "drug":
        mask_a = _disjoint_mask(drug_ids, test_size, rng)
    elif split_by == "both":
        mask_a, mask_b = _joint_disjoint_masks(cell_ids, drug_ids, rng)
        return _build_partitions(df, mask_a, mask_b, cell_ids, drug_ids)
    else:
        raise ValueError(f"Unknown split_by={split_by!r}")

    mask_b = ~mask_a
    return _build_partitions(df, mask_a, mask_b, cell_ids, drug_ids)


def _disjoint_mask(group_ids: np.ndarray, test_size: float, rng: np.random.Generator) -> np.ndarray:
    """Shuffle unique group IDs and assign each entire group to A or B."""
    unique = np.unique(group_ids)
    rng.shuffle(unique)
    cut = int(len(unique) * (1 - test_size))
    a_groups = set(unique[:cut].tolist())
    return np.array([g in a_groups for g in group_ids])


def _joint_disjoint_masks(
    cell_ids: np.ndarray,
    drug_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pivot-table construction for joint disjoint (cell, drug) split.

    Builds a (cell × drug) count matrix, shuffles both axes, then assigns the
    top-left block to A and the bottom-right block to B. Rows not in either
    block are dropped (they share either cells or drugs with both halves).
    """
    cells = np.unique(cell_ids)
    drugs = np.unique(drug_ids)
    rng.shuffle(cells)
    rng.shuffle(drugs)

    half_c = len(cells) // 2
    half_d = len(drugs) // 2
    a_cells = set(cells[:half_c].tolist())
    b_cells = set(cells[half_c:].tolist())
    a_drugs = set(drugs[:half_d].tolist())
    b_drugs = set(drugs[half_d:].tolist())

    cell_in_a = np.array([c in a_cells for c in cell_ids])
    cell_in_b = np.array([c in b_cells for c in cell_ids])
    drug_in_a = np.array([d in a_drugs for d in drug_ids])
    drug_in_b = np.array([d in b_drugs for d in drug_ids])

    mask_a = cell_in_a & drug_in_a
    mask_b = cell_in_b & drug_in_b
    return mask_a, mask_b


def _build_partitions(
    df: pd.DataFrame,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    cell_ids: np.ndarray,
    drug_ids: np.ndarray,
) -> Partitions:
    df_a = df.loc[mask_a].reset_index(drop=True)
    df_b = df.loc[mask_b].reset_index(drop=True)

    cells_a = set(cell_ids[mask_a].tolist())
    cells_b = set(cell_ids[mask_b].tolist())
    drugs_a = set(drug_ids[mask_a].tolist())
    drugs_b = set(drug_ids[mask_b].tolist())

    return Partitions(
        a=df_a,
        b=df_b,
        n_cells_a=len(cells_a),
        n_cells_b=len(cells_b),
        n_drugs_a=len(drugs_a),
        n_drugs_b=len(drugs_b),
        shared_cells=len(cells_a & cells_b),
        shared_drugs=len(drugs_a & drugs_b),
    )
