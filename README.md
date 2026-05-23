# drp-warm-start

Warm-start training with cyclical learning rate (CLR) for cancer drug
response prediction (DRP) under strict disjoint (cell, drug) evaluation.
PyTorch reimplementation of a 2018–2019 DOE Pilot 1 / ECP CANDLE
proof-of-concept.

## Problem

Training many DRP models across the (cell, drug) space — needed for the
CANDLE Pilot 1 challenge workflow — does not scale if each model is trained
from scratch. This repo implements a warm-start scheme: pre-train one model
on partition A of a multi-source pharmaco-omic dataset, then fine-tune from
a saved checkpoint on disjoint partition B using a triangular CLR schedule.
Continue training reaches the same val MAE as the from-scratch reference in
dramatically fewer epochs.

## Method

Three stages:

1. **Strict disjoint partitioning.** Split the (cell, drug) response matrix
   into two partitions A and B with no shared cell lines *and* no shared
   drugs. Pivot-table construction over the response matrix.
2. **Warm phase.** Train an MLP regressor from scratch on partition A with
   SGD + momentum driven by a triangular CLR schedule. Checkpoint every
   epoch.
3. **Continue phase.** Initialize a new model from one of the warm
   checkpoints, fine-tune on partition B, and stop when validation MAE
   reaches `ref_min × 1.02`, where `ref_min` is the converged MAE of a
   from-scratch reference model on partition B.

Reproduces the qualitative behavior of the original 2018 PoC — see
[Provenance](#provenance).

## Data

Input: a single parquet file with one row per (cell, drug) observation and
columns:

- `AUC` (or `AUC1`) — drug response target, ∈ [0, 1]
- `GE_*` — gene expression features
- `DD_*` — drug descriptor features

The Top6 aggregate used in the original PoC can be regenerated from raw
NCI / CCLE / GDSC source files using
<https://github.com/hyoo/topN_generator>:

```bash
python build.py --top_n 6 --drug_descriptor dragon7 \
                --cell_feature rnaseq --cell_feature_subset lincs1000 \
                --format parquet
```

Default Top6 produces 270,426 (cell, drug) samples spanning 355 cell lines
and 1,572 drugs across 6 cancer types, with 942 GE features and 5,270 DD
features.

`topN_generator` is treated as an external upstream tool, not vendored into
this repo. See `CLAUDE.md` for the rationale.

## Install

```bash
# Editable install (recommended for development)
pip install -e ".[dev]"

# Or with uv
uv pip install -e ".[dev]"
```

Python 3.10+, PyTorch 2.2+.

## Usage

```bash
# 1. Partition into disjoint (cell, drug) splits (writes partition_a.parquet
#    and partition_b.parquet; verifies zero shared cells AND zero shared drugs).
python scripts/01_split.py \
    --input data/top_6.parquet \
    --output data/splits/ \
    --split-by both

# 2. Warm-phase training on partition A (checkpoint every epoch).
python scripts/02_train_warm.py \
    --data data/splits/partition_a.parquet \
    --output outputs/warm_clr/ \
    --epochs 300 \
    --lr-mode clr \
    --base-lr 1e-4 --max-lr 1e-3

# 3. Continue-phase training (fine-tune from a warm checkpoint).
#    Lands in the next development phase.
```

### LR scheduling modes

`02_train_warm.py` accepts `--lr-mode {clr,fixed}`:

- `clr` — triangular `torch.optim.lr_scheduler.CyclicLR` oscillating
  between `--base-lr` and `--max-lr` over `--cycle-steps` batches per
  half-cycle.
- `fixed` — no scheduler; `--base-lr` is the constant LR.

The same trainer drives both modes so the SGD-vs-CLR head-to-head is two
invocations differing only in `--lr-mode`:

```bash
python scripts/02_train_warm.py --data ... --output outputs/warm_clr/   --lr-mode clr
python scripts/02_train_warm.py --data ... --output outputs/warm_fixed/ --lr-mode fixed
```

## Repository layout

```
drp-warm-start/
├── README.md
├── CLAUDE.md            # working notes for Claude Code sessions
├── pyproject.toml
├── src/drp_warm/
│   ├── __init__.py
│   ├── data.py          # parquet loading, feature/target split, Dataset
│   ├── split.py         # disjoint (cell, drug) partitioning
│   ├── model.py         # MLP regressor
│   ├── train.py         # warm-phase training loop (CLR or fixed LR)
│   └── cli.py           # CLI entry points
├── scripts/
│   ├── 01_split.py
│   └── 02_train_warm.py
├── tests/
│   ├── test_split.py
│   ├── test_model.py
│   └── test_train.py
└── notebooks/
```

## Notes on environments

- **Linux (Lambda)**: `pip install -e ".[dev]"` from a fresh venv. No
  conflicts.
- **macOS conda**: when PyTorch is pip-installed into a conda env that
  already has numpy / sklearn from conda-forge, you may hit
  `OMP: Error #15` (libomp loaded twice). Either install PyTorch from
  conda-forge alongside the rest, or set
  `KMP_DUPLICATE_LIB_OK=TRUE` for local testing.

## Provenance

Originally developed in 2018–2019 under the DOE Pilot 1 / ECP CANDLE
program as a Keras / TensorFlow 1.x proof-of-concept. The original PoC
reported ≥50× epoch speedup (up to ~65×) and val R² ≈ 0.65 / val MAE ≈
0.070 on AUC under joint disjoint (cell, drug) splits on Top6. This repo
aims to reproduce the qualitative result on the same Top6 data using a
modern PyTorch stack.

## References

- Smith, "Cyclical Learning Rates for Training Neural Networks" (2017),
  <https://arxiv.org/abs/1506.01186>.
- DOE Pilot 1 / ECP CANDLE program documentation.
