# CLAUDE.md — working notes for Claude Code sessions in `drp-warm-start`

Read this at the start of any session in this repo.

## What this project is

A PyTorch reimplementation of a 2018–2019 DOE Pilot 1 / ECP CANDLE
proof-of-concept that accelerates training of deep cancer drug response
prediction (DRP) models across the (cell, drug) space via warm-start
fine-tuning plus a triangular cyclical learning rate (CLR) schedule, under
strict disjoint (cell, drug) evaluation. The original (Keras / TF 1.x) is
preserved separately as a historical archive; this repo is the modern
reproduction.

## Terminology — use these consistently

- **DRP** — drug response prediction.
- **(cell, drug) space** — the 2-D space of (cell-line, drug) combinations
  that the program needs to model. Always in this order, never `(drug,
  cell)`.
- **Strict disjoint (cell, drug) evaluation** — train and test share *no*
  cell lines AND *no* drugs. Distinct from "disjoint by cell" or "disjoint
  by drug" alone.
- **Warm phase / continue phase** — the two stages of the pipeline.
  Partition A is used for warm; partition B for continue (and for the
  reference baseline).
- **`weps`** — warm epoch; the epoch number of the warm checkpoint used to
  seed continue training.
- **`ceps`** — continue epoch; number of epochs of continue training needed
  to reach the reference target val MAE.
- **GE_\*** — gene expression feature columns (no hyphen between "gene"
  and "expression" outside this prefix).
- **DD_\*** — drug descriptor feature columns.

## Code conventions

- Plain professional style. No emojis. No marketing language. No "headline
  contribution" framing in code or docstrings.
- Avoid hyphenated compound nouns where the bare form is conventional:
  prefer "drug response prediction", "drug descriptor", "gene expression".
  Keep compound adjectives that modify a noun (e.g., "warm-start
  fine-tuning", "leakage-mitigated").
- Type hints on public functions; dataclasses for configuration.
- One responsibility per module under `src/drp_warm/`.
- Tests live in `tests/`; use synthetic data where possible.

## Test runner

```bash
pytest tests/                    # full suite
pytest tests/test_train.py -v    # one file
```

On macOS conda + pip-installed PyTorch, prefix with `KMP_DUPLICATE_LIB_OK=TRUE`
to work around the libomp conflict. On Linux / Lambda this is not needed.

## Common commands

```bash
# Install (editable + dev deps)
pip install -e ".[dev]"

# Partition data
python scripts/01_split.py --input data/top_6.auc1.parquet \
    --output data/auc1_splits/ --split-by both

# Warm-phase training (CLR) on partition A
python scripts/02_train_warm.py --data data/auc1_splits/partition_a.parquet \
    --output outputs/auc1_warm_clr/ --epochs 300 --lr-mode clr

# Reference baseline (from-scratch on partition B with CLR)
python scripts/02_train_warm.py --data data/auc1_splits/partition_b.parquet \
    --output outputs/auc1_ref/ --epochs 300 --lr-mode clr

# Continue-phase training (CLR fine-tune from warm checkpoint)
python scripts/03_train_continue.py --warm-dir outputs/auc1_warm_clr/ --weps 300 \
    --data data/auc1_splits/partition_b.parquet --output outputs/auc1_continue_clr_weps300/ \
    --ref-dir outputs/auc1_ref/ --epochs 300 --lr-mode clr
```

## Data contract

`01_split.py` consumes one parquet with columns:

- `AUC` or `AUC1` — drug response target ∈ [0, 1].
- `GE_*` — gene expression features.
- `DD_*` — drug descriptor features.

One row per (cell, drug) observation. Cell identity is recoverable from the
GE feature vector; drug identity from the DD vector. Both `01_split.py` and
the data loader assume this without requiring explicit CELL / DRUG columns.

## Data is *not* in the repo

`data/` and `outputs/` are gitignored. The canonical input is the Top6
parquet with **AUC1** as the target (matches the 2018 PoC's metric),
produced by <https://github.com/hyoo/topN_generator>:

```bash
python build.py --top_n 6 --drug_descriptor dragon7 \
                --cell_feature rnaseq --cell_feature_subset lincs1000 \
                --target AUC1 --format parquet
```

Top6 with AUC1: 271,575 rows × 355 cells × 1,572 drugs across 6 cancer
types, 942 GE features, 5,270 DD features. 12 NaN-target rows are dropped
at load → 271,563 used. See README for the raw input file list.

## On vendoring `topN_generator`

It's not vendored. Reasons:

1. It's third-party code (hyoo) with its own evolution path.
2. The data contract (parquet + `AUC*` / `GE_*` / `DD_*` columns) is the
   actual interface; the producer is interchangeable.
3. Vendoring adds maintenance and authorship burden without buying
   reproducibility — the raw inputs are the bottleneck, not the script.

Treat `topN_generator` as an external upstream. On a fresh Lambda box, the
flow is:

```bash
git clone https://github.com/hyoo/topN_generator
# Copy the 5 raw input files into topN_generator/data/  (see download.sh)
cd topN_generator && python build.py --format parquet
# Move the resulting top_*.parquet into drp-warm-start/data/
```

If a Lambda session needs to regenerate Top6 from scratch, that's the
pattern. Otherwise, copy a pre-built parquet onto the box and skip
regeneration.

## Lambda-specific notes

- Single-GPU is sufficient for both warm and continue phases. No multi-GPU /
  multi-node scaffolding here.
- Use a fresh venv (`python -m venv .venv && source .venv/bin/activate`) or
  a conda env. Avoid mixing conda-forge and pip for PyTorch.
- Checkpointing every epoch produces ~300 files for a 300-epoch warm run;
  ensure `outputs/` is on a disk with enough space (~a few GB for the full
  sweep).
- **PyTorch CUDA wheel mismatch.** The default PyPI torch wheel targets a
  CUDA runtime newer than some Lambda boxes' drivers. If
  `torch.cuda.is_available()` raises `The NVIDIA driver on your system is
  too old`, check `nvidia-smi` for the supported CUDA version and reinstall
  torch from the matching index:
  ```bash
  pip uninstall -y torch
  pip install torch --index-url https://download.pytorch.org/whl/cu124   # or cu121 / cu118
  ```
- **Multi-GPU box, single-GPU job.** Prefer `CUDA_VISIBLE_DEVICES=N python ...
  --device cuda` over `--device cuda:N`. The script stays GPU-agnostic and
  parallel invocations don't need code changes.

## Results (2026-05 reproduction, AUC1 on Top6)

Source of truth: each run's `result.json` + `history.csv` under
`outputs/auc1_*/`. Analysis script: `notebooks/analyze_results.py`
(use `--prefix auc1_`).

### Reference baseline (from-scratch on partition B with CLR)
- `outputs/auc1_ref/`, 300 epochs
- `ref_min` = 0.06432
- First reaches `target = ref_min × 1.02 = 0.06560` at epoch 121

### ceps-vs-weps (CLR continue, target = `ref_min × 1.02`)
| weps | ceps | speedup |
|---:|---:|---:|
| 50  | 72 | 1.68× |
| 150 | 84 | 1.44× |
| 300 | 74 | 1.64× |

Essentially flat — weps choice barely affects ceps past 50 warm epochs.

### CLR vs fixed-LR continue (weps=300, four schedules)
| schedule | ceps | converged | final val MAE |
|---|---:|---|---:|
| CLR (1e-4 → 1e-3) | 74 | ✓ | 0.06552 |
| Fixed 1e-3 (high end of CLR range) | 86 | ✓ | 0.06546 |
| Fixed 5e-4 (mid range) | 113 | ✓ | 0.06555 |
| Fixed 1e-4 (low end / warm-trained LR) | 600 (cap) | ✗ | 0.06631 |

**Main finding: CLR beats every fixed LR within its sweep range.**
Mixing of high- and low-LR phases beats either extreme as a fixed
value. Fixed 1e-4 (the low end) doesn't converge in 600 epochs — one
end of the curve, not a special property of 1e-4.

### Speedup-vs-target (`auc1_continue_clr_weps300` vs `auc1_ref`)
| target val MAE | ref ep | continue ep | speedup |
|---:|---:|---:|---:|
| 0.100  | 26  | 1  | 26×  |
| 0.080  | 37  | 3  | 12×  |
| 0.070  | 61  | 12 | 5.1× ← 2018 PoC's reported reference accuracy |
| 0.0656 | 121 | 74 | 1.6× ← modern tight target (`ref_min × 1.02`) |

Speedup is target-dependent — large at loose targets (pre-trained
checkpoint already partway down the loss curve), modest at tight ones
(both warm-continue and ref approach the same MAE floor). Headline
number: **5× at the 2018 PoC's accuracy bar**, up to 26× at very loose
targets, ~1.6× at the modern tight target.

### What did not reproduce
- The 2018 PoC's headline **≥50× speedup**. Our reproduction tops out
  at ~26× (very loose targets), 5× at the 2018 accuracy bar. Likely
  Keras vs PyTorch initialization / optimizer-state differences. Don't
  claim 50× when discussing this repo's results.

## Out of scope for this repo

- Multi-GPU training, distributed data parallelism.
- Other DRP datasets (Top21, etc.).
- Other model families (LightGBM, RandomForest, attention variants from the
  original codebase).
- Active learning, uncertainty estimation, MoE.

If a request would expand scope beyond this list, push back and confirm
before implementing.
