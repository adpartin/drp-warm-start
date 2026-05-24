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
python scripts/01_split.py --input data/top_6.parquet \
    --output data/splits/ --split-by both

# Warm-phase training (CLR)
python scripts/02_train_warm.py --data data/splits/partition_a.parquet \
    --output outputs/warm_clr/ --epochs 300 --lr-mode clr

# Warm-phase training (fixed-LR baseline; for the Day-5 head-to-head)
python scripts/02_train_warm.py --data data/splits/partition_a.parquet \
    --output outputs/warm_fixed/ --epochs 300 --lr-mode fixed
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

`data/` and `outputs/` are gitignored. The default expectation is the Top6
parquet produced by <https://github.com/hyoo/topN_generator> with:

```bash
python build.py --top_n 6 --drug_descriptor dragon7 \
                --cell_feature rnaseq --cell_feature_subset lincs1000 \
                --format parquet
```

Default Top6 produces 270,426 rows × 355 cells × 1,572 drugs across 6 cancer
types, with 942 GE features and 5,270 DD features. See README for raw input
file list.

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

## Reproduction targets (qualitative)

- Continue phase from `weps ∈ {50, 150, 300}` should converge to the
  reference val MAE in dramatically fewer epochs than the from-scratch
  reference. Original PoC reported ≥50× (up to ~65×).
- A from-scratch reference run with CLR should reach val MAE ≈ 0.07 / val
  R² ≈ 0.65 on AUC over the Top6 disjoint partition.

These are sanity targets, not pass/fail thresholds — environment and
hyperparameter drift can shift absolute numbers.

## Experimental hypothesis (continue-phase / cross-distribution transfer)

The headline question this repo is set up to answer:

> When fine-tuning a partition-A-pretrained model on partition B (no shared
> cells, no shared drugs), does CLR provide a *qualitative* advantage over
> fixed-LR SGD / Adam, or only a quantitative one?

**Hypothesis to test** (carried from the original 2018 PoC's intent): the
high-LR phase of triangular CLR provides escape velocity from the
partition-A basin that fine-LR SGD cannot supply at its converged learning
rate, so naive SGD/Adam fine-tuning stalls while CLR fine-tuning succeeds.

**Literature backing.** Saddle points dominate high-dimensional DL loss
landscapes (Dauphin et al. 2014). LR cycling / warm restarts are a
recognized mechanism for escaping inherited basins during transfer
(Loshchilov & Hutter SGDR 2017; Smith CLR 2017). So the *direction* of the
hypothesis is well-supported.

**Where the hypothesis can overreach.** Literature typically reports CLR /
SGDR as *faster* than fixed-LR, not *categorically required*. A binary
"SGD fails, CLR succeeds" outcome would be stronger than the norm and
should be backed by direct evidence in this repo's results — not asserted
from the 2018 recollection. We have already observed on partition A that
CLR and fixed-LR `5e-4` produce indistinguishable val MAE / R² when
training from scratch; the case for CLR's uniqueness has to live in the
cross-distribution continue phase, if anywhere.

### Experimental matrix (continue phase)

| warm init | continue schedule | continue LR | purpose |
|---|---|---|---|
| from-scratch on B (`outputs/ref/`) | clr | 1e-4 → 1e-3 | baseline; defines `ref_min` |
| `outputs/warm_clr/` `weps=300` | clr | 1e-4 → 1e-3 | original PoC configuration |
| `outputs/warm_clr/` `weps=300` | fixed | 1e-4 | low fixed (tests "SGD stalls" claim) |
| `outputs/warm_clr/` `weps=300` | fixed | 5e-4 | fair fixed-LR baseline (CLR's geo-mean) |
| `outputs/warm_clr/` `weps=300` | fixed | 1e-3 | upper-LR comparison (CLR's max) |

Three additional runs sweep `weps ∈ {50, 150, 300}` with the CLR continue
schedule to reproduce the ceps-vs-weps speedup curve. Result vector per
run: `ceps` (epochs to reach `ref_min × 1.02`), final val MAE, final val R².

### What to watch for in results

1. **All schedules should beat from-scratch.** If even fixed-LR `1e-4`
   reaches the target faster than the reference's full convergence, warm-
   start works regardless of schedule.
2. **CLR's marginal advantage.** If CLR converges only 1.5–3× faster than
   the best fixed LR, the honest story is "warm-start is the lever; CLR
   is the schedule choice." If CLR is the *only* schedule that converges
   at all (others plateau above target), that's the stronger 2018-style
   claim and we can lead with it.
3. **Per-LR variance.** Fixed-LR with too-small `lr` will be slow; with
   too-large `lr` may oscillate. The dispersion across fixed-LR values is
   itself informative — large dispersion suggests CLR's robustness
   matters; small dispersion suggests the choice doesn't.

## Out of scope for this repo

- Multi-GPU training, distributed data parallelism.
- Other DRP datasets (Top21, etc.).
- Other model families (LightGBM, RandomForest, attention variants from the
  original codebase).
- Active learning, uncertainty estimation, MoE.

If a request would expand scope beyond this list, push back and confirm
before implementing.
