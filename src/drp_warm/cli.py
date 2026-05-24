"""Command-line entry points."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from drp_warm.continue_train import continue_train, read_ref_min
from drp_warm.data import load_drp_parquet
from drp_warm.split import partition
from drp_warm.train import TrainConfig, train_warm

log = logging.getLogger("drp_warm")


def split_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Partition a DRP parquet into disjoint cell/drug subsets."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to input parquet.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--split-by",
        choices=["none", "cell", "drug", "both"],
        default="both",
        help="Partition mode (default: both = joint disjoint cell+drug).",
    )
    parser.add_argument("--test-size", type=float, default=0.5, help="Used for cell/drug/none modes.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("loading %s", args.input)
    df, cols = load_drp_parquet(args.input)
    log.info("rows=%d  GE=%d  DD=%d  target=%s", len(df), len(cols.ge_features), len(cols.dd_features), cols.target)

    parts = partition(df, cols, split_by=args.split_by, test_size=args.test_size, seed=args.seed)
    log.info(
        "partition A: rows=%d cells=%d drugs=%d",
        len(parts.a), parts.n_cells_a, parts.n_drugs_a,
    )
    log.info(
        "partition B: rows=%d cells=%d drugs=%d",
        len(parts.b), parts.n_cells_b, parts.n_drugs_b,
    )
    log.info(
        "overlap: shared_cells=%d shared_drugs=%d",
        parts.shared_cells, parts.shared_drugs,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    a_path = args.output / "partition_a.parquet"
    b_path = args.output / "partition_b.parquet"
    parts.a.to_parquet(a_path, index=False)
    parts.b.to_parquet(b_path, index=False)
    log.info("wrote %s and %s", a_path, b_path)


def train_warm_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Warm-phase training with selectable LR schedule (CLR or fixed)."
    )
    parser.add_argument("--data", required=True, type=Path, help="Partition parquet (e.g., partition_a.parquet).")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr-mode", choices=["clr", "fixed"], default="clr")
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--max-lr", type=float, default=1e-3, help="Only used in lr-mode=clr.")
    parser.add_argument("--cycle-steps", type=int, default=2000, help="Batches per CLR half-cycle.")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda / cpu / mps. Default: auto.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("loading %s", args.data)
    df, cols = load_drp_parquet(args.data)
    log.info("rows=%d  features=%d  target=%s", len(df), cols.n_features, cols.target)

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout=args.dropout,
        lr_mode=args.lr_mode,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        cycle_steps=args.cycle_steps,
        momentum=args.momentum,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
    )
    train_warm(df, cols, output_dir=args.output, config=cfg)


def continue_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Continue-phase training: fine-tune a warm checkpoint on partition B with target-driven early stop."
    )
    parser.add_argument("--warm-dir", required=True, type=Path,
                        help="Warm-phase output directory (contains models/ and scaler.pkl).")
    parser.add_argument("--weps", required=True, type=int, help="Warm epoch to load (checkpoint index).")
    parser.add_argument("--data", required=True, type=Path, help="Partition B parquet.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for this continue run.")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--target-val-mae", type=float, help="Convergence target on val MAE (absolute).")
    target_group.add_argument("--ref-dir", type=Path, help="Reference run dir; target derived as ref_min × (1 + --ref-margin).")
    parser.add_argument("--ref-margin", type=float, default=0.02, help="Margin above ref_min when --ref-dir is used.")
    parser.add_argument("--epochs", type=int, default=300, help="Epoch cap.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr-mode", choices=["clr", "fixed"], default="clr")
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--cycle-steps", type=int, default=2000)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("loading %s", args.data)
    df, cols = load_drp_parquet(args.data)
    log.info("rows=%d  features=%d  target=%s", len(df), cols.n_features, cols.target)

    if args.target_val_mae is not None:
        target = args.target_val_mae
        log.info("target_val_mae=%.5f (explicit)", target)
    else:
        ref_min = read_ref_min(args.ref_dir, metric="val_mae")
        target = ref_min * (1.0 + args.ref_margin)
        log.info("target_val_mae=%.5f (ref_min=%.5f × (1 + %.3f))", target, ref_min, args.ref_margin)

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, dropout=args.dropout,
        lr_mode=args.lr_mode, base_lr=args.base_lr, max_lr=args.max_lr,
        cycle_steps=args.cycle_steps, momentum=args.momentum,
        val_fraction=args.val_fraction, seed=args.seed, device=args.device,
    )
    ceps, _ = continue_train(
        warm_dir=args.warm_dir, weps=args.weps, partition=df, columns=cols,
        target_val_mae=target, output_dir=args.output, config=cfg, ref_margin=args.ref_margin,
    )
    log.info("ceps=%d", ceps)


if __name__ == "__main__":
    split_main()
