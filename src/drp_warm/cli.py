"""Command-line entry points."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

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


if __name__ == "__main__":
    split_main()
