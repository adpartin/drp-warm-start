#!/usr/bin/env python
"""CLI entry point: split a DRP parquet into disjoint partitions."""

from drp_warm.cli import split_main

if __name__ == "__main__":
    split_main()
