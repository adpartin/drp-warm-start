"""Analysis of the drp-warm-start continue-phase experiment.

Reads the reference run and the continue runs for a given experiment
prefix, then produces:

1. A summary table of all continue runs (weps, ceps, schedule, convergence).
2. The speedup-vs-target curve from the canonical CLR `weps=300` continue
   run vs the reference.
3. The ceps-vs-weps curve across CLR continue runs.
4. PNG plots under ``outputs/plots/<prefix>``.

Usage (from the repo root):

    python notebooks/analyze_results.py                # default: AUC runs (prefix='')
    python notebooks/analyze_results.py --prefix auc1_  # AUC1 runs

With ``--prefix auc1_`` the script reads:

    outputs/auc1_ref/history.csv
    outputs/auc1_continue_clr_weps{50,150,300}/{history.csv,result.json}
    outputs/auc1_continue_fixed{5e4,1e4}_weps300/{history.csv,result.json}

and writes plots under ``outputs/plots/auc1/``. With an empty prefix it
reads ``outputs/ref/`` and ``outputs/continue_*/`` (the original AUC runs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"


def load_result(run_dir: Path) -> dict:
    with (run_dir / "result.json").open() as fh:
        return json.load(fh)


def load_history(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "history.csv")


def first_epoch_below(history: pd.DataFrame, threshold: float) -> int | None:
    hit = history[history["val_mae"] <= threshold]
    return int(hit.iloc[0]["epoch"]) if len(hit) else None


# ---------------------------------------------------------------------------
# 1. Summary table of all continue runs
# ---------------------------------------------------------------------------
def summary_table(prefix: str) -> pd.DataFrame:
    rows: list[dict] = []
    for d in sorted(OUTPUTS.glob(f"{prefix}continue_*")):
        r = load_result(d)
        rows.append({
            "run": d.name,
            "weps": r["weps"],
            "ceps": r["ceps"],
            "converged": r["converged"],
            "target_val_mae": r["target_val_mae"],
            "final_val_mae": r["final_val_mae"],
            "final_val_r2": r["final_val_r2"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Speedup-vs-target curve (continue_clr_weps300 vs ref)
# ---------------------------------------------------------------------------
def speedup_vs_target_curve(ref_dir: Path, cont_dir: Path) -> pd.DataFrame:
    ref = load_history(ref_dir)
    cont = load_history(cont_dir)
    ref_min = float(ref["val_mae"].min())
    # Sweep from a loose target down to ref_min × 1.02 (the early-stop target).
    # Eight points spaced log-ish in the typical 0.05–0.10 val MAE range.
    targets = sorted({
        0.10, 0.09, 0.08, 0.075, 0.07, 0.068, 0.067, 0.06, 0.055,
        round(ref_min * 1.02, 5),
        round(ref_min * 1.005, 5),
    }, reverse=True)
    rows: list[dict] = []
    for t in targets:
        re = first_epoch_below(ref, t)
        ce = first_epoch_below(cont, t)
        rows.append({
            "target_val_mae": t,
            "ref_epoch": re,
            "continue_epoch": ce,
            "speedup": (re / ce) if (re and ce) else None,
        })
    return pd.DataFrame(rows)


def plot_speedup_vs_target(table: pd.DataFrame, out_path: Path) -> None:
    rows = table.dropna(subset=["speedup"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rows["target_val_mae"], rows["speedup"], "o-", color="C0")
    for _, r in rows.iterrows():
        ax.annotate(
            f"{r['speedup']:.1f}×",
            (r["target_val_mae"], r["speedup"]),
            textcoords="offset points", xytext=(6, 6), fontsize=9,
        )
    ax.set_xlabel("Target val MAE")
    ax.set_ylabel("Speedup (ref_epochs / continue_epochs)")
    ax.set_title("Speedup vs target stringency\n(continue_clr_weps300 vs ref)")
    ax.invert_xaxis()  # tighter target on the right (lower val_mae)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. ceps-vs-weps curve (CLR continue runs only)
# ---------------------------------------------------------------------------
def ceps_vs_weps_table(summary: pd.DataFrame) -> pd.DataFrame:
    clr = summary[summary["run"].str.contains(r"continue_clr_weps\d+$", regex=True)].copy()
    clr = clr.sort_values("weps").reset_index(drop=True)
    return clr[["weps", "ceps", "converged", "final_val_mae", "final_val_r2"]]


def plot_ceps_vs_weps(table: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(table["weps"], table["ceps"], "o-", color="C3")
    for _, r in table.iterrows():
        ax.annotate(
            f"ceps={int(r['ceps'])}",
            (r["weps"], r["ceps"]),
            textcoords="offset points", xytext=(8, 6), fontsize=9,
        )
    ax.set_xlabel("Warm checkpoint epoch (weps)")
    ax.set_ylabel("Continue epochs to target (ceps)")
    ax.set_title("ceps vs weps (CLR continue, target = ref_min × 1.02)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Day-5 head-to-head (weps=300, three schedules)
# ---------------------------------------------------------------------------
def head_to_head_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = summary[summary["run"].str.contains(r"_weps300$", regex=True)].copy()

    def schedule(name: str) -> str:
        if "fixed5e4" in name:
            return "Fixed LR = 5e-4"
        if "fixed1e4" in name:
            return "Fixed LR = 1e-4"
        if "clr" in name:
            return "CLR (1e-4 → 1e-3)"
        return name

    rows["schedule"] = rows["run"].map(schedule)
    return rows[["schedule", "ceps", "converged", "final_val_mae", "final_val_r2"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--prefix", default="",
        help="Output-directory prefix (e.g., 'auc1_' for the AUC1 runs). "
             "Empty = original AUC runs under outputs/continue_*/.",
    )
    args = parser.parse_args(argv)

    prefix = args.prefix
    ref_dir = OUTPUTS / f"{prefix}ref"
    cont_main_dir = OUTPUTS / f"{prefix}continue_clr_weps300"
    plot_dir = OUTPUTS / "plots" / (prefix.rstrip("_") or "auc")
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not ref_dir.exists():
        raise SystemExit(f"reference dir not found: {ref_dir}")
    if not cont_main_dir.exists():
        raise SystemExit(f"canonical continue dir not found: {cont_main_dir}")

    summary = summary_table(prefix)
    print(f"\n=== Continue-run summary (prefix={prefix!r}) ===")
    print(summary.to_string(index=False))

    speedup = speedup_vs_target_curve(ref_dir, cont_main_dir)
    print(f"\n=== Speedup-vs-target curve ({cont_main_dir.name} vs {ref_dir.name}) ===")
    print(speedup.to_string(index=False))
    plot_speedup_vs_target(speedup, plot_dir / "speedup_vs_target.png")

    ceps_curve = ceps_vs_weps_table(summary)
    print("\n=== ceps-vs-weps (CLR continue runs) ===")
    print(ceps_curve.to_string(index=False))
    plot_ceps_vs_weps(ceps_curve, plot_dir / "ceps_vs_weps.png")

    h2h = head_to_head_table(summary)
    print("\n=== Day-5 head-to-head (weps=300, CLR vs fixed-LR continue) ===")
    print(h2h.to_string(index=False))

    print(f"\nPlots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
