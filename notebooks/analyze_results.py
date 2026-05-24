"""Analysis of the drp-warm-start continue-phase experiment.

Reads ``outputs/ref/`` and ``outputs/continue_*/`` directories and produces:

1. A summary table of all continue runs (weps, ceps, schedule, convergence).
2. The speedup-vs-target curve from a single continue run vs the reference.
3. The ceps-vs-weps curve across CLR continue runs.
4. PNG plots under ``outputs/plots/``.

Run from the repo root:

    python notebooks/analyze_results.py

The expected directory layout:

    outputs/ref/history.csv
    outputs/continue_clr_weps50/{history.csv,result.json}
    outputs/continue_clr_weps150/{history.csv,result.json}
    outputs/continue_clr_weps300/{history.csv,result.json}
    outputs/continue_fixed5e4_weps300/{history.csv,result.json}
    outputs/continue_fixed1e4_weps300/{history.csv,result.json}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
PLOTS = OUTPUTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


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
def summary_table() -> pd.DataFrame:
    rows: list[dict] = []
    for d in sorted(OUTPUTS.glob("continue_*")):
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
    targets = [0.10, 0.08, 0.07, 0.06, 0.055, 0.053, 0.052]
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
    clr = summary[summary["run"].str.startswith("continue_clr_weps")].copy()
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
    rows = summary[summary["run"].str.contains("_weps300$", regex=True)].copy()

    def schedule(name: str) -> str:
        if "clr" in name:
            return "CLR (1e-4 → 1e-3)"
        if "fixed5e4" in name:
            return "Fixed LR = 5e-4"
        if "fixed1e4" in name:
            return "Fixed LR = 1e-4"
        return name

    rows["schedule"] = rows["run"].map(schedule)
    return rows[["schedule", "ceps", "converged", "final_val_mae", "final_val_r2"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    summary = summary_table()
    print("\n=== Continue-run summary ===")
    print(summary.to_string(index=False))

    speedup = speedup_vs_target_curve(OUTPUTS / "ref", OUTPUTS / "continue_clr_weps300")
    print("\n=== Speedup-vs-target curve (continue_clr_weps300 vs ref) ===")
    print(speedup.to_string(index=False))
    plot_speedup_vs_target(speedup, PLOTS / "speedup_vs_target.png")

    ceps_curve = ceps_vs_weps_table(summary)
    print("\n=== ceps-vs-weps (CLR continue runs) ===")
    print(ceps_curve.to_string(index=False))
    plot_ceps_vs_weps(ceps_curve, PLOTS / "ceps_vs_weps.png")

    h2h = head_to_head_table(summary)
    print("\n=== Day-5 head-to-head (weps=300, CLR vs fixed-LR continue) ===")
    print(h2h.to_string(index=False))

    print(f"\nPlots saved to {PLOTS}/")


if __name__ == "__main__":
    main()
