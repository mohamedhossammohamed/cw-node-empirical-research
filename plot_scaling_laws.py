"""
Publication-quality scaling law plots for CW-Node vs Dense comparison.

Reads empirical_results.json and produces:
  1. scaling_law_frontier.png — parameter count vs final validation loss
  2. learning_curves_3M.png — training curves for the 3M tier models

Usage:
  .venv/bin/python dst_lab/research_paper_data/plot_scaling_laws.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Architecture display styling — colorblind-friendly, publication quality
ARCH_STYLE = {
    "dense": {
        "label": "Dense Baseline\n(100% External)",
        "color": "#E74C3C",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.8,
        "zorder": 3,
    },
    "cw_70_30": {
        "label": "CW-Node 70/30\n(Out-Heavy)",
        "color": "#2ECC71",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.2,
        "zorder": 4,
    },
    "cw_30_70": {
        "label": "CW-Node 30/70\n(In-Heavy)",
        "color": "#3498DB",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 2.2,
        "zorder": 4,
    },
}

plt.rcParams.update({
    "figure.figsize": (10, 7),
    "figure.dpi": 200,
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_results(path: str) -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)


def compute_entropy_floor(data_dir: str) -> float | None:
    """Estimate entropy floor from micro_val.bin using bz2 compression."""
    import bz2, numpy as np

    val_path = os.path.join(data_dir, "micro_val.bin")
    if not os.path.exists(val_path):
        return None
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    raw = data.tobytes()
    compressed = bz2.compress(raw)
    bpc = (len(compressed) * 8) / (len(data) * math.log(2))
    del data
    return bpc


def plot_scaling_law_frontier(results: List[dict], entropy_floor_bpc: float | None,
                               output_path: str):
    """Plot parameter count vs final validation loss (scaling law)."""
    fig, ax = plt.subplots()

    # Group by architecture, compute mean/std across tiers
    grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if "error" in r or "final_val_loss" not in r:
            continue
        grouped[r["arch"]][r["target_params"]].append(r["final_val_loss"])

    for arch, style in ARCH_STYLE.items():
        if arch not in grouped:
            continue
        tiers = sorted(grouped[arch].keys())
        params = np.array(tiers, dtype=float)
        means = np.array([np.mean(grouped[arch][t]) for t in tiers])

        ax.plot(params, means, linestyle=style["linestyle"],
                color=style["color"], linewidth=style["linewidth"],
                marker=style["marker"], markersize=10,
                markeredgewidth=1.2, markeredgecolor="white",
                label=style["label"], zorder=style["zorder"],
                alpha=0.9)

    # Entropy floor
    if entropy_floor_bpc:
        val_floor = entropy_floor_bpc * math.log(2)
        ax.axhline(y=val_floor, color="#7F8C8D", linestyle=":", linewidth=1.5,
                   alpha=0.7, label=f"Entropy Floor ({entropy_floor_bpc:.2f} BPC)")

    ax.set_xscale("log")
    ax.set_xlabel("Total Parameters")
    ax.set_ylabel("Final Validation Loss (Cross-Entropy)")
    ax.set_title("Micro Scaling Law: CW-Node vs Dense Architecture\n"
                 "(1 epoch on 5M tokens, batch_size=4, AdamW lr=3e-3)")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e3:.0f}K" if x < 1e6 else f"{x/1e6:.1f}M"))
    ax.grid(True, which="major", axis="both")
    ax.legend(framealpha=0.9, edgecolor="gray", fancybox=True, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


def plot_learning_curves(results: List[dict], tier: int = 3_000_000,
                          output_path: str = "learning_curves_3M.png"):
    """Plot training curves for all architectures at a given tier."""
    fig, ax = plt.subplots()

    filtered = [r for r in results if r.get("target_params") == tier
                and "history" in r and len(r["history"]) > 0]

    for arch, style in ARCH_STYLE.items():
        arch_results = [r for r in filtered if r["arch"] == arch]
        if not arch_results:
            continue
        r = arch_results[0]
        steps = [h["step"] for h in r["history"]]
        val_loss = [h["val_loss"] for h in r["history"]]

        ax.plot(steps, val_loss, color=style["color"],
                linewidth=style["linewidth"], linestyle=style["linestyle"],
                label=style["label"], alpha=0.9)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title(f"Learning Curves: {tier//1000}K Parameter Models\n"
                 f"(CW-Node vs Dense, 5M tokens, 1 epoch)")
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.9, edgecolor="gray", fancybox=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


def print_summary(results: List[dict]):
    """Print a formatted summary table."""
    grouped: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for r in results:
        if "error" in r:
            continue
        t = r["target_params"]
        arch = r["arch"]
        grouped[arch][t] = r

    tiers = sorted(set(r["target_params"] for r in results if "error" not in r))

    print(f"\n{'Arch':<12s}", end="")
    for t in tiers:
        print(f"  {t//1000:>4d}K", end="")
    print(f"\n{'-' * 80}")

    for arch in ["dense", "cw_70_30", "cw_30_70"]:
        if arch not in grouped:
            continue
        print(f"{arch:<12s}", end="")
        for t in tiers:
            if t in grouped[arch]:
                vl = grouped[arch][t].get("final_val_loss", 0)
                bpc = grouped[arch][t].get("final_val_bpc", 0)
                print(f"  {vl:>5.4f}", end="")
            else:
                print(f"  {'—':>5s}", end="")
        print()

    # Delta vs Dense
    print(f"\n--- Delta vs Dense (negative = improvement) ---")
    for arch in ["cw_70_30", "cw_30_70"]:
        if arch not in grouped:
            continue
        deltas = []
        for t in tiers:
            if t in grouped["dense"] and t in grouped[arch]:
                d = grouped[arch][t]["final_val_loss"] - grouped["dense"][t]["final_val_loss"]
                deltas.append((t, d))
        if deltas:
            parts = [f"{t//1000}K: {d:+.4f}" for t, d in deltas]
            print(f"  {arch}: {', '.join(parts)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None,
                        help="Path to empirical_results.json")
    parser.add_argument("--no-floor", action="store_true")
    args = parser.parse_args()

    data_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = args.results or os.path.join(data_dir, "empirical_results.json")

    if not os.path.exists(results_path):
        print(f"[ERROR] {results_path} not found. Run micro_sweep.py first.")
        sys.exit(1)

    results = load_results(results_path)
    if not results:
        print("[ERROR] No results loaded.")
        sys.exit(1)

    print_summary(results)

    # Entropy floor
    floor = None
    if not args.no_floor:
        try:
            floor = compute_entropy_floor(data_dir)
            if floor:
                print(f"\nEntropy floor (bz2): {floor:.3f} BPC ({floor * math.log(2):.4f} loss)")
        except Exception as e:
            print(f"[WARN] Entropy floor: {e}")

    # Plot scaling law
    plot_scaling_law_frontier(results, floor,
                               os.path.join(data_dir, "scaling_law_frontier.png"))

    # Plot 3M learning curves
    plot_learning_curves(results, 3_000_000,
                          os.path.join(data_dir, "learning_curves_3M.png"))

    print("\n[DONE] All plots generated.")


if __name__ == "__main__":
    main()
