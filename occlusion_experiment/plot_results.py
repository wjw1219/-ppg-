"""Plot real single-week occlusion results as editable PDF/SVG figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    results = pd.read_csv(args.results)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline = results.loc[results["masked_week"] == 0].iloc[0]
    masked = results.loc[results["masked_week"] > 0].sort_values("masked_week")
    weeks = masked["masked_week"].to_numpy()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(7.2, 3.25), constrained_layout=True)
    for key, color, label in [("roc_auc", "#1f77b4", "ROC-AUC"), ("auprc", "#d95f02", "AUPRC"), ("f1", "#2ca25f", "F1-score")]:
        ax.plot(weeks, masked[key], marker="o", ms=3.2, lw=1.7, color=color, label=label)
        ax.axhline(float(baseline[key]), color=color, ls="--", lw=0.9, alpha=0.65)
    ax.set_xlim(0.5, 26.5)
    ax.set_ylim(max(0.0, float(masked[["roc_auc", "auprc", "f1"]].min().min()) - 0.03), 1.0)
    ax.set_xlabel("Masked weekly node (week)")
    ax.set_ylabel("Discrimination metric")
    ax.grid(axis="y", color="#d9d9d9", lw=0.5, alpha=0.7)
    ax.legend(ncol=3, frameon=False, loc="lower left")
    fig.savefig(out / "single_week_occlusion_performance.pdf", facecolor="white")
    fig.savefig(out / "single_week_occlusion_performance.svg", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.25), constrained_layout=True)
    for key, color, label in [("roc_auc", "#1f77b4", r"$\Delta$ROC-AUC"), ("auprc", "#d95f02", r"$\Delta$AUPRC"), ("f1", "#2ca25f", r"$\Delta$F1"), ("brier", "#7b3294", r"$\Delta$Brier")]:
        delta = (float(baseline[key]) - masked[key]) if key != "brier" else (masked[key] - float(baseline[key]))
        ax.plot(weeks, delta, marker="o", ms=3.0, lw=1.6, color=color, label=label)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xlim(0.5, 26.5)
    ax.set_xlabel("Masked weekly node (week)")
    ax.set_ylabel("Change from full-input reference")
    ax.grid(axis="y", color="#d9d9d9", lw=0.5, alpha=0.7)
    ax.legend(ncol=4, frameon=False, loc="upper left")
    fig.savefig(out / "single_week_occlusion_degradation.pdf", facecolor="white")
    fig.savefig(out / "single_week_occlusion_degradation.svg", facecolor="white")
    plt.close(fig)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
