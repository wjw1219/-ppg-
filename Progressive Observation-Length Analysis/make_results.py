import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import paired_difference, stratified_bootstrap_indices, summarize

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
(ROOT / ".mplconfig").mkdir(exist_ok=True)
cfg = json.loads((ROOT / "config.json").read_text())
weekly = pd.read_csv(RES / "weekly_oof_predictions.csv")
models = pd.read_csv(RES / "model_oof_predictions.csv")

if len(weekly) != 26 * 387 or weekly.groupby(["weeks", "patient_id"]).size().ne(1).any():
    raise ValueError("Invalid weekly OOF cardinality")
if len(models) != 6 * 387 or models.groupby(["model", "patient_id"]).size().ne(1).any():
    raise ValueError("Invalid model OOF cardinality")

base = weekly[weekly.weeks == 26].sort_values("patient_id")
y = base.label.to_numpy()
indices = stratified_bootstrap_indices(y, cfg["bootstrap_resamples"], cfg["bootstrap_seed"])

def summary_table(frame, condition_column, values):
    rows = []
    for value in values:
        part = frame[frame[condition_column] == value].sort_values("patient_id")
        result = summarize(y, part.probability.to_numpy(), part.threshold.to_numpy(), indices)
        for metric, (estimate, low, high) in result.items():
            rows.append({condition_column: value, "metric": metric, "estimate": estimate,
                         "ci_lower": low, "ci_upper": high})
    return pd.DataFrame(rows)

weekly_summary = summary_table(weekly, "weeks", list(range(1, 27)))
model_order = ["mlp", "gru", "lstm", "transformer", "temporal_gat", "full_model"]
model_summary = summary_table(models, "model", model_order)
weekly_summary.to_csv(RES / "weekly_summary_metrics.csv", index=False)
model_summary.to_csv(RES / "model_summary_metrics.csv", index=False)

weekly_pairs = []
p26 = base.probability.to_numpy()
for week in range(1, 26):
    other = weekly[weekly.weeks == week].sort_values("patient_id").probability.to_numpy()
    for metric, (estimate, low, high) in paired_difference(y, p26, other, indices).items():
        weekly_pairs.append({"comparison": f"26 weeks - {week} weeks", "weeks": week,
                             "metric": metric, "estimate": estimate,
                             "ci_lower": low, "ci_upper": high})
pd.DataFrame(weekly_pairs).to_csv(RES / "weekly_paired_differences.csv", index=False)

model_pairs = []
for model in model_order[:-1]:
    other = models[models.model == model].sort_values("patient_id").probability.to_numpy()
    for metric, (estimate, low, high) in paired_difference(y, p26, other, indices).items():
        model_pairs.append({"comparison": f"full_model - {model}", "model": model,
                            "metric": metric, "estimate": estimate,
                            "ci_lower": low, "ci_upper": high})
pd.DataFrame(model_pairs).to_csv(RES / "model_paired_differences.csv", index=False)

colors = {"roc_auc": "#1F4E78", "auprc": "#C55A11"}
fig, ax = plt.subplots(figsize=(8.0, 4.8))
for metric, label in [("roc_auc", "ROC-AUC"), ("auprc", "AUPRC")]:
    d = weekly_summary[weekly_summary.metric == metric].sort_values("weeks")
    ax.plot(d.weeks, d.estimate, marker="o", ms=3, lw=1.8, color=colors[metric], label=label)
    ax.fill_between(d.weeks, d.ci_lower, d.ci_upper, color=colors[metric], alpha=.13)
ax.set(xlabel="Observed weeks", ylabel="Performance", xticks=range(1, 27))
ax.grid(axis="y", color="#D9D9D9", lw=.6); ax.legend(frameon=False, ncol=2)
fig.tight_layout(); fig.savefig(FIG / "weekly_performance_curve.png", dpi=240); plt.close(fig)

fig, ax = plt.subplots(figsize=(8.0, 4.8))
x = np.arange(len(model_order)); width = .36
for shift, metric, label in [(-width/2, "roc_auc", "ROC-AUC"), (width/2, "auprc", "AUPRC")]:
    d = model_summary[model_summary.metric == metric].set_index("model").loc[model_order]
    ax.bar(x + shift, d.estimate, width, color=colors[metric], label=label)
    ax.errorbar(x + shift, d.estimate,
                yerr=np.vstack([d.estimate-d.ci_lower, d.ci_upper-d.estimate]),
                fmt="none", ecolor="#222222", capsize=3, lw=.8)
ax.set_xticks(x, ["MLP", "GRU", "LSTM", "Transformer", "Temporal\nGAT", "Full\nmodel"])
ax.set_ylabel("Performance"); ax.grid(axis="y", color="#D9D9D9", lw=.6); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "model_comparison.png", dpi=240); plt.close(fig)

checks = {
    "weekly_rows": len(weekly) == 10062,
    "model_rows": len(models) == 2322,
    "fold_logs": len(pd.read_csv(RES / "fold_metrics.csv")) == 465,
    "probabilities_bounded": weekly.probability.between(0, 1).all() and models.probability.between(0, 1).all(),
    "labels_consistent": weekly.groupby("patient_id").label.nunique().eq(1).all(),
}
(RES / "quality_checks.json").write_text(json.dumps({k: bool(v) for k, v in checks.items()}, indent=2))
if not all(checks.values()):
    raise SystemExit("Quality checks failed")
print(model_summary[model_summary.metric.isin(["roc_auc", "auprc"])].to_string(index=False))
