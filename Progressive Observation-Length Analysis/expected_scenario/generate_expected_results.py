from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(ROOT.parent))
from src.evaluate import stratified_bootstrap_indices, summarize
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score

OUT, FIG = ROOT / "results", ROOT / "figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); (ROOT / ".mplconfig").mkdir(exist_ok=True)
rng = np.random.default_rng(20260723)
n, positive = 387, 200
y = np.r_[np.ones(positive, dtype=int), np.zeros(n-positive, dtype=int)]
rng.shuffle(y)
signal = np.where(y == 1, 1.0, -1.0)

def probability(strength):
    z = strength * signal + rng.normal(0, 0.7, n)
    return 1 / (1 + np.exp(-z))

def calibrated_probability(target_auc, target_sensitivity, target_specificity, target_brier, seed):
    noise = np.random.default_rng(seed).normal(0, 1, n)
    low, high = 0.0, 3.5
    for _ in range(35):
        distance = (low + high) / 2
        if roc_auc_score(y, distance * signal + noise) < target_auc: low = distance
        else: high = distance
    score = ((low + high) / 2) * signal + noise
    best_bias, best_error = 0.0, float("inf")
    for bias in np.linspace(-1.2, 1.2, 2401):
        pred = score + bias >= 0
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        error = (tp/(tp+fn)-target_sensitivity)**2 + (tn/(tn+fp)-target_specificity)**2
        if error < best_error: best_bias, best_error = bias, error
    shifted = score + best_bias
    best_probability, best_error = None, float("inf")
    for temperature in np.linspace(0.25, 4.0, 1200):
        probs = 1 / (1 + np.exp(-shifted / temperature))
        error = abs(brier_score_loss(y, probs) - target_brier)
        if error < best_error: best_probability, best_error = probs, error
    return best_probability

calibrated_full = np.empty(n); calibration_rng = np.random.default_rng(2295)
positive_ids, negative_ids = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
calibrated_full[positive_ids[:188]] = np.maximum(calibration_rng.beta(12, 3, 188), .501)
calibrated_full[positive_ids[188:]] = np.minimum(calibration_rng.beta(5, 6, 12), .499)
calibrated_full[negative_ids[:176]] = np.minimum(calibration_rng.beta(3, 12, 176), .499)
calibrated_full[negative_ids[176:]] = np.maximum(calibration_rng.beta(6, 5, 11), .501)

weekly = []
auc_targets = [0.653,0.701,0.726,0.717,0.787,0.794,0.796,0.849,0.804,0.870,0.888,0.870,0.927,0.935,0.910,0.955,0.948,0.961,0.972,0.951,0.964,0.975,0.978,0.958,0.977]
sensitivity_targets = [0.590,0.660,0.615,0.685,0.675,0.685,0.700,0.755,0.730,0.745,0.855,0.790,0.820,0.875,0.845,0.890,0.875,0.900,0.900,0.880,0.915,0.915,0.920,0.890,0.925]
specificity_targets = [0.620,0.615,0.717,0.668,0.743,0.749,0.711,0.807,0.727,0.786,0.813,0.738,0.872,0.845,0.845,0.910,0.900,0.895,0.925,0.890,0.895,0.920,0.925,0.900,0.925]
brier_targets = [0.233,0.219,0.213,0.214,0.196,0.191,0.190,0.173,0.188,0.163,0.158,0.163,0.140,0.136,0.145,0.105,0.112,0.098,0.087,0.108,0.097,0.085,0.081,0.102,0.078]
for week in range(1, 27):
    if week == 26:
        weekly_probabilities = calibrated_full
    else:
        i = week - 1
        weekly_probabilities = calibrated_probability(auc_targets[i], sensitivity_targets[i], specificity_targets[i], brier_targets[i], 204000 + week)
    for i, p in enumerate(weekly_probabilities):
        weekly.append({"patient_id": i, "weeks": week, "label": int(y[i]), "probability": float(p)})

model_strength = {"mlp": 0.34, "gru": 0.52, "lstm": 0.56, "transformer": 0.50, "temporal_gat": 0.58, "full_model": 0.76}
models = []
for model, strength in model_strength.items():
    if model == "full_model":
        probs = calibrated_full
    else:
        probs = probability(strength)
    for i, p in enumerate(probs):
        models.append({"patient_id": i, "model": model, "label": int(y[i]), "probability": float(p)})

def bootstrap(labels, probs, reps=2000):
    thresholds = np.full(len(labels), .5)
    indices = stratified_bootstrap_indices(labels, reps, 20260724)
    values = summarize(labels, probs, thresholds, indices)
    return [{"metric": k, "estimate": v[0], "ci_lower": v[1], "ci_upper": v[2]} for k, v in values.items()]

weekly_path, model_path = OUT / "expected_weekly_predictions.csv", OUT / "expected_model_predictions.csv"
weekly_df, model_df = pd.DataFrame(weekly), pd.DataFrame(models)
weekly_df.to_csv(weekly_path, index=False); model_df.to_csv(model_path, index=False)
weekly_y = weekly_df[weekly_df.weeks == 1].sort_values("patient_id").label.to_numpy()
weekly_metric_path, model_metric_path = OUT / "expected_weekly_metrics.csv", OUT / "expected_model_metrics.csv"
existing_m = pd.read_csv(model_metric_path) if model_metric_path.exists() else pd.DataFrame()
existing_w = pd.read_csv(weekly_metric_path) if weekly_metric_path.exists() else pd.DataFrame()
model_refresh = len(sys.argv) > 1 and sys.argv[1] == "models"
requested = [] if model_refresh else ([int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(1, 27)))
weeks_to_compute = requested
models_to_compute = list(model_strength) if model_refresh or existing_m.empty else []
wr = existing_w[~existing_w.weeks.isin(requested)].to_dict("records") if not existing_w.empty else []
for week in weeks_to_compute:
    for row in bootstrap(weekly_y, weekly_df[weekly_df.weeks == week].sort_values("patient_id").probability.to_numpy()): row.update(weeks=week); wr.append(row)
    pd.DataFrame(wr).to_csv(weekly_metric_path, index=False)
mr = [] if model_refresh else (existing_m.to_dict("records") if not existing_m.empty else [])
for model in models_to_compute:
    for row in bootstrap(weekly_y, model_df[model_df.model == model].sort_values("patient_id").probability.to_numpy()): row.update(model=model); mr.append(row)
w, m = pd.DataFrame(wr), pd.DataFrame(mr)
w.to_csv(weekly_metric_path, index=False)
if models_to_compute: m.to_csv(model_metric_path, index=False)

fig, ax = plt.subplots(figsize=(8, 4.8))
for metric, color, label in [("roc_auc", "#1F4E78", "ROC-AUC"), ("auprc", "#C55A11", "AUPRC")]:
    d = w[w.metric == metric].sort_values("weeks"); ax.plot(d.weeks, d.estimate, color=color, marker="o", ms=3, label=label); ax.fill_between(d.weeks, d.ci_lower, d.ci_upper, color=color, alpha=.12)
ax.axvline(16, color="#9B1C1C", ls="--", label="Expected plateau onset: week 16"); ax.set(xlabel="Observed weeks", ylabel="Performance"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig(FIG / "expected_weekly_performance.png", dpi=240); plt.close(fig)
order = list(model_strength); fig, ax = plt.subplots(figsize=(8, 4.8)); x = np.arange(len(order)); width = .36
for shift, metric, color, label in [(-width/2, "roc_auc", "#1F4E78", "ROC-AUC"), (width/2, "auprc", "#C55A11", "AUPRC")]:
    d = m[m.metric == metric].set_index("model").loc[order]; ax.bar(x + shift, d.estimate, width, color=color, label=label); ax.errorbar(x + shift, d.estimate, yerr=[d.estimate-d.ci_lower, d.ci_upper-d.estimate], fmt="none", ecolor="#222", capsize=3)
ax.set_xticks(x, ["MLP", "GRU", "LSTM", "Transformer", "Temporal GAT", "Full model"]); ax.set_ylabel("Performance"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig(FIG / "expected_model_comparison.png", dpi=240); plt.close(fig)
(OUT / "README.txt").write_text("Expected-result simulation only; not generated by model training and must not be reported as empirical evidence.\n")
print(f"generated weekly={len(weekly_df)} models={len(model_df)}")
