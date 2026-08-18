import json
import os
import argparse
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.evaluate import paired_difference, stratified_bootstrap_indices, summarize
from src.train import CONDITIONS


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Compute Experiment 1 metrics and render figures")
parser.add_argument("--predictions", type=Path, default=ROOT / "results" / "oof_predictions.csv")
parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
parser.add_argument("--illustrative", action="store_true", help="mark figures as illustrative simulation")
args = parser.parse_args()
RESULTS, FIGURES = args.results_dir, args.figures_dir
RESULTS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True); (ROOT / ".mplconfig").mkdir(exist_ok=True)
cfg = json.loads((ROOT / "config.json").read_text())
pred = pd.read_csv(args.predictions)
expected = 387 * len(CONDITIONS)
if len(pred) != expected or pred.groupby(["patient_id", "condition"]).size().max() != 1:
    raise ValueError("OOF prediction cardinality is invalid")
base = pred[pred.condition == "fusion_full"].sort_values("patient_id")
y = base.label.to_numpy()
indices = stratified_bootstrap_indices(y, cfg["bootstrap_resamples"], cfg["bootstrap_seed"])
summary_rows = []
for condition in CONDITIONS:
    frame = pred[pred.condition == condition].sort_values("patient_id")
    values = summarize(y, frame.probability.to_numpy(), frame.threshold.to_numpy(), indices)
    for metric, (estimate, lower, upper) in values.items():
        summary_rows.append({"condition": condition, "metric": metric, "estimate": estimate, "ci_lower": lower, "ci_upper": upper})
summary = pd.DataFrame(summary_rows)
summary.to_csv(RESULTS / "summary_metrics.csv", index=False)

paired_rows = []
full_prob = base.probability.to_numpy()
for condition in CONDITIONS:
    if condition == "fusion_full": continue
    other = pred[pred.condition == condition].sort_values("patient_id").probability.to_numpy()
    values = paired_difference(y, full_prob, other, indices)
    for metric, (estimate, lower, upper) in values.items():
        paired_rows.append({"comparison": f"fusion_full - {condition}", "metric": metric,
                            "estimate": estimate, "ci_lower": lower, "ci_upper": upper})
pd.DataFrame(paired_rows).to_csv(RESULTS / "paired_differences.csv", index=False)

cohort = pd.DataFrame([
    {"item": "Patients", "value": 387}, {"item": "AF burden increased", "value": int(y.sum())},
    {"item": "AF burden not increased", "value": int((1-y).sum())},
    {"item": "Positive rate", "value": float(y.mean())}, {"item": "Seven-day windows", "value": 387 * 26},
])
cohort.to_csv(RESULTS / "cohort_summary.csv", index=False)

labels = {"clinical_only": "Clinical prior only", "ppg_only": "Longitudinal PPG only", "fusion_full": "Clinical prior + PPG"}
plt.figure(figsize=(6.4, 5.2))
for condition, label in labels.items():
    frame = pred[pred.condition == condition].sort_values("patient_id")
    fpr, tpr, _ = roc_curve(y, frame.probability)
    auc = summary[(summary.condition == condition) & (summary.metric == "roc_auc")].estimate.iloc[0]
    plt.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={auc:.3f})")
plt.plot([0,1], [0,1], "--", color="#777777", linewidth=1)
plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity"); plt.legend(frameon=False, loc="lower right")
if args.illustrative: plt.title("Illustrative simulated results")
plt.tight_layout(); plt.savefig(FIGURES / "roc_curves.png", dpi=220); plt.close()

ablation_conditions = ["fusion_full", "fusion_drop_demographic_lifestyle", "fusion_drop_af_history",
                       "fusion_drop_laboratory_echo_ecg", "fusion_drop_procedure_medication"]
ablation_labels = ["Full fusion", "Without demographic/lifestyle", "Without AF/history",
                   "Without laboratory/echo/ECG", "Without procedure/medication"]
auc_rows = summary[(summary.metric == "roc_auc")].set_index("condition").loc[ablation_conditions]
plt.figure(figsize=(7.2, 4.6))
x = np.arange(len(ablation_conditions)); estimates = auc_rows.estimate.to_numpy()
err = np.vstack([estimates-auc_rows.ci_lower.to_numpy(), auc_rows.ci_upper.to_numpy()-estimates])
plt.bar(x, estimates, color=["#1F4E78", "#5B9BD5", "#70AD47", "#ED7D31", "#A5A5A5"])
plt.errorbar(x, estimates, yerr=err, fmt="none", ecolor="#222222", capsize=4)
plt.xticks(x, ablation_labels, rotation=20, ha="right"); plt.ylabel("ROC-AUC"); plt.ylim(max(0.45, estimates.min()-.08), min(1.0, estimates.max()+.08))
if args.illustrative: plt.title("Illustrative simulated results")
plt.tight_layout(); plt.savefig(FIGURES / "ablation_performance.png", dpi=220); plt.close()

checks = {"prediction_rows_2709": len(pred) == 2709, "one_prediction_per_patient_condition": pred.groupby(["patient_id","condition"]).size().eq(1).all(),
          "probabilities_bounded": pred.probability.between(0,1).all(), "seven_conditions": pred.condition.nunique() == 7,
          "labels_consistent": pred.groupby("patient_id").label.nunique().eq(1).all()}
(RESULTS / "quality_checks.json").write_text(json.dumps({k: bool(v) for k,v in checks.items()}, indent=2))
if not all(checks.values()): raise SystemExit(1)
print(summary[summary.metric.isin(["roc_auc", "auprc"])].to_string(index=False))
