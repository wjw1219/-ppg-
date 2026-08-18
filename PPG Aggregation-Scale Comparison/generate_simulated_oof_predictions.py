import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = {
    "3day": {"tp": 176, "tn": 159, "auc": 0.945, "auprc": 0.952, "brier": 0.115},
    "7day": {"tp": 188, "tn": 176, "auc": 0.985, "auprc": 0.987, "brier": 0.069},
    "14day": {"tp": 183, "tn": 168, "auc": 0.970, "auprc": 0.975, "brier": 0.085},
}

PARAMETERS = {
    "3day": [-0.92250179, 2.57179666, 0.30173284, 2.06590684, -3.04300498, 2.54004867, -3.38763381, 0.18289640],
    "7day": [1.96984071, 1.16910445, 2.95930855, 1.52693481, 0.03059444, 2.59648002, -1.70686777, 1.93825429],
    "14day": [2.64477875, 1.90002192, 2.88382731, 0.77090881, -0.07108917, 2.98124924, -1.71502637, 2.33043192],
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _scale_probability(labels, scale, seed):
    target, params = TARGETS[scale], PARAMETERS[scale]
    rng = np.random.default_rng(seed)
    positive = rng.permutation(np.flatnonzero(labels == 1))
    negative = rng.permutation(np.flatnonzero(labels == 0))
    tp, fn = positive[:target["tp"]], positive[target["tp"]:]
    tn, fp = negative[:target["tn"]], negative[target["tn"]:]
    probability = np.empty(len(labels), dtype=float)
    groups = [(tp, True, params[0], params[1]), (fn, False, params[2], params[3]),
              (tn, False, params[4], params[5]), (fp, True, params[6], params[7])]
    for indices, above_threshold, center, spread in groups:
        unit = sigmoid(center + spread * rng.normal(size=len(indices)))
        probability[indices] = 0.5 + 0.49 * unit if above_threshold else 0.5 * unit
    return probability


def generate_predictions(outcome, seed=20260724):
    required = {"patient_id", "af_burden_increase_label"}
    if not required.issubset(outcome.columns):
        raise ValueError(f"Outcome must contain {sorted(required)}")
    patient_ids = outcome.patient_id.astype(str).to_numpy()
    labels = outcome.af_burden_increase_label.astype(int).to_numpy()
    if int(labels.sum()) != 200 or int((labels == 0).sum()) != 187:
        raise ValueError("Calibrated simulation requires 200 positive and 187 negative labels")
    rows = []
    for index, scale in enumerate(("3day", "7day", "14day")):
        probability = _scale_probability(labels, scale, seed + index)
        for patient_id, label, value in zip(patient_ids, labels, probability):
            rows.append({"patient_id": patient_id, "scale": scale, "label": int(label),
                         "probability": float(value), "threshold": 0.5})
    metadata = {
        "simulated_predictions": True,
        "model_trained": False,
        "seed": seed,
        "target_metrics": TARGETS,
        "threshold": 0.5,
        "purpose": "Method and figure demonstration only; not empirical model output.",
    }
    return pd.DataFrame(rows), metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True); parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    predictions, metadata = generate_predictions(pd.read_csv(args.outcome), args.seed)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); predictions.to_csv(output, index=False)
    Path(args.metadata).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
