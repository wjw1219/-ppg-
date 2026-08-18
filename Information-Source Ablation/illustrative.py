from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_ORDER = [
    "fusion_full",
    "ppg_only",
    "fusion_drop_demographic_lifestyle",
    "fusion_drop_procedure_medication",
    "fusion_drop_laboratory_echo_ecg",
    "fusion_drop_af_history",
    "clinical_only",
]
CONDITIONS = tuple(EXPECTED_ORDER)
SIGNAL_STRENGTH = dict(zip(EXPECTED_ORDER, [1.55, 1.30, 1.18, 1.08, 0.98, 0.86, 0.62]))


def output_paths(root):
    root = Path(root)
    return {name: root / name for name in ("results", "figures", "report")}


def generate_predictions(labels, seed=20260723):
    required = {"patient_id", "label"}
    if not required.issubset(labels.columns):
        raise ValueError(f"labels must contain {sorted(required)}")
    if labels.patient_id.duplicated().any() or not labels.label.isin([0, 1]).all():
        raise ValueError("patient IDs must be unique and labels must be binary")

    rng = np.random.default_rng(seed)
    y = labels.label.to_numpy(dtype=int)
    direction = 2.0 * y - 1.0
    patient_difficulty = rng.normal(0.0, 1.0, len(labels))
    prevalence = np.clip(y.mean(), 1e-4, 1 - 1e-4)
    intercept = np.log(prevalence / (1 - prevalence))
    rows = []
    for condition in EXPECTED_ORDER:
        logits = intercept + SIGNAL_STRENGTH[condition] * direction + patient_difficulty
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        for patient_id, label, probability in zip(labels.patient_id, y, probabilities):
            rows.append({"patient_id": patient_id, "condition": condition, "label": int(label),
                         "probability": float(probability), "threshold": 0.5})
    return pd.DataFrame(rows)
