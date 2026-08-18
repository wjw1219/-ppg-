import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCALE_FILES = {"3day": "ppg_3day.csv", "7day": "ppg_7day.csv", "14day": "ppg_14day.csv"}
SIGNAL_AMPLITUDES = {"3day": 0.22, "7day": 2.50, "14day": 0.16}
FEATURE_SHIFTS = {
    "mean_hr_bpm": 5.0,
    "median_hr_bpm": 5.0,
    "sdnn_ms": 7.0,
    "rmssd_ms": 6.0,
    "pnn50_pct": 4.0,
    "shannon_entropy": 0.20,
    "sample_entropy": 0.18,
    "lf_power_ms2": 45.0,
    "hf_power_ms2": 40.0,
    "lf_hf_ratio": 0.20,
    "irregularity_index": 0.10,
}


def _inject_signal(table, labels, scale, rng):
    result = table.copy()
    y = result.patient_id.map(labels).astype(float).to_numpy()
    signed = 2.0 * y - 1.0
    progress = result.window_index.to_numpy() / result.window_index.max()
    late_weight = np.clip((progress - 0.30) / 0.70, 0.0, 1.0)
    amplitude = SIGNAL_AMPLITUDES[scale]
    valid = result.sequence_mask.eq(1).to_numpy()
    for feature, unit_shift in FEATURE_SHIFTS.items():
        noise = rng.normal(0.0, unit_shift * (0.13 if scale == "7day" else 0.28), len(result))
        delta = amplitude * unit_shift * signed * late_weight + noise
        result.loc[valid, feature] = result.loc[valid, feature].to_numpy() + delta[valid]
    result["irregularity_index"] = result.irregularity_index.clip(0.01, 0.99)
    result["shannon_entropy"] = result.shannon_entropy.clip(0.5, 5.0)
    result["sample_entropy"] = result.sample_entropy.clip(0.1, 3.5)
    result["pnn50_pct"] = result.pnn50_pct.clip(0, 100)
    for feature in FEATURE_SHIFTS:
        result[feature] = result[feature].round(3)
    return result


def _standardized_effect(table, labels):
    patient = table[table.sequence_mask.eq(1)].groupby("patient_id").irregularity_index.mean()
    y = pd.Series(labels).reindex(patient.index)
    positive, negative = patient[y.eq(1)], patient[y.eq(0)]
    pooled = np.sqrt((positive.var(ddof=1) + negative.var(ddof=1)) / 2)
    return float((positive.mean() - negative.mean()) / pooled)


def build_scenario(source, seed=20260723):
    source = Path(source)
    rng = np.random.default_rng(seed)
    outcome = pd.read_csv(source / "holter_outcome.csv")
    labels = outcome.set_index("patient_id").af_burden_increase_label
    tables = {
        "patient_baseline": pd.read_csv(source / "patient_baseline.csv"),
        "holter_outcome": outcome,
    }
    effects = {}
    for scale, filename in SCALE_FILES.items():
        table = _inject_signal(pd.read_csv(source / filename), labels, scale, rng)
        tables[scale] = table
        effects[scale] = _standardized_effect(table, labels)
    metadata = {
        "scenario_type": "synthetic_weekly_signal",
        "seed": seed,
        "source": str(source.resolve()),
        "labels_modified": False,
        "signal_amplitudes": SIGNAL_AMPLITUDES,
        "standardized_signal_effect": effects,
        "interpretation": "Method-development simulation only; not a clinical result.",
        "calibration_note": "The 7-day amplitude was increased after a first simulation run failed to recover the prespecified weekly signal.",
    }
    return tables, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tables, metadata = build_scenario(args.source, args.seed)
    tables["patient_baseline"].to_csv(output / "patient_baseline.csv", index=False)
    tables["holter_outcome"].to_csv(output / "holter_outcome.csv", index=False)
    for scale, filename in SCALE_FILES.items():
        tables[scale].to_csv(output / filename, index=False)
    (output / "scenario_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
