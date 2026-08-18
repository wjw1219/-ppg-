from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CLINICAL_GROUPS = {
    "demographic_lifestyle": ["age_years", "sex", "bmi_kg_m2", "smoking", "alcohol_use"],
    "af_history": ["af_type", "af_duration_years", "hypertension", "heart_failure", "diabetes",
                   "stroke_tia_history", "previous_ablation_count"],
    "laboratory_echo_ecg": ["hs_crp_mg_l", "nt_probnp_pg_ml", "lad_mm", "lavi_ml_m2", "lvef_pct"],
    "procedure_medication": ["ablation_energy", "discharge_antiarrhythmic", "discharge_anticoagulant"],
}
PPG_FEATURES = ["mean_hr_bpm", "median_hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct",
                "shannon_entropy", "sample_entropy", "lf_power_ms2", "hf_power_ms2",
                "lf_hf_ratio", "irregularity_index", "adherence_rate", "validity_rate"]
OUTCOME_ONLY = {"discharge_af_burden_pct", "month6_af_burden_pct", "af_burden_change_pct",
                "af_burden_increase_label"}


def load_raw(input_dir):
    root = Path(input_dir)
    baseline = pd.read_csv(root / "patient_baseline.csv")
    outcome = pd.read_csv(root / "holter_outcome.csv")
    ppg = pd.read_csv(root / "ppg_7day.csv")
    if baseline.patient_id.nunique() != 387 or len(outcome) != 387:
        raise ValueError("Expected 387 unique patients and 387 outcomes")
    required = {c for cols in CLINICAL_GROUPS.values() for c in cols}
    missing = required - set(baseline.columns)
    if missing:
        raise ValueError(f"Missing clinical columns: {sorted(missing)}")
    counts = ppg.groupby("patient_id").size()
    if not (counts == 26).all():
        raise ValueError("Every patient must have exactly 26 seven-day windows")
    ppg = ppg.sort_values(["patient_id", "window_index"])
    labels = outcome.set_index("patient_id")["af_burden_increase_label"]
    folds = outcome.set_index("patient_id")["cv_fold"]
    ids = baseline.patient_id.tolist()
    return baseline.set_index("patient_id").loc[ids], ppg, labels.loc[ids], folds.loc[ids]


class GroupPreprocessor:
    def __init__(self, columns):
        self.columns = columns
        self.numeric = []
        self.categorical = []

    def fit(self, df):
        self.numeric = [c for c in self.columns if pd.api.types.is_numeric_dtype(df[c])]
        self.categorical = [c for c in self.columns if c not in self.numeric]
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        if self.numeric:
            x = self.num_imputer.fit_transform(df[self.numeric])
            self.scaler.fit(x)
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if self.categorical:
            x = self.cat_imputer.fit_transform(df[self.categorical])
            self.encoder.fit(x)
        return self

    def transform(self, df):
        blocks = []
        if self.numeric:
            blocks.append(self.scaler.transform(self.num_imputer.transform(df[self.numeric])))
        if self.categorical:
            blocks.append(self.encoder.transform(self.cat_imputer.transform(df[self.categorical])))
        return np.concatenate(blocks, axis=1).astype(np.float32)


@dataclass
class ProcessedData:
    patient_ids: list
    clinical: np.ndarray
    group_slices: list
    ppg: np.ndarray
    mask: np.ndarray
    labels: np.ndarray


class FoldPreprocessor:
    def __init__(self, retained_groups):
        self.retained_groups = retained_groups

    def fit(self, baseline, ppg, train_ids):
        train = baseline.loc[train_ids]
        self.group_processors = []
        for group in self.retained_groups:
            self.group_processors.append(GroupPreprocessor(CLINICAL_GROUPS[group]).fit(train))
        train_ppg = ppg[ppg.patient_id.isin(train_ids)]
        valid = train_ppg.sequence_mask.eq(1)
        self.ppg_medians = train_ppg.loc[valid, PPG_FEATURES].median().fillna(0.0)
        filled = train_ppg.loc[valid, PPG_FEATURES].fillna(self.ppg_medians)
        self.ppg_scaler = StandardScaler().fit(filled)
        return self

    def transform(self, baseline, ppg, labels, ids):
        parts, slices, start = [], [], 0
        for processor in self.group_processors:
            block = processor.transform(baseline.loc[ids])
            parts.append(block)
            slices.append((start, start + block.shape[1]))
            start += block.shape[1]
        clinical = np.concatenate(parts, axis=1).astype(np.float32) if parts else np.zeros((len(ids), 0), np.float32)
        selected = ppg[ppg.patient_id.isin(ids)].copy().sort_values(["patient_id", "window_index"])
        selected[PPG_FEATURES] = selected[PPG_FEATURES].fillna(self.ppg_medians)
        scaled = self.ppg_scaler.transform(selected[PPG_FEATURES]).astype(np.float32)
        scaled[selected.sequence_mask.to_numpy() == 0] = 0.0
        ppg_array = scaled.reshape(len(ids), 26, len(PPG_FEATURES))
        mask = selected.sequence_mask.to_numpy(np.float32).reshape(len(ids), 26)
        y = labels.loc[ids].to_numpy(np.float32)
        return ProcessedData(list(ids), clinical, slices, ppg_array, mask, y)
