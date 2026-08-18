from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLINICAL_GROUPS = {
    "demographic_lifestyle": ["age_years", "sex", "bmi_kg_m2", "smoking", "alcohol_use"],
    "af_history": ["af_type", "af_duration_years", "hypertension", "heart_failure", "diabetes", "stroke_tia_history", "previous_ablation_count"],
    "laboratory_echo_ecg": ["hs_crp_mg_l", "nt_probnp_pg_ml", "lad_mm", "lavi_ml_m2", "lvef_pct"],
    "procedure_medication": ["ablation_energy", "discharge_antiarrhythmic", "discharge_anticoagulant"],
}
PPG_FEATURES = ["mean_hr_bpm", "median_hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct", "shannon_entropy",
                "sample_entropy", "lf_power_ms2", "hf_power_ms2", "lf_hf_ratio", "irregularity_index", "adherence_rate", "validity_rate"]

def load_common(input_dir):
    root = Path(input_dir)
    baseline = pd.read_csv(root / "patient_baseline.csv").set_index("patient_id")
    outcome = pd.read_csv(root / "holter_outcome.csv").set_index("patient_id")
    ids = baseline.index.tolist()
    return baseline.loc[ids], outcome.loc[ids, "af_burden_increase_label"], outcome.loc[ids, "cv_fold"]

def load_scale(input_dir, filename, nodes):
    ppg = pd.read_csv(Path(input_dir) / filename).sort_values(["patient_id", "window_index"])
    counts = ppg.groupby("patient_id").size()
    if len(counts) != 387 or not (counts == nodes).all(): raise ValueError(f"Expected 387 patients x {nodes} nodes")
    if ppg.end_day.max() != 182 or ppg.start_day.min() != 1: raise ValueError("Scale does not cover days 1-182")
    return ppg

class GroupPreprocessor:
    def __init__(self, columns): self.columns = columns
    def fit(self, df):
        self.numeric = [c for c in self.columns if pd.api.types.is_numeric_dtype(df[c])]
        self.categorical = [c for c in self.columns if c not in self.numeric]
        self.num_imp = SimpleImputer(strategy="median"); self.scaler = StandardScaler()
        if self.numeric: self.scaler.fit(self.num_imp.fit_transform(df[self.numeric]))
        self.cat_imp = SimpleImputer(strategy="most_frequent"); self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if self.categorical: self.enc.fit(self.cat_imp.fit_transform(df[self.categorical]))
        return self
    def transform(self, df):
        blocks=[]
        if self.numeric: blocks.append(self.scaler.transform(self.num_imp.transform(df[self.numeric])))
        if self.categorical: blocks.append(self.enc.transform(self.cat_imp.transform(df[self.categorical])))
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
    def __init__(self, nodes): self.nodes = nodes
    def fit(self, baseline, ppg, train_ids):
        self.group_processors = [GroupPreprocessor(cols).fit(baseline.loc[train_ids]) for cols in CLINICAL_GROUPS.values()]
        train = ppg[ppg.patient_id.isin(train_ids)]; valid = train.sequence_mask.eq(1)
        self.ppg_medians = train.loc[valid, PPG_FEATURES].median().fillna(0)
        self.ppg_scaler = StandardScaler().fit(train.loc[valid, PPG_FEATURES].fillna(self.ppg_medians)); return self
    def transform(self, baseline, ppg, labels, ids):
        blocks, slices, start = [], [], 0
        for proc in self.group_processors:
            block = proc.transform(baseline.loc[ids]); blocks.append(block); slices.append((start, start + block.shape[1])); start += block.shape[1]
        selected = ppg[ppg.patient_id.isin(ids)].copy().sort_values(["patient_id", "window_index"])
        selected[PPG_FEATURES] = selected[PPG_FEATURES].fillna(self.ppg_medians)
        scaled = self.ppg_scaler.transform(selected[PPG_FEATURES]).astype(np.float32)
        mask = selected.sequence_mask.to_numpy(np.float32); scaled[mask == 0] = 0
        return ProcessedData(list(ids), np.concatenate(blocks, axis=1), slices,
                             scaled.reshape(len(ids), self.nodes, len(PPG_FEATURES)), mask.reshape(len(ids), self.nodes),
                             labels.loc[ids].to_numpy(np.float32))
