from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLINICAL_GROUPS={"demographic_lifestyle":["age_years","sex","bmi_kg_m2","smoking","alcohol_use"],"af_history":["af_type","af_duration_years","hypertension","heart_failure","diabetes","stroke_tia_history","previous_ablation_count"],"laboratory_echo_ecg":["hs_crp_mg_l","nt_probnp_pg_ml","lad_mm","lavi_ml_m2","lvef_pct"],"procedure_medication":["ablation_energy","discharge_antiarrhythmic","discharge_anticoagulant"]}
PPG_FEATURES=["mean_hr_bpm","median_hr_bpm","sdnn_ms","rmssd_ms","pnn50_pct","shannon_entropy","sample_entropy","lf_power_ms2","hf_power_ms2","lf_hf_ratio","irregularity_index","adherence_rate","validity_rate"]

def load_common(root):
    b=pd.read_csv(Path(root)/"patient_baseline.csv").set_index("patient_id"); o=pd.read_csv(Path(root)/"holter_outcome.csv").set_index("patient_id"); ids=b.index.tolist(); return b.loc[ids],o.loc[ids,"af_burden_increase_label"],o.loc[ids,"cv_fold"]
def load_ppg(root):
    p=pd.read_csv(Path(root)/"ppg_7day.csv").sort_values(["patient_id","window_index"]); assert len(p)==10062 and (p.groupby("patient_id").size()==26).all(); return p

class GroupPreprocessor:
    def __init__(self,cols): self.cols=cols
    def fit(self,df):
        self.num=[c for c in self.cols if pd.api.types.is_numeric_dtype(df[c])]; self.cat=[c for c in self.cols if c not in self.num]; self.nimp=SimpleImputer(strategy="median"); self.scaler=StandardScaler(); self.cimp=SimpleImputer(strategy="most_frequent"); self.enc=OneHotEncoder(handle_unknown="ignore",sparse_output=False)
        if self.num:self.scaler.fit(self.nimp.fit_transform(df[self.num]))
        if self.cat:self.enc.fit(self.cimp.fit_transform(df[self.cat])); return self
        return self
    def transform(self,df):
        blocks=[]
        if self.num:blocks.append(self.scaler.transform(self.nimp.transform(df[self.num])))
        if self.cat:blocks.append(self.enc.transform(self.cimp.transform(df[self.cat])))
        return np.concatenate(blocks,axis=1).astype(np.float32)
@dataclass
class Data:
    ids:list; clinical:np.ndarray; group_slices:list; ppg:np.ndarray; mask:np.ndarray; labels:np.ndarray; weeks:int
class FoldPreprocessor:
    def __init__(self,weeks):self.weeks=weeks
    def fit(self,b,p,ids):
        self.g=[GroupPreprocessor(x).fit(b.loc[ids]) for x in CLINICAL_GROUPS.values()]; q=p[p.patient_id.isin(ids)]; v=q.sequence_mask.eq(1); self.med=q.loc[v,PPG_FEATURES].median().fillna(0); self.ps=StandardScaler().fit(q.loc[v,PPG_FEATURES].fillna(self.med)); return self
    def transform(self,b,p,y,ids):
        blocks=[]; sl=[]; s=0
        for g in self.g:
            x=g.transform(b.loc[ids]); blocks.append(x); sl.append((s,s+x.shape[1])); s+=x.shape[1]
        q=p[p.patient_id.isin(ids)&(p.window_index<=self.weeks)].copy().sort_values(["patient_id","window_index"]); q[PPG_FEATURES]=q[PPG_FEATURES].fillna(self.med); x=self.ps.transform(q[PPG_FEATURES]).astype(np.float32); mask=q.sequence_mask.to_numpy(np.float32); x[mask==0]=0
        return Data(list(ids),np.concatenate(blocks,axis=1),sl,x.reshape(len(ids),self.weeks,len(PPG_FEATURES)),mask.reshape(len(ids),self.weeks),y.loc[ids].to_numpy(np.float32),self.weeks)
