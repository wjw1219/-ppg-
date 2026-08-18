"""Patient-level nested-CV single-week PPG occlusion experiment.

The script intentionally requires real input data. It never fabricates labels,
probabilities, or metrics. See README.md for the expected schema.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def require_runtime_dependencies():
    try:
        import torch  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "本实验需要 torch 和 scikit-learn。请先安装 occlusion_experiment/requirements.txt。"
        ) from exc


require_runtime_dependencies()
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"不支持的数据格式: {path.suffix}")


@dataclass
class PatientData:
    patient_ids: np.ndarray
    y: np.ndarray
    clinical: np.ndarray
    ppg: np.ndarray
    ppg_valid: np.ndarray
    clinical_names: List[str]
    ppg_names: List[str]


def numeric_feature_columns(df: pd.DataFrame, excluded: Iterable[str]) -> List[str]:
    excluded = set(excluded)
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def load_patient_data(cfg: Dict, root: Path) -> PatientData:
    data_dir = root / cfg["data_dir"]
    cohort_path = data_dir / cfg["cohort_file"]
    weekly_path = data_dir / cfg["weekly_file"]
    cohort = read_table(cohort_path)
    weekly = read_table(weekly_path)
    pid, label, week = cfg["id_column"], cfg["label_column"], cfg["week_column"]
    for col in (pid, label):
        if col not in cohort.columns:
            raise ValueError(f"cohort 文件缺少字段: {col}")
    for col in (pid, week):
        if col not in weekly.columns:
            raise ValueError(f"ppg_weekly 文件缺少字段: {col}")
    if cohort[pid].duplicated().any():
        raise ValueError("cohort.csv 必须每位患者一行，发现重复 patient_id。")
    cohort[label] = pd.to_numeric(cohort[label], errors="raise").astype(int)
    if not set(cohort[label].unique()).issubset({0, 1}):
        raise ValueError("label 必须只包含 0 和 1。")
    if weekly.duplicated([pid, week]).any():
        dup = weekly.loc[weekly.duplicated([pid, week], keep=False), [pid, week]]
        raise ValueError(f"发现重复 patient_id-week 组合，必须先明确聚合规则:\n{dup.head()}")
    weekly[week] = pd.to_numeric(weekly[week], errors="raise").astype(int)
    n_weeks = int(cfg["n_weeks"])
    bad_week = weekly.loc[~weekly[week].between(1, n_weeks), week]
    if not bad_week.empty:
        raise ValueError(f"week 必须在1–{n_weeks}之间，发现: {sorted(bad_week.unique())}")
    clinical_cols = cfg.get("clinical_columns") or numeric_feature_columns(cohort, [pid, label])
    ppg_cols = cfg.get("ppg_columns") or numeric_feature_columns(weekly, [pid, week])
    if not clinical_cols or not ppg_cols:
        raise ValueError("未识别到临床或PPG数值特征，请在config.json中显式设置列名。")
    missing_clinical = set(clinical_cols) - set(cohort.columns)
    missing_ppg = set(ppg_cols) - set(weekly.columns)
    if missing_clinical or missing_ppg:
        raise ValueError(f"特征列不存在。clinical={missing_clinical}, ppg={missing_ppg}")

    patient_ids = cohort[pid].astype(str).to_numpy()
    n = len(cohort)
    ppg = np.full((n, n_weeks, len(ppg_cols)), np.nan, dtype=np.float32)
    valid = np.zeros((n, n_weeks), dtype=bool)
    index = {p: i for i, p in enumerate(patient_ids)}
    unknown = set(weekly[pid].astype(str)) - set(index)
    if unknown:
        raise ValueError(f"ppg_weekly中存在cohort没有的patient_id，例如: {sorted(unknown)[:5]}")
    for row in weekly.itertuples(index=False):
        record = row._asdict()
        i = index[str(record[pid])]
        w = int(record[week]) - 1
        ppg[i, w, :] = np.asarray([record[c] for c in ppg_cols], dtype=np.float32)
        valid[i, w] = np.isfinite(ppg[i, w, :]).any()
    clinical = cohort[clinical_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return PatientData(patient_ids, cohort[label].to_numpy(dtype=np.int64), clinical, ppg, valid, clinical_cols, ppg_cols)


class FoldPreprocessor:
    def __init__(self, add_missing_indicators: bool = True):
        self.add_missing_indicators = add_missing_indicators
        self.c_mean = None
        self.c_std = None
        self.p_mean = None
        self.p_std = None

    def fit(self, clinical: np.ndarray, ppg: np.ndarray, valid: np.ndarray):
        self.c_mean = np.nanmean(clinical, axis=0)
        self.c_std = np.nanstd(clinical, axis=0)
        self.c_mean = np.where(np.isfinite(self.c_mean), self.c_mean, 0.0)
        self.c_std = np.where(np.isfinite(self.c_std) & (self.c_std > 1e-8), self.c_std, 1.0)
        flat = ppg[valid]
        if flat.size == 0:
            self.p_mean = np.zeros(ppg.shape[-1], dtype=np.float32)
            self.p_std = np.ones(ppg.shape[-1], dtype=np.float32)
        else:
            self.p_mean = np.nanmean(flat, axis=0)
            self.p_std = np.nanstd(flat, axis=0)
            self.p_mean = np.where(np.isfinite(self.p_mean), self.p_mean, 0.0)
            self.p_std = np.where(np.isfinite(self.p_std) & (self.p_std > 1e-8), self.p_std, 1.0)
        return self

    def transform(self, clinical: np.ndarray, ppg: np.ndarray, valid: np.ndarray):
        c_missing = ~np.isfinite(clinical)
        c = np.where(c_missing, self.c_mean, clinical)
        c = (c - self.c_mean) / self.c_std
        if self.add_missing_indicators:
            c = np.concatenate([c, c_missing.astype(np.float32)], axis=1)
        p_missing = ~np.isfinite(ppg)
        p = np.where(p_missing, self.p_mean, ppg)
        p = (p - self.p_mean) / self.p_std
        p = np.where(np.isfinite(p), p, 0.0).astype(np.float32)
        p = np.where(valid[..., None], p, 0.0)
        return c.astype(np.float32), p.astype(np.float32), valid.astype(bool)


class ClinicalPriorTemporalGraph(nn.Module):
    """Causal temporal attention model with a clinical-prior token.

    The clinical token is retained at position 0. PPG week positions are never
    reindexed; invalid weeks are supplied through key padding masks.
    """
    def __init__(self, clinical_dim, ppg_dim, n_weeks, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim必须能被num_heads整除。")
        self.n_weeks = n_weeks
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ppg_encoder = nn.Sequential(
            nn.Linear(ppg_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position = nn.Parameter(torch.zeros(n_weeks + 1, hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=4 * hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, clinical, ppg, valid):
        c = self.clinical_encoder(clinical).unsqueeze(1)
        p = self.ppg_encoder(ppg)
        tokens = torch.cat([c, p], dim=1) + self.position.unsqueeze(0)
        key_padding = torch.cat([torch.ones((valid.size(0), 1), dtype=torch.bool, device=valid.device), valid], dim=1)
        key_padding = ~key_padding
        causal = torch.triu(torch.ones((self.n_weeks + 1, self.n_weeks + 1), dtype=torch.bool, device=valid.device), diagonal=1)
        h = self.temporal(tokens, mask=causal, src_key_padding_mask=key_padding)
        h = self.norm(h)
        v = valid.float().unsqueeze(-1)
        pooled = (h[:, 1:] * v).sum(dim=1) / v.sum(dim=1).clamp_min(1.0)
        return self.head(torch.cat([h[:, 0], pooled], dim=1)).squeeze(1)


def make_loader(c, p, valid, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(c), torch.from_numpy(p), torch.from_numpy(valid), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for c, p, valid, _ in loader:
        logits = model(c.to(device), p.to(device), valid.to(device))
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def youden_threshold(y, prob):
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y, prob)
    score = tpr - fpr
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    return float(np.clip(thresholds[np.argmax(np.where(finite, score, -np.inf))], 0.0, 1.0))


def train_epochs(model, loader, optimizer, criterion, device, epochs):
    model.train()
    for _ in range(epochs):
        for c, p, valid, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(c.to(device), p.to(device), valid.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def fit_with_early_stopping(model, train_loader, val_loader, y_val, cfg, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    best_state, best_loss, best_epoch, wait = None, math.inf, 0, 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        for c, p, valid, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(c.to(device), p.to(device), valid.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        val_prob = predict(model, val_loader, device)
        val_loss = float(-(y_val * np.log(np.clip(val_prob, 1e-7, 1 - 1e-7)) + (1 - y_val) * np.log(np.clip(1 - val_prob, 1e-7, 1 - 1e-7))).mean())
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg["early_stopping_patience"]:
                break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = max(1, epoch)
    model.load_state_dict(best_state)
    return model, best_epoch


def metric_dict(y, prob, threshold):
    threshold_arr = np.asarray(threshold, dtype=float)
    if threshold_arr.ndim == 0:
        threshold_arr = np.full(len(prob), float(threshold_arr))
    pred = (prob >= threshold_arr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
        "auprc": float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, prob)),
        "tp": int(tp), "fn": int(fn), "tn": int(tn), "fp": int(fp),
        "threshold": float(np.mean(threshold_arr)), "n": int(len(y)), "positive": int(y.sum()),
    }


def bootstrap_ci(y, prob, threshold, reps, seed):
    rng = np.random.default_rng(seed)
    threshold_arr = np.asarray(threshold, dtype=float)
    values = {k: [] for k in ["roc_auc", "auprc", "sensitivity", "specificity", "f1", "brier"]}
    for _ in range(reps):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        sampled_threshold = threshold_arr if threshold_arr.ndim == 0 else threshold_arr[idx]
        m = metric_dict(y[idx], prob[idx], sampled_threshold)
        for key in values:
            values[key].append(m[key])
    out = {}
    for key, vals in values.items():
        out[f"{key}_ci_low"] = float(np.nanpercentile(vals, 2.5)) if vals else np.nan
        out[f"{key}_ci_high"] = float(np.nanpercentile(vals, 97.5)) if vals else np.nan
    return out


def fit_outer_model(data: PatientData, inner_train_idx, outer_train_idx, val_idx, test_idx, cfg, seed, device):
    set_seed(seed)
    # Inner preprocessing is used only to select the number of epochs and threshold.
    prep_inner = FoldPreprocessor(cfg["add_clinical_missing_indicators"])
    c_tr, p_tr, v_tr = prep_inner.fit(data.clinical[inner_train_idx], data.ppg[inner_train_idx], data.ppg_valid[inner_train_idx]).transform(data.clinical[inner_train_idx], data.ppg[inner_train_idx], data.ppg_valid[inner_train_idx])
    c_va, p_va, v_va = prep_inner.transform(data.clinical[val_idx], data.ppg[val_idx], data.ppg_valid[val_idx])
    model = ClinicalPriorTemporalGraph(c_tr.shape[1], p_tr.shape[2], cfg["n_weeks"], cfg["hidden_dim"], cfg["num_heads"], cfg["num_layers"], cfg["dropout"]).to(device)
    tr_loader = make_loader(c_tr, p_tr, v_tr, data.y[inner_train_idx], cfg["batch_size"], True)
    va_loader = make_loader(c_va, p_va, v_va, data.y[val_idx], cfg["batch_size"], False)
    model, best_epoch = fit_with_early_stopping(model, tr_loader, va_loader, data.y[val_idx].astype(float), cfg, device)
    val_prob = predict(model, va_loader, device)
    threshold = youden_threshold(data.y[val_idx], val_prob) if cfg["threshold_mode"] == "validation_youden" else float(cfg["fixed_threshold"])

    # Refit with all outer-training patients using the selected epoch count.
    prep_outer = FoldPreprocessor(cfg["add_clinical_missing_indicators"])
    c_all, p_all, v_all = prep_outer.fit(data.clinical[outer_train_idx], data.ppg[outer_train_idx], data.ppg_valid[outer_train_idx]).transform(data.clinical[outer_train_idx], data.ppg[outer_train_idx], data.ppg_valid[outer_train_idx])
    c_test, p_test, v_test = prep_outer.transform(data.clinical[test_idx], data.ppg[test_idx], data.ppg_valid[test_idx])
    final_model = ClinicalPriorTemporalGraph(c_all.shape[1], p_all.shape[2], cfg["n_weeks"], cfg["hidden_dim"], cfg["num_heads"], cfg["num_layers"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(final_model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    train_epochs(final_model, make_loader(c_all, p_all, v_all, data.y[outer_train_idx], cfg["batch_size"], True), opt, criterion, device, best_epoch)
    return final_model, prep_outer, c_test, p_test, v_test, threshold, best_epoch


def run(cfg: Dict, root: Path):
    data = load_patient_data(cfg, root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = root / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    n = len(data.y)
    oof = {w: np.full(n, np.nan, dtype=float) for w in range(0, cfg["n_weeks"] + 1)}
    threshold_by_patient = np.full(n, np.nan, dtype=float)
    fold_by_patient = np.full(n, -1, dtype=int)
    logs = []
    fold_metric_rows = []
    splitter = StratifiedKFold(cfg["n_splits"], shuffle=True, random_state=2026)
    for fold, (train_outer, test_idx) in enumerate(splitter.split(np.zeros(n), data.y), start=1):
        inner_train, val_idx = train_test_split(train_outer, test_size=cfg["inner_val_fraction"], stratify=data.y[train_outer], random_state=fold * 100)
        fold_predictions = {w: [] for w in range(0, cfg["n_weeks"] + 1)}
        fold_thresholds = []
        for seed in cfg["seeds"]:
            model, prep, c_test, p_test, valid_test, threshold, best_epoch = fit_outer_model(data, inner_train, train_outer, val_idx, test_idx, cfg, int(seed + fold), device)
            base_loader = make_loader(c_test, p_test, valid_test, data.y[test_idx], cfg["batch_size"], False)
            fold_predictions[0].append(predict(model, base_loader, device))
            for masked_week in range(1, cfg["n_weeks"] + 1):
                masked_valid = valid_test.copy()
                masked_valid[:, masked_week - 1] = False
                masked_ppg = p_test.copy()
                masked_ppg[:, masked_week - 1, :] = 0.0
                loader = make_loader(c_test, masked_ppg, masked_valid, data.y[test_idx], cfg["batch_size"], False)
                fold_predictions[masked_week].append(predict(model, loader, device))
            fold_thresholds.append(threshold)
            logs.append({"fold": fold, "seed": int(seed), "best_epoch": int(best_epoch), "threshold": float(threshold), "n_train": len(inner_train), "n_val": len(val_idx), "n_test": len(test_idx), "device": str(device)})
        threshold = float(np.mean(fold_thresholds))
        threshold_by_patient[test_idx] = threshold
        fold_by_patient[test_idx] = fold
        for w in fold_predictions:
            fold_prob = np.mean(np.stack(fold_predictions[w], axis=0), axis=0)
            oof[w][test_idx] = fold_prob
            fold_metric = metric_dict(data.y[test_idx], fold_prob, threshold)
            fold_metric.update({"fold": fold, "masked_week": 0 if w == 0 else w})
            fold_metric_rows.append(fold_metric)

    long_rows, metric_rows = [], []
    y = data.y
    for w in range(0, cfg["n_weeks"] + 1):
        prob = oof[w]
        if np.isnan(prob).any():
            raise RuntimeError(f"分析{w}存在未生成的折外预测。")
        threshold = threshold_by_patient
        m = metric_dict(y, prob, threshold)
        m.update({"masked_week": 0 if w == 0 else w})
        m.update(bootstrap_ci(y, prob, threshold, int(cfg["bootstrap_reps"]), 7000 + w))
        metric_rows.append(m)
        for i, pid in enumerate(data.patient_ids):
            long_rows.append({"patient_id": pid, "fold": int(fold_by_patient[i]), "label": int(y[i]), "masked_week": 0 if w == 0 else w, "probability": float(prob[i]), "threshold": float(threshold[i]), "predicted": int(prob[i] >= threshold[i])})
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics_by_week.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_metric_rows).to_csv(out_dir / "metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(long_rows).to_csv(out_dir / "patient_oof_predictions_long.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(logs).to_csv(out_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"patient_id": data.patient_ids, "label": data.y}).to_csv(out_dir / "cohort_used.csv", index=False, encoding="utf-8-sig")
    print(f"Completed: {out_dir}")
    print(f"Patients={n}, positives={int(y.sum())}, negatives={int((1-y).sum())}, device={device}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="occlusion_experiment/config.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    with (root / args.config).open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    run(cfg, root)


if __name__ == "__main__":
    main()
