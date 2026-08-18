import copy
import json
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import FoldPreprocessor
from .models import AblationModel


CONDITIONS = {
    "clinical_only": {"mode": "clinical", "groups": ["demographic_lifestyle", "af_history", "laboratory_echo_ecg", "procedure_medication"]},
    "ppg_only": {"mode": "ppg", "groups": []},
    "fusion_full": {"mode": "fusion", "groups": ["demographic_lifestyle", "af_history", "laboratory_echo_ecg", "procedure_medication"]},
    "fusion_drop_demographic_lifestyle": {"mode": "fusion", "groups": ["af_history", "laboratory_echo_ecg", "procedure_medication"]},
    "fusion_drop_af_history": {"mode": "fusion", "groups": ["demographic_lifestyle", "laboratory_echo_ecg", "procedure_medication"]},
    "fusion_drop_laboratory_echo_ecg": {"mode": "fusion", "groups": ["demographic_lifestyle", "af_history", "procedure_medication"]},
    "fusion_drop_procedure_medication": {"mode": "fusion", "groups": ["demographic_lifestyle", "af_history", "laboratory_echo_ecg"]},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def youden_threshold(y, p):
    thresholds = np.unique(np.r_[0.0, p, 1.0])
    best_t, best_j = 0.5, -np.inf
    for t in thresholds:
        pred = p >= t
        tp = ((pred == 1) & (y == 1)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        if sens + spec - 1 > best_j:
            best_t, best_j = float(t), sens + spec - 1
    return best_t


def train_one(condition, train_data, val_data, cfg, seed):
    set_seed(seed)
    mode = CONDITIONS[condition]["mode"]
    group_dims = [b - a for a, b in train_data.group_slices] if mode in {"clinical", "fusion"} else []
    model = AblationModel(mode, group_dims, hidden_dim=cfg["hidden_dim"], heads=cfg["attention_heads"], dropout=cfg["dropout"])
    x_clin = torch.tensor(train_data.clinical, dtype=torch.float32)
    x_ppg = torch.tensor(train_data.ppg, dtype=torch.float32)
    x_mask = torch.tensor(train_data.mask, dtype=torch.float32)
    y = torch.tensor(train_data.labels, dtype=torch.float32)
    train_ds = TensorDataset(x_clin, x_ppg, x_mask, y)
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed))
    pos = float(train_data.labels.sum())
    neg = float(len(train_data.labels) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best_state, best_auc, patience = None, -np.inf, 0
    val_clin = torch.tensor(val_data.clinical, dtype=torch.float32)
    val_ppg = torch.tensor(val_data.ppg, dtype=torch.float32)
    val_mask = torch.tensor(val_data.mask, dtype=torch.float32)
    for epoch in range(cfg["max_epochs"]):
        model.train()
        for cb, pb, mb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(cb, pb, mb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(val_clin, val_ppg, val_mask)).cpu().numpy()
        auc = roc_auc_score(val_data.labels, val_prob) if len(np.unique(val_data.labels)) == 2 else 0.5
        if auc > best_auc + 1e-5:
            best_auc, best_state, patience = auc, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(val_clin, val_ppg, val_mask)).cpu().numpy()
        train_prob = torch.sigmoid(model(x_clin, x_ppg, x_mask)).cpu().numpy()
    return model, val_prob, train_prob, epoch + 1, best_auc


def run_condition(condition, baseline, ppg, labels, folds, cfg):
    ids = baseline.index.tolist()
    pred_records, fold_records, logs = [], [], []
    final_probs = {pid: [] for pid in ids}
    final_thresholds = {pid: [] for pid in ids}
    for fold in range(1, cfg["outer_folds"] + 1):
        test_ids = [pid for pid in ids if int(folds.loc[pid]) == fold]
        train_ids = [pid for pid in ids if int(folds.loc[pid]) != fold]
        split = StratifiedShuffleSplit(n_splits=1, test_size=cfg["validation_fraction"], random_state=fold)
        tr_rel, va_rel = next(split.split(np.zeros(len(train_ids)), labels.loc[train_ids].to_numpy()))
        fit_ids, val_ids = [train_ids[i] for i in tr_rel], [train_ids[i] for i in va_rel]
        prep = FoldPreprocessor(CONDITIONS[condition]["groups"]).fit(baseline, ppg, fit_ids)
        fit_data = prep.transform(baseline, ppg, labels, fit_ids)
        val_data = prep.transform(baseline, ppg, labels, val_ids)
        test_data = prep.transform(baseline, ppg, labels, test_ids)
        for seed in cfg["training_seeds"]:
            model, val_prob, _, epochs, val_auc = train_one(condition, fit_data, val_data, cfg, seed)
            model.eval()
            with torch.no_grad():
                test_prob = torch.sigmoid(model(torch.tensor(test_data.clinical), torch.tensor(test_data.ppg), torch.tensor(test_data.mask))).numpy()
            threshold = youden_threshold(val_data.labels, val_prob)
            for pid, prob in zip(test_ids, test_prob):
                final_probs[pid].append(float(prob))
                final_thresholds[pid].append(threshold)
            fold_records.append({"condition": condition, "fold": fold, "seed": seed, "validation_auc": val_auc, "epochs": epochs, "threshold": threshold, "test_n": len(test_ids)})
            logs.append({"condition": condition, "fold": fold, "seed": seed, "status": "complete"})
    for pid in ids:
        pred_records.append({"patient_id": pid, "condition": condition, "label": int(labels.loc[pid]),
                             "probability": float(np.mean(final_probs[pid])), "threshold": float(np.mean(final_thresholds[pid]))})
    return pred_records, fold_records, logs


def run_all(baseline, ppg, labels, folds, cfg):
    preds, fold_metrics, logs = [], [], []
    for condition in CONDITIONS:
        p, f, l = run_condition(condition, baseline, ppg, labels, folds, cfg)
        preds.extend(p); fold_metrics.extend(f); logs.extend(l)
    return preds, fold_metrics, logs
