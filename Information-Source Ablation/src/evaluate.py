import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, roc_auc_score)


def metrics(y, p, thresholds):
    pred = (p >= thresholds).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
        "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y, pred, zero_division=0), "accuracy": accuracy_score(y, pred),
        "brier": brier_score_loss(y, p),
    }


def stratified_bootstrap_indices(y, n_resamples, seed):
    rng = np.random.default_rng(seed)
    negative, positive = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    return [np.r_[rng.choice(negative, len(negative), replace=True),
                  rng.choice(positive, len(positive), replace=True)] for _ in range(n_resamples)]


def summarize(y, p, thresholds, indices):
    point = metrics(y, p, thresholds)
    boot = {name: [] for name in point}
    for idx in indices:
        values = metrics(y[idx], p[idx], thresholds[idx])
        for name, value in values.items():
            boot[name].append(value)
    return {name: (value, float(np.percentile(boot[name], 2.5)), float(np.percentile(boot[name], 97.5)))
            for name, value in point.items()}


def paired_difference(y, p_full, p_other, indices):
    point = {"roc_auc_difference": roc_auc_score(y, p_full) - roc_auc_score(y, p_other),
             "auprc_difference": average_precision_score(y, p_full) - average_precision_score(y, p_other)}
    boot = {k: [] for k in point}
    for idx in indices:
        boot["roc_auc_difference"].append(roc_auc_score(y[idx], p_full[idx]) - roc_auc_score(y[idx], p_other[idx]))
        boot["auprc_difference"].append(average_precision_score(y[idx], p_full[idx]) - average_precision_score(y[idx], p_other[idx]))
    return {name: (value, float(np.percentile(boot[name], 2.5)), float(np.percentile(boot[name], 97.5)))
            for name, value in point.items()}
