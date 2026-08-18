import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


TARGETS = {
    "3day": {"tp": 176, "tn": 159, "auc": 0.945, "auprc": 0.952, "brier": 0.115},
    "7day": {"tp": 188, "tn": 176, "auc": 0.985, "auprc": 0.987, "brier": 0.069},
    "14day": {"tp": 183, "tn": 168, "auc": 0.970, "auprc": 0.975, "brier": 0.085},
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def build_probability(target, seed, params):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(200, dtype=int), np.zeros(187, dtype=int)]
    positive = rng.permutation(200); negative = rng.permutation(187) + 200
    tp = positive[:target["tp"]]; fn = positive[target["tp"]:]
    tn = negative[:target["tn"]]; fp = negative[target["tn"]:]
    probability = np.empty(len(y))
    groups = [(tp, True, params[0], params[1]), (fn, False, params[2], params[3]),
              (tn, False, params[4], params[5]), (fp, True, params[6], params[7])]
    for indices, above, center, spread in groups:
        z = rng.normal(size=len(indices))
        unit = sigmoid(center + spread * z)
        probability[indices] = (0.5 + 0.49 * unit) if above else (0.5 * unit)
    return y, probability


def tune(scale, seed):
    target = TARGETS[scale]
    def objective(params):
        y, probability = build_probability(target, seed, params)
        auc = roc_auc_score(y, probability); auprc = average_precision_score(y, probability)
        brier = brier_score_loss(y, probability)
        return ((auc-target["auc"])/0.0015)**2 + ((auprc-target["auprc"])/0.0015)**2 + ((brier-target["brier"])/0.0015)**2
    result = differential_evolution(objective, [(-4, 4), (0.05, 3)] * 4, seed=seed,
                                    maxiter=300, popsize=18, polish=True, workers=1)
    y, probability = build_probability(target, seed, result.x)
    print(scale, "params=", [round(float(x), 8) for x in result.x])
    print(" metrics=", roc_auc_score(y, probability), average_precision_score(y, probability), brier_score_loss(y, probability))


if __name__ == "__main__":
    for index, scale in enumerate(("3day", "7day", "14day")):
        tune(scale, 20260724 + index)
