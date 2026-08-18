import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import metrics, stratified_bootstrap_indices


class EvaluationTests(unittest.TestCase):
    def test_known_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([.1, .2, .8, .9])
        values = metrics(y, p, np.full(4, .5))
        self.assertEqual(values["roc_auc"], 1.0)
        self.assertEqual(values["sensitivity"], 1.0)
        self.assertEqual(values["specificity"], 1.0)

    def test_bootstrap_is_stratified_and_reproducible(self):
        y = np.array([0] * 7 + [1] * 5)
        a = stratified_bootstrap_indices(y, 10, 4)
        b = stratified_bootstrap_indices(y, 10, 4)
        self.assertTrue(all(np.array_equal(x, z) for x, z in zip(a, b)))
        self.assertTrue(all((y[idx] == 0).sum() == 7 and (y[idx] == 1).sum() == 5 for idx in a))


if __name__ == "__main__": unittest.main()
