from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

class ExpectedScenarioConsistencyTest(unittest.TestCase):
    def test_week26_reuses_full_model_predictions(self):
        weekly = pd.read_csv(ROOT / "results/expected_weekly_predictions.csv")
        models = pd.read_csv(ROOT / "results/expected_model_predictions.csv")
        w26 = weekly[weekly.weeks == 26].sort_values("patient_id")
        full = models[models.model == "full_model"].sort_values("patient_id")
        self.assertTrue(np.array_equal(w26.patient_id.to_numpy(), full.patient_id.to_numpy()))
        self.assertTrue(np.allclose(w26.probability.to_numpy(), full.probability.to_numpy(), atol=0, rtol=0))

    def test_plateau_has_small_nonzero_fluctuation(self):
        metrics = pd.read_csv(ROOT / "results/expected_weekly_metrics.csv")
        auc = metrics[(metrics.metric == "roc_auc") & metrics.weeks.between(16, 26)].estimate
        self.assertGreater(auc.max() - auc.min(), 0.02)
        self.assertLess(auc.max() - auc.min(), 0.05)

if __name__ == "__main__":
    unittest.main()
