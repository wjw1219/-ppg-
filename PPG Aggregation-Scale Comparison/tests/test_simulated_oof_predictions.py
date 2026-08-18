import sys
import unittest
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_simulated_oof_predictions import generate_predictions


class SimulatedOOFPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        outcome = pd.read_csv(ROOT.parent / "outputs/evoaf_synthetic_387/intermediate/holter_outcome.csv")
        cls.result, cls.metadata = generate_predictions(outcome, seed=20260724)

    def test_preserves_patients_labels_and_cardinality(self):
        self.assertEqual(len(self.result), 387 * 3)
        self.assertTrue(self.result.groupby(["patient_id", "scale"]).size().eq(1).all())
        self.assertTrue(self.result.groupby("patient_id").label.nunique().eq(1).all())

    def test_probabilities_and_thresholds_are_bounded(self):
        self.assertTrue(self.result.probability.between(0, 1).all())
        self.assertTrue(self.result.threshold.between(0, 1).all())

    def test_seven_day_has_best_discrimination(self):
        metrics = {}
        for scale, frame in self.result.groupby("scale"):
            metrics[scale] = (
                roc_auc_score(frame.label, frame.probability),
                average_precision_score(frame.label, frame.probability),
            )
        self.assertGreater(metrics["7day"][0], metrics["14day"][0])
        self.assertGreater(metrics["7day"][0], metrics["3day"][0])
        self.assertGreater(metrics["7day"][1], metrics["14day"][1])
        self.assertGreater(metrics["7day"][1], metrics["3day"][1])
        self.assertTrue(self.metadata["simulated_predictions"])

    def test_seven_day_matches_prespecified_fusion_target(self):
        frame = self.result[self.result.scale.eq("7day")]
        y = frame.label.to_numpy(); probability = frame.probability.to_numpy()
        prediction = (probability >= frame.threshold.to_numpy()).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        self.assertEqual((tp, fn, tn, fp), (188, 12, 176, 11))
        self.assertAlmostEqual(roc_auc_score(y, probability), 0.985, delta=0.003)
        self.assertAlmostEqual(average_precision_score(y, probability), 0.987, delta=0.003)
        self.assertAlmostEqual(f1_score(y, prediction), 0.942, delta=0.001)
        self.assertAlmostEqual(brier_score_loss(y, probability), 0.069, delta=0.003)


if __name__ == "__main__":
    unittest.main()
