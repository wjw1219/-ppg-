from pathlib import Path
import unittest

import numpy as np
import pandas as pd


class IllustrativeOutputsTest(unittest.TestCase):
    def test_generate_predictions_has_expected_schema_and_cardinality(self):
        from illustrative import CONDITIONS, generate_predictions

        labels = pd.DataFrame({"patient_id": [f"P{i:03d}" for i in range(40)], "label": [0, 1] * 20})
        result = generate_predictions(labels, seed=17)

        self.assertEqual(set(result.columns), {"patient_id", "condition", "label", "probability", "threshold"})
        self.assertEqual(len(result), len(labels) * len(CONDITIONS))
        self.assertTrue(result.groupby(["patient_id", "condition"]).size().eq(1).all())
        self.assertTrue(result.probability.between(0, 1).all())
        self.assertTrue(result.threshold.between(0, 1).all())


    def test_generate_predictions_follows_prespecified_auc_order(self):
        from illustrative import EXPECTED_ORDER, generate_predictions
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(9)
        labels = pd.DataFrame({"patient_id": [f"P{i:03d}" for i in range(387)],
                               "label": rng.binomial(1, 0.42, 387)})
        result = generate_predictions(labels, seed=20260723)
        auc = {condition: roc_auc_score(frame.label, frame.probability)
               for condition, frame in result.groupby("condition")}

        self.assertTrue(all(auc[a] > auc[b] for a, b in zip(EXPECTED_ORDER, EXPECTED_ORDER[1:])))


    def test_output_paths_are_isolated(self):
        from illustrative import output_paths

        root = Path("illustrative_simulation")
        paths = output_paths(root)
        self.assertEqual(paths["results"], root / "results")
        self.assertEqual(paths["figures"], root / "figures")
        self.assertEqual(paths["report"], root / "report")
        self.assertTrue(all("illustrative_simulation" in str(path) for path in paths.values()))

    def test_report_metric_format_includes_confidence_interval(self):
        from build_illustrative_report import format_ci

        self.assertEqual(format_ci(0.81234, 0.7501, 0.8702), "0.812 (0.750-0.870)")


if __name__ == "__main__":
    unittest.main()
