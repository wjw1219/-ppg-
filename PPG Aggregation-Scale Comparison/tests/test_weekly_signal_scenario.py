import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_weekly_signal_scenario import build_scenario


class WeeklySignalScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT.parent / "outputs/evoaf_synthetic_387/intermediate"
        cls.tables, cls.metadata = build_scenario(source, seed=20260723)

    def test_patient_counts_and_windows_are_preserved(self):
        expected = {"3day": 61, "7day": 26, "14day": 13}
        for scale, nodes in expected.items():
            table = self.tables[scale]
            self.assertEqual(table.patient_id.nunique(), 387)
            self.assertTrue(table.groupby("patient_id").size().eq(nodes).all())

    def test_labels_and_identifiers_are_not_changed(self):
        source = pd.read_csv(ROOT.parent / "outputs/evoaf_synthetic_387/intermediate/holter_outcome.csv")
        result = self.tables["holter_outcome"]
        pd.testing.assert_frame_equal(source, result, check_dtype=False)

    def test_seven_day_signal_is_prespecified_as_strongest(self):
        effects = self.metadata["standardized_signal_effect"]
        self.assertGreater(effects["7day"], 2.5)
        self.assertGreater(effects["7day"], effects["3day"])
        self.assertGreater(effects["7day"], effects["14day"])
        self.assertEqual(self.metadata["scenario_type"], "synthetic_weekly_signal")


if __name__ == "__main__":
    unittest.main()
