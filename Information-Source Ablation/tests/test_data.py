import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import CLINICAL_GROUPS, OUTCOME_ONLY, load_raw


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline, cls.ppg, cls.labels, cls.folds = load_raw(ROOT.parent / "outputs/evoaf_synthetic_387/intermediate")

    def test_patient_and_window_counts(self):
        self.assertEqual(len(self.baseline), 387)
        self.assertTrue((self.ppg.groupby("patient_id").size() == 26).all())

    def test_groups_are_disjoint_and_no_outcome(self):
        flat = [x for values in CLINICAL_GROUPS.values() for x in values]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertTrue(set(flat).isdisjoint(OUTCOME_ONLY))

    def test_folds_cover_each_patient_once(self):
        self.assertEqual(set(self.folds.unique()), {1, 2, 3, 4, 5})
        self.assertEqual(len(self.folds), self.folds.index.nunique())


if __name__ == "__main__":
    unittest.main()
