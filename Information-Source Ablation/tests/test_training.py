import sys
import unittest
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import FoldPreprocessor, load_raw
from src.train import train_one


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline, cls.ppg, cls.labels, _ = load_raw(ROOT.parent / "outputs/evoaf_synthetic_387/intermediate")

    def test_short_training_returns_finite_predictions(self):
        ids = self.baseline.index.tolist()[:120]
        split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=1)
        a, b = next(split.split(ids, self.labels.loc[ids]))
        train_ids, val_ids = [ids[i] for i in a], [ids[i] for i in b]
        groups = ["demographic_lifestyle", "af_history", "laboratory_echo_ecg", "procedure_medication"]
        prep = FoldPreprocessor(groups).fit(self.baseline, self.ppg, train_ids)
        train = prep.transform(self.baseline, self.ppg, self.labels, train_ids)
        val = prep.transform(self.baseline, self.ppg, self.labels, val_ids)
        cfg = {"hidden_dim": 8, "attention_heads": 2, "dropout": .1, "batch_size": 32,
               "learning_rate": .001, "weight_decay": .0001, "max_epochs": 3,
               "early_stopping_patience": 2}
        _, probabilities, _, epochs, auc = train_one("fusion_full", train, val, cfg, 11)
        self.assertEqual(len(probabilities), len(val_ids))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertGreaterEqual(auc, 0)
        self.assertGreaterEqual(epochs, 1)


if __name__ == "__main__":
    unittest.main()
