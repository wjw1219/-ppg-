import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import AblationModel


class ModelTests(unittest.TestCase):
    def test_all_modes_return_one_logit_per_patient(self):
        clinical = torch.randn(8, 20)
        ppg = torch.randn(8, 26, 13)
        mask = torch.ones(8, 26)
        for mode in ("clinical", "ppg", "fusion"):
            model = AblationModel(mode, [5, 6, 5, 4])
            self.assertEqual(tuple(model(clinical, ppg, mask).shape), (8,))

    def test_masked_values_do_not_change_temporal_output(self):
        model = AblationModel("ppg", [1]).eval()
        x = torch.randn(3, 26, 13)
        mask = torch.ones(3, 26)
        mask[:, -3:] = 0
        changed = x.clone()
        changed[:, -3:] = 1000
        with torch.no_grad():
            a = model(torch.empty(3, 0), x, mask)
            b = model(torch.empty(3, 0), changed, mask)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
