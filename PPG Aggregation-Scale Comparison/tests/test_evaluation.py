import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.evaluate import metrics,stratified_bootstrap_indices

class EvaluationTests(unittest.TestCase):
    def test_metrics(self):
        y=np.array([0,0,1,1]); p=np.array([.1,.2,.8,.9]); v=metrics(y,p,np.full(4,.5)); self.assertEqual(v["roc_auc"],1); self.assertEqual(v["sensitivity"],1)
    def test_shared_bootstrap_indices(self):
        y=np.array([0]*6+[1]*5); a=stratified_bootstrap_indices(y,8,9); b=stratified_bootstrap_indices(y,8,9); self.assertTrue(all(np.array_equal(x,z) for x,z in zip(a,b)))

if __name__=="__main__": unittest.main()
