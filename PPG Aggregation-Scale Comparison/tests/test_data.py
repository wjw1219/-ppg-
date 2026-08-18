import json, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data import load_common,load_scale

class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input=ROOT.parent/"outputs/evoaf_synthetic_387/intermediate"; cls.cfg=json.loads((ROOT/"config.json").read_text())
        cls.baseline,cls.labels,cls.folds=load_common(cls.input)
    def test_common_patients_and_folds(self):
        self.assertEqual(len(self.baseline),387); self.assertEqual(set(self.folds.unique()),{1,2,3,4,5})
    def test_scale_node_counts_and_coverage(self):
        for _,meta in self.cfg["scales"].items():
            df=load_scale(self.input,meta["file"],meta["nodes"])
            self.assertTrue((df.groupby("patient_id").size()==meta["nodes"]).all()); self.assertEqual(df.end_day.max(),182)
    def test_same_patient_set_across_scales(self):
        sets=[]
        for _,meta in self.cfg["scales"].items(): sets.append(set(load_scale(self.input,meta["file"],meta["nodes"]).patient_id))
        self.assertTrue(sets[0]==sets[1]==sets[2]==set(self.baseline.index))

if __name__=="__main__": unittest.main()
