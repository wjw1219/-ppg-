import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import load_common,load_ppg,FoldPreprocessor
class T(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.b,cls.y,cls.f=load_common(ROOT.parent/"outputs/evoaf_synthetic_387/intermediate");cls.p=load_ppg(ROOT.parent/"outputs/evoaf_synthetic_387/intermediate")
 def test_prefix_sizes(self):
  for k in (1,13,26):
   self.assertEqual(self.p[self.p.window_index<=k].window_index.max(),k);self.assertTrue((self.p.groupby("patient_id").size()==26).all())
 def test_no_holter_columns(self):self.assertNotIn("discharge_af_burden_pct",self.b.columns);self.assertNotIn("month6_af_burden_pct",self.p.columns)
if __name__=="__main__":unittest.main()
