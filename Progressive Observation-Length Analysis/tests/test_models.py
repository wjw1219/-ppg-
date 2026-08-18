import sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models import ComparisonModel
class T(unittest.TestCase):
 def test_all_models(self):
  for kind in ("mlp","gru","lstm","transformer","temporal_gat","full_model"):
   m=ComparisonModel(kind,20,[5,6,5,4],26).eval();c=torch.randn(3,20);p=torch.randn(3,26,13);mask=torch.ones(3,26);self.assertEqual(tuple(m(c,p,mask).shape),(3,))
 def test_full_prefixes(self):
  for k in (1,13,26):
   m=ComparisonModel("full_model",20,[5,6,5,4],k).eval();v=m(torch.randn(2,20),torch.randn(2,k,13),torch.ones(2,k));self.assertEqual(tuple(v.shape),(2,))
 def test_mask_invariance(self):
  for k in (1,13,26):
   m=ComparisonModel("full_model",20,[5,6,5,4],k).eval();c=torch.randn(2,20);p=torch.randn(2,k,13);mask=torch.ones(2,k);mask[:,-1]=0;q=p.clone();q[:,-1]=999
   with torch.no_grad():a=m(c,p,mask);b=m(c,q,mask)
   self.assertTrue(torch.allclose(a,b,atol=1e-5))
if __name__=="__main__":unittest.main()
