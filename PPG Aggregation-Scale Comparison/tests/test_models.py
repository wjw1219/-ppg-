import sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.models import AblationModel

class ModelTests(unittest.TestCase):
    def test_three_sequence_lengths(self):
        for nodes in (61,26,13):
            m=AblationModel("fusion",[5,6,5,4],weeks=nodes).eval(); c=torch.randn(4,20); p=torch.randn(4,nodes,13); mask=torch.ones(4,nodes)
            self.assertEqual(tuple(m(c,p,mask).shape),(4,))
    def test_mask_invariance_each_scale(self):
        for nodes in (61,26,13):
            m=AblationModel("fusion",[5,6,5,4],weeks=nodes).eval(); c=torch.randn(2,20); p=torch.randn(2,nodes,13); mask=torch.ones(2,nodes); mask[:,-2:]=0
            changed=p.clone(); changed[:,-2:]=999
            with torch.no_grad(): a=m(c,p,mask); b=m(c,changed,mask)
            self.assertTrue(torch.allclose(a,b,atol=1e-5))

if __name__=="__main__": unittest.main()
