import sys,unittest
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import load_common,load_ppg,FoldPreprocessor
from src.train import fit
class T(unittest.TestCase):
 def test_short_full_fit(self):
  inp=ROOT.parent/"outputs/evoaf_synthetic_387/intermediate";b,y,_=load_common(inp);p=load_ppg(inp);ids=b.index[:100].tolist();a,z=next(StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=1).split(ids,y.loc[ids]));ti=[ids[i] for i in a];vi=[ids[i] for i in z];pr=FoldPreprocessor(5).fit(b,p,ti);tr=pr.transform(b,p,y,ti);va=pr.transform(b,p,y,vi);cfg={"hidden_dim":8,"attention_heads":2,"dropout":.1,"batch_size":32,"learning_rate":.001,"weight_decay":.0001,"max_epochs":2,"early_stopping_patience":2};_,prob,_,auc=fit(tr,va,"full_model",cfg,3,5);self.assertEqual(len(prob),len(vi));self.assertTrue(((prob>=0)&(prob<=1)).all());self.assertGreaterEqual(auc,0)
if __name__=="__main__":unittest.main()
