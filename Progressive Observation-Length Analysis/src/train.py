import copy,random
import numpy as np,torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from .data import FoldPreprocessor
from .models import ComparisonModel

def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def threshold(y,p):
    best,bj=.5,-1
    for t in np.unique(np.r_[0,p,1]):
        q=p>=t; tp=((q==1)&(y==1)).sum();fn=((q==0)&(y==1)).sum();tn=((q==0)&(y==0)).sum();fp=((q==1)&(y==0)).sum();j=tp/max(tp+fn,1)+tn/max(tn+fp,1)-1
        if j>bj:best,bj=float(t),j
    return best
def fit(train,val,kind,cfg,seed,weeks):
    seed_all(seed); dims=[b-a for a,b in train.group_slices]; model=ComparisonModel(kind,train.clinical.shape[1],dims,weeks,h=cfg["hidden_dim"],heads=cfg["attention_heads"],drop=cfg["dropout"])
    tc,tp,tm,ty=[torch.tensor(x,dtype=torch.float32) for x in (train.clinical,train.ppg,train.mask,train.labels)]; vc,vp,vm=[torch.tensor(x,dtype=torch.float32) for x in (val.clinical,val.ppg,val.mask)]
    loader=DataLoader(TensorDataset(tc,tp,tm,ty),batch_size=cfg["batch_size"],shuffle=True,generator=torch.Generator().manual_seed(seed)); pos=train.labels.sum(); loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(train.labels)-pos)/max(pos,1)])); opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"]); best=-1; state=None; wait=0
    for ep in range(cfg["max_epochs"]):
        model.train()
        for c,p,m,y in loader:
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(c,p,m),y); loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        model.eval();
        with torch.no_grad():vp0=torch.sigmoid(model(vc,vp,vm)).numpy()
        auc=roc_auc_score(val.labels,vp0)
        if auc>best+1e-5:best=auc;state=copy.deepcopy(model.state_dict());wait=0
        else:wait+=1
        if wait>=cfg["early_stopping_patience"]:break
    model.load_state_dict(state);model.eval();
    with torch.no_grad():vp0=torch.sigmoid(model(vc,vp,vm)).numpy()
    return model,vp0,ep+1,best
def run_condition(name,weeks,kind,b,p,y,folds,cfg):
    ids=b.index.tolist(); probs={i:[] for i in ids}; thrs={i:[] for i in ids}; logs=[]
    for fold in range(1,cfg["outer_folds"]+1):
        test=[i for i in ids if int(folds.loc[i])==fold]; outer=[i for i in ids if int(folds.loc[i])!=fold]; a,z=next(StratifiedShuffleSplit(n_splits=1,test_size=cfg["validation_fraction"],random_state=fold).split(np.zeros(len(outer)),y.loc[outer])); ti=[outer[i] for i in a]; vi=[outer[i] for i in z]; prep=FoldPreprocessor(weeks).fit(b,p,ti); tr=prep.transform(b,p,y,ti); va=prep.transform(b,p,y,vi); te=prep.transform(b,p,y,test)
        for seed in cfg["training_seeds"]:
            model,vp,epochs,vauc=fit(tr,va,kind,cfg,seed,weeks); t=threshold(va.labels,vp);model.eval();
            with torch.no_grad():testp=torch.sigmoid(model(torch.tensor(te.clinical),torch.tensor(te.ppg),torch.tensor(te.mask))).numpy()
            for pid,prob in zip(test,testp):probs[pid].append(float(prob));thrs[pid].append(t)
            logs.append({"condition":name,"weeks":weeks,"model":kind,"fold":fold,"seed":seed,"validation_auc":vauc,"epochs":epochs,"threshold":t,"test_n":len(test)})
    rows=[{"patient_id":i,"condition":name,"weeks":weeks,"model":kind,"label":int(y.loc[i]),"probability":float(np.mean(probs[i])),"threshold":float(np.mean(thrs[i]))} for i in ids]; return rows,logs
