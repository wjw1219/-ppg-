import copy, random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .data import FoldPreprocessor
from .models import AblationModel

def set_seed(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def youden_threshold(y, p):
    best_t, best_j = .5, -1
    for t in np.unique(np.r_[0, p, 1]):
        pred = p >= t; tp=((pred==1)&(y==1)).sum(); fn=((pred==0)&(y==1)).sum(); tn=((pred==0)&(y==0)).sum(); fp=((pred==1)&(y==0)).sum()
        j = tp/max(tp+fn,1) + tn/max(tn+fp,1) - 1
        if j > best_j: best_t, best_j = float(t), j
    return best_t

def train_one(train, val, cfg, seed, nodes):
    set_seed(seed); dims=[b-a for a,b in train.group_slices]
    model=AblationModel("fusion",dims,hidden_dim=cfg["hidden_dim"],heads=cfg["attention_heads"],dropout=cfg["dropout"],weeks=nodes)
    tc,tp,tm,ty=map(lambda x:torch.tensor(x,dtype=torch.float32),(train.clinical,train.ppg,train.mask,train.labels))
    vc,vp,vm=map(lambda x:torch.tensor(x,dtype=torch.float32),(val.clinical,val.ppg,val.mask))
    loader=DataLoader(TensorDataset(tc,tp,tm,ty),batch_size=cfg["batch_size"],shuffle=True,generator=torch.Generator().manual_seed(seed))
    pos=train.labels.sum(); loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(train.labels)-pos)/max(pos,1)]))
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    best_auc,best_state,patience=-1,None,0
    for epoch in range(cfg["max_epochs"]):
        model.train()
        for cb,pb,mb,yb in loader:
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(cb,pb,mb),yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2); opt.step()
        model.eval()
        with torch.no_grad(): prob=torch.sigmoid(model(vc,vp,vm)).numpy()
        auc=roc_auc_score(val.labels,prob)
        if auc > best_auc + 1e-5: best_auc,best_state,patience=auc,copy.deepcopy(model.state_dict()),0
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): val_prob=torch.sigmoid(model(vc,vp,vm)).numpy()
    return model,val_prob,epoch+1,best_auc

def run_scale(scale, nodes, baseline, ppg, labels, folds, cfg):
    ids=baseline.index.tolist(); probs={p:[] for p in ids}; thresholds={p:[] for p in ids}; fold_rows=[]
    for fold in range(1,cfg["outer_folds"]+1):
        test_ids=[p for p in ids if int(folds.loc[p])==fold]; outer_train=[p for p in ids if int(folds.loc[p])!=fold]
        split=StratifiedShuffleSplit(n_splits=1,test_size=cfg["validation_fraction"],random_state=fold)
        ai,bi=next(split.split(np.zeros(len(outer_train)),labels.loc[outer_train])); fit_ids=[outer_train[i] for i in ai]; val_ids=[outer_train[i] for i in bi]
        prep=FoldPreprocessor(nodes).fit(baseline,ppg,fit_ids)
        train=prep.transform(baseline,ppg,labels,fit_ids); val=prep.transform(baseline,ppg,labels,val_ids); test=prep.transform(baseline,ppg,labels,test_ids)
        for seed in cfg["training_seeds"]:
            model,val_prob,epochs,val_auc=train_one(train,val,cfg,seed,nodes); threshold=youden_threshold(val.labels,val_prob)
            model.eval()
            with torch.no_grad(): test_prob=torch.sigmoid(model(torch.tensor(test.clinical),torch.tensor(test.ppg),torch.tensor(test.mask))).numpy()
            for pid,p in zip(test_ids,test_prob): probs[pid].append(float(p)); thresholds[pid].append(threshold)
            fold_rows.append({"scale":scale,"fold":fold,"seed":seed,"validation_auc":val_auc,"epochs":epochs,"threshold":threshold,"test_n":len(test_ids)})
    rows=[{"patient_id":p,"scale":scale,"label":int(labels.loc[p]),"probability":float(np.mean(probs[p])),"threshold":float(np.mean(thresholds[p]))} for p in ids]
    return rows,fold_rows
