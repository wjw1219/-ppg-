import json
from pathlib import Path
import pandas as pd
from src.data import load_common, load_scale
from src.train import run_scale

ROOT=Path(__file__).resolve().parent; INPUT=ROOT.parent/"outputs/evoaf_synthetic_387/intermediate"
cfg=json.loads((ROOT/"config.json").read_text()); baseline,labels,folds=load_common(INPUT)
preds=[]; fold_rows=[]
for scale,meta in cfg["scales"].items():
    ppg=load_scale(INPUT,meta["file"],meta["nodes"]); p,f=run_scale(scale,meta["nodes"],baseline,ppg,labels,folds,cfg); preds.extend(p); fold_rows.extend(f)
out=ROOT/"results"; out.mkdir(exist_ok=True)
pd.DataFrame(preds).to_csv(out/"oof_predictions.csv",index=False); pd.DataFrame(fold_rows).to_csv(out/"fold_metrics.csv",index=False)
(out/"run_config.json").write_text(json.dumps(cfg,indent=2)); print(f"saved {len(preds)} predictions and {len(fold_rows)} fold-seed records")
