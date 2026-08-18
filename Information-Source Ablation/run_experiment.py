import json
from pathlib import Path

import pandas as pd

from src.data import load_raw
from src.train import run_all


ROOT = Path(__file__).resolve().parent
cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
baseline, ppg, labels, folds = load_raw(ROOT.parent / "outputs/evoaf_synthetic_387/intermediate")
preds, fold_metrics, logs = run_all(baseline, ppg, labels, folds, cfg)
out = ROOT / "results"
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(preds).to_csv(out / "oof_predictions.csv", index=False)
pd.DataFrame(fold_metrics).to_csv(out / "fold_metrics.csv", index=False)
pd.DataFrame(logs).to_csv(out / "training_log.csv", index=False)
(out / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(f"saved {len(preds)} patient-condition predictions and {len(fold_metrics)} fold-seed records")
