import json
from pathlib import Path

import pandas as pd

from src.data import load_common, load_scale
from src.train import run_scale


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data"
cfg = json.loads((ROOT / "config.json").read_text())
baseline, labels, folds = load_common(INPUT)
predictions, fold_rows = [], []
for scale, meta in cfg["scales"].items():
    ppg = load_scale(INPUT, meta["file"], meta["nodes"])
    scale_predictions, scale_folds = run_scale(
        scale, meta["nodes"], baseline, ppg, labels, folds, cfg
    )
    predictions.extend(scale_predictions)
    fold_rows.extend(scale_folds)

results = ROOT / "results"
results.mkdir(exist_ok=True)
pd.DataFrame(predictions).to_csv(results / "oof_predictions.csv", index=False)
pd.DataFrame(fold_rows).to_csv(results / "fold_metrics.csv", index=False)
(results / "run_config.json").write_text(json.dumps(cfg, indent=2))
print(f"saved {len(predictions)} predictions and {len(fold_rows)} fold-seed records")
