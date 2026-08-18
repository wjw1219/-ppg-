import json
from pathlib import Path

import pandas as pd

from src.data import load_common, load_ppg
from src.train import run_condition


ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent / "outputs/evoaf_synthetic_387/intermediate"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CFG = json.loads((ROOT / "config.json").read_text())


def read_rows(path):
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def save(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


baseline, labels, folds = load_common(INPUT)
ppg = load_ppg(INPUT)
weekly_path = OUT / "weekly_oof_predictions.csv"
model_path = OUT / "model_oof_predictions.csv"
log_path = OUT / "fold_metrics.csv"
weekly = read_rows(weekly_path)
models = read_rows(model_path)
logs = read_rows(log_path)

completed_weekly = {int(row["weeks"]) for row in weekly}
for weeks in CFG["prefix_weeks"]:
    if weeks in completed_weekly:
        print(f"skip completed full_model week {weeks}", flush=True)
        continue
    print(f"start full_model week {weeks}", flush=True)
    rows, condition_logs = run_condition(
        f"full_model_week_{weeks}", weeks, "full_model",
        baseline, ppg, labels, folds, CFG
    )
    weekly += rows
    logs += condition_logs
    save(weekly, weekly_path)
    save(logs, log_path)
    print(f"saved full_model week {weeks}", flush=True)

completed_models = {str(row["model"]) for row in models}
for kind in CFG["comparison_models"]:
    if kind in completed_models:
        print(f"skip completed model {kind}", flush=True)
        continue
    print(f"start model {kind}", flush=True)
    if kind == "full_model":
        rows = [row for row in weekly if int(row["weeks"]) == 26]
    else:
        rows, condition_logs = run_condition(
            kind, 26, kind, baseline, ppg, labels, folds, CFG
        )
        logs += condition_logs
        save(logs, log_path)
    models += rows
    save(models, model_path)
    print(f"saved model {kind}", flush=True)

(OUT / "run_config.json").write_text(json.dumps(CFG, indent=2))
print(f"weekly={len(weekly)} models={len(models)} fits={len(logs)}", flush=True)
