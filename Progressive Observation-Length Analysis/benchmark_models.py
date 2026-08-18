import json
import time
from pathlib import Path

import pandas as pd
import psutil
import torch

from src.data import FoldPreprocessor, load_common, load_ppg
from src.models import ComparisonModel

ROOT = Path(__file__).resolve().parent
cfg = json.loads((ROOT / "config.json").read_text())
inp = ROOT.parent / "outputs/evoaf_synthetic_387/intermediate"
b, y, folds = load_common(inp); p = load_ppg(inp)
ids = b.index.tolist(); prep = FoldPreprocessor(26).fit(b, p, ids)
d = prep.transform(b, p, y, ids[:1])
c = torch.tensor(d.clinical); x = torch.tensor(d.ppg); m = torch.tensor(d.mask)
rows = []
for kind in cfg["comparison_models"]:
    model = ComparisonModel(kind, c.shape[1], [z-a for a,z in d.group_slices], 26,
                            h=cfg["hidden_dim"], heads=cfg["attention_heads"],
                            drop=cfg["dropout"]).eval()
    with torch.no_grad():
        for _ in range(20): model(c, x, m)
        start_mem = psutil.Process().memory_info().rss
        times = []
        for _ in range(100):
            t = time.perf_counter(); model(c, x, m); times.append((time.perf_counter()-t)*1000)
        peak = psutil.Process().memory_info().rss
    rows.append({"model": kind, "parameters": sum(v.numel() for v in model.parameters()),
                 "cpu_single_patient_ms_median": float(pd.Series(times).median()),
                 "cpu_single_patient_ms_iqr": float(pd.Series(times).quantile(.75)-pd.Series(times).quantile(.25)),
                 "process_rss_mb": peak / 1024**2,
                 "rss_increment_mb": max(0, peak-start_mem) / 1024**2,
                 "gpu_memory": "not measured (CPU environment)"})
pd.DataFrame(rows).to_csv(ROOT / "results/resource_usage.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
