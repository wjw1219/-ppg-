from pathlib import Path

import pandas as pd

from illustrative import generate_predictions, output_paths


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "illustrative_simulation"
source = pd.read_csv(ROOT / "results" / "oof_predictions.csv")
labels = (source[source.condition == "fusion_full"][["patient_id", "label"]]
          .sort_values("patient_id").reset_index(drop=True))
predictions = generate_predictions(labels)
paths = output_paths(TARGET)
for path in paths.values():
    path.mkdir(parents=True, exist_ok=True)
predictions.to_csv(paths["results"] / "oof_predictions.csv", index=False)
print(f"saved {len(predictions)} illustrative patient-condition predictions")
