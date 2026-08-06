#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd
from catboost import CatBoostClassifier

MODEL_DIR = Path("models/squeeze_v1")
OUTPUT_DIR = Path("reports")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(MODEL_DIR / "features.json") as f:
    FEATURE_COLUMNS = json.load(f)

model = CatBoostClassifier()
model.load_model(str(MODEL_DIR / "model.cbm"))

df = pd.read_parquet(
    "data/research/topology_v2_squeeze_events/squeeze_event_dataset.parquet"
)

time_col = None

for candidate in [
    "event_time",
    "timestamp",
    "ts",
    "datetime",
]:
    if candidate in df.columns:
        time_col = candidate
        break

if time_col is None:
    raise RuntimeError("No timestamp column found")

latest_df = (
    df
    .sort_values(time_col)
    .groupby("symbol", as_index=False)
    .tail(1)
    .copy()
)

X = latest_df[FEATURE_COLUMNS]

proba = model.predict_proba(X)

classes = list(model.classes_)

event_cols = [
    i
    for i, cls in enumerate(classes)
    if cls != 0
]

scores = proba[:, event_cols].sum(axis=1)

preds = model.predict(X)

results = []

for idx in range(len(latest_df)):

    pred = preds[idx]

    if hasattr(pred, "__len__"):
        pred = pred[0]

    if pred == -1:
        label = "LONG_SQUEEZE"
    elif pred == 1:
        label = "SHORT_SQUEEZE"
    else:
        label = "NO_EVENT"

    results.append({
        "symbol": str(latest_df.iloc[idx]["symbol"]),
        "score": round(float(scores[idx]), 6),
        "prediction": label,
        "timestamp": str(latest_df.iloc[idx][time_col]),
    })

results.sort(
    key=lambda x: x["score"],
    reverse=True
)

for rank, row in enumerate(results, start=1):

    row["rank"] = rank

    if rank == 1:
        row["status"] = "CRITICAL"
    elif rank == 2:
        row["status"] = "ALERT"
    elif rank == 3:
        row["status"] = "WATCH"
    else:
        row["status"] = "NORMAL"

payload = {
    "engine": "textara-squeeze-v1",
    "generated_at": datetime.now(
        timezone.utc
    ).isoformat(),
    "symbol_count": len(results),
    "radar": results,
}

output_file = OUTPUT_DIR / "latest_radar.json"

with open(output_file, "w") as f:
    json.dump(
        payload,
        f,
        indent=2
    )

print()
print("RADAR JSON EXPORTED")
print(output_file)
print()

print(
    json.dumps(
        payload,
        indent=2
    )
)
