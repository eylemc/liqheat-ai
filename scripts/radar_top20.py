#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd
from catboost import CatBoostClassifier

MODEL_DIR = Path("models/squeeze_v1")

with open(MODEL_DIR / "features.json") as f:
    FEATURE_COLUMNS = json.load(f)

model = CatBoostClassifier()
model.load_model(str(MODEL_DIR / "model.cbm"))

df = pd.read_parquet(
    "data/research/topology_v2_squeeze_events/squeeze_event_dataset.parquet"
)

time_col = "event_time"

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
    i for i, cls in enumerate(classes)
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
        "symbol": latest_df.iloc[idx]["symbol"],
        "score": float(scores[idx]),
        "prediction": label
    })

results.sort(
    key=lambda x: x["score"],
    reverse=True
)

for rank, row in enumerate(results, start=1):

    if rank == 1:
        row["status"] = "CRITICAL"
    elif rank == 2:
        row["status"] = "ALERT"
    elif rank == 3:
        row["status"] = "WATCH"
    else:
        row["status"] = "NORMAL"

print()
print("TEXTARA RADAR")
print()

print(
    f'{"RANK":<6}'
    f'{"SYMBOL":<12}'
    f'{"SCORE":<10}'
    f'{"STATUS":<10}'
    f'{"PREDICTION":<20}'
)

print("-" * 72)

for rank, item in enumerate(results, start=1):

    print(
        f'{rank:<6}'
        f'{item["symbol"]:<12}'
        f'{item["score"]:<10.4f}'
        f'{item["status"]:<10}'
        f'{item["prediction"]:<20}'
    )
