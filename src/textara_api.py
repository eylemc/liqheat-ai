from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI

app = FastAPI(
    title="Textara AI Engine",
    version="0.1"
)

MODEL_DIR = Path("models/squeeze_v1")

with open(MODEL_DIR / "features.json") as f:
    FEATURE_COLUMNS = json.load(f)

model = CatBoostClassifier()
model.load_model(str(MODEL_DIR / "model.cbm"))

DATASET_PATH = (
    "data/research/topology_v2_squeeze_events/"
    "squeeze_event_dataset.parquet"
)


def build_radar():

    df = pd.read_parquet(DATASET_PATH)

    latest_df = (
        df
        .sort_values("logged_at")
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
            "logged_at": str(latest_df.iloc[idx]["logged_at"]),
            "score": round(float(scores[idx]), 6),
            "prediction": label,
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

    return {
        "engine": "textara-squeeze-v1",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "symbol_count": len(results),
        "radar": results,
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/radar")
def radar():
    return build_radar()
