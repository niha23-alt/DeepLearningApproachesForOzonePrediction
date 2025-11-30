import os
import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model


app = FastAPI(title="Ozone Prediction API", version="1.0.0")

# Allow local frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Cached artifacts
_predictions: List[dict] = []
_model_loaded: bool = False


def _to_date(year: int, doy: int) -> str:
    base = datetime.date(int(year), 1, 1)
    return (base + datetime.timedelta(days=int(doy) - 1)).isoformat()


def _prepare_predictions() -> None:
    global _predictions, _model_loaded

    # Load dataset
    csv_path = os.path.join("sample_dataset", "2020-Northwest_China_Ozone_data.csv")
    df = pd.read_csv(csv_path, index_col=0)

    # Select columns as in training script
    cols = [
        'O3', 'year',  'doy', 'dem1', 'dem2', 'dem3', 'dem4', 'dem5', 'dem6','dem7', 'dem8', 'dem9', 'dem91',
        'lu1', 'lu2', 'lu3', 'lu4', 'lu5', 'lu6', 'lu7', 'lu8', 'lu9', 'lu91',
        'aod51', 'aod52', 'aod53', 'aod54', 'aod55', 'aod56','aod57', 'aod58', 'aod59', 'aod591',
        'pop1', 'pop2', 'pop3', 'pop4', 'pop5', 'pop6', 'pop7', 'pop8', 'pop9', 'pop91',
        'ph1', 'ph2', 'ph3', 'ph4', 'ph5', 'ph6','ph7', 'ph8', 'ph9', 'ph91',
        'so1', 'so2', 'so3', 'so4', 'so5', 'so6', 'so7', 'so8', 'so9', 'so91',
        'o31', 'o32', 'o33', 'o34', 'o35', 'o36', 'o37', 'o38', 'o39', 'o391',
        'aod471', 'aod472', 'aod473', 'aod474', 'aod475', 'aod476','aod477', 'aod478', 'aod479', 'aod4791',
        'hm', 'pr', 'tem', 'ws',
        'ndvi1', 'ndvi2', 'ndvi3', 'ndvi4', 'ndvi5', 'ndvi6', 'ndvi7', 'ndvi8', 'ndvi9',
        'workd', 'osm1','osm2','osm3','osm4','osm5','osm6','osm7','osm8','osm9','osm91'
    ]

    df = df[cols]

    # Remove outliers consistent with training
    df = df.drop(df[(df['O3'] > 300)].index, axis=0)

    # Train/test split on 2020 data (already only 2020)
    train_set, test_set = train_test_split(df, test_size=0.20, random_state=39)

    scaler = MinMaxScaler()
    normalized_train = scaler.fit_transform(train_set.values)
    normalized_test = scaler.transform(test_set.values)

    test_x = normalized_test[:, 1:]
    test_y = normalized_test[:, 0]

    # Load trained model
    model_path = os.path.join('result', '2-conv1d-CNN-new_2025-11-30_15-59-41_33_0.055_0.005.h5')
    model = load_model(model_path, custom_objects={'mse': 'mse'})
    _model_loaded = True

    # Predict
    probs = model.predict(test_x, batch_size=32, verbose=0).flatten()

    # Inverse transform to original scale for label column
    predicted_norm = normalized_test.copy()
    predicted_norm[:, 0] = probs
    predicted = scaler.inverse_transform(predicted_norm)
    predicted_y = predicted[:, 0]

    # Build response records with date
    # Recover year/doy from the raw (unscaled) test_set
    test_df = test_set.copy()
    years = test_df['year'].to_numpy()
    doys = test_df['doy'].to_numpy()
    actual_y = test_df['O3'].to_numpy()

    records = []
    for i in range(len(test_df)):
        try:
            date_str = _to_date(int(years[i]), int(doys[i]))
        except Exception:
            date_str = str(years[i])
        records.append({
            'date': date_str,
            'actualValue': float(actual_y[i]),
            'predictedValue': float(predicted_y[i])
        })

    # Sort by date and keep cached list
    _predictions = sorted(records, key=lambda r: r['date'])


@app.get("/health")
def health():
    return {"status": "ok", "modelLoaded": _model_loaded, "count": len(_predictions)}


@app.get("/predict")
def get_predictions(
    limit: Optional[int] = Query(30, ge=1, le=500),
    start: Optional[str] = Query(None, description="ISO start date e.g. 2020-12-01"),
    end: Optional[str] = Query(None, description="ISO end date e.g. 2020-12-31"),
):
    if not _predictions:
        _prepare_predictions()
    data = _predictions
    if start:
        data = [r for r in data if r['date'] >= start]
    if end:
        data = [r for r in data if r['date'] <= end]
    return data[-limit:]


# Eagerly prepare on startup
@app.on_event("startup")
def startup_event():
    try:
        _prepare_predictions()
    except Exception as e:
        # Lazy initialization will try again on first request
        pass
