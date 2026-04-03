from functools import lru_cache
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
import joblib
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel

from mlops.config import MODELS_DIR

DEFAULT_MODEL_PATH = MODELS_DIR / "fraud_model.joblib"

app = FastAPI(title="Fraud Detection API", version="1.0.0")


class TransactionFeatures(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class PredictionResponse(BaseModel):
    prediction: int
    fraud_probability: float


@lru_cache(maxsize=1)
def load_model():
    model_uri = os.getenv("MODEL_URI")
    if model_uri:
        return mlflow.pyfunc.load_model(model_uri)

    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train model with DVC pipeline or set MODEL_URI."
        )
    return joblib.load(model_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TransactionFeatures) -> PredictionResponse:
    model = load_model()
    features_df = pd.DataFrame([payload.model_dump()])

    try:
        if hasattr(model, "predict_proba"):
            prediction = int(model.predict(features_df)[0])
            fraud_probability = float(model.predict_proba(features_df)[0][1])
        else:
            prediction_raw = model.predict(features_df)
            # mlflow.pyfunc usually returns a pandas object
            prediction = int(prediction_raw.iloc[0] if hasattr(prediction_raw, "iloc") else prediction_raw[0])
            fraud_probability = float(prediction)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error

    return PredictionResponse(prediction=prediction, fraud_probability=fraud_probability)
