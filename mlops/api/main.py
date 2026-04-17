from functools import lru_cache
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
import joblib
import mlflow.pyfunc
import pandas as pd
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from mlops.config import MODELS_DIR

DEFAULT_MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
FEATURE_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]

app = FastAPI(title="Fraud Detection API", version="1.0.0")


class TransactionFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_vector: list[float] | None = Field(
        default=None,
        validation_alias=AliasChoices("feature_vector", "features"),
    )
    Time: float | None = None
    V1: float | None = None
    V2: float | None = None
    V3: float | None = None
    V4: float | None = None
    V5: float | None = None
    V6: float | None = None
    V7: float | None = None
    V8: float | None = None
    V9: float | None = None
    V10: float | None = None
    V11: float | None = None
    V12: float | None = None
    V13: float | None = None
    V14: float | None = None
    V15: float | None = None
    V16: float | None = None
    V17: float | None = None
    V18: float | None = None
    V19: float | None = None
    V20: float | None = None
    V21: float | None = None
    V22: float | None = None
    V23: float | None = None
    V24: float | None = None
    V25: float | None = None
    V26: float | None = None
    V27: float | None = None
    V28: float | None = None
    Amount: float | None = None

    @field_validator("feature_vector")
    @classmethod
    def validate_feature_vector(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != len(FEATURE_COLUMNS):
            raise ValueError(
                f"feature_vector must contain exactly {len(FEATURE_COLUMNS)} values"
            )
        return value

    @model_validator(mode="after")
    def populate_scalar_features(self) -> "TransactionFeatures":
        if self.feature_vector is not None:
            for column_name, value in zip(
                FEATURE_COLUMNS,
                self.feature_vector,
                strict=True,
            ):
                if getattr(self, column_name) is None:
                    setattr(self, column_name, value)

        missing_fields = [
            column_name
            for column_name in FEATURE_COLUMNS
            if getattr(self, column_name) is None
        ]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        return self

    def to_feature_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{column_name: getattr(self, column_name) for column_name in FEATURE_COLUMNS}]
        )


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
    features_df = payload.to_feature_frame()

    try:
        if hasattr(model, "predict_proba"):
            prediction = int(model.predict(features_df)[0])
            fraud_probability = float(model.predict_proba(features_df)[0][1])
        else:
            prediction_raw = model.predict(features_df)
            # mlflow.pyfunc usually returns a pandas object
            prediction = int(
                prediction_raw.iloc[0]
                if hasattr(prediction_raw, "iloc")
                else prediction_raw[0]
            )
            fraud_probability = float(prediction)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error

    return PredictionResponse(prediction=prediction, fraud_probability=fraud_probability)
