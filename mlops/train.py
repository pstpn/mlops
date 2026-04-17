import json
from pathlib import Path

import joblib
from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from mlops.config import INTERIM_DATA_DIR, MODELS_DIR, REPORTS_DIR

app = typer.Typer()


@app.command()
def main(
    train_features_path: Path = INTERIM_DATA_DIR / "train_features.csv",
    train_target_path: Path = INTERIM_DATA_DIR / "train_target.csv",
    test_features_path: Path = INTERIM_DATA_DIR / "test_features.csv",
    test_target_path: Path = INTERIM_DATA_DIR / "test_target.csv",
    model_path: Path = MODELS_DIR / "fraud_model.joblib",
    metrics_path: Path = REPORTS_DIR / "metrics.json",
    experiment_name: str = "fraud-detection",
    tracking_uri: str = "http://127.0.0.1:5001",
    n_estimators: int = 400,
    max_depth: int = 12,
    min_samples_leaf: int = 1,
    random_state: int = 42,
):
    x_train = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_target_path).squeeze("columns").astype(int)
    x_test = pd.read_csv(test_features_path)
    y_test = pd.read_csv(test_target_path).squeeze("columns").astype(int)

    numeric_features = ["Time", "Amount"]
    preprocessor = ColumnTransformer(
        transformers=[("numeric", StandardScaler(), numeric_features)],
        remainder="passthrough",
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    logger.info("Training model")
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="random_forest_pipeline"):
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "random_state": random_state,
                "classifier": "RandomForestClassifier",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="bundle")

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as file_obj:
            json.dump(metrics, file_obj, indent=2)
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")

    logger.success(f"Model saved to {model_path}")
    logger.success(f"Metrics saved to {metrics_path}")
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    app()