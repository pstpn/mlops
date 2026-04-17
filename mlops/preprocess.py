import json
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mlops.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_dataset(frame: pd.DataFrame, frame_name: str) -> None:
    if frame.empty:
        raise ValueError(f"{frame_name} dataset is empty")

    if "Class" not in frame.columns:
        raise ValueError(f"Column 'Class' not found in {frame_name} dataset")

    if "Amount" not in frame.columns:
        raise ValueError(f"Column 'Amount' not found in {frame_name} dataset")

    empty_columns = [column for column in frame.columns if frame[column].isna().all()]
    if empty_columns:
        raise ValueError(
            f"Dataset {frame_name} contains columns with only missing values: {', '.join(empty_columns)}"
        )

    if not pd.api.types.is_numeric_dtype(frame["Amount"]):
        raise TypeError(f"Column 'Amount' in {frame_name} dataset must be numeric")


def drop_duplicate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop_duplicates().reset_index(drop=True)


def split_features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_columns = [column for column in frame.columns if column != "Class"]
    features = frame[feature_columns]
    target = frame["Class"].astype(int)
    return features, target, feature_columns


def save_preprocessed_artifacts(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train.to_csv(output_dir / "train_features.csv", index=False)
    y_train.to_csv(output_dir / "train_target.csv", index=False)
    x_test.to_csv(output_dir / "test_features.csv", index=False)
    y_test.to_csv(output_dir / "test_target.csv", index=False)

    with (output_dir / "feature_columns.json").open("w", encoding="utf-8") as file_obj:
        json.dump(feature_columns, file_obj)


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    output_dir: Path = INTERIM_DATA_DIR,
):
    logger.info(f"Reading train data from {train_path}")
    train_df = read_dataset(train_path)
    logger.info(f"Reading test data from {test_path}")
    test_df = read_dataset(test_path)

    for frame_name, frame in [("train", train_df), ("test", test_df)]:
        validate_dataset(frame, frame_name)

    train_df = drop_duplicate_rows(train_df)
    test_df = drop_duplicate_rows(test_df)

    x_train, y_train, feature_columns = split_features_and_target(train_df)
    x_test, y_test, _ = split_features_and_target(test_df)

    save_preprocessed_artifacts(x_train, y_train, x_test, y_test, feature_columns, output_dir)

    logger.info(
        "Prepared data with shapes: "
        f"x_train={x_train.shape}, y_train={y_train.shape}, "
        f"x_test={x_test.shape}, y_test={y_test.shape}"
    )
    logger.success(f"Saved preprocessed artifacts to {output_dir}")


if __name__ == "__main__":
    app()