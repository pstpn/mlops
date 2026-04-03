import json
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mlops.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    output_dir: Path = INTERIM_DATA_DIR,
):
    logger.info(f"Reading train data from {train_path}")
    train_df = pd.read_csv(train_path)
    logger.info(f"Reading test data from {test_path}")
    test_df = pd.read_csv(test_path)

    for frame_name, frame in [("train", train_df), ("test", test_df)]:
        if "Class" not in frame.columns:
            raise ValueError(f"Column 'Class' not found in {frame_name} dataset")

    train_df = train_df.drop_duplicates().reset_index(drop=True)
    test_df = test_df.drop_duplicates().reset_index(drop=True)

    feature_columns = [column for column in train_df.columns if column != "Class"]
    x_train = train_df[feature_columns]
    y_train = train_df["Class"].astype(int)

    x_test = test_df[feature_columns]
    y_test = test_df["Class"].astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    x_train.to_csv(output_dir / "train_features.csv", index=False)
    y_train.to_csv(output_dir / "train_target.csv", index=False)
    x_test.to_csv(output_dir / "test_features.csv", index=False)
    y_test.to_csv(output_dir / "test_target.csv", index=False)

    with (output_dir / "feature_columns.json").open("w", encoding="utf-8") as file_obj:
        json.dump(feature_columns, file_obj)

    logger.info(
        "Prepared data with shapes: "
        f"x_train={x_train.shape}, y_train={y_train.shape}, "
        f"x_test={x_test.shape}, y_test={y_test.shape}"
    )
    logger.success(f"Saved preprocessed artifacts to {output_dir}")


if __name__ == "__main__":
    app()