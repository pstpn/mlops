from pathlib import Path

from loguru import logger
import pandas as pd
from scipy.io import arff
import typer

from mlops.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset",
    output_dir: Path = PROCESSED_DATA_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
):
    logger.info(f"Loading dataset from {input_path}")
    data, _ = arff.loadarff(input_path)
    df = pd.DataFrame(data)

    if "Class" not in df.columns:
        raise ValueError("Expected target column 'Class' in dataset")

    df["Class"] = pd.to_numeric(df["Class"]).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    major = df[df["Class"] == 0]
    minor = df[df["Class"] == 1]

    train_major = major.sample(frac=1 - test_size, random_state=random_state)
    test_major = major.drop(train_major.index)

    train_minor = minor.sample(frac=1 - test_size, random_state=random_state)
    test_minor = minor.drop(train_minor.index)

    train_df = (
        pd.concat([train_major, train_minor])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat([test_major, test_minor])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    logger.info(
        f"Class share (train/test): {train_df['Class'].mean():.4f} / {test_df['Class'].mean():.4f}"
    )
    logger.success(f"Saved split datasets to {output_dir}")


if __name__ == "__main__":
    app()