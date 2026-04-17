from pathlib import Path

import pandas as pd
import pytest

from mlops.preprocess import (
    drop_duplicate_rows,
    main,
    read_dataset,
    save_preprocessed_artifacts,
    split_features_and_target,
    validate_dataset,
)


def make_frame(
    *,
    amount_values: list[object] | None = None,
    class_values: list[int] | None = None,
) -> pd.DataFrame:
    amount_values = amount_values or [10.0, 20.0]
    class_values = class_values or [0, 1]
    return pd.DataFrame(
        {
            "Time": [1.0, 2.0],
            "V1": [0.1, 0.2],
            "Amount": amount_values,
            "Class": class_values,
        }
    )


def test_read_dataset_reads_csv(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    expected = make_frame()
    expected.to_csv(source, index=False)

    loaded = read_dataset(source)

    pd.testing.assert_frame_equal(loaded, expected)


@pytest.mark.parametrize(
    ("frame", "frame_name", "expected_message"),
    [
        (pd.DataFrame(), "train", "train dataset is empty"),
        (
            pd.DataFrame({"Time": [1.0], "Amount": [10.0], "Class": [0], "V1": [None]}),
            "test",
            "contains columns with only missing values",
        ),
        (
            pd.DataFrame({"Time": [1.0], "Amount": ["abc"], "Class": [0], "V1": [0.1]}),
            "train",
            "must be numeric",
        ),
    ],
)
def test_validate_dataset_rejects_bad_inputs(
    frame: pd.DataFrame,
    frame_name: str,
    expected_message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=expected_message):
        validate_dataset(frame, frame_name)


def test_validate_dataset_accepts_single_class() -> None:
    frame = make_frame(class_values=[1, 1])

    validate_dataset(frame, "train")


def test_drop_duplicate_rows_removes_duplicates() -> None:
    frame = pd.concat([make_frame().iloc[[0]], make_frame().iloc[[0]]], ignore_index=True)

    deduplicated = drop_duplicate_rows(frame)

    assert len(deduplicated) == 1


def test_split_features_and_target_handles_single_class() -> None:
    frame = make_frame(class_values=[1, 1])

    features, target, feature_columns = split_features_and_target(frame)

    assert feature_columns == ["Time", "V1", "Amount"]
    assert list(features.columns) == feature_columns
    assert target.nunique() == 1
    assert target.iloc[0] == 1


def test_save_preprocessed_artifacts_writes_files(tmp_path: Path) -> None:
    frame = make_frame()
    features, target, feature_columns = split_features_and_target(frame)

    save_preprocessed_artifacts(features, target, features, target, feature_columns, tmp_path)

    assert (tmp_path / "train_features.csv").exists()
    assert (tmp_path / "train_target.csv").exists()
    assert (tmp_path / "test_features.csv").exists()
    assert (tmp_path / "test_target.csv").exists()
    assert (
        (tmp_path / "feature_columns.json").read_text(encoding="utf-8")
        == '["Time", "V1", "Amount"]'
    )


def test_main_creates_expected_artifacts(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    output_dir = tmp_path / "out"

    make_frame().to_csv(train_path, index=False)
    make_frame(class_values=[1, 1]).to_csv(test_path, index=False)

    main(train_path=train_path, test_path=test_path, output_dir=output_dir)

    assert (output_dir / "train_features.csv").exists()
    assert (output_dir / "test_features.csv").exists()
    assert (output_dir / "feature_columns.json").exists()