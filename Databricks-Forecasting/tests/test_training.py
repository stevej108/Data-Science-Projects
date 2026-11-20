import pandas as pd
import pytest

from databricks_forecasting.config import (
    MODEL_FEATURES,
)

from databricks_forecasting.training import (
    create_modeling_dataset,
    create_train_valid_split,
    score_model,
)


class FakeModel:

    def predict(
        self,
        X,
    ):

        return (
            X["Traffic_Lag_1"]
            .to_numpy()
        )


def create_training_panel():

    rows = []

    for month in pd.date_range(
        "2026-01-01",
        periods=6,
        freq="MS",
    ):

        row = {
            feature: 1.0
            for feature in MODEL_FEATURES
        }

        row.update({
            "DMA": "Region_001",
            "Month": month,
            "Traffic": 100.0,
            "Traffic_Lag_1": 90.0,
        })

        rows.append(row)

    return pd.DataFrame(rows)


def test_create_modeling_dataset():

    panel = create_training_panel()

    result = create_modeling_dataset(
        panel
    )

    assert len(result) == 6

    assert (
        result["Month"]
        .is_monotonic_increasing
    )


def test_missing_feature_rejected():

    panel = (
        create_training_panel()
        .drop(
            columns=[
                "Traffic_Lag_1"
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):

        create_modeling_dataset(
            panel
        )


def test_temporal_split():

    panel = create_training_panel()

    model_df = (
        create_modeling_dataset(
            panel
        )
    )

    train, valid = (
        create_train_valid_split(
            model_df,
            cutoff_date="2026-05-01",
        )
    )

    assert len(train) == 4
    assert len(valid) == 2

    assert (
        train["Month"].max()
        < valid["Month"].min()
    )


def test_invalid_cutoff_rejected():

    panel = create_training_panel()

    model_df = (
        create_modeling_dataset(
            panel
        )
    )

    with pytest.raises(
        ValueError,
        match="Training dataset is empty",
    ):

        create_train_valid_split(
            model_df,
            cutoff_date="2025-01-01",
        )


def test_score_model():

    panel = create_training_panel()

    model_df = (
        create_modeling_dataset(
            panel
        )
    )

    _, valid = (
        create_train_valid_split(
            model_df,
            cutoff_date="2026-05-01",
        )
    )

    predictions, metrics = (
        score_model(
            FakeModel(),
            valid,
        )
    )

    assert len(predictions) == 2

    assert (
        predictions["Forecast"]
        == 90.0
    ).all()

    assert metrics["MAE"] == 10.0