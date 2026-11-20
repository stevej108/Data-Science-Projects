import pandas as pd

from xgboost import XGBRegressor

from databricks_forecasting.config import (
    MODEL_FEATURES,
)

from databricks_forecasting.evaluation import (
    calculate_metrics,
)


def create_modeling_dataset(
    panel: pd.DataFrame,
) -> pd.DataFrame:

    required_columns = (
        MODEL_FEATURES
        + [
            "Traffic",
            "DMA",
            "Month",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in panel.columns
    ]

    if missing_columns:
        raise ValueError(
            "Modeling dataset is missing "
            f"required columns: {missing_columns}"
        )

    model_df = (
        panel[required_columns]
        .copy()
    )

    model_df["Month"] = pd.to_datetime(
        model_df["Month"]
    )

    model_df = (
        model_df
        .sort_values(
            [
                "Month",
                "DMA",
            ]
        )
        .reset_index(drop=True)
    )

    model_df = (
        model_df
        .dropna()
        .reset_index(drop=True)
    )

    if model_df.empty:
        raise ValueError(
            "Modeling dataset is empty "
            "after removing missing values."
        )

    return model_df


def create_train_valid_split(
    model_df: pd.DataFrame,
    cutoff_date,
):

    cutoff_date = pd.Timestamp(
        cutoff_date
    )

    train = (
        model_df[
            model_df["Month"]
            < cutoff_date
        ]
        .copy()
    )

    valid = (
        model_df[
            model_df["Month"]
            >= cutoff_date
        ]
        .copy()
    )

    if train.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if valid.empty:
        raise ValueError(
            "Validation dataset is empty."
        )

    if (
        train["Month"].max()
        >= valid["Month"].min()
    ):
        raise ValueError(
            "Training and validation "
            "periods overlap."
        )

    return train, valid


def train_xgboost(
    train: pd.DataFrame,
    params: dict,
):

    X_train = (
        train[
            MODEL_FEATURES
        ]
    )

    y_train = (
        train["Traffic"]
    )

    model = XGBRegressor(
        **params
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def score_model(
    model,
    valid: pd.DataFrame,
):

    X_valid = (
        valid[
            MODEL_FEATURES
        ]
    )

    predictions = (
        model.predict(
            X_valid
        )
    )

    result = (
        valid[
            [
                "DMA",
                "Month",
            ]
        ]
        .copy()
    )

    result["Actual"] = (
        valid["Traffic"]
        .to_numpy()
    )

    result["Forecast"] = (
        predictions
    )

    result["Error"] = (
        result["Forecast"]
        - result["Actual"]
    )

    result["Absolute_Error"] = (
        result["Error"]
        .abs()
    )

    metrics = calculate_metrics(
        actual=result["Actual"],
        forecast=result["Forecast"],
    )

    return (
        result,
        metrics,
    )