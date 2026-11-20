import pandas as pd

from databricks_forecasting.config import (
    STATIC_FEATURES,
)


def create_future_periods(
    panel: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:

    if panel.empty:
        raise ValueError(
            "Historical panel is empty."
        )

    required_columns = {
        "DMA",
        "Month",
    }

    missing_columns = (
        required_columns
        - set(panel.columns)
    )

    if missing_columns:
        raise ValueError(
            "Historical panel is missing "
            f"required columns: "
            f"{sorted(missing_columns)}"
        )

    if horizon <= 0:
        raise ValueError(
            "Forecast horizon must be positive."
        )

    panel = panel.copy()

    panel["Month"] = pd.to_datetime(
        panel["Month"]
    )

    latest_month = (
        panel["Month"].max()
    )

    future_months = pd.date_range(
        start=(
            latest_month
            + pd.DateOffset(months=1)
        ),
        periods=horizon,
        freq="MS",
    )

    dmas = (
        panel["DMA"]
        .dropna()
        .unique()
    )

    future = (
        pd.MultiIndex.from_product(
            [
                dmas,
                future_months,
            ],
            names=[
                "DMA",
                "Month",
            ],
        )
        .to_frame(
            index=False
        )
    )

    expected_rows = (
        len(dmas)
        * horizon
    )

    if len(future) != expected_rows:
        raise RuntimeError(
            "Future-period creation produced "
            "an unexpected row count."
        )

    return future


def populate_static_features(
    future: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:

    latest_attributes = (
        panel
        .sort_values("Month")
        .groupby("DMA")
        .tail(1)
        [STATIC_FEATURES]
        .copy()
    )

    future = future.merge(
        latest_attributes,
        on="DMA",
        how="left",
        validate="many_to_one",
    )

    feature_columns = [
        column
        for column in STATIC_FEATURES
        if column != "DMA"
    ]

    missing_counts = (
        future[feature_columns]
        .isna()
        .sum()
    )

    invalid_columns = (
        missing_counts[
            missing_counts > 0
        ]
        .to_dict()
    )

    if invalid_columns:
        raise ValueError(
            "Static feature population "
            "produced missing values: "
            f"{invalid_columns}"
        )

    return future


def add_calendar_features(
    future: pd.DataFrame,
) -> pd.DataFrame:

    future = future.copy()

    future["Month"] = pd.to_datetime(
        future["Month"]
    )

    future["Year"] = (
        future["Month"].dt.year
    )

    future["Month_Number"] = (
        future["Month"].dt.month
    )

    future["Quarter"] = (
        future["Month"].dt.quarter
    )

    future["Holiday_Season"] = (
        future["Month_Number"]
        .isin([11, 12])
        .astype(int)
    )

    if not (
        future["Year"]
        == future["Month"].dt.year
    ).all():
        raise AssertionError(
            "Year validation failed."
        )

    if not (
        future["Month_Number"]
        == future["Month"].dt.month
    ).all():
        raise AssertionError(
            "Month number validation failed."
        )

    if not (
        future["Quarter"]
        == future["Month"].dt.quarter
    ).all():
        raise AssertionError(
            "Quarter validation failed."
        )

    expected_holiday = (
        future["Month"].dt.month
        .isin([11, 12])
        .astype(int)
    )

    if not (
        future["Holiday_Season"]
        == expected_holiday
    ).all():
        raise AssertionError(
            "Holiday season validation failed."
        )

    return future