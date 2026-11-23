import pandas as pd


def create_future_periods(
    panel,
    horizon,
):
    """
    Create one future row per DMA and month.
    """

    if horizon <= 0:
        raise ValueError(
            "Forecast horizon must be "
            "greater than zero."
        )

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
            "required columns: "
            f"{sorted(missing_columns)}"
        )

    latest_month = (
        panel["Month"]
        .max()
    )

    future_months = (
        pd.date_range(
            start=(
                latest_month
                + pd.DateOffset(
                    months=1
                )
            ),
            periods=horizon,
            freq="MS",
        )
    )

    dmas = (
        panel["DMA"]
        .dropna()
        .unique()
    )

    if len(dmas) == 0:
        raise ValueError(
            "Historical panel contains "
            "no valid DMAs."
        )

    future = (
        pd.MultiIndex
        .from_product(
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

    return future


def populate_static_features(
    future,
    panel,
    static_features,
):
    """
    Populate each DMA's future rows with its
    most recently observed static attributes.
    """

    required_columns = (
        set(static_features)
        | {"Month"}
    )

    missing_columns = (
        required_columns
        - set(panel.columns)
    )

    if missing_columns:
        raise ValueError(
            "Historical panel is missing "
            "static feature columns: "
            f"{sorted(missing_columns)}"
        )

    latest_attributes = (
        panel
        .sort_values(
            [
                "DMA",
                "Month",
            ]
        )
        .groupby(
            "DMA",
            as_index=False,
        )
        .tail(1)
        [static_features]
        .copy()
    )

    populated = (
        future.merge(
            latest_attributes,
            on="DMA",
            how="left",
            validate="many_to_one",
        )
    )

    missing_dmas = (
        populated.loc[
            populated[
                static_features
            ]
            .isna()
            .all(axis=1),
            "DMA",
        ]
        .unique()
        .tolist()
    )

    if missing_dmas:
        raise ValueError(
            "Static features could not be "
            "populated for DMAs: "
            f"{missing_dmas}"
        )

    return populated


def add_calendar_features(
    future,
):
    """
    Add model calendar features.
    """

    future = future.copy()

    future["Month"] = (
        pd.to_datetime(
            future["Month"]
        )
    )

    future["Year"] = (
        future["Month"]
        .dt.year
    )

    future["Month_Number"] = (
        future["Month"]
        .dt.month
    )

    future["Quarter"] = (
        future["Month"]
        .dt.quarter
    )

    future["Holiday_Season"] = (
        future["Month_Number"]
        .isin(
            [
                11,
                12,
            ]
        )
        .astype(int)
    )

    return future


def prepare_future_frame(
    panel,
    horizon,
    static_features,
):
    """
    Build the complete future forecasting frame.
    """

    future = create_future_periods(
        panel=panel,
        horizon=horizon,
    )

    future = populate_static_features(
        future=future,
        panel=panel,
        static_features=(
            static_features
        ),
    )

    future = add_calendar_features(
        future
    )

    return future


def generate_batch_forecast(
    panel,
    future,
    champion_model,
    model_features,
):
    """
    Generate recursive monthly forecasts.

    Each prediction becomes Traffic history for
    the next forecast horizon. Transactions use
    the latest-observation assumption.
    """

    required_history_columns = {
        "DMA",
        "Month",
        "Traffic",
        "Transactions",
    }

    missing_columns = (
        required_history_columns
        - set(panel.columns)
    )

    if missing_columns:
        raise ValueError(
            "Historical panel is missing "
            "required columns: "
            f"{sorted(missing_columns)}"
        )

    working = (
        panel
        .copy()
        .sort_values(
            [
                "DMA",
                "Month",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    working["Month"] = (
        pd.to_datetime(
            working["Month"]
        )
    )

    future = future.copy()

    future["Month"] = (
        pd.to_datetime(
            future["Month"]
        )
    )

    forecast_rows = []

    future_months = sorted(
        future["Month"]
        .unique()
    )

    for horizon_number, month in enumerate(
        future_months,
        start=1,
    ):
        print(
            "Forecasting horizon "
            f"{horizon_number}: "
            f"{month}"
        )

        month_rows = (
            future.loc[
                future["Month"]
                == month
            ]
            .copy()
        )

        monthly_forecasts = []

        for _, row in (
            month_rows.iterrows()
        ):
            future_row = row.copy()

            dma = (
                future_row["DMA"]
            )

            history = (
                working.loc[
                    working["DMA"]
                    == dma
                ]
                .sort_values("Month")
                .copy()
            )

            if history.empty:
                raise ValueError(
                    "No historical records "
                    f"found for DMA {dma}."
                )

            traffic_history = (
                history["Traffic"]
                .dropna()
            )

            transaction_history = (
                history["Transactions"]
                .dropna()
            )

            if traffic_history.empty:
                raise ValueError(
                    "No valid Traffic history "
                    f"found for DMA {dma}."
                )

            if transaction_history.empty:
                raise ValueError(
                    "No valid Transactions "
                    "history found for DMA "
                    f"{dma}."
                )

            # ---------------------------------
            # Recursive traffic features
            # ---------------------------------

            future_row[
                "Traffic_Lag_1"
            ] = (
                traffic_history
                .iloc[-1]
            )

            future_row[
                "Rolling_Traffic_3"
            ] = (
                traffic_history
                .tail(3)
                .mean()
            )

            # ---------------------------------
            # Transaction assumption
            # ---------------------------------

            latest_transactions = (
                transaction_history
                .iloc[-1]
            )

            future_row[
                "Transactions_Lag_1"
            ] = latest_transactions

            future_row[
                "Transactions_Lag_3"
            ] = (
                transaction_history
                .tail(3)
                .mean()
            )

            future_row[
                "Transactions_Lag_6"
            ] = (
                transaction_history
                .tail(6)
                .mean()
            )

            # ---------------------------------
            # Exact model input
            # ---------------------------------

            model_input = (
                pd.DataFrame(
                    [future_row]
                )
                .loc[
                    :,
                    model_features,
                ]
            )

            prediction = (
                champion_model.predict(
                    model_input
                )
            )

            if (
                "Forecast"
                not in prediction.columns
            ):
                raise ValueError(
                    "Model output does not "
                    "contain a Forecast column."
                )

            forecast = float(
                prediction[
                    "Forecast"
                ]
                .iloc[0]
            )

            # ---------------------------------
            # Forecast output
            # ---------------------------------

            output_row = (
                future_row.copy()
            )

            output_row[
                "Forecast"
            ] = forecast

            output_row[
                "Forecast_Horizon"
            ] = horizon_number

            monthly_forecasts.append(
                output_row
            )

            # ---------------------------------
            # Extend recursive history
            # ---------------------------------

            history_row = (
                future_row.copy()
            )

            history_row[
                "Traffic"
            ] = forecast

            history_row[
                "Transactions"
            ] = latest_transactions

            working = (
                pd.concat(
                    [
                        working,
                        pd.DataFrame(
                            [history_row]
                        ),
                    ],
                    ignore_index=True,
                )
            )

        forecast_rows.extend(
            monthly_forecasts
        )

    forecast = pd.DataFrame(
        forecast_rows
    )

    if forecast.empty:
        raise ValueError(
            "Batch forecasting produced "
            "no output rows."
        )

    expected_rows = (
        future["DMA"].nunique()
        * future["Month"].nunique()
    )

    if len(forecast) != expected_rows:
        raise ValueError(
            "Unexpected forecast row count. "
            f"Expected {expected_rows}, "
            f"received {len(forecast)}."
        )

    return forecast


def create_national_forecast(
    forecast,
):
    """
    Aggregate DMA forecasts to national month.
    """

    required_columns = {
        "Month",
        "Forecast",
        "Forecast_Horizon",
    }

    missing_columns = (
        required_columns
        - set(forecast.columns)
    )

    if missing_columns:
        raise ValueError(
            "Forecast is missing required "
            "national aggregation columns: "
            f"{sorted(missing_columns)}"
        )

    national_forecast = (
        forecast
        .groupby(
            [
                "Month",
                "Forecast_Horizon",
            ],
            as_index=False,
        )
        .agg(
            Forecast=(
                "Forecast",
                "sum",
            ),
            DMA_Count=(
                "DMA",
                "nunique",
            ),
        )
        .sort_values(
            "Month"
        )
        .reset_index(
            drop=True
        )
    )

    return national_forecast