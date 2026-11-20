import pandas as pd

from databricks_forecasting.config import (
    MODEL_FEATURES,
)


def generate_batch_forecast(
    panel: pd.DataFrame,
    future: pd.DataFrame,
    champion_model,
) -> pd.DataFrame:

    working = (
        panel
        .copy()
        .sort_values(
            ["DMA", "Month"]
        )
        .reset_index(drop=True)
    )

    forecast_rows = []

    future_months = sorted(
        future["Month"].unique()
    )

    for horizon_number, month in enumerate(
        future_months,
        start=1,
    ):

        month_rows = (
            future[
                future["Month"] == month
            ]
            .copy()
        )

        for _, future_row in (
            month_rows.iterrows()
        ):

            dma = future_row["DMA"]

            history = (
                working[
                    working["DMA"] == dma
                ]
                .sort_values("Month")
            )

            if history.empty:
                raise ValueError(
                    f"No history found for DMA {dma}."
                )

            if len(history) < 6:
                raise ValueError(
                    f"DMA {dma} has fewer than "
                    "six historical periods."
                )

            # Traffic features

            future_row["Traffic_Lag_1"] = (
                history["Traffic"]
                .iloc[-1]
            )

            future_row["Rolling_Traffic_3"] = (
                history["Traffic"]
                .tail(3)
                .mean()
            )

            # Transaction assumption:
            # latest observation is carried forward.

            latest_transactions = (
                history["Transactions"]
                .iloc[-1]
            )

            future_row[
                "Transactions_Lag_1"
            ] = latest_transactions

            future_row[
                "Transactions_Lag_3"
            ] = (
                history["Transactions"]
                .tail(3)
                .mean()
            )

            future_row[
                "Transactions_Lag_6"
            ] = (
                history["Transactions"]
                .tail(6)
                .mean()
            )

            model_input = (
                pd.DataFrame(
                    [future_row]
                )[MODEL_FEATURES]
            )

            prediction = (
                champion_model.predict(
                    model_input
                )
            )

            if isinstance(
                prediction,
                pd.DataFrame,
            ):

                forecast = float(
                    prediction[
                        "Forecast"
                    ].iloc[0]
                )

            else:

                forecast = float(
                    prediction[0]
                )

            output_row = (
                future_row.copy()
            )

            output_row["Forecast"] = (
                forecast
            )

            output_row[
                "Forecast_Horizon"
            ] = horizon_number

            forecast_rows.append(
                output_row
            )

            # Extend history so that future
            # traffic forecasts become future lags.

            history_row = (
                future_row.copy()
            )

            history_row["Traffic"] = (
                forecast
            )

            history_row[
                "Transactions"
            ] = latest_transactions

            working = pd.concat(
                [
                    working,
                    pd.DataFrame(
                        [history_row]
                    ),
                ],
                ignore_index=True,
            )

    return pd.DataFrame(
        forecast_rows
    )


def create_national_forecast(
    batch_forecast: pd.DataFrame,
) -> pd.DataFrame:

    return (
        batch_forecast
        .groupby(
            "Month",
            as_index=False,
        )
        .agg(
            Forecast=(
                "Forecast",
                "sum",
            )
        )
        .sort_values("Month")
        .reset_index(drop=True)
    )


def validate_batch_forecast(
    batch_forecast: pd.DataFrame,
    dma_count: int,
    horizon: int,
) -> None:

    expected_rows = (
        dma_count
        * horizon
    )

    if len(batch_forecast) != expected_rows:

        raise AssertionError(
            "Unexpected forecast row count. "
            f"Expected {expected_rows}; "
            f"received {len(batch_forecast)}."
        )

    if not (
        batch_forecast[
            "Forecast"
        ]
        .notna()
        .all()
    ):

        raise AssertionError(
            "Forecast contains null values."
        )

    if not (
        batch_forecast[
            "Forecast"
        ]
        .gt(0)
        .all()
    ):

        raise AssertionError(
            "Forecast contains nonpositive values."
        )

    if (
        batch_forecast[
            "Forecast_Horizon"
        ].nunique()
        != horizon
    ):

        raise AssertionError(
            "Unexpected forecast horizon count."
        )