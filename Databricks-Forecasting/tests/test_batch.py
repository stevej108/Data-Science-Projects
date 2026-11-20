import pandas as pd

from databricks_forecasting.batch import (
    create_national_forecast,
    generate_batch_forecast,
    validate_batch_forecast,
)

from databricks_forecasting.features import (
    add_calendar_features,
    create_future_periods,
    populate_static_features,
)


class FakeForecastModel:

    def predict(
        self,
        model_input,
    ):

        # Deterministic test model:
        # forecast equals Traffic_Lag_1.

        return pd.DataFrame({
            "Forecast":
                model_input[
                    "Traffic_Lag_1"
                ].astype(float)
        })


def create_test_panel():

    months = pd.date_range(
        start="2026-01-01",
        periods=6,
        freq="MS",
    )

    rows = []

    for dma, dma_code, traffic in [
        ("Region_001", 0, 100.0),
        ("Region_002", 1, 200.0),
    ]:

        for month_number, month in enumerate(
            months,
            start=1,
        ):

            rows.append({

                "DMA": dma,
                "DMA_Code": dma_code,
                "Month": month,

                "Traffic":
                    traffic + month_number,

                "Transactions":
                    50.0,

                "Store_Count": 10,

                "Total_Selling_SqFt":
                    300000.0,

                "Avg_Selling_SqFt":
                    30000.0,

                "DMA_Population":
                    2500000.0,

                "TradeAreaPopulation":
                    5000000.0,

                "Stores_per_Million":
                    4.0,

                "CX_Composite": 3.5,

                "Comp_Sales_Index":
                    1.0,

                "Avg_Competitors":
                    0.5,

                "Competitor_Density":
                    0.05,

                "Pct_A_Facility":
                    0.7,

                "Year": month.year,

                "Month_Number":
                    month.month,

                "Quarter":
                    month.quarter,

                "Holiday_Season":
                    int(
                        month.month
                        in [11, 12]
                    ),
            })

    return pd.DataFrame(rows)


def test_recursive_batch_forecast():

    panel = create_test_panel()

    future = create_future_periods(
        panel,
        horizon=2,
    )

    future = populate_static_features(
        future,
        panel,
    )

    future = add_calendar_features(
        future
    )

    model = FakeForecastModel()

    forecast = (
        generate_batch_forecast(
            panel=panel,
            future=future,
            champion_model=model,
        )
    )

    validate_batch_forecast(
        batch_forecast=forecast,
        dma_count=2,
        horizon=2,
    )

    assert len(forecast) == 4

    assert (
        forecast[
            "Forecast_Horizon"
        ].tolist()
        == [1, 1, 2, 2]
    )


def test_recursive_prediction_becomes_next_lag():

    panel = create_test_panel()

    future = create_future_periods(
        panel,
        horizon=2,
    )

    future = populate_static_features(
        future,
        panel,
    )

    future = add_calendar_features(
        future
    )

    model = FakeForecastModel()

    forecast = (
        generate_batch_forecast(
            panel,
            future,
            model,
        )
    )

    region_001 = (
        forecast[
            forecast["DMA"]
            == "Region_001"
        ]
        .sort_values("Month")
        .reset_index(drop=True)
    )

    assert (
        region_001.loc[
            1,
            "Traffic_Lag_1",
        ]
        ==
        region_001.loc[
            0,
            "Forecast",
        ]
    )


def test_national_forecast():

    forecast = pd.DataFrame({

        "Month": pd.to_datetime([
            "2026-07-01",
            "2026-07-01",
            "2026-08-01",
            "2026-08-01",
        ]),

        "Forecast": [
            100.0,
            200.0,
            110.0,
            220.0,
        ],
    })

    national = (
        create_national_forecast(
            forecast
        )
    )

    assert (
        national[
            "Forecast"
        ].tolist()
        == [300.0, 330.0]
    )