from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from databricks_forecasting.forecasting import (
    add_calendar_features,
    create_future_periods,
    create_national_forecast,
    generate_batch_forecast,
    populate_static_features,
    prepare_future_frame,
    enforce_model_input_dtypes,
)


STATIC_FEATURES = [
    "DMA",
    "Store_Count",
    "Total_Selling_SqFt",
    "Avg_Selling_SqFt",
    "DMA_Population",
    "TradeAreaPopulation",
    "Stores_per_Million",
    "CX_Composite",
    "Comp_Sales_Index",
    "Avg_Competitors",
    "Competitor_Density",
    "Pct_A_Facility",
]


MODEL_FEATURES = [
    "Traffic_Lag_1",
    "Rolling_Traffic_3",
    "Transactions_Lag_1",
    "Transactions_Lag_3",
    "Transactions_Lag_6",
    "Store_Count",
    "Total_Selling_SqFt",
    "Avg_Selling_SqFt",
    "DMA_Population",
    "TradeAreaPopulation",
    "Stores_per_Million",
    "CX_Composite",
    "Comp_Sales_Index",
    "Avg_Competitors",
    "Competitor_Density",
    "Pct_A_Facility",
    "Year",
    "Month_Number",
    "Quarter",
    "Holiday_Season",
]


def make_panel():

    rows = []

    for dma in [
        "DMA_A",
        "DMA_B",
    ]:

        for month_number in range(
            1,
            7,
        ):

            rows.append({
                "DMA":
                    dma,

                "Month":
                    pd.Timestamp(
                        2026,
                        month_number,
                        1,
                    ),

                "Traffic":
                    (
                        100.0
                        + month_number
                    ),

                "Transactions":
                    (
                        50.0
                        + month_number
                    ),

                "Store_Count":
                    10,

                "Total_Selling_SqFt":
                    100000.0,

                "Avg_Selling_SqFt":
                    10000.0,

                "DMA_Population":
                    500000.0,

                "TradeAreaPopulation":
                    400000.0,

                "Stores_per_Million":
                    20.0,

                "CX_Composite":
                    0.80,

                "Comp_Sales_Index":
                    1.05,

                "Avg_Competitors":
                    3.0,

                "Competitor_Density":
                    0.25,

                "Pct_A_Facility":
                    0.60,
            })

    return pd.DataFrame(
        rows
    )


# -------------------------------------------------
# 1. Future periods begin after history
# -------------------------------------------------

def test_future_periods_begin_after_history():

    panel = make_panel()

    future = (
        create_future_periods(
            panel=panel,
            horizon=3,
        )
    )

    assert (
        future["Month"].min()
        > panel["Month"].max()
    )

    assert (
        future["Month"].min()
        == pd.Timestamp(
            "2026-07-01"
        )
    )


# -------------------------------------------------
# 2. DMA x horizon row count is correct
# -------------------------------------------------

def test_future_period_row_count():

    panel = make_panel()

    future = (
        create_future_periods(
            panel=panel,
            horizon=6,
        )
    )

    expected_rows = (
        panel["DMA"].nunique()
        * 6
    )

    assert (
        len(future)
        == expected_rows
    )

    assert (
        future["Month"].nunique()
        == 6
    )

    assert (
        future["DMA"].nunique()
        == 2
    )


# -------------------------------------------------
# 3. Latest static attributes are selected
# -------------------------------------------------

def test_latest_static_features_are_used():

    panel = make_panel()

    # Change only the most recent observation
    # for DMA_A.
    mask = (
        (panel["DMA"] == "DMA_A")
        & (
            panel["Month"]
            == pd.Timestamp(
                "2026-06-01"
            )
        )
    )

    panel.loc[
        mask,
        "Store_Count",
    ] = 99

    future = (
        create_future_periods(
            panel=panel,
            horizon=1,
        )
    )

    populated = (
        populate_static_features(
            future=future,
            panel=panel,
            static_features=(
                STATIC_FEATURES
            ),
        )
    )

    dma_a = (
        populated.loc[
            populated["DMA"]
            == "DMA_A"
        ]
        .iloc[0]
    )

    assert (
        dma_a["Store_Count"]
        == 99
    )


# -------------------------------------------------
# 4. Calendar features are correct
# -------------------------------------------------

def test_calendar_features_are_correct():

    future = pd.DataFrame({
        "DMA": [
            "DMA_A",
            "DMA_A",
            "DMA_A",
        ],

        "Month": pd.to_datetime([
            "2026-10-01",
            "2026-11-01",
            "2026-12-01",
        ]),
    })

    result = (
        add_calendar_features(
            future
        )
    )

    assert (
        result["Year"].tolist()
        == [
            2026,
            2026,
            2026,
        ]
    )

    assert (
        result[
            "Month_Number"
        ].tolist()
        == [
            10,
            11,
            12,
        ]
    )

    assert (
        result["Quarter"].tolist()
        == [
            4,
            4,
            4,
        ]
    )

    assert (
        result[
            "Holiday_Season"
        ].tolist()
        == [
            0,
            1,
            1,
        ]
    )

    assert (
        result["Year"].dtype
        == np.dtype("int32")
    )

    assert (
        result["Month_Number"].dtype
        == np.dtype("int32")
    )

    assert (
        result["Quarter"].dtype
        == np.dtype("int32")
    )

    assert (
        result["Holiday_Season"].dtype
        == np.dtype("int64")
    )


# -------------------------------------------------
# 5. Recursive forecasts feed later horizons
# -------------------------------------------------

def test_recursive_forecast_used_as_next_lag():

    panel = make_panel()

    # Use one DMA so model calls are easy
    # to inspect.
    panel = (
        panel.loc[
            panel["DMA"]
            == "DMA_A"
        ]
        .copy()
    )

    future = (
        prepare_future_frame(
            panel=panel,
            horizon=2,
            static_features=(
                STATIC_FEATURES
            ),
        )
    )

    model_inputs = []

    class MockModel:

        def predict(
            self,
            model_input,
        ):

            model_inputs.append(
                model_input.copy()
            )

            # First horizon -> 200
            # Second horizon -> 300
            forecast = (
                200.0
                if len(model_inputs) == 1
                else 300.0
            )

            return pd.DataFrame({
                "Forecast": [
                    forecast
                ]
            })

    forecast = (
        generate_batch_forecast(
            panel=panel,
            future=future,
            champion_model=(
                MockModel()
            ),
            model_features=(
                MODEL_FEATURES
            ),
        )
    )

    assert len(
        model_inputs
    ) == 2

    # Horizon 1 predicts 200.
    assert (
        forecast.iloc[0][
            "Forecast"
        ]
        == 200.0
    )

    # That forecast becomes the Traffic lag
    # used by horizon 2.
    assert (
        model_inputs[1]
        ["Traffic_Lag_1"]
        .iloc[0]
        == 200.0
    )


# -------------------------------------------------
# 6. Transactions stay at latest observation
# -------------------------------------------------

def test_transactions_use_latest_observation():

    panel = make_panel()

    panel = (
        panel.loc[
            panel["DMA"]
            == "DMA_A"
        ]
        .copy()
    )

    future = (
        prepare_future_frame(
            panel=panel,
            horizon=2,
            static_features=(
                STATIC_FEATURES
            ),
        )
    )

    model_inputs = []

    class MockModel:

        def predict(
            self,
            model_input,
        ):

            model_inputs.append(
                model_input.copy()
            )

            return pd.DataFrame({
                "Forecast": [
                    200.0
                ]
            })

    generate_batch_forecast(
        panel=panel,
        future=future,
        champion_model=(
            MockModel()
        ),
        model_features=(
            MODEL_FEATURES
        ),
    )

    latest_transactions = (
        panel
        .sort_values("Month")
        ["Transactions"]
        .iloc[-1]
    )

    assert (
        model_inputs[0]
        ["Transactions_Lag_1"]
        .iloc[0]
        == latest_transactions
    )

    assert (
        model_inputs[1]
        ["Transactions_Lag_1"]
        .iloc[0]
        == latest_transactions
    )


# -------------------------------------------------
# 7. National forecast equals DMA sum
# -------------------------------------------------

def test_national_forecast_equals_dma_sum():

    forecast = pd.DataFrame({
        "DMA": [
            "DMA_A",
            "DMA_B",
            "DMA_A",
            "DMA_B",
        ],

        "Month": pd.to_datetime([
            "2026-07-01",
            "2026-07-01",
            "2026-08-01",
            "2026-08-01",
        ]),

        "Forecast": [
            100.0,
            200.0,
            150.0,
            250.0,
        ],

        "Forecast_Horizon": [
            1,
            1,
            2,
            2,
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
        == [
            300.0,
            400.0,
        ]
    )

    assert (
        national[
            "DMA_Count"
        ].tolist()
        == [
            2,
            2,
        ]
    )


# -------------------------------------------------
# 8. Invalid horizon fails
# -------------------------------------------------

def test_nonpositive_forecast_horizon_raises():

    panel = make_panel()

    with pytest.raises(
        ValueError,
        match=(
            "Forecast horizon must be "
            "greater than zero"
        ),
    ):
        create_future_periods(
            panel=panel,
            horizon=0,
        )

def test_model_input_dtypes_match_mlflow_signature():

    model_input = pd.DataFrame({
        "Traffic_Lag_1": [100],
        "Rolling_Traffic_3": [100],
        "Transactions_Lag_1": [50],
        "Transactions_Lag_3": [50],
        "Transactions_Lag_6": [50],
        "Store_Count": [33],
        "Total_Selling_SqFt": [100000],
        "Avg_Selling_SqFt": [10000],
        "DMA_Population": [500000],
        "TradeAreaPopulation": [400000],
        "Stores_per_Million": [20],
        "CX_Composite": [0.8],
        "Comp_Sales_Index": [1.05],
        "Avg_Competitors": [3],
        "Competitor_Density": [0.25],
        "Pct_A_Facility": [0.60],
        "Year": [2026],
        "Month_Number": [8],
        "Quarter": [3],
        "Holiday_Season": [0],
    })

    result = (
        enforce_model_input_dtypes(
            model_input
        )
    )

    assert (
        result["Year"].dtype
        == np.dtype("int32")
    )

    assert (
        result["Month_Number"].dtype
        == np.dtype("int32")
    )

    assert (
        result["Quarter"].dtype
        == np.dtype("int32")
    )

    assert (
        result["Store_Count"].dtype
        == np.dtype("int64")
    )

    assert (
        result["Holiday_Season"].dtype
        == np.dtype("int64")
    )

    float_columns = [
        "Traffic_Lag_1",
        "Rolling_Traffic_3",
        "Transactions_Lag_1",
        "Transactions_Lag_3",
        "Transactions_Lag_6",
        "Total_Selling_SqFt",
        "Avg_Selling_SqFt",
        "DMA_Population",
        "TradeAreaPopulation",
        "Stores_per_Million",
        "CX_Composite",
        "Comp_Sales_Index",
        "Avg_Competitors",
        "Competitor_Density",
        "Pct_A_Facility",
    ]

    for column in float_columns:
        assert (
            result[column].dtype
            == np.dtype("float64")
        )        