import pandas as pd
import pytest

from databricks_forecasting.features import (
    add_calendar_features,
    create_future_periods,
    populate_static_features,
)


def create_test_panel():

    months = pd.to_datetime([
        "2026-05-01",
        "2026-06-01",
        "2026-07-01",
    ])

    rows = []

    for dma, dma_code in [
        ("Region_001", 0),
        ("Region_002", 1),
    ]:

        for month in months:

            rows.append({
                "DMA": dma,
                "DMA_Code": dma_code,
                "Month": month,

                "Store_Count": 10,

                "Total_Selling_SqFt": 300000.0,
                "Avg_Selling_SqFt": 30000.0,

                "DMA_Population": 2500000.0,
                "TradeAreaPopulation": 5000000.0,

                "Stores_per_Million": 4.0,

                "CX_Composite": 3.5,
                "Comp_Sales_Index": 1.0,

                "Avg_Competitors": 0.5,
                "Competitor_Density": 0.05,

                "Pct_A_Facility": 0.7,
            })

    return pd.DataFrame(rows)


def test_create_future_periods():

    panel = create_test_panel()

    future = create_future_periods(
        panel=panel,
        horizon=3,
    )

    assert len(future) == 6

    assert future["DMA"].nunique() == 2

    assert future["Month"].nunique() == 3

    assert future["Month"].min() == (
        pd.Timestamp("2026-08-01")
    )

    assert future["Month"].max() == (
        pd.Timestamp("2026-10-01")
    )


def test_create_future_periods_rejects_zero_horizon():

    panel = create_test_panel()

    with pytest.raises(
        ValueError,
        match="Forecast horizon must be positive",
    ):

        create_future_periods(
            panel=panel,
            horizon=0,
        )


def test_populate_static_features():

    panel = create_test_panel()

    future = create_future_periods(
        panel=panel,
        horizon=2,
    )

    populated = populate_static_features(
        future=future,
        panel=panel,
    )

    assert len(populated) == 4

    assert populated[
        "DMA_Code"
    ].notna().all()

    assert populated[
        "Store_Count"
    ].notna().all()

    assert (
        populated[
            "Total_Selling_SqFt"
        ] == 300000.0
    ).all()


def test_add_calendar_features():

    future = pd.DataFrame({
        "DMA": [
            "Region_001",
            "Region_001",
            "Region_001",
        ],

        "Month": pd.to_datetime([
            "2026-10-01",
            "2026-11-01",
            "2026-12-01",
        ]),
    })

    result = add_calendar_features(
        future
    )

    assert (
        result["Year"].tolist()
        == [2026, 2026, 2026]
    )

    assert (
        result["Month_Number"].tolist()
        == [10, 11, 12]
    )

    assert (
        result["Quarter"].tolist()
        == [4, 4, 4]
    )

    assert (
        result["Holiday_Season"].tolist()
        == [0, 1, 1]
    )