import numpy as np
import pandas as pd
import pytest

from databricks_forecasting.config import (
    MODEL_FEATURES,
)

from databricks_forecasting.model_wrapper import (
    MarketingForecastPyFunc,
)


class FakeXGBoostModel:

    def predict(
        self,
        model_input,
    ):

        return (
            model_input[
                "Traffic_Lag_1"
            ]
            .to_numpy()
            * 1.10
        )


def create_model_input():

    row = {
        feature: 1.0
        for feature in MODEL_FEATURES
    }

    row["Traffic_Lag_1"] = 100.0

    return pd.DataFrame(
        [row]
    )


def test_prepare_features_preserves_order():

    model_input = create_model_input()

    shuffled_input = (
        model_input[
            list(
                reversed(
                    model_input.columns
                )
            )
        ]
    )

    wrapper = MarketingForecastPyFunc(
        xgb_model=
            FakeXGBoostModel()
    )

    prepared = (
        wrapper.prepare_features(
            shuffled_input
        )
    )

    assert (
        prepared.columns.tolist()
        == MODEL_FEATURES
    )


def test_prediction():

    model_input = create_model_input()

    wrapper = MarketingForecastPyFunc(
        xgb_model=
            FakeXGBoostModel()
    )

    prediction = wrapper.predict(
        context=None,
        model_input=model_input,
    )

    assert list(
        prediction.columns
    ) == ["Forecast"]

    assert np.isclose(
        prediction.loc[
            0,
            "Forecast",
        ],
        110.0,
    )


def test_missing_feature_rejected():

    model_input = (
        create_model_input()
        .drop(
            columns=[
                "Traffic_Lag_1"
            ]
        )
    )

    wrapper = MarketingForecastPyFunc(
        xgb_model=
            FakeXGBoostModel()
    )

    with pytest.raises(
        ValueError,
        match="Missing required feature",
    ):

        wrapper.predict(
            context=None,
            model_input=model_input,
        )


def test_nonnumeric_feature_rejected():

    model_input = (
        create_model_input()
    )

    # Allow the test DataFrame to contain
    # intentionally invalid input.
    model_input["Store_Count"] = (
        model_input["Store_Count"]
        .astype(object)
    )

    model_input.loc[
        0,
        "Store_Count",
    ] = "invalid"

    wrapper = MarketingForecastPyFunc(
        xgb_model=
            FakeXGBoostModel()
    )

    with pytest.raises(
        ValueError,
        match="Null or nonnumeric",
    ):

        wrapper.predict(
            context=None,
            model_input=model_input,
        )