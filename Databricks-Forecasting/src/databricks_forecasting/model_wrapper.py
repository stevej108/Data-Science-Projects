import mlflow.pyfunc
import numpy as np
import pandas as pd

from databricks_forecasting.config import (
    MODEL_FEATURES,
)


class MarketingForecastPyFunc(
    mlflow.pyfunc.PythonModel
):

    def __init__(
        self,
        xgb_model,
        feature_order=None,
    ):
        self.xgb_model = xgb_model

        self.feature_order = list(
            feature_order
            if feature_order is not None
            else MODEL_FEATURES
        )


    def prepare_features(
        self,
        model_input,
    ):

        if not isinstance(
            model_input,
            pd.DataFrame,
        ):
            model_input = pd.DataFrame(
                model_input
            )

        df = model_input.copy()

        missing_columns = [
            feature
            for feature in self.feature_order
            if feature not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required feature columns: "
                f"{missing_columns}"
            )

        model_features = (
            df[self.feature_order]
            .copy()
        )

        for feature in self.feature_order:

            model_features[feature] = (
                pd.to_numeric(
                    model_features[feature],
                    errors="coerce",
                )
            )

        invalid_counts = (
            model_features
            .isna()
            .sum()
        )

        invalid_columns = (
            invalid_counts[
                invalid_counts > 0
            ]
            .to_dict()
        )

        if invalid_columns:
            raise ValueError(
                "Null or nonnumeric values found: "
                f"{invalid_columns}"
            )

        return model_features


    def predict(
        self,
        context,
        model_input,
        params=None,
    ):

        model_features = (
            self.prepare_features(
                model_input
            )
        )

        forecasts = (
            self.xgb_model.predict(
                model_features
            )
        )

        return pd.DataFrame({
            "Forecast": (
                np.asarray(
                    forecasts
                )
                .astype(float)
            )
        })