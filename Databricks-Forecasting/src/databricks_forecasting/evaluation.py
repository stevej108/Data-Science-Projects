import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def calculate_metrics(
    actual,
    forecast,
):

    actual = np.asarray(
        actual
    )

    forecast = np.asarray(
        forecast
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            forecast,
        )
    )

    mae = mean_absolute_error(
        actual,
        forecast,
    )

    nonzero_mask = (
        actual != 0
    )

    if nonzero_mask.any():

        mape = (
            mean_absolute_percentage_error(
                actual[
                    nonzero_mask
                ],
                forecast[
                    nonzero_mask
                ],
            )
        )

    else:

        mape = np.nan

    actual_total = (
        np.abs(actual).sum()
    )

    if actual_total > 0:

        wape = (
            np.abs(
                actual - forecast
            ).sum()
            / actual_total
        )

    else:

        wape = np.nan

    r2 = r2_score(
        actual,
        forecast,
    )

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "WAPE": float(wape),
        "R2": float(r2),
    }


def evaluate_naive(
    validation_data,
):

    return calculate_metrics(
        actual=(
            validation_data[
                "Traffic"
            ]
        ),

        forecast=(
            validation_data[
                "Traffic_Lag_1"
            ]
        ),
    )


def relative_wape_improvement(
    candidate_wape,
    benchmark_wape,
):

    if benchmark_wape == 0:
        raise ValueError(
            "Benchmark WAPE cannot be zero."
        )

    return (
        benchmark_wape
        - candidate_wape
    ) / benchmark_wape