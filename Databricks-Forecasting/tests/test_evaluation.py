import numpy as np
import pandas as pd

from databricks_forecasting.evaluation import (
    calculate_metrics,
    evaluate_naive,
    relative_wape_improvement,
)


def test_perfect_predictions():

    actual = np.array(
        [100, 200, 300]
    )

    forecast = np.array(
        [100, 200, 300]
    )

    metrics = calculate_metrics(
        actual,
        forecast,
    )

    assert metrics["RMSE"] == 0
    assert metrics["MAE"] == 0
    assert metrics["MAPE"] == 0
    assert metrics["WAPE"] == 0
    assert metrics["R2"] == 1


def test_wape():

    actual = np.array(
        [100, 200]
    )

    forecast = np.array(
        [90, 220]
    )

    metrics = calculate_metrics(
        actual,
        forecast,
    )

    expected_wape = (
        30 / 300
    )

    assert np.isclose(
        metrics["WAPE"],
        expected_wape,
    )


def test_naive_evaluation():

    validation = pd.DataFrame({
        "Traffic": [
            100,
            200,
        ],

        "Traffic_Lag_1": [
            90,
            190,
        ],
    })

    metrics = evaluate_naive(
        validation
    )

    assert np.isclose(
        metrics["MAE"],
        10.0,
    )


def test_relative_wape_improvement():

    improvement = (
        relative_wape_improvement(
            candidate_wape=0.1291,
            benchmark_wape=0.1558,
        )
    )

    assert np.isclose(
        improvement,
        0.1713735558,
    )