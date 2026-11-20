import math
from types import SimpleNamespace

import pytest

from jobs.validate_candidate import (
    get_required_metric,
    validate_candidate,
)


def make_mock_run(
    validation_wape=0.1229,
    naive_wape=0.1525,
    relative_wape_improvement=0.1938,
    status="FINISHED",
):
    """
    Create a lightweight mock MLflow run.
    """

    return SimpleNamespace(
        info=SimpleNamespace(
            status=status,
        ),
        data=SimpleNamespace(
            metrics={
                "validation_wape":
                    validation_wape,

                "naive_wape":
                    naive_wape,

                "relative_wape_improvement":
                    relative_wape_improvement,
            }
        ),
    )


# -------------------------------------------------
# 1. Good candidate passes
# -------------------------------------------------

def test_good_candidate_passes(
    monkeypatch,
):

    mock_run = make_mock_run()

    monkeypatch.setattr(
        "jobs.validate_candidate.mlflow.get_run",
        lambda run_id: mock_run,
    )

    result = validate_candidate(
        "test-run-id"
    )

    assert result is True


# -------------------------------------------------
# 2. Candidate worse than naive fails
# -------------------------------------------------

def test_candidate_worse_than_naive_fails(
    monkeypatch,
):

    mock_run = make_mock_run(
        validation_wape=0.16,
        naive_wape=0.15,
        relative_wape_improvement=-0.0667,
    )

    monkeypatch.setattr(
        "jobs.validate_candidate.mlflow.get_run",
        lambda run_id: mock_run,
    )

    result = validate_candidate(
        "test-run-id"
    )

    assert result is False


# -------------------------------------------------
# 3. Candidate above WAPE threshold fails
# -------------------------------------------------

def test_candidate_above_wape_threshold_fails(
    monkeypatch,
):

    mock_run = make_mock_run(
        validation_wape=0.16,
        naive_wape=0.20,
        relative_wape_improvement=0.20,
    )

    monkeypatch.setattr(
        "jobs.validate_candidate.mlflow.get_run",
        lambda run_id: mock_run,
    )

    result = validate_candidate(
        "test-run-id"
    )

    assert result is False


# -------------------------------------------------
# 4. Insufficient relative improvement fails
# -------------------------------------------------

def test_insufficient_relative_improvement_fails(
    monkeypatch,
):

    mock_run = make_mock_run(
        validation_wape=0.14,
        naive_wape=0.145,
        relative_wape_improvement=0.0345,
    )

    monkeypatch.setattr(
        "jobs.validate_candidate.mlflow.get_run",
        lambda run_id: mock_run,
    )

    result = validate_candidate(
        "test-run-id"
    )

    assert result is False


# -------------------------------------------------
# 5. Missing required metric raises error
# -------------------------------------------------

def test_missing_required_metric_raises():

    metrics = {
        "validation_wape": 0.12,
        "naive_wape": 0.15,
    }

    with pytest.raises(
        ValueError,
        match="relative_wape_improvement",
    ):
        get_required_metric(
            metrics,
            "relative_wape_improvement",
        )


# -------------------------------------------------
# 6. Non-finite metric raises error
# -------------------------------------------------

@pytest.mark.parametrize(
    "bad_value",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_nonfinite_metric_raises(
    bad_value,
):

    metrics = {
        "validation_wape":
            bad_value,
    }

    with pytest.raises(
        ValueError,
        match="not finite",
    ):
        get_required_metric(
            metrics,
            "validation_wape",
        )