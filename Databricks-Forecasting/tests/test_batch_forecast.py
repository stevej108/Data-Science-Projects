from types import SimpleNamespace

import pytest

from jobs.batch_forecast import (
    run_batch_forecast,
)


def test_batch_model_version_must_match_champion(
    monkeypatch,
):

    mlflow_client = (
        SimpleNamespace(
            get_model_version_by_alias=(
                lambda **kwargs:
                    SimpleNamespace(
                        version="10"
                    )
            )
        )
    )

    fake_spark = (
        SimpleNamespace()
    )

    monkeypatch.setattr(
        (
            "jobs.batch_forecast."
            "mlflow.set_registry_uri"
        ),
        lambda uri: None,
    )

    with pytest.raises(
        ValueError,
        match=(
            "does not match Champion"
        ),
    ):
        run_batch_forecast(
            model_version="9",
            spark=fake_spark,
            mlflow_client=(
                mlflow_client
            ),
        )