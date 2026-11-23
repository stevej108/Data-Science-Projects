from types import SimpleNamespace

import pytest

from jobs.update_endpoint import (
    update_endpoint,
)


def make_mlflow_client(
    champion_version,
):

    return SimpleNamespace(
        get_model_version_by_alias=(
            lambda **kwargs:
                SimpleNamespace(
                    version=(
                        champion_version
                    )
                )
        )
    )


def test_promoted_version_must_match_champion(
    monkeypatch,
):

    mlflow_client = (
        make_mlflow_client(
            champion_version="5"
        )
    )

    workspace_client = (
        SimpleNamespace()
    )

    monkeypatch.setattr(
        (
            "jobs.update_endpoint."
            "mlflow.set_registry_uri"
        ),
        lambda uri: None,
    )

    with pytest.raises(
        ValueError,
        match=(
            "does not match the "
            "current Champion"
        ),
    ):
        update_endpoint(
            promoted_version="4",
            workspace_client=(
                workspace_client
            ),
            mlflow_client=(
                mlflow_client
            ),
        )


def test_matching_champion_is_deployed(
    monkeypatch,
):

    mlflow_client = (
        make_mlflow_client(
            champion_version="5"
        )
    )

    workspace_client = (
        SimpleNamespace()
    )

    expected_result = {
        "endpoint_name":
            "marketing-traffic-forecast",

        "previous_version":
            "4",

        "deployed_version":
            "5",

        "updated":
            True,
    }

    captured = {}

    def mock_update_endpoint_to_version(
        **kwargs,
    ):

        captured.update(
            kwargs
        )

        return expected_result

    monkeypatch.setattr(
        (
            "jobs.update_endpoint."
            "mlflow.set_registry_uri"
        ),
        lambda uri: None,
    )

    monkeypatch.setattr(
        (
            "jobs.update_endpoint."
            "update_endpoint_to_version"
        ),
        mock_update_endpoint_to_version,
    )

    result = update_endpoint(
        promoted_version="5",
        workspace_client=(
            workspace_client
        ),
        mlflow_client=(
            mlflow_client
        ),
    )

    assert (
        captured[
            "target_version"
        ]
        == "5"
    )

    assert result == expected_result