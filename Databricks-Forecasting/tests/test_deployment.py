from types import SimpleNamespace

import pytest

from databricks_forecasting.deployment import (
    build_updated_served_entities,
    get_deployed_model_version,
    update_endpoint_to_version,
)


MODEL_NAME = (
    "workspace.default."
    "marketing_traffic_forecast"
)

ENDPOINT_NAME = (
    "marketing-traffic-forecast"
)


def make_entity(
    version="4",
):

    return SimpleNamespace(
        name=(
            "marketing-traffic-forecast"
        ),

        entity_name=(
            MODEL_NAME
        ),

        entity_version=(
            version
        ),

        workload_size=(
            "Small"
        ),

        workload_type=(
            None
        ),

        scale_to_zero_enabled=(
            True
        ),

        environment_vars=(
            None
        ),
    )


def make_endpoint(
    version="4",
):

    return SimpleNamespace(
        config=SimpleNamespace(
            served_entities=[
                make_entity(
                    version=version
                )
            ],

            traffic_config=(
                SimpleNamespace(
                    routes=[]
                )
            ),
        )
    )


# -------------------------------------------------
# 1. Deployed version is extracted
# -------------------------------------------------

def test_get_deployed_model_version():

    endpoint = make_endpoint(
        version="4"
    )

    result = (
        get_deployed_model_version(
            endpoint=endpoint,
            registered_model_name=(
                MODEL_NAME
            ),
        )
    )

    assert result == "4"


# -------------------------------------------------
# 2. Missing model raises
# -------------------------------------------------

def test_missing_registered_model_raises():

    endpoint = make_endpoint(
        version="4"
    )

    with pytest.raises(
        ValueError,
        match="is not currently deployed",
    ):
        get_deployed_model_version(
            endpoint=endpoint,
            registered_model_name=(
                "workspace.default."
                "different_model"
            ),
        )


# -------------------------------------------------
# 3. Updated entity uses target version
# -------------------------------------------------

def test_build_updated_entity_uses_target_version(
    monkeypatch,
):

    endpoint = make_endpoint(
        version="4"
    )

    captured = {}

    def mock_served_entity_input(
        **kwargs,
    ):

        captured.update(
            kwargs
        )

        return SimpleNamespace(
            **kwargs
        )

    monkeypatch.setattr(
        (
            "databricks_forecasting."
            "deployment."
            "ServedEntityInput"
        ),
        mock_served_entity_input,
    )

    result = (
        build_updated_served_entities(
            endpoint=endpoint,
            registered_model_name=(
                MODEL_NAME
            ),
            target_version="5",
        )
    )

    assert len(result) == 1

    assert (
        captured["entity_version"]
        == "5"
    )

    assert (
        captured["entity_name"]
        == MODEL_NAME
    )

    assert (
        captured[
            "workload_size"
        ]
        == "Small"
    )


# -------------------------------------------------
# 4. Already deployed is a no-op
# -------------------------------------------------

def test_already_deployed_is_noop():

    endpoint = make_endpoint(
        version="5"
    )

    update_calls = []

    serving_endpoints = (
        SimpleNamespace(
            get=(
                lambda name:
                    endpoint
            ),

            update_config_and_wait=(
                lambda **kwargs:
                    update_calls.append(
                        kwargs
                    )
            ),
        )
    )

    workspace_client = (
        SimpleNamespace(
            serving_endpoints=(
                serving_endpoints
            )
        )
    )

    result = (
        update_endpoint_to_version(
            workspace_client=(
                workspace_client
            ),
            endpoint_name=(
                ENDPOINT_NAME
            ),
            registered_model_name=(
                MODEL_NAME
            ),
            target_version="5",
        )
    )

    assert result["updated"] is False

    assert (
        result[
            "deployed_version"
        ]
        == "5"
    )

    assert update_calls == []


# -------------------------------------------------
# 5. Old version triggers update
# -------------------------------------------------

def test_old_version_triggers_update(
    monkeypatch,
):

    old_endpoint = make_endpoint(
        version="4"
    )

    new_endpoint = make_endpoint(
        version="5"
    )

    endpoints = iter([
        old_endpoint,
        new_endpoint,
    ])

    update_calls = []

    serving_endpoints = (
        SimpleNamespace(
            get=(
                lambda name:
                    next(endpoints)
            ),

            update_config_and_wait=(
                lambda **kwargs:
                    update_calls.append(
                        kwargs
                    )
            ),
        )
    )

    workspace_client = (
        SimpleNamespace(
            serving_endpoints=(
                serving_endpoints
            )
        )
    )

    monkeypatch.setattr(
        (
            "databricks_forecasting."
            "deployment."
            "ServedEntityInput"
        ),
        lambda **kwargs:
            SimpleNamespace(
                **kwargs
            ),
    )

    result = (
        update_endpoint_to_version(
            workspace_client=(
                workspace_client
            ),
            endpoint_name=(
                ENDPOINT_NAME
            ),
            registered_model_name=(
                MODEL_NAME
            ),
            target_version="5",
        )
    )

    assert len(update_calls) == 1

    assert result["updated"] is True

    assert (
        result[
            "previous_version"
        ]
        == "4"
    )

    assert (
        result[
            "deployed_version"
        ]
        == "5"
    )


# -------------------------------------------------
# 6. Verification mismatch raises
# -------------------------------------------------

def test_deployment_verification_mismatch_raises(
    monkeypatch,
):

    unchanged_endpoint = (
        make_endpoint(
            version="4"
        )
    )

    serving_endpoints = (
        SimpleNamespace(
            get=(
                lambda name:
                    unchanged_endpoint
            ),

            update_config_and_wait=(
                lambda **kwargs:
                    None
            ),
        )
    )

    workspace_client = (
        SimpleNamespace(
            serving_endpoints=(
                serving_endpoints
            )
        )
    )

    monkeypatch.setattr(
        (
            "databricks_forecasting."
            "deployment."
            "ServedEntityInput"
        ),
        lambda **kwargs:
            SimpleNamespace(
                **kwargs
            ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "verification failed"
        ),
    ):
        update_endpoint_to_version(
            workspace_client=(
                workspace_client
            ),
            endpoint_name=(
                ENDPOINT_NAME
            ),
            registered_model_name=(
                MODEL_NAME
            ),
            target_version="5",
        )