from types import SimpleNamespace

import pytest

from jobs.promote_candidate import (
    promote_candidate,
)


def make_mock_run(
    status="FINISHED",
):
    """
    Create a lightweight mock MLflow run.
    """

    return SimpleNamespace(
        info=SimpleNamespace(
            status=status,
        ),
    )


def make_mock_model_version(
    version="5",
):
    """
    Create a lightweight mock MLflow
    registered model version.
    """

    return SimpleNamespace(
        version=version,
    )


# -------------------------------------------------
# 1. Finished run registers successfully
# -------------------------------------------------

def test_finished_run_registers_successfully(
    monkeypatch,
):

    mock_run = make_mock_run()

    registered_version = (
        make_mock_model_version(
            version="5"
        )
    )

    champion_version = (
        make_mock_model_version(
            version="5"
        )
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.get_run",
        lambda run_id: mock_run,
    )

    register_calls = []

    def mock_register_model(
        model_uri,
        name,
    ):

        register_calls.append(
            {
                "model_uri": model_uri,
                "name": name,
            }
        )

        return registered_version

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.register_model",
        mock_register_model,
    )

    mock_client = SimpleNamespace(
        set_registered_model_alias=(
            lambda **kwargs: None
        ),
        get_model_version_by_alias=(
            lambda **kwargs:
                champion_version
        ),
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.MlflowClient",
        lambda: mock_client,
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.set_registry_uri",
        lambda uri: None,
    )

    promote_candidate(
        "test-run-id"
    )

    assert len(register_calls) == 1

    assert (
        register_calls[0]["model_uri"]
        == "runs:/test-run-id/model"
    )


# -------------------------------------------------
# 2. Non-finished run is rejected
# -------------------------------------------------

def test_nonfinished_run_rejected(
    monkeypatch,
):

    mock_run = make_mock_run(
        status="FAILED"
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.get_run",
        lambda run_id: mock_run,
    )

    with pytest.raises(
        ValueError,
        match="Expected 'FINISHED'",
    ):
        promote_candidate(
            "test-run-id"
        )


# -------------------------------------------------
# 3. Correct registered-model name is used
# -------------------------------------------------

def test_correct_registered_model_name_used(
    monkeypatch,
):

    mock_run = make_mock_run()

    registered_version = (
        make_mock_model_version(
            version="5"
        )
    )

    champion_version = (
        make_mock_model_version(
            version="5"
        )
    )

    captured = {}

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.get_run",
        lambda run_id: mock_run,
    )

    def mock_register_model(
        model_uri,
        name,
    ):

        captured["name"] = name

        return registered_version

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.register_model",
        mock_register_model,
    )

    mock_client = SimpleNamespace(
        set_registered_model_alias=(
            lambda **kwargs: None
        ),
        get_model_version_by_alias=(
            lambda **kwargs:
                champion_version
        ),
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.MlflowClient",
        lambda: mock_client,
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.set_registry_uri",
        lambda uri: None,
    )

    promote_candidate(
        "test-run-id"
    )

    assert (
        captured["name"]
        == (
            "workspace.default."
            "marketing_traffic_forecast"
        )
    )


# -------------------------------------------------
# 4. Champion alias assigned to new version
# -------------------------------------------------

def test_champion_alias_assigned(
    monkeypatch,
):

    mock_run = make_mock_run()

    registered_version = (
        make_mock_model_version(
            version="7"
        )
    )

    champion_version = (
        make_mock_model_version(
            version="7"
        )
    )

    alias_call = {}

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.get_run",
        lambda run_id: mock_run,
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.register_model",
        lambda **kwargs:
            registered_version,
    )

    def mock_set_alias(
        name,
        alias,
        version,
    ):

        alias_call["name"] = name
        alias_call["alias"] = alias
        alias_call["version"] = version

    mock_client = SimpleNamespace(
        set_registered_model_alias=(
            mock_set_alias
        ),
        get_model_version_by_alias=(
            lambda **kwargs:
                champion_version
        ),
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.MlflowClient",
        lambda: mock_client,
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.set_registry_uri",
        lambda uri: None,
    )

    promote_candidate(
        "test-run-id"
    )

    assert (
        alias_call["alias"]
        == "Champion"
    )

    assert (
        alias_call["version"]
        == "7"
    )


# -------------------------------------------------
# 5. Champion verification mismatch raises
# -------------------------------------------------

def test_champion_verification_mismatch_raises(
    monkeypatch,
):

    mock_run = make_mock_run()

    registered_version = (
        make_mock_model_version(
            version="8"
        )
    )

    wrong_champion = (
        make_mock_model_version(
            version="7"
        )
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.get_run",
        lambda run_id: mock_run,
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.register_model",
        lambda **kwargs:
            registered_version,
    )

    mock_client = SimpleNamespace(
        set_registered_model_alias=(
            lambda **kwargs: None
        ),
        get_model_version_by_alias=(
            lambda **kwargs:
                wrong_champion
        ),
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.MlflowClient",
        lambda: mock_client,
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.set_registry_uri",
        lambda uri: None,
    )

    with pytest.raises(
        RuntimeError,
        match="verification failed",
    ):
        promote_candidate(
            "test-run-id"
        )


# -------------------------------------------------
# 6. New model version is returned
# -------------------------------------------------

def test_new_model_version_returned(
    monkeypatch,
):

    mock_run = make_mock_run()

    registered_version = (
        make_mock_model_version(
            version="9"
        )
    )

    champion_version = (
        make_mock_model_version(
            version="9"
        )
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.get_run",
        lambda run_id: mock_run,
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.register_model",
        lambda **kwargs:
            registered_version,
    )

    mock_client = SimpleNamespace(
        set_registered_model_alias=(
            lambda **kwargs: None
        ),
        get_model_version_by_alias=(
            lambda **kwargs:
                champion_version
        ),
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.MlflowClient",
        lambda: mock_client,
    )

    monkeypatch.setattr(
        "jobs.promote_candidate.mlflow.set_registry_uri",
        lambda uri: None,
    )

    result = promote_candidate(
        "test-run-id"
    )

    assert result == "9"