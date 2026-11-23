import argparse
import sys

import mlflow
from mlflow import MlflowClient

from pathlib import Path


try:
    _THIS_FILE = (
        Path(__file__)
        .resolve()
    )
except NameError:
    _THIS_FILE = Path(
        "/Workspace/Users/stevej108@gmail.com/"
        "Data-Science-Projects/"
        "Databricks-Forecasting/jobs/"
        "promote_candidate.py"
    ).resolve()


PROJECT_ROOT = (
    _THIS_FILE
    .parents[1]
)

SRC_PATH = (
    PROJECT_ROOT
    / "src"
)

sys.path.insert(
    0,
    str(SRC_PATH),
)

from databricks_forecasting.config import (
    MODEL_ALIAS,
    REGISTERED_MODEL_NAME,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Promote a validated candidate "
            "model to the Champion alias."
        )
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help=(
            "MLflow run ID for the "
            "validated candidate model."
        ),
    )

    return parser.parse_args()


def promote_candidate(
    run_id,
):

    # -------------------------------------------------
    # Verify candidate run
    # -------------------------------------------------

    run = mlflow.get_run(
        run_id
    )

    if run.info.status != "FINISHED":
        raise ValueError(
            "Candidate run status is "
            f"'{run.info.status}'. "
            "Expected 'FINISHED'."
        )


    # -------------------------------------------------
    # Registry configuration
    # -------------------------------------------------

    mlflow.set_registry_uri(
        "databricks-uc"
    )

    client = MlflowClient()


    # -------------------------------------------------
    # Register candidate model
    # -------------------------------------------------

    model_uri = (
        f"runs:/{run_id}/model"
    )

    model_version = (
        mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME,
        )
    )

    version = (
        str(model_version.version)
    )


    # -------------------------------------------------
    # Assign Champion alias
    # -------------------------------------------------

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
        version=version,
    )


    # -------------------------------------------------
    # Verify promotion
    # -------------------------------------------------

    champion = (
        client.get_model_version_by_alias(
            name=REGISTERED_MODEL_NAME,
            alias=MODEL_ALIAS,
        )
    )

    if str(champion.version) != version:
        raise RuntimeError(
            "Champion alias verification "
            "failed. "
            f"Expected version {version}, "
            f"but found version "
            f"{champion.version}."
        )


    # -------------------------------------------------
    # Promotion summary
    # -------------------------------------------------

    print("=" * 60)
    print(
        "CANDIDATE PROMOTION COMPLETE"
    )
    print("=" * 60)

    print(
        "Run ID:",
        run_id,
    )

    print(
        "Registered model:",
        REGISTERED_MODEL_NAME,
    )

    print(
        "New model version:",
        version,
    )

    print(
        "Alias:",
        MODEL_ALIAS,
    )

    print(
        "Champion version:",
        champion.version,
    )

    print("=" * 60)

    return version


def main():

    args = parse_args()

    try:

        version = (
            promote_candidate(
                args.run_id
            )
        )

    except Exception as exc:

        print("=" * 60)
        print(
            "CANDIDATE PROMOTION ERROR"
        )
        print("=" * 60)

        print(
            str(exc)
        )

        print("=" * 60)

        sys.exit(1)

    return version


if __name__ == "__main__":
    main()