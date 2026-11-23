import argparse
import sys

from pathlib import Path

import mlflow

from databricks.sdk import (
    WorkspaceClient,
)

from mlflow import (
    MlflowClient,
)


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
        "update_endpoint.py"
    ).resolve()


PROJECT_ROOT = (
    _THIS_FILE
    .parents[1]
)

SRC_PATH = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_PATH) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_PATH),
    )


from databricks_forecasting.config import (
    MODEL_ALIAS,
    REGISTERED_MODEL_NAME,
    SERVING_ENDPOINT_NAME,
)

from databricks_forecasting.deployment import (
    update_endpoint_to_version,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Update the Databricks serving "
            "endpoint to the promoted model "
            "version."
        )
    )

    parser.add_argument(
        "--model-version",
        required=True,
        help=(
            "Registered model version that "
            "should be deployed."
        ),
    )

    return parser.parse_args()


def update_endpoint(
    promoted_version,
    workspace_client=None,
    mlflow_client=None,
):

    promoted_version = (
        str(promoted_version)
    )


    # -------------------------------------------------
    # Clients
    # -------------------------------------------------

    mlflow.set_registry_uri(
        "databricks-uc"
    )

    if mlflow_client is None:
        mlflow_client = (
            MlflowClient()
        )

    if workspace_client is None:
        workspace_client = (
            WorkspaceClient()
        )


    # -------------------------------------------------
    # Confirm promoted version is Champion
    # -------------------------------------------------

    champion = (
        mlflow_client
        .get_model_version_by_alias(
            name=REGISTERED_MODEL_NAME,
            alias=MODEL_ALIAS,
        )
    )

    champion_version = (
        str(champion.version)
    )

    if (
        champion_version
        != promoted_version
    ):
        raise ValueError(
            "Promoted model version does "
            "not match the current Champion. "
            f"Promoted version: "
            f"{promoted_version}. "
            f"Champion version: "
            f"{champion_version}."
        )


    # -------------------------------------------------
    # Update endpoint
    # -------------------------------------------------

    result = (
        update_endpoint_to_version(
            workspace_client=(
                workspace_client
            ),
            endpoint_name=(
                SERVING_ENDPOINT_NAME
            ),
            registered_model_name=(
                REGISTERED_MODEL_NAME
            ),
            target_version=(
                promoted_version
            ),
        )
    )


    # -------------------------------------------------
    # Deployment summary
    # -------------------------------------------------

    print("=" * 60)
    print(
        "SERVING ENDPOINT UPDATE COMPLETE"
    )
    print("=" * 60)

    print(
        "Endpoint:",
        result[
            "endpoint_name"
        ],
    )

    print(
        "Champion version:",
        champion_version,
    )

    print(
        "Previous deployed version:",
        result[
            "previous_version"
        ],
    )

    print(
        "Current deployed version:",
        result[
            "deployed_version"
        ],
    )

    print(
        "Endpoint updated:",
        result[
            "updated"
        ],
    )

    print("=" * 60)

    return result


def main():

    args = parse_args()

    try:

        return update_endpoint(
            promoted_version=(
                args.model_version
            )
        )

    except Exception as exc:

        print("=" * 60)
        print(
            "SERVING ENDPOINT UPDATE ERROR"
        )
        print("=" * 60)

        print(
            str(exc)
        )

        print("=" * 60)

        sys.exit(1)


if __name__ == "__main__":
    main()