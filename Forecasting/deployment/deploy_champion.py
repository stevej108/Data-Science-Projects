import json
import os
from datetime import timedelta
from pathlib import Path

import mlflow
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    Route,
    ServedEntityInput,
    TrafficConfig,
)
from dotenv import load_dotenv
from mlflow import MlflowClient


# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

load_dotenv()

DATABRICKS_HOST = (
    os.getenv("DATABRICKS_HOST", "")
    .rstrip("/")
)

DATABRICKS_DEPLOYMENT_TOKEN = os.getenv(
    "DATABRICKS_DEPLOYMENT_TOKEN"
)

REGISTERED_MODEL_NAME = (
    "workspace.default.marketing_traffic_forecast"
)

MODEL_ALIAS = "Champion"

ENDPOINT_NAME = (
    "marketing-traffic-forecast"
)

WORKLOAD_SIZE = "Small"

SCALE_TO_ZERO_ENABLED = True

DEPLOYMENT_TIMEOUT_MINUTES = 30

SMOKE_TEST_PATH = (
    Path(__file__).parent
    / "smoke_test_payload.json"
)


# -----------------------------------------------------
# Validate local configuration
# -----------------------------------------------------

if not DATABRICKS_HOST:
    raise ValueError(
        "DATABRICKS_HOST was not found."
    )

if not DATABRICKS_DEPLOYMENT_TOKEN:
    raise ValueError(
        "DATABRICKS_DEPLOYMENT_TOKEN was not found."
    )

if not SMOKE_TEST_PATH.exists():
    raise FileNotFoundError(
        "Smoke-test payload was not found at "
        f"{SMOKE_TEST_PATH}."
    )


# -----------------------------------------------------
# Configure clients
# -----------------------------------------------------

os.environ["DATABRICKS_HOST"] = (
    DATABRICKS_HOST
)

os.environ["DATABRICKS_TOKEN"] = (
    DATABRICKS_DEPLOYMENT_TOKEN
)

mlflow.set_registry_uri(
    "databricks-uc"
)

registry_client = MlflowClient()

workspace_client = WorkspaceClient(
    host=DATABRICKS_HOST,
    token=DATABRICKS_DEPLOYMENT_TOKEN,
)


# -----------------------------------------------------
# Resolve Champion alias
# -----------------------------------------------------

champion = (
    registry_client
    .get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
    )
)

champion_version = str(
    champion.version
)

champion_run_id = champion.run_id

print(
    f"Champion version: {champion_version}"
)

print(
    f"Champion source run: {champion_run_id}"
)


# -----------------------------------------------------
# Read current endpoint configuration
# -----------------------------------------------------

endpoint = (
    workspace_client
    .serving_endpoints
    .get(
        name=ENDPOINT_NAME
    )
)

if not endpoint.config:
    raise RuntimeError(
        "Serving endpoint has no active "
        "configuration."
    )

served_entities = (
    endpoint.config.served_entities
    or []
)

if len(served_entities) != 1:
    raise RuntimeError(
        "This deployment script expects exactly "
        "one served entity. "
        f"Found {len(served_entities)}."
    )

current_entity = served_entities[0]

current_version = str(
    current_entity.entity_version
)

print(
    f"Currently deployed version: "
    f"{current_version}"
)


# -----------------------------------------------------
# Idempotency check
# -----------------------------------------------------

if current_version == champion_version:

    print(
        "Endpoint already serves the current "
        "Champion version."
    )

else:

    served_entity_name = (
        f"marketing-traffic-forecast-"
        f"v{champion_version}"
    )

    print(
        f"Updating endpoint from version "
        f"{current_version} to "
        f"{champion_version}..."
    )

    updated_endpoint = (
        workspace_client
        .serving_endpoints
        .update_config_and_wait(
            name=ENDPOINT_NAME,

            served_entities=[
                ServedEntityInput(
                    name=served_entity_name,

                    entity_name=(
                        REGISTERED_MODEL_NAME
                    ),

                    entity_version=(
                        champion_version
                    ),

                    workload_size=(
                        WORKLOAD_SIZE
                    ),

                    scale_to_zero_enabled=(
                        SCALE_TO_ZERO_ENABLED
                    ),
                )
            ],

            traffic_config=TrafficConfig(
                routes=[
                    Route(
                        served_model_name=(
                            served_entity_name
                        ),

                        traffic_percentage=100,
                    )
                ]
            ),

            timeout=timedelta(
                minutes=(
                    DEPLOYMENT_TIMEOUT_MINUTES
                )
            ),
        )
    )

    print(
        "Endpoint update completed."
    )


# -----------------------------------------------------
# Verify final endpoint version
# -----------------------------------------------------

final_endpoint = (
    workspace_client
    .serving_endpoints
    .get(
        name=ENDPOINT_NAME
    )
)

final_entities = (
    final_endpoint.config.served_entities
    or []
)

if len(final_entities) != 1:
    raise RuntimeError(
        "Unexpected served-entity count after "
        "deployment."
    )

deployed_version = str(
    final_entities[0].entity_version
)

print(
    f"Version after deployment: "
    f"{deployed_version}"
)

if deployed_version != champion_version:
    raise RuntimeError(
        "Deployment completed, but endpoint "
        "does not serve the Champion version. "
        f"Expected {champion_version}; "
        f"received {deployed_version}."
    )


# -----------------------------------------------------
# Load smoke-test payload
# -----------------------------------------------------

with open(
    SMOKE_TEST_PATH,
    "r",
    encoding="utf-8",
) as file:

    smoke_payload = json.load(file)


# -----------------------------------------------------
# Invoke endpoint
# -----------------------------------------------------

invocation_url = (
    f"{DATABRICKS_HOST}"
    f"/serving-endpoints/"
    f"{ENDPOINT_NAME}"
    f"/invocations"
)

headers = {
    "Authorization":
        f"Bearer {DATABRICKS_DEPLOYMENT_TOKEN}",

    "Content-Type":
        "application/json",
}

response = requests.post(
    invocation_url,
    headers=headers,
    json=smoke_payload,
    timeout=60,
)

if not response.ok:

    raise RuntimeError(
        "Smoke test failed. "
        f"Status={response.status_code}; "
        f"Response={response.text}"
    )

smoke_result = response.json()

print(
    "Smoke-test response:",
    json.dumps(
        smoke_result,
        indent=2,
    ),
)


# -----------------------------------------------------
# Validate response structure
# -----------------------------------------------------

predictions = smoke_result.get(
    "predictions"
)

if not predictions:
    raise RuntimeError(
        "Smoke test returned no predictions."
    )

first_prediction = predictions[0]

if isinstance(
    first_prediction,
    dict,
):

    forecast = first_prediction.get(
        "Forecast"
    )

else:

    forecast = first_prediction


if forecast is None:
    raise RuntimeError(
        "Smoke test response did not contain "
        "a Forecast value."
    )


print(
    f"Smoke-test forecast: {forecast}"
)

print(
    "DEPLOYMENT SUCCESSFUL"
)

print(
    f"{REGISTERED_MODEL_NAME} "
    f"version {champion_version} is serving "
    f"on endpoint {ENDPOINT_NAME}."
)