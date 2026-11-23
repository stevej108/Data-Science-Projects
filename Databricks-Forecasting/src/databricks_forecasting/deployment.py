from databricks.sdk.service.serving import (
    ServedEntityInput,
)


def get_served_entities(
    endpoint,
):
    """
    Return the endpoint's currently deployed
    served entities.

    Raises:
        ValueError: If the endpoint has no
        active serving configuration.
    """

    if endpoint.config is None:
        raise ValueError(
            "Serving endpoint does not have "
            "an active configuration."
        )

    served_entities = (
        endpoint.config.served_entities
    )

    if not served_entities:
        raise ValueError(
            "Serving endpoint does not have "
            "any served entities."
        )

    return served_entities


def get_deployed_model_version(
    endpoint,
    registered_model_name,
):
    """
    Return the deployed version for the
    requested registered model.

    Raises:
        ValueError: If the registered model
        is not deployed on the endpoint.
    """

    served_entities = (
        get_served_entities(
            endpoint
        )
    )

    matching_entities = [
        entity
        for entity in served_entities
        if (
            entity.entity_name
            == registered_model_name
        )
    ]

    if not matching_entities:
        raise ValueError(
            "Registered model "
            f"'{registered_model_name}' "
            "is not currently deployed on "
            "the endpoint."
        )

    if len(matching_entities) > 1:
        raise ValueError(
            "Multiple served entities were "
            "found for registered model "
            f"'{registered_model_name}'."
        )

    return str(
        matching_entities[
            0
        ].entity_version
    )


def build_updated_served_entities(
    endpoint,
    registered_model_name,
    target_version,
):
    """
    Create an updated served-entity list
    while preserving the endpoint's existing
    compute and environment configuration.
    """

    served_entities = (
        get_served_entities(
            endpoint
        )
    )

    updated_entities = []

    target_found = False

    for entity in served_entities:

        entity_version = (
            str(entity.entity_version)
        )

        if (
            entity.entity_name
            == registered_model_name
        ):
            entity_version = (
                str(target_version)
            )

            target_found = True

        updated_entities.append(
            ServedEntityInput(
                name=entity.name,

                entity_name=(
                    entity.entity_name
                ),

                entity_version=(
                    entity_version
                ),

                workload_size=(
                    entity.workload_size
                ),

                workload_type=(
                    entity.workload_type
                ),

                scale_to_zero_enabled=(
                    entity.scale_to_zero_enabled
                ),

                environment_vars=(
                    entity.environment_vars
                ),
            )
        )

    if not target_found:
        raise ValueError(
            "Registered model "
            f"'{registered_model_name}' "
            "is not currently deployed on "
            "the endpoint."
        )

    return updated_entities


def update_endpoint_to_version(
    workspace_client,
    endpoint_name,
    registered_model_name,
    target_version,
):
    """
    Update a serving endpoint to the requested
    registered-model version.

    The operation is idempotent. If the
    endpoint already serves the target
    version, no update is submitted.

    Returns:
        A dictionary containing deployment
        details.
    """

    target_version = (
        str(target_version)
    )


    # -------------------------------------------------
    # Inspect current endpoint
    # -------------------------------------------------

    endpoint = (
        workspace_client
        .serving_endpoints
        .get(
            name=endpoint_name
        )
    )

    deployed_version = (
        get_deployed_model_version(
            endpoint=endpoint,
            registered_model_name=(
                registered_model_name
            ),
        )
    )


    # -------------------------------------------------
    # No-op when already current
    # -------------------------------------------------

    if deployed_version == target_version:

        return {
            "endpoint_name":
                endpoint_name,

            "previous_version":
                deployed_version,

            "deployed_version":
                deployed_version,

            "updated":
                False,
        }


    # -------------------------------------------------
    # Build updated configuration
    # -------------------------------------------------

    updated_entities = (
        build_updated_served_entities(
            endpoint=endpoint,
            registered_model_name=(
                registered_model_name
            ),
            target_version=(
                target_version
            ),
        )
    )

    traffic_config = (
        endpoint.config.traffic_config
    )


    # -------------------------------------------------
    # Update and wait
    # -------------------------------------------------

    workspace_client.serving_endpoints.update_config_and_wait(
        name=endpoint_name,
        served_entities=(
            updated_entities
        ),
        traffic_config=(
            traffic_config
        ),
    )


    # -------------------------------------------------
    # Verify deployed version
    # -------------------------------------------------

    updated_endpoint = (
        workspace_client
        .serving_endpoints
        .get(
            name=endpoint_name
        )
    )

    verified_version = (
        get_deployed_model_version(
            endpoint=updated_endpoint,
            registered_model_name=(
                registered_model_name
            ),
        )
    )

    if verified_version != target_version:
        raise RuntimeError(
            "Endpoint deployment "
            "verification failed. "
            f"Expected version "
            f"{target_version}, but "
            f"endpoint serves version "
            f"{verified_version}."
        )

    return {
        "endpoint_name":
            endpoint_name,

        "previous_version":
            deployed_version,

        "deployed_version":
            verified_version,

        "updated":
            True,
    }