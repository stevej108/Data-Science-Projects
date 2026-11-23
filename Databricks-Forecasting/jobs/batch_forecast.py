import argparse
import sys

from pathlib import Path

import mlflow
import pandas as pd

from mlflow import MlflowClient

from pyspark.sql import (
    SparkSession,
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
        "batch_forecast.py"
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
    FORECAST_HORIZON,
    FORECAST_TABLE,
    MODEL_ALIAS,
    MODEL_FEATURES,
    NATIONAL_FORECAST_TABLE,
    REGISTERED_MODEL_NAME,
    SOURCE_TABLE,
    STATIC_FEATURES,
)

from databricks_forecasting.forecasting import (
    create_national_forecast,
    generate_batch_forecast,
    prepare_future_frame,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate future traffic forecasts "
            "and write Delta forecast tables."
        )
    )

    parser.add_argument(
        "--model-version",
        required=True,
        help=(
            "Deployed registered model version "
            "to use for batch forecasting."
        ),
    )

    return parser.parse_args()


def write_delta_table(
    spark,
    pandas_df,
    table_name,
):
    """
    Overwrite a managed Delta table with the
    latest authoritative forecast snapshot.
    """

    if pandas_df.empty:
        raise ValueError(
            f"Cannot write empty table "
            f"'{table_name}'."
        )

    spark_df = (
        spark.createDataFrame(
            pandas_df
        )
    )

    (
        spark_df
        .write
        .format("delta")
        .mode("overwrite")
        .option(
            "overwriteSchema",
            "true",
        )
        .saveAsTable(
            table_name
        )
    )

    written_rows = (
        spark.table(
            table_name
        )
        .count()
    )

    if written_rows != len(pandas_df):
        raise RuntimeError(
            "Delta row-count verification "
            f"failed for '{table_name}'. "
            f"Expected {len(pandas_df)}, "
            f"found {written_rows}."
        )

    return written_rows


def run_batch_forecast(
    model_version,
    spark=None,
    mlflow_client=None,
):
    """
    Load the confirmed Champion version,
    produce the future forecast and overwrite
    the current Delta forecast snapshots.
    """

    model_version = str(
        model_version
    )

    if spark is None:
        spark = (
            SparkSession.builder
            .getOrCreate()
        )

    mlflow.set_registry_uri(
        "databricks-uc"
    )

    if mlflow_client is None:
        mlflow_client = (
            MlflowClient()
        )

    # -------------------------------------------------
    # Verify model lineage
    # -------------------------------------------------

    champion = (
        mlflow_client
        .get_model_version_by_alias(
            name=(
                REGISTERED_MODEL_NAME
            ),
            alias=MODEL_ALIAS,
        )
    )

    champion_version = str(
        champion.version
    )

    if (
        champion_version
        != model_version
    ):
        raise ValueError(
            "Batch forecast model version "
            "does not match Champion. "
            f"Requested version: "
            f"{model_version}. "
            f"Champion version: "
            f"{champion_version}."
        )

    # Use the explicit version passed through
    # Tasks 03 and 04, not a floating alias.
    model_uri = (
        f"models:/{REGISTERED_MODEL_NAME}/"
        f"{model_version}"
    )

    champion_model = (
        mlflow.pyfunc.load_model(
            model_uri
        )
    )

    # -------------------------------------------------
    # Historical panel
    # -------------------------------------------------

    panel = (
        spark.table(
            SOURCE_TABLE
        )
        .toPandas()
    )

    panel["Month"] = (
        pd.to_datetime(
            panel["Month"]
        )
    )

    print(
        "Historical rows:",
        f"{len(panel):,}",
    )

    print(
        "Latest historical month:",
        panel["Month"].max(),
    )

    # -------------------------------------------------
    # Future forecast frame
    # -------------------------------------------------

    future = (
        prepare_future_frame(
            panel=panel,
            horizon=(
                FORECAST_HORIZON
            ),
            static_features=(
                STATIC_FEATURES
            ),
        )
    )

    print(
        "Future months:",
        future["Month"].nunique(),
    )

    print(
        "Future DMAs:",
        future["DMA"].nunique(),
    )

    print(
        "Future rows:",
        f"{len(future):,}",
    )

    if (
        future["Month"].min()
        <= panel["Month"].max()
    ):
        raise ValueError(
            "Future forecast periods overlap "
            "historical periods."
        )

    # -------------------------------------------------
    # Recursive batch inference
    # -------------------------------------------------

    forecast = (
        generate_batch_forecast(
            panel=panel,
            future=future,
            champion_model=(
                champion_model
            ),
            model_features=(
                MODEL_FEATURES
            ),
        )
    )

    forecast[
        "Model_Version"
    ] = model_version

    national_forecast = (
        create_national_forecast(
            forecast
        )
    )

    national_forecast[
        "Model_Version"
    ] = model_version

    # -------------------------------------------------
    # Delta outputs
    # -------------------------------------------------

    forecast_rows = (
        write_delta_table(
            spark=spark,
            pandas_df=forecast,
            table_name=(
                FORECAST_TABLE
            ),
        )
    )

    national_rows = (
        write_delta_table(
            spark=spark,
            pandas_df=(
                national_forecast
            ),
            table_name=(
                NATIONAL_FORECAST_TABLE
            ),
        )
    )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    print("=" * 60)
    print(
        "BATCH FORECAST COMPLETE"
    )
    print("=" * 60)

    print(
        "Model version:",
        model_version,
    )

    print(
        "Forecast table:",
        FORECAST_TABLE,
    )

    print(
        "Forecast rows:",
        forecast_rows,
    )

    print(
        "National forecast table:",
        NATIONAL_FORECAST_TABLE,
    )

    print(
        "National rows:",
        national_rows,
    )

    print("=" * 60)

    return {
        "model_version":
            model_version,

        "forecast_rows":
            forecast_rows,

        "national_forecast_rows":
            national_rows,

        "forecast_table":
            FORECAST_TABLE,

        "national_forecast_table":
            NATIONAL_FORECAST_TABLE,
    }


def main():

    args = parse_args()

    try:
        return run_batch_forecast(
            model_version=(
                args.model_version
            )
        )

    except Exception as exc:
        print("=" * 60)
        print(
            "BATCH FORECAST ERROR"
        )
        print("=" * 60)

        print(
            str(exc)
        )

        print("=" * 60)

        sys.exit(1)


if __name__ == "__main__":
    main()