import mlflow
import mlflow.pyfunc
import pandas as pd
import xgboost

from mlflow.models import (
    infer_signature,
)

import sys
from pathlib import Path


try:
    _THIS_FILE = (
        Path(__file__)
        .resolve()
    )
except NameError:
    # __file__ is undefined when this file
    # is run interactively (e.g. from the
    # workspace editor), so fall back to
    # this file's known workspace path.
    _THIS_FILE = Path(
        "/Workspace/Users/stevej108@gmail.com/"
        "Data-Science-Projects/"
        "Databricks-Forecasting/jobs/"
        "train_candidate.py"
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

# This entry point is designed to execute on
# Databricks compute. Core modeling logic is
# tested locally under src/databricks_forecasting.

from pyspark.sql import (
    SparkSession,
)

from databricks_forecasting.config import (
    CANDIDATE_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_FEATURES,
    SOURCE_TABLE,
    VALIDATION_CUTOFF,
)

from databricks_forecasting.evaluation import (
    evaluate_naive,
    relative_wape_improvement,
)

from databricks_forecasting.model_wrapper import (
    MarketingForecastPyFunc,
)

from databricks_forecasting.training import (
    create_modeling_dataset,
    create_train_valid_split,
    score_model,
    train_xgboost,
)


def main():

    # -------------------------------------------------
    # Spark
    # -------------------------------------------------

    spark = (
        SparkSession.builder
        .getOrCreate()
    )


    # -------------------------------------------------
    # Load training data
    # -------------------------------------------------

    panel = (
        spark.table(
            SOURCE_TABLE
        )
        .toPandas()
    )

    print(
        f"Source rows: {len(panel):,}"
    )


    # -------------------------------------------------
    # Build modeling dataset
    # -------------------------------------------------

    model_df = (
        create_modeling_dataset(
            panel
        )
    )

    train, valid = (
        create_train_valid_split(
            model_df=model_df,
            cutoff_date=(
                VALIDATION_CUTOFF
            ),
        )
    )


    # -------------------------------------------------
    # Benchmark
    # -------------------------------------------------

    naive_metrics = (
        evaluate_naive(
            valid
        )
    )


    # -------------------------------------------------
    # Candidate training
    # -------------------------------------------------

    candidate_model = (
        train_xgboost(
            train=train,
            params=CANDIDATE_PARAMS,
        )
    )

    predictions, candidate_metrics = (
        score_model(
            model=candidate_model,
            valid=valid,
        )
    )


    relative_improvement = (
        relative_wape_improvement(
            candidate_wape=(
                candidate_metrics[
                    "WAPE"
                ]
            ),
            benchmark_wape=(
                naive_metrics[
                    "WAPE"
                ]
            ),
        )
    )


    # -------------------------------------------------
    # Package candidate
    # -------------------------------------------------

    candidate_wrapper = (
        MarketingForecastPyFunc(
            xgb_model=(
                candidate_model
            ),
            feature_order=(
                MODEL_FEATURES
            ),
        )
    )

    input_example = (
        valid[
            MODEL_FEATURES
        ]
        .head(5)
        .copy()
    )

    output_example = (
        candidate_wrapper.predict(
            context=None,
            model_input=input_example,
        )
    )

    signature = infer_signature(
        input_example,
        output_example,
    )


    # -------------------------------------------------
    # MLflow
    # -------------------------------------------------

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="xgb_candidate"
    ) as run:

        mlflow.log_params(
            CANDIDATE_PARAMS
        )

        mlflow.log_param(
            "validation_cutoff",
            VALIDATION_CUTOFF,
        )

        mlflow.log_param(
            "feature_count",
            len(MODEL_FEATURES),
        )

        mlflow.log_param(
            "training_rows",
            len(train),
        )

        mlflow.log_param(
            "validation_rows",
            len(valid),
        )

        mlflow.log_metrics({

            "validation_rmse":
                candidate_metrics[
                    "RMSE"
                ],

            "validation_mae":
                candidate_metrics[
                    "MAE"
                ],

            "validation_mape":
                candidate_metrics[
                    "MAPE"
                ],

            "validation_wape":
                candidate_metrics[
                    "WAPE"
                ],

            "validation_r2":
                candidate_metrics[
                    "R2"
                ],

            "naive_wape":
                naive_metrics[
                    "WAPE"
                ],

            "relative_wape_improvement":
                relative_improvement,
        })


        # ---------------------------------------------
        # Log packaged model
        # ---------------------------------------------

        mlflow.pyfunc.log_model(
            name="model",

            python_model=(
                candidate_wrapper
            ),

            signature=signature,

            input_example=(
                input_example
            ),

            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"pandas=={pd.__version__}",
                f"xgboost=={xgboost.__version__}",
            ],
        )


        candidate_run_id = (
            run.info.run_id
        )


    # -------------------------------------------------
    # Execution summary
    # -------------------------------------------------

    print("=" * 60)
    print("CANDIDATE TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Run ID:",
        candidate_run_id,
    )

    print(
        "Candidate WAPE:",
        f"{candidate_metrics['WAPE']:.2%}",
    )

    print(
        "Naive WAPE:",
        f"{naive_metrics['WAPE']:.2%}",
    )

    print(
        "Relative improvement:",
        f"{relative_improvement:.2%}",
    )

    print("=" * 60)


if __name__ == "__main__":
    main()