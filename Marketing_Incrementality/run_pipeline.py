# =============================================================================
# MARKETING INCREMENTALITY PIPELINE ORCHESTRATOR
# =============================================================================

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

LOG_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "logs"
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PIPELINE_LOG_FILE = (
    LOG_DIR
    / "marketing_incrementality_pipeline.log"
)


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger(
    "marketing_incrementality_pipeline"
)

logger.setLevel(
    logging.INFO
)

logger.handlers.clear()

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s"
)

console_handler = logging.StreamHandler()

console_handler.setFormatter(
    formatter
)

file_handler = logging.FileHandler(
    PIPELINE_LOG_FILE,
    mode="w",
    encoding="utf-8",
)

file_handler.setFormatter(
    formatter
)

logger.addHandler(
    console_handler
)

logger.addHandler(
    file_handler
)


# =============================================================================
# PIPELINE DEFINITION
# =============================================================================

PIPELINE = [

    {
        "id": "01",
        "name": "Load Data",
        "script": "01_load_data.py",
        "stage": "Data Preparation",
    },

    {
        "id": "02",
        "name": "Clean Data",
        "script": "02_clean_data.py",
        "stage": "Data Preparation",
    },

    {
        "id": "02a",
        "name": "Standardize DMAs",
        "script": "02a_standardize_dmas.py",
        "stage": "Data Preparation",
    },

    {
        "id": "03",
        "name": "Feature Engineering",
        "script": "03_feature_engineering.py",
        "stage": "Data Preparation",
    },

    {
        "id": "04",
        "name": "Exploratory Data Analysis",
        "script": "04_eda.py",
        "stage": "Exploratory Analysis",
    },

    {
        "id": "05",
        "name": "Seasonality Analysis",
        "script": "05_seasonality.py",
        "stage": "Exploratory Analysis",
    },

    {
        "id": "06",
        "name": "Cross Correlation",
        "script": "06_cross_correlation.py",
        "stage": "Exploratory Analysis",
    },

    {
        "id": "07",
        "name": "Panel Regression",
        "script": "07_panel_regression.py",
        "stage": "Modeling",
    },

    {
        "id": "08",
        "name": "Difference in Differences",
        "script": "08_difference_in_differences.py",
        "stage": "Modeling",
    },

    {
        "id": "09",
        "name": "Causal Impact / BSTS",
        "script": "09_causal_impact.py",
        "stage": "Causal Modeling",
    },

    {
        "id": "10",
        "name": "Causal Impact Validation",
        "script": "10_causal_impact_validation.py",
        "stage": "Causal Modeling",
    },

    {
        "id": "11",
        "name": "Causal Impact Robustness",
        "script": "11_causal_impact_robustness.py",
        "stage": "Causal Modeling",
    },

    {
        "id": "12",
        "name": "Reporting",
        "script": "12_reporting.py",
        "stage": "Reporting",
    },

    {
        "id": "13",
        "name": "Comp Traffic Analysis",
        "script": "13_comp_traffic_analysis.py",
        "stage": "Comp Traffic",
    },

    {
        "id": "14",
        "name": "BSTS / Comp Comparison",
        "script": "14_bsts_comp_comparison.py",
        "stage": "Comp Traffic",
    },

    {
        "id": "15",
        "name": "DMA Spend Explanatory Power",
        "script": "15_dma_spend_explanatory_power.py",
        "stage": "Spend Diagnostics",
    },

    {
            "id": "16",
            "name": "DMA Opportunity Analysis",
            "script": "16_dma_opportunity_analysis.py",
            "stage": "DMA Analytics",
    },

]


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """
    Parse command-line arguments.

    Examples
    --------
    Full pipeline:
        python run_pipeline.py

    Start from script 09:
        python run_pipeline.py --from 09

    Run through script 11:
        python run_pipeline.py --to 11

    Run scripts 09 through 11:
        python run_pipeline.py --from 09 --to 11

    Run only script 14:
        python run_pipeline.py --only 14
    """

    parser = argparse.ArgumentParser(
        description=(
            "Execute the Marketing Incrementality "
            "analysis pipeline."
        )
    )

    parser.add_argument(
        "--from",
        dest="from_step",
        type=str,
        default=None,
        help=(
            "Start pipeline execution from this step. "
            "Example: --from 09"
        ),
    )

    parser.add_argument(
        "--to",
        dest="to_step",
        type=str,
        default=None,
        help=(
            "Stop pipeline execution after this step. "
            "Example: --to 11"
        ),
    )

    parser.add_argument(
        "--only",
        dest="only_step",
        type=str,
        default=None,
        help=(
            "Run only one pipeline step. "
            "Example: --only 14"
        ),
    )

    return parser.parse_args()


# =============================================================================
# PIPELINE HELPERS
# =============================================================================

def get_step_index(
    step_id,
):
    """
    Return the pipeline list index for a given step ID.
    """

    for index, step in enumerate(
        PIPELINE
    ):

        if step["id"] == step_id:

            return index

    raise ValueError(
        f"Unknown pipeline step: {step_id}"
    )


def select_pipeline_steps(
    from_step=None,
    to_step=None,
    only_step=None,
):
    """
    Select pipeline steps based on CLI arguments.
    """

    if only_step is not None:

        index = get_step_index(
            only_step
        )

        return [
            PIPELINE[index]
        ]

    start_index = 0

    end_index = (
        len(PIPELINE) - 1
    )

    if from_step is not None:

        start_index = get_step_index(
            from_step
        )

    if to_step is not None:

        end_index = get_step_index(
            to_step
        )

    if start_index > end_index:

        raise ValueError(
            "--from step occurs after --to step."
        )

    return PIPELINE[
        start_index:
        end_index + 1
    ]


def validate_pipeline(
    steps,
):
    """
    Validate that all requested scripts exist.
    """

    missing_scripts = []

    for step in steps:

        script_path = (
            PROJECT_ROOT
            / step["script"]
        )

        if not script_path.exists():

            missing_scripts.append(
                str(script_path)
            )

    if missing_scripts:

        logger.error(
            "Missing pipeline scripts:"
        )

        for path in missing_scripts:

            logger.error(
                "  %s",
                path,
            )

        raise FileNotFoundError(
            "One or more pipeline scripts "
            "could not be found."
        )


def format_duration(
    seconds,
):
    """
    Format elapsed seconds for logging.
    """

    seconds = int(
        round(seconds)
    )

    minutes, seconds = divmod(
        seconds,
        60,
    )

    hours, minutes = divmod(
        minutes,
        60,
    )

    if hours > 0:

        return (
            f"{hours}h "
            f"{minutes}m "
            f"{seconds}s"
        )

    if minutes > 0:

        return (
            f"{minutes}m "
            f"{seconds}s"
        )

    return (
        f"{seconds}s"
    )


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

def run_script(
    step,
    position,
    total_steps,
):
    """
    Execute one pipeline script.

    Uses the same Python interpreter that launched
    this orchestrator.
    """

    script_path = (
        PROJECT_ROOT
        / step["script"]
    )

    logger.info("")
    logger.info("=" * 80)

    logger.info(
        "[%d/%d] %s | %s",
        position,
        total_steps,
        step["id"],
        step["name"],
    )

    logger.info(
        "Stage: %s",
        step["stage"],
    )

    logger.info(
        "Script: %s",
        script_path.name,
    )

    logger.info("=" * 80)

    start_time = time.perf_counter()

    try:

        subprocess.run(
            [
                sys.executable,
                str(script_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    except subprocess.CalledProcessError as exc:

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        logger.error("")
        logger.error("-" * 80)

        logger.error(
            "FAILED: %s",
            step["script"],
        )

        logger.error(
            "Exit code: %s",
            exc.returncode,
        )

        logger.error(
            "Runtime: %s",
            format_duration(
                elapsed
            ),
        )

        logger.error("-" * 80)

        raise

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    logger.info("")
    logger.info(
        "SUCCESS: %s",
        step["script"],
    )

    logger.info(
        "Runtime: %s",
        format_duration(
            elapsed
        ),
    )

    return elapsed


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def run_pipeline(
    steps,
):
    """
    Execute selected pipeline steps sequentially.
    """

    pipeline_start = (
        time.perf_counter()
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info(
        "MARKETING INCREMENTALITY PIPELINE"
    )
    logger.info("=" * 80)

    logger.info(
        "Project root: %s",
        PROJECT_ROOT,
    )

    logger.info(
        "Python executable: %s",
        sys.executable,
    )

    logger.info(
        "Steps scheduled: %d",
        len(steps),
    )

    logger.info(
        "Pipeline log: %s",
        PIPELINE_LOG_FILE,
    )

    logger.info("")

    logger.info(
        "Execution plan:"
    )

    for step in steps:

        logger.info(
            "  %s | %-30s | %s",
            step["id"],
            step["name"],
            step["stage"],
        )

    completed = []

    total_steps = len(
        steps
    )

    try:

        for position, step in enumerate(
            steps,
            start=1,
        ):

            elapsed = run_script(
                step=step,
                position=position,
                total_steps=total_steps,
            )

            completed.append(
                {
                    "id":
                        step["id"],

                    "script":
                        step["script"],

                    "name":
                        step["name"],

                    "runtime_seconds":
                        elapsed,
                }
            )

    except subprocess.CalledProcessError:

        pipeline_elapsed = (
            time.perf_counter()
            -
            pipeline_start
        )

        logger.error("")
        logger.error("=" * 80)
        logger.error(
            "PIPELINE FAILED"
        )
        logger.error("=" * 80)

        logger.error(
            "Completed steps: %d of %d",
            len(completed),
            total_steps,
        )

        logger.error(
            "Total runtime before failure: %s",
            format_duration(
                pipeline_elapsed
            ),
        )

        logger.error(
            "Execution stopped to prevent "
            "downstream scripts from running "
            "on incomplete outputs."
        )

        raise

    pipeline_elapsed = (
        time.perf_counter()
        -
        pipeline_start
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info(
        "PIPELINE COMPLETE"
    )
    logger.info("=" * 80)

    logger.info(
        "Completed steps: %d",
        len(completed),
    )

    logger.info(
        "Total runtime: %s",
        format_duration(
            pipeline_elapsed
        ),
    )

    logger.info("")
    logger.info(
        "Step runtimes:"
    )

    for result in completed:

        logger.info(
            "  %-4s %-40s %s",
            result["id"],
            result["script"],
            format_duration(
                result[
                    "runtime_seconds"
                ]
            ),
        )

    logger.info("")
    logger.info("=" * 80)
    logger.info(
        "SUCCESS"
    )
    logger.info("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    try:

        steps = select_pipeline_steps(
            from_step=args.from_step,
            to_step=args.to_step,
            only_step=args.only_step,
        )

        validate_pipeline(
            steps
        )

        run_pipeline(
            steps
        )

    except Exception:

        logger.exception(
            "Marketing incrementality "
            "pipeline execution failed."
        )

        raise


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()