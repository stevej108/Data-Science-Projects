import argparse
import math
import sys

import mlflow


# -------------------------------------------------
# Validation thresholds
# -------------------------------------------------

MAX_VALIDATION_WAPE = 0.15
MIN_RELATIVE_WAPE_IMPROVEMENT = 0.05


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Validate an MLflow candidate "
            "run for promotion."
        )
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help=(
            "MLflow run ID for the "
            "candidate model."
        ),
    )

    return parser.parse_args()


def get_required_metric(
    metrics,
    metric_name,
):

    if metric_name not in metrics:
        raise ValueError(
            f"Required metric "
            f"'{metric_name}' was not "
            f"found in the MLflow run."
        )

    value = metrics[
        metric_name
    ]

    if not math.isfinite(value):
        raise ValueError(
            f"Metric '{metric_name}' "
            f"is not finite: {value}"
        )

    return value


def validate_candidate(
    run_id,
):

    # -------------------------------------------------
    # Load candidate run
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

    metrics = (
        run.data.metrics
    )


    # -------------------------------------------------
    # Required metrics
    # -------------------------------------------------

    validation_wape = (
        get_required_metric(
            metrics,
            "validation_wape",
        )
    )

    naive_wape = (
        get_required_metric(
            metrics,
            "naive_wape",
        )
    )

    relative_improvement = (
        get_required_metric(
            metrics,
            "relative_wape_improvement",
        )
    )


    # -------------------------------------------------
    # Validation gates
    # -------------------------------------------------

    checks = {
        (
            "Candidate beats "
            "naive baseline"
        ):
            validation_wape
            < naive_wape,

        (
            "Validation WAPE <= "
            f"{MAX_VALIDATION_WAPE:.0%}"
        ):
            validation_wape
            <= MAX_VALIDATION_WAPE,

        (
            "Relative WAPE improvement "
            f">= "
            f"{MIN_RELATIVE_WAPE_IMPROVEMENT:.0%}"
        ):
            relative_improvement
            >= MIN_RELATIVE_WAPE_IMPROVEMENT,
    }


    # -------------------------------------------------
    # Validation summary
    # -------------------------------------------------

    print("=" * 60)
    print(
        "CANDIDATE VALIDATION"
    )
    print("=" * 60)

    print(
        "Run ID:",
        run_id,
    )

    print(
        "Validation WAPE:",
        f"{validation_wape:.2%}",
    )

    print(
        "Naive WAPE:",
        f"{naive_wape:.2%}",
    )

    print(
        "Relative improvement:",
        f"{relative_improvement:.2%}",
    )

    print("-" * 60)


    all_passed = True

    for (
        check_name,
        passed,
    ) in checks.items():

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{status}: "
            f"{check_name}"
        )

        if not passed:
            all_passed = False


    print("-" * 60)

    if all_passed:

        print(
            "VALIDATION RESULT: PASS"
        )

        print(
            "Candidate is eligible "
            "for promotion."
        )

    else:

        print(
            "VALIDATION RESULT: FAIL"
        )

        print(
            "Candidate is NOT eligible "
            "for promotion."
        )

    print("=" * 60)

    return all_passed


def main():

    args = parse_args()

    try:

        passed = (
            validate_candidate(
                args.run_id
            )
        )

    except Exception as exc:

        print("=" * 60)
        print(
            "CANDIDATE VALIDATION ERROR"
        )
        print("=" * 60)

        print(
            str(exc)
        )

        print("=" * 60)

        sys.exit(1)


    if not passed:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()