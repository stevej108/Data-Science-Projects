"""
09_causal_impact.py

Causal Impact Analysis
----------------------

Purpose
-------
Evaluate if traffic changes more in markets exposed to increased marketing compared with markets that were less exposed.

Outputs
-------
Regression results
Treatment effect estimates
Parallel trends diagnostics
Figures
Tables
HTML report
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import arviz as az
import pymc as pm
import pytensor
import json

from config import (
    PROCESSED_DATA_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    OUTPUT_DIR,
    MODEL_DIR
    
)

def configure_pytensor():
    """
    Configure PyTensor for environments without
    an available C/C++ compiler.
    """

    logger.info(
        "Configuring PyTensor..."
    )

    pytensor.config.cxx = ""

    logger.info(
        "PyTensor CXX compiler disabled."
    )

#=========================================================
# Output Folders
#=========================================================

REPORT_DIR = OUTPUT_DIR / "reports"


def create_output_folders():

    folders = [

        # FIGURE_DIR / "causal_impact",

        # FIGURE_DIR / "causal_impact" / "parallel_trends",

        # FIGURE_DIR / "causal_impact" / "effects",

        # FIGURE_DIR / "causal_impact" / "diagnostics",

        # TABLE_DIR / "causal_impact",

        # REPORT_DIR

    ]

    for folder in folders:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger(__name__)

#=========================================================
# Load Data
#=========================================================

def load_model_panel():

    logger.info("Loading model panel...")

    panel_df = pd.read_parquet(

        PROCESSED_DATA_DIR /
        "model_panel.parquet"

    )

    panel_df["Month"] = pd.to_datetime(
        panel_df["Month"]
    )

    panel= panel_df.copy()
    panel = panel[panel["DMA"].notna()]
    panel = panel[panel["Traffic"].notna()]

    logger.info(
        f"Loaded {len(panel):,} rows"
    )

    logger.info(
        f"{panel['DMA'].nunique()} DMAs"
    )

    jan_2026 = (
        panel.loc[
            panel["Month"].eq(pd.Timestamp("2026-01-01")),
            [
                "DMA",
                "Month",
                "Spend",
                "Traffic",
                
            ]
        ]
        .sort_values("DMA")
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("JANUARY 2026 TRAFFIC QA")
    logger.info("=" * 70)

    logger.info(
        "\n%s",
        jan_2026.to_string(index=False)
    )

    logger.info("")
    logger.info(
        f"January 2026 rows: {len(jan_2026)}"
    )

    logger.info(
        f"January 2026 missing Traffic: "
        f"{jan_2026['Traffic'].isna().sum()}"
    )

    return panel

#=========================================================
# Validation
#=========================================================

def validate_panel(panel):

    logger.info("=" * 70)
    logger.info("MODEL PANEL")
    logger.info("=" * 70)

    logger.info(
        f"Rows     : {len(panel):,}"
    )

    logger.info(
        f"DMAs     : {panel['DMA'].nunique()}"
    )

    logger.info(
        f"Months   : {panel['Month'].nunique()}"
    )

    logger.info(
        f"Date Min : {panel['Month'].min().date()}"
    )

    logger.info(
        f"Date Max : {panel['Month'].max().date()}"
    )

    logger.info("")



#=========================================================
# Figure Helper
#=========================================================

def save_plot(

    fig,

    folder,

    filename

):

    path = (

        FIGURE_DIR
        / "causal_impact"
        / folder
        / f"{filename}.png"

    )

    fig.tight_layout()

    fig.savefig(

        path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close(fig)

    logger.info(
        f"Saved {filename}"
    )


#=========================================================
# Prepare DMA Time Series
#=========================================================

def prepare_dma_timeseries(
    panel,
    intervention_date,
    traffic_col="Traffic",
    spend_col="Spend"
):
    """
    Prepare monthly DMA-level time series for BSTS modeling.

    This function prepares the complete DMA-level dataset.
    Individual treated DMAs are selected later in the workflow
    and passed to prepare_causal_impact_inputs().

    Parameters
    ----------
    panel : pd.DataFrame
        Model panel containing DMA/month observations.

    intervention_date : pd.Timestamp
        Date separating pre- and post-intervention periods.

    traffic_col : str
        Traffic outcome column.

    spend_col : str
        Marketing spend column.

    Returns
    -------
    pd.DataFrame
        Clean DMA-level modeling dataset containing all DMAs.
    """

    logger.info(
        "Preparing DMA-level time series..."
    )

    #-----------------------------------------------------
    # Validate inputs
    #-----------------------------------------------------

    if not isinstance(panel, pd.DataFrame):

        raise TypeError(
            "panel must be a pandas DataFrame."
        )

    required_columns = [
        "DMA",
        "Month",
        traffic_col,
        spend_col
    ]

    missing = [
        col
        for col in required_columns
        if col not in panel.columns
    ]

    if missing:

        raise ValueError(
            f"Panel is missing required columns: {missing}"
        )

    #-----------------------------------------------------
    # Copy data
    #-----------------------------------------------------

    dma_data = panel[
        required_columns
    ].copy()

    #-----------------------------------------------------
    # Standardize dates
    #-----------------------------------------------------

    dma_data["Month"] = pd.to_datetime(
        dma_data["Month"],
        errors="coerce"
    )

    intervention_date = pd.Timestamp(
        intervention_date
    )

    #-----------------------------------------------------
    # Validate dates
    #-----------------------------------------------------

    if dma_data["Month"].isna().any():

        invalid_dates = (
            dma_data["Month"].isna().sum()
        )

        raise ValueError(
            f"Found {invalid_dates} invalid Month values."
        )

    #-----------------------------------------------------
    # Sort data
    #-----------------------------------------------------

    dma_data = (
        dma_data
        .sort_values(
            ["DMA", "Month"]
        )
        .reset_index(drop=True)
    )

    #-----------------------------------------------------
    # Check for duplicate DMA-month observations
    #-----------------------------------------------------

    duplicates = (
        dma_data
        .duplicated(
            subset=["DMA", "Month"]
        )
        .sum()
    )

    if duplicates > 0:

        raise ValueError(
            f"Found {duplicates} duplicate "
            "DMA-month observations."
        )

    #-----------------------------------------------------
    # Intervention indicator
    #-----------------------------------------------------

    dma_data["Post"] = (
        dma_data["Month"] >= intervention_date
    ).astype(int)

    #-----------------------------------------------------
    # DMA-level time index
    #
    # Each DMA receives its own sequential time index.
    #-----------------------------------------------------

    dma_data["Time_Index"] = (
        dma_data
        .groupby("DMA")
        .cumcount()
    )

    #-----------------------------------------------------
    # Month of year
    #-----------------------------------------------------

    dma_data["Month_of_Year"] = (
        dma_data["Month"].dt.month
    )

    #-----------------------------------------------------
    # Missing-value diagnostics
    #-----------------------------------------------------

    missing_traffic = (
        dma_data[traffic_col]
        .isna()
        .sum()
    )

    missing_spend = (
        dma_data[spend_col]
        .isna()
        .sum()
    )

    if missing_traffic > 0:

        logger.warning(
            f"Traffic contains {missing_traffic} "
            "missing observations."
        )

    if missing_spend > 0:

        logger.warning(
            f"Spend contains {missing_spend} "
            "missing observations."
        )

    #-----------------------------------------------------
    # Dataset diagnostics
    #-----------------------------------------------------

    n_dmas = (
        dma_data["DMA"]
        .nunique()
    )

    n_months = (
        dma_data["Month"]
        .nunique()
    )

    pre_months = (
        dma_data.loc[
            dma_data["Post"] == 0,
            "Month"
        ]
        .nunique()
    )

    post_months = (
        dma_data.loc[
            dma_data["Post"] == 1,
            "Month"
        ]
        .nunique()
    )

    logger.info(
        f"DMA time series prepared: "
        f"{n_dmas} DMAs | "
        f"{n_months} months"
    )

    logger.info(
        f"Period: "
        f"{dma_data['Month'].min().strftime('%Y-%m')} "
        f"to "
        f"{dma_data['Month'].max().strftime('%Y-%m')}"
    )

    logger.info(
        f"Pre-treatment months: {pre_months}"
    )

    logger.info(
        f"Post-treatment months: {post_months}"
    )

    logger.info(
        f"Observations: {len(dma_data)}"
    )

    #-----------------------------------------------------
    # Export prepared dataset
    #-----------------------------------------------------

    output_path = (
        TABLE_DIR /
        "bsts" /
        "dma_timeseries.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dma_data.to_csv(
        output_path,
        index=False
    )

    logger.info(
        f"Saved DMA time series to {output_path}"
    )

    return dma_data



def get_treated_dmas(
    panel,
    intervention_date,
    spend_col="Spend",
    treatment_quantile=0.75
):
    """
    Identify treated DMAs for the BSTS / Causal Impact analysis.

    Treatment is defined using the DMA-level change in average
    marketing spend between the pre- and post-intervention periods.

    A DMA is classified as treated when its Spend_Increase is
    greater than or equal to the specified treatment quantile.

    Parameters
    ----------
    panel : pd.DataFrame
        Model panel containing DMA, Month, and Spend.

    intervention_date : pd.Timestamp
        Date separating pre- and post-intervention periods.

    spend_col : str, default="Spend"
        Marketing spend column.

    treatment_quantile : float, default=0.75
        Quantile threshold used to identify treated DMAs.

    Returns
    -------
    treated_dmas : list
        List of treated DMA names.

    treatment_summary : pd.DataFrame
        DMA-level treatment summary containing pre/post spend,
        spend increase, and treatment indicator.
    """

    logger.info(
        "Identifying treated DMAs..."
    )

    #=====================================================
    # Validate inputs
    #=====================================================

    required_columns = [
        "DMA",
        "Month",
        spend_col
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in panel.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns for treatment "
            f"identification: {missing_columns}"
        )

    intervention_date = pd.Timestamp(
        intervention_date
    )

    if not 0 < treatment_quantile < 1:

        raise ValueError(
            "treatment_quantile must be between 0 and 1."
        )

    #=====================================================
    # Prepare data
    #=====================================================

    data = panel[
        [
            "DMA",
            "Month",
            spend_col
        ]
    ].copy()

    data["Month"] = pd.to_datetime(
        data["Month"]
    )

    #=====================================================
    # Create pre/post indicators
    #=====================================================

    data["Post"] = (
        data["Month"] >= intervention_date
    ).astype(int)

    #=====================================================
    # Calculate DMA-level pre/post spend
    #=====================================================

    spend_summary = (

        data

        .groupby(
            ["DMA", "Post"],
            as_index=False
        )[spend_col]

        .mean()

        .pivot(
            index="DMA",
            columns="Post",
            values=spend_col
        )

        .rename(
            columns={
                0: "Pre_Spend",
                1: "Post_Spend"
            }
        )

        .reset_index()

    )

    #=====================================================
    # Validate pre/post coverage
    #=====================================================

    missing_pre = spend_summary[
        "Pre_Spend"
    ].isna()

    missing_post = spend_summary[
        "Post_Spend"
    ].isna()

    if missing_pre.any():

        missing_dmas = (
            spend_summary.loc[
                missing_pre,
                "DMA"
            ]
            .tolist()
        )

        logger.warning(
            f"{len(missing_dmas)} DMAs have no "
            f"pre-treatment spend observations."
        )

    if missing_post.any():

        missing_dmas = (
            spend_summary.loc[
                missing_post,
                "DMA"
            ]
            .tolist()
        )

        logger.warning(
            f"{len(missing_dmas)} DMAs have no "
            f"post-treatment spend observations."
        )

    #=====================================================
    # Calculate spend change
    #=====================================================

    spend_summary["Spend_Increase"] = (
        spend_summary["Post_Spend"]
        -
        spend_summary["Pre_Spend"]
    )

    #=====================================================
    # Treatment threshold
    #=====================================================

    valid_increases = (
        spend_summary[
            "Spend_Increase"
        ]
        .dropna()
    )

    if valid_increases.empty:

        raise ValueError(
            "Unable to calculate DMA-level spend "
            "increases."
        )

    treatment_threshold = (
        valid_increases
        .quantile(treatment_quantile)
    )

    logger.info(
        f"Treatment spend-increase threshold "
        f"({treatment_quantile:.0%} quantile): "
        f"{treatment_threshold:,.2f}"
    )

    #=====================================================
    # Assign treatment
    #=====================================================

    spend_summary["Treatment"] = (
        spend_summary["Spend_Increase"]
        >= treatment_threshold
    ).astype(int)

    #=====================================================
    # Sort treatment summary
    #=====================================================

    treatment_summary = (
        spend_summary
        .sort_values(
            "Spend_Increase",
            ascending=False
        )
        .reset_index(drop=True)
    )

    treated_dmas = (
        treatment_summary.loc[
            treatment_summary["Treatment"] == 1,
            "DMA"
        ]
        .tolist()
    )

    #=====================================================
    # Validate treatment group
    #=====================================================

    if len(treated_dmas) == 0:

        raise ValueError(
            "No treated DMAs were identified."
        )

    #=====================================================
    # Logging
    #=====================================================

    logger.info(
        f"Treated DMAs identified: "
        f"{len(treated_dmas)}"
    )

    logger.info(
        f"Control DMAs identified: "
        f"{len(treatment_summary) - len(treated_dmas)}"
    )

    logger.info(
        "Treated DMA list:"
    )

    for dma in treated_dmas:

        spend_increase = (
            treatment_summary.loc[
                treatment_summary["DMA"] == dma,
                "Spend_Increase"
            ]
            .iloc[0]
        )

        logger.info(
            f"  {dma}: "
            f"{spend_increase:,.0f} spend increase"
        )

    #=====================================================
    # Export
    #=====================================================

    output_path = (
        TABLE_DIR
        / "bsts"
        / "treated_dmas.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    treatment_summary.to_csv(
        output_path,
        index=False
    )

    logger.info(
        f"Treated DMA summary saved to "
        f"{output_path}"
    )

    return (
        treated_dmas,
        treatment_summary
    )


def get_candidate_controls(
    panel,
    treated_dmas
):
    """
    Identify candidate control DMAs for the BSTS /
    Causal Impact analysis.

    Candidate controls are all DMAs that are not included
    in the treated DMA list.

    This function establishes the eligible control pool.
    It does not perform correlation analysis or select
    the final control DMAs.

    Parameters
    ----------
    panel : pd.DataFrame
        Model panel containing DMA/month observations.

    treated_dmas : list
        List of treated DMA names identified upstream.

    Returns
    -------
    candidate_controls : list
        List of eligible untreated DMA names.

    control_summary : pd.DataFrame
        Summary of candidate control DMAs.
    """

    logger.info(
        "Identifying candidate control DMAs..."
    )

    #-----------------------------------------------------
    # Validate required columns
    #-----------------------------------------------------

    required_columns = [
        "DMA",
        "Month",
        "Traffic"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in panel.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns for candidate "
            f"control identification: {missing_columns}"
        )

    #-----------------------------------------------------
    # Validate treated DMA list
    #-----------------------------------------------------

    if treated_dmas is None:

        raise ValueError(
            "treated_dmas cannot be None."
        )

    treated_dmas = set(
        treated_dmas
    )

    if len(treated_dmas) == 0:

        raise ValueError(
            "treated_dmas is empty. Cannot identify "
            "candidate controls."
        )

    #-----------------------------------------------------
    # Identify all DMAs
    #-----------------------------------------------------

    all_dmas = set(
        panel["DMA"]
        .dropna()
        .unique()
    )

    if len(all_dmas) == 0:

        raise ValueError(
            "No DMA observations found in panel."
        )

    #-----------------------------------------------------
    # Initial candidate pool
    #
    # All DMAs not identified as treated.
    #-----------------------------------------------------

    candidate_dmas = sorted(
        all_dmas - treated_dmas
    )

    if len(candidate_dmas) == 0:

        raise ValueError(
            "No candidate control DMAs remain after "
            "excluding treated DMAs."
        )

    #-----------------------------------------------------
    # Build candidate summary
    #-----------------------------------------------------

    control_summary = (

        panel.loc[
            panel["DMA"].isin(candidate_dmas)
        ]

        .groupby("DMA")

        .agg(

            Observations=(
                "Traffic",
                "count"
            ),

            Months=(
                "Month",
                "nunique"
            ),

            Missing_Traffic=(
                "Traffic",
                lambda x: x.isna().sum()
            ),

            Avg_Traffic=(
                "Traffic",
                "mean"
            )

        )

        .reset_index()

    )

    #-----------------------------------------------------
    # Add treatment flag for reporting only
    #
    # This is derived from treated_dmas rather than
    # requiring Treatment to exist in the panel.
    #-----------------------------------------------------

    control_summary["Is_Treated"] = (
        control_summary["DMA"]
        .isin(treated_dmas)
        .astype(int)
    )

    #-----------------------------------------------------
    # Validate no treated DMAs slipped through
    #-----------------------------------------------------

    treated_in_controls = (

        control_summary.loc[
            control_summary["Is_Treated"] == 1,
            "DMA"
        ]

        .tolist()

    )

    if treated_in_controls:

        raise ValueError(
            "Treated DMAs found in candidate control pool: "
            f"{treated_in_controls}"
        )

    #-----------------------------------------------------
    # Logging
    #-----------------------------------------------------

    logger.info(
        f"Total DMAs in panel: "
        f"{len(all_dmas)}"
    )

    logger.info(
        f"Treated DMAs excluded: "
        f"{len(treated_dmas)}"
    )

    logger.info(
        f"Candidate control DMAs identified: "
        f"{len(candidate_dmas)}"
    )

    #-----------------------------------------------------
    # Export
    #-----------------------------------------------------

    output_path = (
        TABLE_DIR
        / "bsts"
        / "candidate_controls.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    control_summary.to_csv(
        output_path,
        index=False
    )

    logger.info(
        f"Candidate control summary saved to "
        f"{output_path}"
    )

    return (
        candidate_dmas,
        control_summary
    )

#=========================================================
# Build Control Matrix
#=========================================================

def build_control_matrix(
    dma_timeseries,
    candidate_controls,
    intervention_date
):
    """
    Build a DMA-by-month control matrix for BSTS/Causal Impact.

    The matrix contains monthly traffic for candidate control
    DMAs during the pre-treatment period. Each DMA becomes a
    separate control series that can later be evaluated for
    correlation and suitability as a counterfactual.

    Parameters
    ----------
    dma_timeseries : pd.DataFrame
        DMA-level monthly time series containing at minimum:
        DMA, Month, and Traffic.

    candidate_controls : list
        DMA names identified as potential control markets.

    intervention_date : pd.Timestamp
        Start date of the intervention.

    Returns
    -------
    pd.DataFrame
        Wide-format control matrix with Month as the index
        and each candidate control DMA as a column.
    """

    logger.info(
        "Building control matrix..."
    )

    #-----------------------------------------------------
    # Validate inputs
    #-----------------------------------------------------

    required_columns = [
        "DMA",
        "Month",
        "Traffic"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in dma_timeseries.columns
    ]

    if missing_columns:

        raise ValueError(
            "dma_timeseries is missing required columns: "
            f"{missing_columns}"
        )

    if not candidate_controls:

        raise ValueError(
            "candidate_controls is empty. "
            "Cannot build control matrix."
        )

    #-----------------------------------------------------
    # Copy and standardize dates
    #-----------------------------------------------------

    data = dma_timeseries.copy()

    data["Month"] = pd.to_datetime(
        data["Month"]
    )

    intervention_date = pd.Timestamp(
        intervention_date
    )

    #-----------------------------------------------------
    # Restrict to candidate control DMAs
    #-----------------------------------------------------

    controls = data[
        data["DMA"].isin(candidate_controls)
    ].copy()

    if controls.empty:

        raise ValueError(
            "No observations found for candidate control DMAs."
        )

    logger.info(
        f"Candidate control DMAs: "
        f"{controls['DMA'].nunique()}"
    )

    #-----------------------------------------------------
    # Restrict to pre-treatment period
    #-----------------------------------------------------

    controls = controls[
        controls["Month"] < intervention_date
    ].copy()

    if controls.empty:

        raise ValueError(
            "No pre-treatment observations available "
            "for candidate controls."
        )

    #-----------------------------------------------------
    # Check for duplicate DMA-month observations
    #-----------------------------------------------------

    duplicates = (
        controls
        .duplicated(
            subset=["DMA", "Month"]
        )
        .sum()
    )

    if duplicates > 0:

        raise ValueError(
            f"Found {duplicates} duplicate "
            "DMA-month observations in candidate controls."
        )

    #-----------------------------------------------------
    # Pivot into DMA x Month matrix
    #-----------------------------------------------------

    control_matrix = (

        controls

        .pivot(
            index="Month",
            columns="DMA",
            values="Traffic"
        )

        .sort_index()

    )

    #-----------------------------------------------------
    # Remove columns with no usable observations
    #-----------------------------------------------------

    before_columns = control_matrix.shape[1]

    control_matrix = control_matrix.dropna(
        axis=1,
        how="all"
    )

    removed_columns = (
        before_columns -
        control_matrix.shape[1]
    )

    if removed_columns > 0:

        logger.warning(
            f"Removed {removed_columns} control DMAs "
            "with no usable observations."
        )

    #-----------------------------------------------------
    # Diagnostics
    #-----------------------------------------------------

    logger.info(
        f"Control matrix shape: "
        f"{control_matrix.shape}"
    )

    logger.info(
        f"Pre-treatment months: "
        f"{control_matrix.index.min().strftime('%Y-%m')} "
        f"to "
        f"{control_matrix.index.max().strftime('%Y-%m')}"
    )

    logger.info(
        f"Control DMAs in matrix: "
        f"{control_matrix.shape[1]}"
    )

    #-----------------------------------------------------
    # Export
    #-----------------------------------------------------

    output_path = (
        TABLE_DIR /
        "bsts" /
        "control_matrix.csv"
    )

    control_matrix.to_csv(
        output_path
    )

    logger.info(
        f"Saved control matrix to {output_path}"
    )

    return control_matrix



# =========================================================
# Validate Control Matrix
# =========================================================

def validate_control_matrix(
    control_matrix,
    expected_months=None,
    min_observations=12
):
    """
    Validate the control matrix before control selection
    and BSTS / causal impact modeling.

    Parameters
    ----------
    control_matrix : pd.DataFrame
        Monthly control DMA traffic matrix.

        Expected structure:
            Index   -> Month
            Columns -> DMA names
            Values  -> Traffic

    expected_months : int, optional
        Expected number of monthly observations.

    min_observations : int, default=12
        Minimum number of non-missing observations required
        for each control DMA.

    Returns
    -------
    validation : dict
        Validation summary containing structural,
        missingness, variance, and alignment checks.
    """

    logger.info(
        "Validating control matrix..."
    )

    # =====================================================
    # Basic validation
    # =====================================================

    if not isinstance(
        control_matrix,
        pd.DataFrame
    ):
        raise TypeError(
            "control_matrix must be a pandas DataFrame."
        )

    if control_matrix.empty:
        raise ValueError(
            "Control matrix is empty."
        )

    # Work on a copy so validation does not modify
    # the modeling data.
    matrix = control_matrix.copy()

    # =====================================================
    # Index validation
    # =====================================================

    if not isinstance(
        matrix.index,
        pd.DatetimeIndex
    ):

        try:

            matrix.index = pd.to_datetime(
                matrix.index
            )

        except Exception as exc:

            raise TypeError(
                "Control matrix index must be "
                "datetime-like."
            ) from exc

    # -----------------------------------------------------
    # Duplicate months
    # -----------------------------------------------------

    duplicate_months = (
        matrix.index
        .duplicated()
        .sum()
    )

    if duplicate_months > 0:

        raise ValueError(
            f"Control matrix contains "
            f"{duplicate_months} duplicate months."
        )

    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------

    matrix = matrix.sort_index()

    # =====================================================
    # Expected monthly structure
    # =====================================================

    observed_months = len(matrix)

    if expected_months is not None:

        if observed_months != expected_months:

            logger.warning(
                "Expected %d months but found %d.",
                expected_months,
                observed_months
            )

    # -----------------------------------------------------
    # Check for missing months
    # -----------------------------------------------------

    expected_index = pd.date_range(

        start=matrix.index.min(),

        end=matrix.index.max(),

        freq="MS"

    )

    missing_months = (
        expected_index
        .difference(matrix.index)
    )

    if len(missing_months) > 0:

        logger.warning(
            "Control matrix is missing %d months.",
            len(missing_months)
        )

    # =====================================================
    # DMA validation
    # =====================================================

    n_controls = matrix.shape[1]

    if n_controls == 0:

        raise ValueError(
            "Control matrix contains no DMA columns."
        )

    logger.info(
        "Control DMAs: %d",
        n_controls
    )

    logger.info(
        "Months: %d",
        observed_months
    )

    # =====================================================
    # Missing-value validation
    # =====================================================

    missing_counts = (
        matrix
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    controls_with_missing = (
        missing_counts[
            missing_counts > 0
        ]
    )

    # -----------------------------------------------------
    # Insufficient observations
    # -----------------------------------------------------

    observation_counts = (
        matrix
        .notna()
        .sum()
    )

    insufficient_controls = (
        observation_counts[
            observation_counts < min_observations
        ]
    )

    # =====================================================
    # Variance validation
    # =====================================================

    variance = matrix.var(
        skipna=True
    )

    zero_variance_controls = (
        variance[
            variance <= 0
        ]
    )

    # =====================================================
    # Numeric validation
    # =====================================================

    non_numeric_columns = []

    for column in matrix.columns:

        if not pd.api.types.is_numeric_dtype(
            matrix[column]
        ):

            non_numeric_columns.append(
                column
            )

    if non_numeric_columns:

        raise TypeError(
            "Control matrix contains "
            "non-numeric columns: "
            f"{non_numeric_columns}"
        )

    # =====================================================
    # Negative traffic validation
    # =====================================================

    negative_values = (
        (matrix < 0)
        .sum()
        .sum()
    )

    if negative_values > 0:

        logger.warning(
            "Control matrix contains "
            "%d negative traffic observations.",
            negative_values
        )

    # =====================================================
    # Summary
    # =====================================================

    validation = {

        "Valid": True,

        "Months": observed_months,

        "Expected_Months":
            expected_months,

        "Controls":
            n_controls,

        "Start_Date":
            matrix.index.min(),

        "End_Date":
            matrix.index.max(),

        "Missing_Months":
            len(missing_months),

        "Controls_With_Missing":
            len(controls_with_missing),

        "Insufficient_Controls":
            len(insufficient_controls),

        "Zero_Variance_Controls":
            len(zero_variance_controls),

        "Negative_Values":
            int(negative_values)

    }

    # =====================================================
    # Determine overall validity
    # =====================================================

    structural_failure = (

        len(non_numeric_columns) > 0

        or

        duplicate_months > 0

        or

        n_controls == 0

    )

    if structural_failure:

        validation["Valid"] = False

    # Missing data is a warning at this stage rather
    # than an automatic failure. The control-selection
    # function can decide whether individual controls
    # should be removed.
    if (
        len(insufficient_controls) > 0
    ):

        logger.warning(
            "%d controls have fewer than "
            "%d observations.",
            len(insufficient_controls),
            min_observations
        )

    if (
        len(zero_variance_controls) > 0
    ):

        logger.warning(
            "%d controls have zero variance.",
            len(zero_variance_controls)
        )

    # =====================================================
    # Export validation summary
    # =====================================================

    validation_df = pd.DataFrame({

        "Metric": [

            "Valid",
            "Months",
            "Controls",
            "Start_Date",
            "End_Date",
            "Missing_Months",
            "Controls_With_Missing",
            "Insufficient_Controls",
            "Zero_Variance_Controls",
            "Negative_Values"

        ],

        "Value": [

            validation["Valid"],
            validation["Months"],
            validation["Controls"],
            validation["Start_Date"],
            validation["End_Date"],
            validation["Missing_Months"],
            validation["Controls_With_Missing"],
            validation["Insufficient_Controls"],
            validation["Zero_Variance_Controls"],
            validation["Negative_Values"]

        ]

    })

    validation_df.to_csv(

        TABLE_DIR /
        "bsts" /
        "control_matrix_validation.csv",

        index=False

    )

    # =====================================================
    # Logging
    # =====================================================

    logger.info(
        "Control matrix validation complete."
    )

    logger.info(
        "Matrix shape: %d months x %d controls",
        observed_months,
        n_controls
    )

    logger.info(
        "Missing months: %d",
        len(missing_months)
    )

    logger.info(
        "Controls with missing values: %d",
        len(controls_with_missing)
    )

    logger.info(
        "Zero-variance controls: %d",
        len(zero_variance_controls)
    )

    if validation["Valid"]:

        logger.info(
            "Control matrix passed structural validation."
        )

    return validation


# =========================================================
# Select Controls
# =========================================================

def select_controls(
    treated_dmas,
    dma_timeseries,
    control_matrix,
    n_controls=5,
    min_correlation=0.50
):
    """
    Select the best candidate control DMAs for each
    treated DMA using pre-treatment traffic.

    Controls are ranked primarily by Pearson correlation
    with the treated DMA during the pre-treatment period.

    Parameters
    ----------
    treated_dmas : list
        List of treated DMA names.

    dma_timeseries : pd.DataFrame
        DMA-level monthly traffic data containing:
        DMA, Month, Traffic.

    control_matrix : pd.DataFrame
        Pre-treatment control matrix.

        Index:
            Month

        Columns:
            Candidate control DMAs

    n_controls : int, default=5
        Number of control DMAs to select for each
        treated DMA.

    min_correlation : float, default=0.50
        Minimum pre-treatment correlation required for
        a candidate control to be eligible.

    Returns
    -------
    selected_controls : dict
        Dictionary mapping each treated DMA to its
        selected control DMAs.

    control_scores : pd.DataFrame
        Full candidate-control scoring table.
    """

    logger.info(
        "Selecting control DMAs..."
    )

    # =====================================================
    # Validate inputs
    # =====================================================

    if not isinstance(
        dma_timeseries,
        pd.DataFrame
    ):

        raise TypeError(
            "dma_timeseries must be a pandas DataFrame."
        )

    if not isinstance(
        control_matrix,
        pd.DataFrame
    ):

        raise TypeError(
            "control_matrix must be a pandas DataFrame."
        )

    if not treated_dmas:

        raise ValueError(
            "treated_dmas is empty."
        )

    if n_controls < 1:

        raise ValueError(
            "n_controls must be at least 1."
        )

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = [
        "DMA",
        "Month",
        "Traffic"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in dma_timeseries.columns
    ]

    if missing_columns:

        raise ValueError(
            "dma_timeseries is missing required columns: "
            f"{missing_columns}"
        )

    # -----------------------------------------------------
    # Standardize dates
    # -----------------------------------------------------

    data = dma_timeseries.copy()

    data["Month"] = pd.to_datetime(
        data["Month"]
    )

    control_matrix = control_matrix.copy()

    control_matrix.index = pd.to_datetime(
        control_matrix.index
    )

    control_matrix = (
        control_matrix
        .sort_index()
    )

    # =====================================================
    # Restrict treated DMA data to pre-treatment months
    # =====================================================

    treated_data = data[
        data["DMA"].isin(treated_dmas)
    ].copy()

    treated_data = treated_data[
        treated_data["Month"].isin(
            control_matrix.index
        )
    ].copy()

    # =====================================================
    # Selection results
    # =====================================================

    selected_controls = {}

    all_scores = []

    # =====================================================
    # Evaluate each treated DMA
    # =====================================================

    for treated_dma in treated_dmas:

        logger.info(
            f"Evaluating controls for {treated_dma}..."
        )

        # -------------------------------------------------
        # Extract treated DMA series
        # -------------------------------------------------

        treated_series = (

            treated_data.loc[
                treated_data["DMA"] == treated_dma
            ]

            .set_index("Month")["Traffic"]

            .sort_index()

        )

        if treated_series.empty:

            logger.warning(
                f"No observations found for "
                f"treated DMA: {treated_dma}"
            )

            selected_controls[
                treated_dma
            ] = []

            continue

        # -------------------------------------------------
        # Evaluate every candidate control
        # -------------------------------------------------

        dma_scores = []

        for control_dma in control_matrix.columns:

            control_series = (
                control_matrix[control_dma]
            )

            paired = pd.concat(

                [
                    treated_series,
                    control_series
                ],

                axis=1,

                keys=[
                    "Treated",
                    "Control"
                ]

            ).dropna()

            observations = len(
                paired
            )

            # ---------------------------------------------
            # Require sufficient overlap
            # ---------------------------------------------

            if observations < 3:
                continue

            treated_values = (
                paired["Treated"]
            )

            control_values = (
                paired["Control"]
            )

            # ---------------------------------------------
            # Correlation
            # ---------------------------------------------

            correlation = (
                treated_values
                .corr(
                    control_values
                )
            )

            # ---------------------------------------------
            # RMSE
            # ---------------------------------------------

            rmse = np.sqrt(
                np.mean(
                    (
                        treated_values
                        -
                        control_values
                    ) ** 2
                )
            )

            # ---------------------------------------------
            # Mean absolute difference
            # ---------------------------------------------

            mean_absolute_difference = np.mean(
                np.abs(
                    treated_values
                    -
                    control_values
                )
            )

            dma_scores.append({

                "Treated_DMA":
                    treated_dma,

                "Control_DMA":
                    control_dma,

                "Correlation":
                    correlation,

                "RMSE":
                    rmse,

                "Mean_Absolute_Difference":
                    mean_absolute_difference,

                "Observations":
                    observations

            })

        # =================================================
        # No valid candidates
        # =================================================

        if not dma_scores:

            logger.warning(
                f"No valid control candidates found "
                f"for {treated_dma}."
            )

            selected_controls[
                treated_dma
            ] = []

            continue

        # =================================================
        # Rank candidates
        # =================================================

        dma_scores = pd.DataFrame(
            dma_scores
        )

        dma_scores = dma_scores.sort_values(

            [
                "Correlation",
                "RMSE"
            ],

            ascending=[
                False,
                True
            ]

        ).reset_index(
            drop=True
        )

        dma_scores["Rank"] = (
            dma_scores.index + 1
        )

        # =================================================
        # Apply correlation threshold
        # =================================================

        eligible = dma_scores.loc[
            dma_scores["Correlation"]
            >= min_correlation
        ].copy()

        # =================================================
        # Select controls
        # =================================================

        selected = (

            eligible

            .head(n_controls)

            ["Control_DMA"]

            .tolist()

        )

        dma_scores["Selected"] = (
            dma_scores["Control_DMA"]
            .isin(selected)
            .astype(int)
        )

        selected_controls[
            treated_dma
        ] = selected

        all_scores.append(
            dma_scores
        )

        # =================================================
        # Logging
        # =================================================

        if selected:

            logger.info(
                f"{treated_dma} selected controls: "
                f"{', '.join(selected)}"
            )

        else:

            logger.warning(
                f"No controls met the minimum "
                f"correlation threshold for "
                f"{treated_dma}."
            )

    # =====================================================
    # Combine scoring results
    # =====================================================

    if all_scores:

        control_scores = pd.concat(

            all_scores,

            ignore_index=True

        )

    else:

        control_scores = pd.DataFrame(

            columns=[
                "Treated_DMA",
                "Control_DMA",
                "Correlation",
                "RMSE",
                "Mean_Absolute_Difference",
                "Observations",
                "Rank",
                "Selected"
            ]

        )

    # =====================================================
    # Export detailed scores
    # =====================================================

    control_scores.to_csv(

        TABLE_DIR /
        "bsts" /
        "control_selection_scores.csv",

        index=False

    )

    # =====================================================
    # Create selected-control table
    # =====================================================

    selection_rows = []

    for treated_dma, controls in (
        selected_controls.items()
    ):

        for rank, control_dma in enumerate(
            controls,
            start=1
        ):

            selection_rows.append({

                "Treated_DMA":
                    treated_dma,

                "Control_DMA":
                    control_dma,

                "Rank":
                    rank

            })

    selected_df = pd.DataFrame(
        selection_rows
    )

    selected_df.to_csv(

        TABLE_DIR /
        "bsts" /
        "selected_controls.csv",

        index=False

    )

    # =====================================================
    # Summary
    # =====================================================

    total_treated = len(
        selected_controls
    )

    treated_with_controls = sum(

        len(controls) > 0

        for controls
        in selected_controls.values()

    )

    logger.info(
        "Control selection complete."
    )

    logger.info(
        f"Treated DMAs evaluated: "
        f"{total_treated}"
    )

    logger.info(
        f"Treated DMAs with valid controls: "
        f"{treated_with_controls}"
    )

    logger.info(
        f"Requested controls per DMA: "
        f"{n_controls}"
    )

    return (
        selected_controls,
        control_scores
    )



# =========================================================
# Prepare BSTS / Causal Impact Inputs
# =========================================================

def prepare_causal_impact_inputs(
    dma_timeseries,
    target_dma,
    selected_controls,
    intervention_date
):
    """
    Prepare BSTS inputs for a single treated DMA.

    Constructs a regular monthly time series containing the
    treated DMA and its selected control DMAs, then separates
    the data into pre- and post-intervention periods.

    Parameters
    ----------
    dma_timeseries : pd.DataFrame
        DMA-level monthly traffic data containing:
        DMA, Month, Traffic.

    target_dma : str
        Treated DMA being modeled.

    selected_controls : dict
        Dictionary mapping each treated DMA to its selected
        control DMAs.

    intervention_date : pd.Timestamp
        Start date of the intervention.

    Returns
    -------
    prepared_inputs : dict
        Modeling inputs required by configure_bsts_model().
    """

    logger.info(
        f"Preparing BSTS inputs for {target_dma}..."
    )

    #=====================================================
    # Validate arguments
    #=====================================================

    if not isinstance(
        dma_timeseries,
        pd.DataFrame
    ):
        raise TypeError(
            "dma_timeseries must be a pandas DataFrame."
        )

    if not isinstance(
        selected_controls,
        dict
    ):
        raise TypeError(
            "selected_controls must be a dictionary "
            "mapping treated DMAs to control DMAs."
        )

    if target_dma not in selected_controls:
        raise ValueError(
            f"No selected controls found for "
            f"{target_dma}."
        )

    intervention_date = pd.Timestamp(
        intervention_date
    )

    required_columns = [
        "DMA",
        "Month",
        "Traffic"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in dma_timeseries.columns
    ]

    if missing_columns:
        raise ValueError(
            "dma_timeseries is missing required "
            f"columns: {missing_columns}"
        )

    #=====================================================
    # Standardize data
    #=====================================================

    data = dma_timeseries.copy()

    data["Month"] = pd.to_datetime(
        data["Month"]
    )

    data = (
        data
        .sort_values(
            ["DMA", "Month"]
        )
        .reset_index(drop=True)
    )

    #=====================================================
    # Validate duplicate DMA-month observations
    #=====================================================

    duplicates = (
        data
        .duplicated(
            subset=["DMA", "Month"]
        )
    )

    if duplicates.any():

        duplicate_count = int(
            duplicates.sum()
        )

        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "DMA-month observations."
        )

    #=====================================================
    # Retrieve selected controls
    #=====================================================

    controls = list(
        selected_controls[target_dma]
    )

    if not controls:

        raise ValueError(
            f"No controls selected for {target_dma}."
        )

    logger.info(
        f"{target_dma}: "
        f"{len(controls)} selected controls."
    )

    logger.info(
        f"Controls: {controls}"
    )

    #=====================================================
    # Validate DMA availability
    #=====================================================

    required_dmas = [
        target_dma
    ] + controls

    available_dmas = set(
        data["DMA"]
        .dropna()
        .unique()
    )

    missing_dmas = [
        dma
        for dma in required_dmas
        if dma not in available_dmas
    ]

    if missing_dmas:

        raise ValueError(
            f"{target_dma}: missing required DMAs: "
            f"{missing_dmas}"
        )

    #=====================================================
    # Determine complete analysis period
    #=====================================================

    analysis_start = (
        data["Month"].min()
    )

    analysis_end = (
        data["Month"].max()
    )

    expected_months = pd.date_range(
        start=analysis_start,
        end=analysis_end,
        freq="MS"
    )

    logger.info(
        f"{target_dma}: expected analysis period "
        f"{analysis_start:%Y-%m} to "
        f"{analysis_end:%Y-%m}"
    )

    #=====================================================
    # Construct modeling matrix
    #=====================================================

    subset = data[
        data["DMA"].isin(
            required_dmas
        )
    ].copy()

    matrix = (
        subset
        .pivot(
            index="Month",
            columns="DMA",
            values="Traffic"
        )
        .reindex(expected_months)
    )

    matrix.index.name = "Month"

    #=====================================================
    # Validate required columns after pivot
    #=====================================================

    missing_after_pivot = [
        dma
        for dma in required_dmas
        if dma not in matrix.columns
    ]

    if missing_after_pivot:

        raise ValueError(
            f"{target_dma}: required DMAs disappeared "
            f"during matrix construction: "
            f"{missing_after_pivot}"
        )

    matrix = matrix[
        required_dmas
    ].copy()

    #=====================================================
    # Identify missing structural periods
    #=====================================================

    structural_missing = {}

    for dma in required_dmas:

        dma_months = set(
            data.loc[
                data["DMA"] == dma,
                "Month"
            ]
        )

        missing_months = [
            month
            for month in expected_months
            if month not in dma_months
        ]

        if missing_months:

            structural_missing[dma] = (
                missing_months
            )

    if structural_missing:

        logger.warning(
            f"{target_dma}: structural monthly "
            "gaps detected."
        )

        for dma, missing_months in (
            structural_missing.items()
        ):

            logger.warning(
                f"  {dma}: "
                f"{len(missing_months)} missing months "
                f"{missing_months}"
            )

    #=====================================================
    # Validate intervention coverage
    #=====================================================

    pre_mask = (
        matrix.index < intervention_date
    )

    post_mask = (
        matrix.index >= intervention_date
    )

    pre_period = (
        matrix.index[pre_mask]
    )

    post_period = (
        matrix.index[post_mask]
    )

    if len(pre_period) == 0:

        raise ValueError(
            f"{target_dma}: no pre-treatment "
            "observations available."
        )

    if len(post_period) == 0:

        raise ValueError(
            f"{target_dma}: no post-treatment "
            "observations available."
        )

    #=====================================================
    # Validate target pre-treatment traffic
    #=====================================================

    target_pre = matrix.loc[
        pre_mask,
        target_dma
    ]

    if target_pre.isna().any():

        missing_dates = (
            target_pre[
                target_pre.isna()
            ]
            .index
            .tolist()
        )

        raise ValueError(
            f"{target_dma}: target traffic contains "
            f"{len(missing_dates)} missing pre-treatment "
            f"observations: {missing_dates}"
        )

    #=====================================================
    # Validate target post-treatment traffic
    #=====================================================

    target_post = matrix.loc[
        post_mask,
        target_dma
    ]

    if target_post.isna().any():

        missing_dates = (
            target_post[
                target_post.isna()
            ]
            .index
            .tolist()
        )

        raise ValueError(
            f"{target_dma}: target traffic contains "
            f"{len(missing_dates)} missing post-treatment "
            f"observations: {missing_dates}"
        )

    #=====================================================
    # Validate control traffic
    #=====================================================

    control_missing = {}

    for control in controls:

        missing_count = int(
            matrix[
                control
            ]
            .isna()
            .sum()
        )

        if missing_count > 0:

            control_missing[
                control
            ] = missing_count

    if control_missing:

        raise ValueError(
            f"{target_dma}: selected controls contain "
            f"missing traffic values: "
            f"{control_missing}"
        )

    #=====================================================
    # Extract modeling arrays
    #=====================================================

    y_full = (
        matrix[
            target_dma
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_pre = (
        matrix.loc[
            pre_mask,
            target_dma
        ]
        .to_numpy(
            dtype=float
        )
    )

    controls_full = (
        matrix[
            controls
        ]
        .to_numpy(
            dtype=float
        )
    )

    controls_pre = (
        matrix.loc[
            pre_mask,
            controls
        ]
        .to_numpy(
            dtype=float
        )
    )

    #=====================================================
    # Final validation
    #=====================================================

    if np.isnan(y_full).any():

        raise ValueError(
            f"{target_dma}: y_full contains "
            "missing values."
        )

    if np.isnan(y_pre).any():

        raise ValueError(
            f"{target_dma}: y_pre contains "
            "missing values."
        )

    if np.isnan(controls_full).any():

        raise ValueError(
            f"{target_dma}: controls_full contains "
            "missing values."
        )

    if np.isnan(controls_pre).any():

        raise ValueError(
            f"{target_dma}: controls_pre contains "
            "missing values."
        )

    #=====================================================
    # Build prepared input object
    #=====================================================

    prepared_inputs = {

        "target_dma":
            target_dma,

        "intervention_date":
            intervention_date,

        "dates":
            matrix.index,

        "y_pre":
            y_pre,

        "y_full":
            y_full,

        "controls_pre":
            controls_pre,

        "controls_full":
            controls_full,

        "pre_period":
            pre_period,

        "post_period":
            post_period,

        "control_dmas":
            controls,

        "model_matrix":
            matrix

    }

    #=====================================================
    # Logging
    #=====================================================

    logger.info(
        f"{target_dma}: BSTS inputs prepared."
    )

    logger.info(
        f"  Total periods     : {len(matrix)}"
    )

    logger.info(
        f"  Pre-treatment     : {len(pre_period)}"
    )

    logger.info(
        f"  Post-treatment    : {len(post_period)}"
    )

    logger.info(
        f"  Controls          : {len(controls)}"
    )

    logger.info(
        f"  Target missing    : "
        f"{int(matrix[target_dma].isna().sum())}"
    )

    logger.info(
        f"  Control missing   : "
        f"{int(matrix[controls].isna().sum().sum())}"
    )

    return prepared_inputs


#=========================================================
# Validate / Clean BSTS Model Inputs
#=========================================================

def validate_bsts_inputs(
    treated_series,
    control_matrix,
    intervention_date,
    min_pre_periods=11,
    min_control_series=2
):
    """
    Validate and clean the final inputs to the PyMC BSTS model.

    This function acts as the final quality-control gate before
    model estimation.

    Parameters
    ----------
    treated_series : pd.Series
        Monthly traffic series for the treated DMA.

    control_matrix : pd.DataFrame
        Monthly control series, with Month as the index and
        one DMA per column.

    intervention_date : pd.Timestamp
        Start date of the intervention.

    min_pre_periods : int, default=12
        Minimum number of complete pre-treatment observations
        required for model estimation.

    min_control_series : int, default=2
        Minimum number of usable control DMAs required.

    Returns
    -------
    treated_series : pd.Series
        Cleaned treated DMA series.

    control_matrix : pd.DataFrame
        Cleaned control matrix.

    """

    logger.info(
        "Validating final BSTS model inputs..."
    )

    #-----------------------------------------------------
    # Validate object types
    #-----------------------------------------------------

    if not isinstance(
        treated_series,
        pd.Series
    ):

        raise TypeError(
            "treated_series must be a pandas Series."
        )

    if not isinstance(
        control_matrix,
        pd.DataFrame
    ):

        raise TypeError(
            "control_matrix must be a pandas DataFrame."
        )

    #-----------------------------------------------------
    # Standardize intervention date
    #-----------------------------------------------------

    intervention_date = pd.Timestamp(
        intervention_date
    )

    #-----------------------------------------------------
    # Standardize indexes
    #-----------------------------------------------------

    treated_series = treated_series.copy()

    control_matrix = control_matrix.copy()

    treated_series.index = pd.to_datetime(
        treated_series.index
    )

    control_matrix.index = pd.to_datetime(
        control_matrix.index
    )

    #-----------------------------------------------------
    # Sort chronologically
    #-----------------------------------------------------

    treated_series = (
        treated_series
        .sort_index()
    )

    control_matrix = (
        control_matrix
        .sort_index()
    )

    #-----------------------------------------------------
    # Validate duplicate dates
    #-----------------------------------------------------

    if treated_series.index.duplicated().any():

        raise ValueError(
            "Treated series contains duplicate dates."
        )

    if control_matrix.index.duplicated().any():

        raise ValueError(
            "Control matrix contains duplicate dates."
        )

    #-----------------------------------------------------
    # Align treated series and controls
    #-----------------------------------------------------

    common_index = (
        treated_series.index
        .intersection(
            control_matrix.index
        )
        .sort_values()
    )

    if len(common_index) == 0:

        raise ValueError(
            "Treated series and control matrix "
            "have no overlapping dates."
        )

    treated_series = treated_series.loc[
        common_index
    ]

    control_matrix = control_matrix.loc[
        common_index
    ]

    logger.info(
        f"Common modeling periods: "
        f"{len(common_index)}"
    )

    #-----------------------------------------------------
    # Validate intervention date
    #-----------------------------------------------------

    pre_periods = (
        common_index <
        intervention_date
    ).sum()

    post_periods = (
        common_index >=
        intervention_date
    ).sum()

    if pre_periods == 0:

        raise ValueError(
            "No pre-treatment observations exist "
            "before the intervention date."
        )

    if post_periods == 0:

        raise ValueError(
            "No post-treatment observations exist "
            "after the intervention date."
        )

    logger.info(
        f"Pre-treatment periods: {pre_periods}"
    )

    logger.info(
        f"Post-treatment periods: {post_periods}"
    )

    #-----------------------------------------------------
    # Minimum pre-treatment history
    #-----------------------------------------------------

    if pre_periods < min_pre_periods:

        raise ValueError(
            f"Only {pre_periods} pre-treatment periods "
            f"available. At least {min_pre_periods} "
            "are required."
        )

    #-----------------------------------------------------
    # Remove control DMAs with missing observations
    #
    # For BSTS, we want complete predictors rather than
    # allowing NaNs to reach the PyMC model.
    #-----------------------------------------------------

    before_controls = (
        control_matrix.shape[1]
    )

    control_matrix = (
        control_matrix
        .dropna(
            axis=1,
            how="any"
        )
    )

    removed_controls = (
        before_controls -
        control_matrix.shape[1]
    )

    if removed_controls > 0:

        logger.warning(
            f"Removed {removed_controls} control DMAs "
            "containing missing observations."
        )

    #-----------------------------------------------------
    # Validate remaining controls
    #-----------------------------------------------------

    if (
        control_matrix.shape[1]
        <
        min_control_series
    ):

        raise ValueError(
            f"Only {control_matrix.shape[1]} usable "
            f"control DMAs remain. At least "
            f"{min_control_series} are required."
        )

    #-----------------------------------------------------
    # Remove rows with missing treated observations
    #-----------------------------------------------------

    valid_treated = (
        treated_series.notna()
    )

    treated_series = (
        treated_series.loc[
            valid_treated
        ]
    )

    control_matrix = (
        control_matrix.loc[
            treated_series.index
        ]
    )

    #-----------------------------------------------------
    # Final missing-value check
    #-----------------------------------------------------

    if treated_series.isna().any():

        raise ValueError(
            "Treated series still contains missing values "
            "after cleanup."
        )

    if control_matrix.isna().any().any():

        raise ValueError(
            "Control matrix still contains missing values "
            "after cleanup."
        )

    #-----------------------------------------------------
    # Final index validation
    #-----------------------------------------------------

    if not treated_series.index.equals(
        control_matrix.index
    ):

        raise ValueError(
            "Treated series and control matrix indexes "
            "are not identical after cleanup."
        )

    #-----------------------------------------------------
    # Validate numeric data
    #-----------------------------------------------------

    if not pd.api.types.is_numeric_dtype(
        treated_series
    ):

        raise TypeError(
            "Treated series must contain numeric values."
        )

    non_numeric_controls = [
        column
        for column in control_matrix.columns
        if not pd.api.types.is_numeric_dtype(
            control_matrix[column]
        )
    ]

    if non_numeric_controls:

        raise TypeError(
            "Control matrix contains non-numeric columns: "
            f"{non_numeric_controls}"
        )

    #-----------------------------------------------------
    # Final diagnostics
    #-----------------------------------------------------

    logger.info(
        "Final BSTS input validation complete."
    )

    logger.info(
        f"Treated observations: "
        f"{len(treated_series)}"
    )

    logger.info(
        f"Control DMAs: "
        f"{control_matrix.shape[1]}"
    )

    logger.info(
        f"Model period: "
        f"{treated_series.index.min().strftime('%Y-%m')} "
        f"to "
        f"{treated_series.index.max().strftime('%Y-%m')}"
    )

    logger.info(
        f"Pre-treatment periods: "
        f"{pre_periods}"
    )

    logger.info(
        f"Post-treatment periods: "
        f"{post_periods}"
    )

    #-----------------------------------------------------
    # Export validation summary
    #-----------------------------------------------------

    validation_summary = pd.DataFrame({

        "Metric": [

            "Treated Observations",
            "Control DMAs",
            "Pre_Treatment_Periods",
            "Post_Treatment_Periods",
            "Model_Start",
            "Model_End"

        ],

        "Value": [

            len(treated_series),

            control_matrix.shape[1],

            pre_periods,

            post_periods,

            treated_series.index.min(),

            treated_series.index.max()

        ]

    })

    validation_summary.to_csv(

        TABLE_DIR /
        "bsts" /
        "input_validation.csv",

        index=False

    )

    return (
        treated_series,
        control_matrix
    )


#=========================================================
# Configure BSTS Model
#=========================================================


def configure_bsts_model(
    prepared_inputs,
    include_trend=True,
    include_seasonality=False,
    seasonal_period=12,
    seasonal_harmonics=1,
    intercept_sigma=0.5,
    coefficient_sigma=0.30,
    trend_sigma=0.20,
    seasonal_sigma=0.20,
    observation_sigma=0.50,
    min_observations_per_parameter=2.0,
):
    """
    Configure a regularized Bayesian regression-style BSTS model
    for a single treated DMA.

    Model specification:

        y_t =
            intercept
            + deterministic trend
            + control-DMA effects
            + optional Fourier seasonality
            + Gaussian observation noise

    All continuous predictors and the target are standardized using
    pre-treatment observations only.

    The model intentionally excludes stochastic local-level and
    stochastic local-trend components. This keeps the model tractable
    when the pre-treatment period is short.

    Parameters
    ----------
    prepared_inputs : dict
        Output from prepare_causal_impact_inputs().

    include_trend : bool, default=True
        Include a deterministic standardized linear trend.

    include_seasonality : bool, default=False
        Include deterministic Fourier seasonality.

        With only ~11 pre-treatment observations, annual seasonality
        should generally remain disabled.

    seasonal_period : int, default=12
        Number of observations in one seasonal cycle.

    seasonal_harmonics : int, default=1
        Number of Fourier harmonics.

    intercept_sigma : float, default=0.5
        Prior standard deviation for the intercept.

    coefficient_sigma : float, default=0.30
        Prior standard deviation for control-DMA coefficients.

    trend_sigma : float, default=0.20
        Prior standard deviation for the deterministic trend coefficient.

    seasonal_sigma : float, default=0.20
        Prior standard deviation for Fourier coefficients.

    observation_sigma : float, default=0.50
        Scale parameter for the HalfNormal prior on observation noise.

    min_observations_per_parameter : float, default=2.0
        Minimum desired ratio of pre-treatment observations to
        stochastic model parameters.

        This is a safeguard rather than a mathematical requirement.

    Returns
    -------
    model : pm.Model

    model_data : dict
        Metadata required for fitting and counterfactual generation.
    """

    logger.info("=" * 70)
    logger.info("Configuring BSTS model...")
    logger.info("=" * 70)

    # =====================================================
    # Extract Inputs
    # =====================================================

    target_dma = prepared_inputs["target_dma"]

    intervention_date = pd.Timestamp(
        prepared_inputs["intervention_date"]
    )

    dates = pd.DatetimeIndex(
        prepared_inputs["dates"]
    )

    y_pre = np.asarray(
        prepared_inputs["y_pre"],
        dtype=float,
    )

    y_full = np.asarray(
        prepared_inputs["y_full"],
        dtype=float,
    )

    controls_pre = prepared_inputs.get(
        "controls_pre"
    )

    controls_full = prepared_inputs.get(
        "controls_full"
    )

    control_dmas = prepared_inputs.get(
        "control_dmas",
        [],
    )

    # =====================================================
    # Validate Target
    # =====================================================

    n_pre = len(y_pre)

    if n_pre < 3:

        raise ValueError(
            f"{target_dma}: insufficient pre-treatment "
            "observations for BSTS model."
        )

    if len(y_full) != len(dates):

        raise ValueError(
            f"{target_dma}: y_full and dates have "
            f"different lengths "
            f"({len(y_full)} vs {len(dates)})."
        )

    if not np.isfinite(y_pre).all():

        raise ValueError(
            f"{target_dma}: pre-treatment target "
            "contains non-finite values."
        )

    # =====================================================
    # Validate Controls
    # =====================================================

    X_pre = None
    X_full = None

    if controls_pre is not None:

        if controls_full is None:

            raise ValueError(
                f"{target_dma}: controls_pre was provided "
                "but controls_full is missing."
            )

        X_pre = np.asarray(
            controls_pre,
            dtype=float,
        )

        X_full = np.asarray(
            controls_full,
            dtype=float,
        )

        if X_pre.ndim == 1:

            X_pre = X_pre.reshape(-1, 1)

        if X_full.ndim == 1:

            X_full = X_full.reshape(-1, 1)

        if X_pre.shape[0] != n_pre:

            raise ValueError(
                f"{target_dma}: controls_pre has "
                f"{X_pre.shape[0]} rows but y_pre has "
                f"{n_pre} observations."
            )

        if X_full.shape[0] != len(y_full):

            raise ValueError(
                f"{target_dma}: controls_full has "
                f"{X_full.shape[0]} rows but y_full has "
                f"{len(y_full)} observations."
            )

        if X_pre.shape[1] != X_full.shape[1]:

            raise ValueError(
                f"{target_dma}: controls_pre and "
                "controls_full have different numbers "
                "of columns."
            )

        if X_pre.shape[1] == 0:

            X_pre = None
            X_full = None

        else:

            if not np.isfinite(X_pre).all():

                raise ValueError(
                    f"{target_dma}: controls_pre contains "
                    "non-finite values."
                )

            if not np.isfinite(X_full).all():

                raise ValueError(
                    f"{target_dma}: controls_full contains "
                    "non-finite values."
                )

    # =====================================================
    # Standardize Target
    # =====================================================

    y_mean = float(
        np.mean(y_pre)
    )

    y_std = float(
        np.std(
            y_pre,
            ddof=0,
        )
    )

    if not np.isfinite(y_std) or y_std <= 0:

        raise ValueError(
            f"{target_dma}: target has zero or invalid "
            "pre-treatment variance."
        )

    y_pre_scaled = (
        y_pre - y_mean
    ) / y_std

    # =====================================================
    # Standardize Controls
    #
    # IMPORTANT:
    # Fit scaling parameters using pre-treatment data only.
    # =====================================================

    X_mean = None
    X_std = None
    X_pre_scaled = None
    X_full_scaled = None

    n_controls = 0

    if X_pre is not None:

        X_mean = np.mean(
            X_pre,
            axis=0,
        )

        X_std = np.std(
            X_pre,
            axis=0,
            ddof=0,
        )

        if not np.isfinite(X_std).all():

            raise ValueError(
                f"{target_dma}: control standard deviations "
                "contain non-finite values."
            )

        constant_controls = (
            X_std <= 0
        )

        if constant_controls.any():

            constant_indices = np.where(
                constant_controls
            )[0].tolist()

            logger.warning(
                f"{target_dma}: control columns "
                f"{constant_indices} have zero variance."
            )

            X_std = X_std.copy()

            X_std[
                constant_controls
            ] = 1.0

        X_pre_scaled = (
            X_pre - X_mean
        ) / X_std

        X_full_scaled = (
            X_full - X_mean
        ) / X_std

        if not np.isfinite(
            X_pre_scaled
        ).all():

            raise ValueError(
                f"{target_dma}: standardized pre-treatment "
                "controls contain invalid values."
            )

        if not np.isfinite(
            X_full_scaled
        ).all():

            raise ValueError(
                f"{target_dma}: standardized full controls "
                "contain invalid values."
            )

        n_controls = X_pre_scaled.shape[1]

    # =====================================================
    # Deterministic Trend
    #
    # Centered and standardized using pre-treatment period.
    # =====================================================

    time_index_pre = np.arange(
        n_pre,
        dtype=float,
    )

    time_mean = float(
        time_index_pre.mean()
    )

    time_std = float(
        time_index_pre.std()
    )

    if include_trend:

        if time_std <= 0:

            raise ValueError(
                f"{target_dma}: unable to standardize "
                "time trend."
            )

        time_scaled = (
            time_index_pre - time_mean
        ) / time_std

    else:

        time_scaled = None

    # =====================================================
    # Fourier Seasonality
    # =====================================================

    seasonal_matrix = None
    n_seasonal_terms = 0

    if include_seasonality:

        if seasonal_period <= 1:

            raise ValueError(
                "seasonal_period must be greater than 1."
            )

        if seasonal_harmonics < 1:

            raise ValueError(
                "seasonal_harmonics must be at least 1."
            )

        if seasonal_harmonics >= (
            seasonal_period / 2
        ):

            raise ValueError(
                "seasonal_harmonics must be less than "
                "half of seasonal_period."
            )

        seasonal_terms = []

        for harmonic in range(
            1,
            seasonal_harmonics + 1,
        ):

            seasonal_terms.append(
                np.sin(
                    2
                    * np.pi
                    * harmonic
                    * time_index_pre
                    / seasonal_period
                )
            )

            seasonal_terms.append(
                np.cos(
                    2
                    * np.pi
                    * harmonic
                    * time_index_pre
                    / seasonal_period
                )
            )

        seasonal_matrix = np.column_stack(
            seasonal_terms
        )

        n_seasonal_terms = (
            seasonal_matrix.shape[1]
        )

    # =====================================================
    # Estimate Model Complexity
    #
    # Stochastic parameters:
    #
    #   intercept
    #   trend_beta
    #   beta[]
    #   seasonal_beta[]
    #   sigma_obs
    # =====================================================

    parameter_count = (
        1
        + int(include_trend)
        + n_controls
        + n_seasonal_terms
        + 1
    )

    observations_per_parameter = (
        n_pre / parameter_count
    )

    # -----------------------------------------------------
    # Complexity diagnostic
    #
    # This is intentionally a warning rather than a hard
    # failure. Bayesian regularization allows the model to
    # estimate more parameters than a simple observations /
    # parameters heuristic would permit.
    # -----------------------------------------------------

    if (
        observations_per_parameter
        < min_observations_per_parameter
    ):

        logger.warning(
            f"{target_dma}: model complexity is high relative "
            f"to the available pre-treatment period. "
            f"{n_pre} observations / "
            f"{parameter_count} stochastic parameters = "
            f"{observations_per_parameter:.2f} "
            f"observations per parameter. "
            f"Recommended minimum: "
            f"{min_observations_per_parameter:.2f}."
        )

    # =====================================================
    # Build PyMC Model
    # =====================================================

    with pm.Model() as model:

        # -------------------------------------------------
        # Intercept
        # -------------------------------------------------

        intercept = pm.Normal(
            "intercept",
            mu=0.0,
            sigma=intercept_sigma,
        )

        # -------------------------------------------------
        # Deterministic Trend
        # -------------------------------------------------

        if include_trend:

            trend_beta = pm.Normal(
                "trend_beta",
                mu=0.0,
                sigma=trend_sigma,
            )

            trend_component = (
                trend_beta
                * time_scaled
            )

        else:

            trend_beta = None
            trend_component = 0.0

        # -------------------------------------------------
        # Control DMA Effects
        # -------------------------------------------------

        if X_pre_scaled is not None:

            beta = pm.Normal(
                "beta",
                mu=0.0,
                sigma=coefficient_sigma,
                shape=n_controls,
            )

            control_component = pm.math.dot(
                X_pre_scaled,
                beta,
            )

        else:

            beta = None
            control_component = 0.0

        # -------------------------------------------------
        # Fourier Seasonality
        # -------------------------------------------------

        if seasonal_matrix is not None:

            seasonal_beta = pm.Normal(
                "seasonal_beta",
                mu=0.0,
                sigma=seasonal_sigma,
                shape=n_seasonal_terms,
            )

            seasonal_component = pm.math.dot(
                seasonal_matrix,
                seasonal_beta,
            )

        else:

            seasonal_beta = None
            seasonal_component = 0.0

        # -------------------------------------------------
        # Expected Traffic
        # -------------------------------------------------

        mu = (
            intercept
            + trend_component
            + control_component
            + seasonal_component
        )

        # -------------------------------------------------
        # Observation Noise
        # -------------------------------------------------

        sigma_obs = pm.HalfNormal(
            "sigma_obs",
            sigma=observation_sigma,
        )

        # -------------------------------------------------
        # Observation Model
        # -------------------------------------------------

        pm.Normal(
            "traffic",
            mu=mu,
            sigma=sigma_obs,
            observed=y_pre_scaled,
        )

    # =====================================================
    # Model Metadata
    # =====================================================

    model_data = {

        # -------------------------------------------------
        # Identification
        # -------------------------------------------------

        "target_dma":
            target_dma,

        "control_dmas":
            control_dmas,

        "dates":
            dates,

        "intervention_date":
            intervention_date,

        "pre_period":
            prepared_inputs["pre_period"],

        "post_period":
            prepared_inputs["post_period"],

        # -------------------------------------------------
        # Raw Target
        # -------------------------------------------------

        "y_pre":
            y_pre,

        "y_full":
            y_full,

        # -------------------------------------------------
        # Target Scaling
        # -------------------------------------------------

        "y_mean":
            y_mean,

        "y_std":
            y_std,

        "y_pre_scaled":
            y_pre_scaled,

        # -------------------------------------------------
        # Raw Controls
        # -------------------------------------------------

        "controls_pre":
            X_pre,

        "controls_full":
            X_full,

        # -------------------------------------------------
        # Control Scaling
        # -------------------------------------------------

        "controls_mean":
            X_mean,

        "controls_std":
            X_std,

        "controls_pre_scaled":
            X_pre_scaled,

        "controls_full_scaled":
            X_full_scaled,

        # -------------------------------------------------
        # Time
        # -------------------------------------------------

        "time_index_pre":
            time_index_pre,

        "time_mean":
            time_mean,

        "time_std":
            time_std,

        "time_scaled":
            time_scaled,

        # -------------------------------------------------
        # Seasonality
        # -------------------------------------------------

        "seasonal_matrix":
            seasonal_matrix,

        # -------------------------------------------------
        # Model Structure
        # -------------------------------------------------

        "n_pre":
            n_pre,

        "n_controls":
            n_controls,

        "n_seasonal_terms":
            n_seasonal_terms,

        "parameter_count":
            parameter_count,

        "observations_per_parameter":
            observations_per_parameter,

        "include_trend":
            include_trend,

        "include_seasonality":
            include_seasonality,

        "seasonal_period":
            seasonal_period,

        "seasonal_harmonics":
            seasonal_harmonics,

        # -------------------------------------------------
        # Prior Configuration
        # -------------------------------------------------

        "intercept_sigma":
            intercept_sigma,

        "coefficient_sigma":
            coefficient_sigma,

        "trend_sigma":
            trend_sigma,

        "seasonal_sigma":
            seasonal_sigma,

        "observation_sigma":
            observation_sigma,

        # -------------------------------------------------
        # Model Version
        # -------------------------------------------------

        "model_type":
            "deterministic_trend_control_fourier_bsts",

        "model_version":
            "v3_regularized",
    }

    # =====================================================
    # Logging
    # =====================================================

    logger.info(
        f"BSTS model configured for {target_dma}"
    )

    logger.info(
        f"Pre-treatment observations: {n_pre}"
    )

    logger.info(
        f"Control DMA predictors: {n_controls}"
    )

    logger.info(
        f"Trend enabled: {include_trend}"
    )

    logger.info(
        f"Seasonality enabled: {include_seasonality}"
    )

    if include_seasonality:

        logger.info(
            f"Seasonal period: {seasonal_period}"
        )

        logger.info(
            f"Seasonal harmonics: {seasonal_harmonics}"
        )

        logger.info(
            f"Seasonal terms: {n_seasonal_terms}"
        )

    logger.info(
        f"Model parameters: {parameter_count}"
    )

    logger.info(
        f"Observations per parameter: "
        f"{observations_per_parameter:.2f}"
    )

    logger.info(
        "Stochastic local-level component: disabled"
    )

    logger.info(
        "Stochastic local-trend component: disabled"
    )

    logger.info("=" * 70)

    return model, model_data


#=========================================================
# Holdout Validation
#=========================================================
def validate_holdout(
    prepared_inputs,
    holdout_periods=3,
    include_trend=True,
    include_seasonality=False,
    seasonal_period=12,
    seasonal_harmonics=1,
    draws=1000,
    tune=1000,
    chains=4,
    cores=None,
    target_accept=0.95,
    random_seed=42,
    min_holdout_r2=0.50,
    max_holdout_rmse_ratio=0.30,
    max_holdout_mae_ratio=0.25,
):
    """
    Validate BSTS out-of-sample predictive performance using a
    pre-treatment holdout period.

    The final `holdout_periods` observations of the pre-treatment
    period are withheld from model fitting. A new BSTS model is then
    fitted using only the earlier pre-treatment observations and
    evaluated against the withheld observations.

    Parameters
    ----------
    prepared_inputs : dict
        Output from prepare_causal_impact_inputs().

    holdout_periods : int, default=3
        Number of pre-treatment observations to reserve for
        out-of-sample validation.

    include_trend : bool, default=True
        Include deterministic trend in the validation model.

    include_seasonality : bool, default=False
        Include Fourier seasonality.

    seasonal_period : int, default=12
        Seasonal period.

    seasonal_harmonics : int, default=1
        Number of Fourier harmonics.

    draws : int, default=1000
        Posterior draws per chain.

    tune : int, default=1000
        Tuning iterations per chain.

    chains : int, default=4
        Number of MCMC chains.

    cores : int, optional
        Number of parallel sampling cores.

    target_accept : float, default=0.95
        NUTS target acceptance probability.

    random_seed : int, default=42
        Random seed.

    min_holdout_r2 : float, default=0.50
        Minimum acceptable holdout R².

    max_holdout_rmse_ratio : float, default=0.30
        Maximum acceptable RMSE relative to mean holdout traffic.

    max_holdout_mae_ratio : float, default=0.25
        Maximum acceptable MAE relative to mean holdout traffic.

    Returns
    -------
    validation : dict
        Structured holdout validation results.

    validation_summary : pd.DataFrame
        One-row summary suitable for consolidation across DMAs.
    """

    logger.info("=" * 70)
    logger.info("Validating BSTS out-of-sample performance...")
    logger.info("=" * 70)

    # =====================================================
    # Validate Inputs
    # =====================================================

    if prepared_inputs is None:
        raise ValueError(
            "prepared_inputs is None."
        )

    required_keys = {
        "target_dma",
        "dates",
        "y_pre",
        "controls_pre",
        "controls_full",
        "control_dmas",
        "intervention_date",
        "pre_period",
        "post_period",
    }

    missing_keys = (
        required_keys -
        set(prepared_inputs.keys())
    )

    if missing_keys:
        raise ValueError(
            "prepared_inputs is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    target_dma = prepared_inputs["target_dma"]

    dates = pd.DatetimeIndex(
        prepared_inputs["dates"]
    )

    y_pre = np.asarray(
        prepared_inputs["y_pre"],
        dtype=float,
    )

    controls_pre = prepared_inputs["controls_pre"]

    if controls_pre is None:
        X_pre = None
    else:
        X_pre = np.asarray(
            controls_pre,
            dtype=float,
        )

        if X_pre.ndim == 1:
            X_pre = X_pre.reshape(-1, 1)

    # =====================================================
    # Validate Holdout Configuration
    # =====================================================

    if not isinstance(holdout_periods, int):
        raise TypeError(
            "holdout_periods must be an integer."
        )

    if holdout_periods < 1:
        raise ValueError(
            "holdout_periods must be at least 1."
        )

    n_pre = len(y_pre)

    if n_pre < 4:
        raise ValueError(
            f"{target_dma}: at least 4 pre-treatment "
            "observations are required for holdout validation."
        )

    if holdout_periods >= n_pre:
        raise ValueError(
            f"{target_dma}: holdout_periods="
            f"{holdout_periods} leaves no observations "
            "for model training."
        )

    # =====================================================
    # Align Pre-Treatment Dates
    #
    # IMPORTANT:
    #
    # dates contains the FULL modeling period.
    # y_pre contains ONLY the pre-treatment period.
    #
    # Therefore dates must be restricted to n_pre before
    # validating its length against y_pre.
    # =====================================================

    pre_dates = dates[:n_pre]

    if len(pre_dates) != n_pre:
        raise ValueError(
            f"{target_dma}: unable to align pre-treatment "
            f"dates ({len(pre_dates)}) with y_pre "
            f"({n_pre})."
        )

    # =====================================================
    # Validate Controls
    # =====================================================

    if X_pre is not None:

        if X_pre.shape[0] != n_pre:
            raise ValueError(
                f"{target_dma}: controls_pre contains "
                f"{X_pre.shape[0]} rows but y_pre contains "
                f"{n_pre} observations."
            )

        if not np.isfinite(X_pre).all():
            raise ValueError(
                f"{target_dma}: controls_pre contains "
                "non-finite values."
            )

    if not np.isfinite(y_pre).all():
        raise ValueError(
            f"{target_dma}: y_pre contains "
            "non-finite values."
        )

    # =====================================================
    # Define Train / Holdout Periods
    # =====================================================

    n_train = (
        n_pre -
        holdout_periods
    )

    train_slice = slice(
        0,
        n_train,
    )

    holdout_slice = slice(
        n_train,
        n_pre,
    )

    train_dates = pre_dates[train_slice]
    holdout_dates = pre_dates[holdout_slice]

    y_train = y_pre[train_slice]
    y_holdout = y_pre[holdout_slice]

    if X_pre is not None:

        X_train = X_pre[train_slice]
        X_holdout = X_pre[holdout_slice]

    else:

        X_train = None
        X_holdout = None

    # =====================================================
    # Validate Training / Holdout Sizes
    # =====================================================

    if len(y_train) < 3:
        raise ValueError(
            f"{target_dma}: only {len(y_train)} observations "
            "remain for model training."
        )

    if len(y_holdout) < 1:
        raise ValueError(
            f"{target_dma}: holdout period is empty."
        )

    logger.info(
        f"{target_dma}: pre-treatment observations: "
        f"{n_pre}"
    )

    logger.info(
        f"{target_dma}: training observations: "
        f"{len(y_train)}"
    )

    logger.info(
        f"{target_dma}: holdout observations: "
        f"{len(y_holdout)}"
    )

    logger.info(
        f"{target_dma}: training period: "
        f"{train_dates.min().date()} to "
        f"{train_dates.max().date()}"
    )

    logger.info(
        f"{target_dma}: holdout period: "
        f"{holdout_dates.min().date()} to "
        f"{holdout_dates.max().date()}"
    )

    # =====================================================
    # Construct Training Inputs
    #
    # We intentionally create a NEW prepared-inputs object
    # containing only the training period.
    #
    # This prevents the holdout observations from entering
    # model fitting or scaling.
    # =====================================================

    training_inputs = dict(
        prepared_inputs
    )

    training_inputs["dates"] = (
        train_dates
    )

    training_inputs["y_pre"] = (
        y_train
    )

    training_inputs["y_full"] = (
        y_train
    )

    training_inputs["pre_period"] = (
        (
            train_dates.min(),
            train_dates.max(),
        )
    )

    training_inputs["post_period"] = (
        None
    )

    training_inputs["intervention_date"] = (
        train_dates.max()
        + pd.DateOffset(months=1)
    )

    if X_train is not None:

        training_inputs["controls_pre"] = (
            X_train
        )

        training_inputs["controls_full"] = (
            X_train
        )

    else:

        training_inputs["controls_pre"] = None
        training_inputs["controls_full"] = None

    # =====================================================
    # Configure Training Model
    # =====================================================

    model, model_data = configure_bsts_model(
        prepared_inputs=training_inputs,
        include_trend=include_trend,
        include_seasonality=include_seasonality,
        seasonal_period=seasonal_period,
        seasonal_harmonics=seasonal_harmonics,
    )

    # =====================================================
    # Fit Training Model
    # =====================================================

    model_results = fit_bsts_model(
        model=model,
        model_data=model_data,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        random_seed=random_seed,
    )

    idata = model_results["idata"]

    # =====================================================
    # Extract Posterior Parameters
    # =====================================================

    posterior = idata.posterior

    # -----------------------------------------------------
    # Posterior Mean Intercept
    # -----------------------------------------------------

    intercept_mean = float(
        posterior["intercept"]
        .mean()
        .values
    )

    # -----------------------------------------------------
    # Posterior Mean Trend
    # -----------------------------------------------------

    trend_beta_mean = 0.0

    if include_trend:

        trend_beta_mean = float(
            posterior["trend_beta"]
            .mean()
            .values
        )

    # -----------------------------------------------------
    # Posterior Mean Control Coefficients
    # -----------------------------------------------------

    beta_mean = None

    if (
        X_train is not None
        and
        X_train.shape[1] > 0
    ):

        beta_mean = np.asarray(
            posterior["beta"]
            .mean(
                dim=("chain", "draw")
            )
            .values,
            dtype=float,
        )

    # -----------------------------------------------------
    # Posterior Mean Seasonal Coefficients
    # -----------------------------------------------------

    seasonal_beta_mean = None

    if include_seasonality:

        seasonal_beta_mean = np.asarray(
            posterior["seasonal_beta"]
            .mean(
                dim=("chain", "draw")
            )
            .values,
            dtype=float,
        )

    # =====================================================
    # Reconstruct Training Scaling
    # =====================================================

    y_mean = float(
        model_data["y_mean"]
    )

    y_std = float(
        model_data["y_std"]
    )

    X_mean = model_data["controls_mean"]
    X_std = model_data["controls_std"]

    # =====================================================
    # Scale Holdout Controls
    #
    # IMPORTANT:
    # Scaling parameters come exclusively from training
    # observations.
    # =====================================================

    if X_holdout is not None:

        if X_mean is None or X_std is None:
            raise ValueError(
                f"{target_dma}: holdout controls exist but "
                "training control scaling parameters are missing."
            )

        X_holdout_scaled = (
            X_holdout -
            X_mean
        ) / X_std

    else:

        X_holdout_scaled = None

    # =====================================================
    # Construct Holdout Trend
    #
    # Continue the training trend rather than resetting
    # the time index.
    # =====================================================

    train_time_index = np.arange(
        n_train,
        dtype=float,
    )

    holdout_time_index = np.arange(
        n_train,
        n_pre,
        dtype=float,
    )

    time_mean = float(
        model_data["time_mean"]
    )

    time_std = float(
        model_data["time_std"]
    )

    if include_trend:

        holdout_time_scaled = (
            holdout_time_index -
            time_mean
        ) / time_std

    else:

        holdout_time_scaled = None

    # =====================================================
    # Construct Holdout Seasonality
    # =====================================================

    holdout_seasonal_matrix = None

    if include_seasonality:

        seasonal_terms = []

        for harmonic in range(
            1,
            seasonal_harmonics + 1,
        ):

            seasonal_terms.append(
                np.sin(
                    2
                    * np.pi
                    * harmonic
                    * holdout_time_index
                    / seasonal_period
                )
            )

            seasonal_terms.append(
                np.cos(
                    2
                    * np.pi
                    * harmonic
                    * holdout_time_index
                    / seasonal_period
                )
            )

        holdout_seasonal_matrix = np.column_stack(
            seasonal_terms
        )

    # =====================================================
    # Generate Holdout Predictions
    # =====================================================

    predictions_scaled = np.full(
        len(y_holdout),
        intercept_mean,
        dtype=float,
    )

    if include_trend:

        predictions_scaled += (
            trend_beta_mean *
            holdout_time_scaled
        )

    if beta_mean is not None:

        predictions_scaled += (
            X_holdout_scaled @
            beta_mean
        )

    if seasonal_beta_mean is not None:

        predictions_scaled += (
            holdout_seasonal_matrix @
            seasonal_beta_mean
        )

    # =====================================================
    # Convert Predictions Back to Traffic Scale
    # =====================================================

    predictions = (
        predictions_scaled *
        y_std
        +
        y_mean
    )

    # =====================================================
    # Calculate Holdout Metrics
    # =====================================================

    errors = (
        predictions -
        y_holdout
    )

    absolute_errors = np.abs(
        errors
    )

    squared_errors = (
        errors ** 2
    )

    holdout_rmse = float(
        np.sqrt(
            np.mean(
                squared_errors
            )
        )
    )

    holdout_mae = float(
        np.mean(
            absolute_errors
        )
    )

    mean_holdout = float(
        np.mean(y_holdout)
    )

    if mean_holdout == 0:
        raise ValueError(
            f"{target_dma}: mean holdout traffic is zero."
        )

    holdout_rmse_ratio = (
        holdout_rmse /
        abs(mean_holdout)
    )

    holdout_mae_ratio = (
        holdout_mae /
        abs(mean_holdout)
    )

    # =====================================================
    # Holdout R²
    # =====================================================

    ss_res = float(
        np.sum(
            squared_errors
        )
    )

    ss_tot = float(
        np.sum(
            (
                y_holdout -
                mean_holdout
            ) ** 2
        )
    )

    if ss_tot > 0:

        holdout_r2 = float(
            1.0 -
            (
                ss_res /
                ss_tot
            )
        )

    else:

        holdout_r2 = np.nan

    # =====================================================
    # Prediction Correlation
    # =====================================================

    if (
        np.std(y_holdout) > 0
        and
        np.std(predictions) > 0
    ):

        holdout_correlation = float(
            np.corrcoef(
                y_holdout,
                predictions,
            )[0, 1]
        )

    else:

        holdout_correlation = np.nan

    # =====================================================
    # Validation Criteria
    # =====================================================

    r2_pass = (
        np.isfinite(holdout_r2)
        and
        holdout_r2 >= min_holdout_r2
    )

    rmse_pass = (
        np.isfinite(holdout_rmse_ratio)
        and
        holdout_rmse_ratio <= max_holdout_rmse_ratio
    )

    mae_pass = (
        np.isfinite(holdout_mae_ratio)
        and
        holdout_mae_ratio <= max_holdout_mae_ratio
    )

    holdout_valid = (
        r2_pass
        and
        rmse_pass
        and
        mae_pass
    )

    # =====================================================
    # Structured Result
    # =====================================================

    validation = {

        "dma":
            target_dma,

        "n_pre":
            n_pre,

        "n_train":
            n_train,

        "n_holdout":
            holdout_periods,

        "train_start":
            train_dates.min(),

        "train_end":
            train_dates.max(),

        "holdout_start":
            holdout_dates.min(),

        "holdout_end":
            holdout_dates.max(),

        "holdout_rmse":
            holdout_rmse,

        "holdout_mae":
            holdout_mae,

        "holdout_r2":
            holdout_r2,

        "holdout_rmse_ratio":
            holdout_rmse_ratio,

        "holdout_mae_ratio":
            holdout_mae_ratio,

        "holdout_correlation":
            holdout_correlation,

        "r2_pass":
            r2_pass,

        "rmse_pass":
            rmse_pass,

        "mae_pass":
            mae_pass,

        "holdout_valid":
            holdout_valid,

        "min_holdout_r2":
            min_holdout_r2,

        "max_holdout_rmse_ratio":
            max_holdout_rmse_ratio,

        "max_holdout_mae_ratio":
            max_holdout_mae_ratio,
    }

    validation_summary = pd.DataFrame(
        [validation]
    )

    # =====================================================
    # Logging
    # =====================================================

    if holdout_valid:

        logger.info(
            f"{target_dma}: holdout validation PASSED."
        )

    else:

        logger.warning(
            f"{target_dma}: holdout validation FAILED."
        )

    logger.info(
        f"  Holdout R²       : "
        f"{holdout_r2:.4f}"
    )

    logger.info(
        f"  Holdout RMSE     : "
        f"{holdout_rmse:,.2f} "
        f"({holdout_rmse_ratio:.2%})"
    )

    logger.info(
        f"  Holdout MAE      : "
        f"{holdout_mae:,.2f} "
        f"({holdout_mae_ratio:.2%})"
    )

    logger.info(
        f"  Correlation      : "
        f"{holdout_correlation:.4f}"
    )

    return (
        validation,
        validation_summary,
    )
#=========================================================
# Fit BSTS Model
#=========================================================
def fit_bsts_model(
    model,
    model_data,
    draws=2000,
    tune=2000,
    chains=4,
    cores=None,
    target_accept=0.95,
    random_seed=42,
):
    """
    Fit the configured v2 deterministic BSTS model using PyMC.

    Sampling only. Posterior diagnostics, counterfactual generation,
    and causal-impact extraction are handled downstream.
    """

    logger.info("=" * 70)
    logger.info("Fitting BSTS model...")
    logger.info("=" * 70)

    # =====================================================
    # Validate Inputs
    # =====================================================

    if model is None:
        raise ValueError("BSTS model is None.")

    if not isinstance(model_data, dict):
        raise TypeError("model_data must be a dictionary.")

    required_keys = {
        "target_dma",
        "model_type",
        "model_version",
        "n_pre",
        "n_controls",
        "include_trend",
        "include_seasonality",
        "seasonal_harmonics",
    }

    missing_keys = required_keys.difference(model_data)

    if missing_keys:
        raise ValueError(
            "model_data is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    target_dma = model_data["target_dma"]
    model_type = model_data["model_type"]
    model_version = model_data["model_version"]

    # =====================================================
    # Validate Model Version
    # =====================================================

    expected_model_type = (
        "deterministic_trend_control_fourier_bsts"
    )

    expected_model_version = {
    "v2",
        "v3_regularized",
    }

    if model_type != expected_model_type:
        raise ValueError(
            f"{target_dma}: unexpected model_type "
            f"'{model_type}'. Expected "
            f"'{expected_model_type}'."
        )

    if model_version not in expected_model_version:
        raise ValueError(
            f"{target_dma}: unexpected model_version "
            f"'{model_version}'. Expected "
            f"'{expected_model_version}'."
        )

    # =====================================================
    # Validate Sampling Parameters
    # =====================================================

    if not isinstance(draws, int) or draws <= 0:
        raise ValueError(
            "draws must be a positive integer."
        )

    if not isinstance(tune, int) or tune <= 0:
        raise ValueError(
            "tune must be a positive integer."
        )

    if not isinstance(chains, int) or chains < 2:
        raise ValueError(
            "chains must be an integer >= 2."
        )

    if cores is not None:
        if not isinstance(cores, int) or cores < 1:
            raise ValueError(
                "cores must be None or a positive integer."
            )

    if not 0 < target_accept < 1:
        raise ValueError(
            "target_accept must be between 0 and 1."
        )

    # =====================================================
    # Log Model Structure
    # =====================================================

    logger.info(
        f"Fitting BSTS v{model_version} for {target_dma}"
    )

    logger.info(
        f"  Model type       : {model_type}"
    )

    logger.info(
        f"  Pre-treatment    : {model_data['n_pre']}"
    )

    logger.info(
        f"  Control DMAs     : {model_data['n_controls']}"
    )

    logger.info(
        f"  Trend enabled    : "
        f"{model_data['include_trend']}"
    )

    logger.info(
        f"  Seasonality      : "
        f"{model_data['include_seasonality']}"
    )

    if model_data["include_seasonality"]:
        logger.info(
            f"  Seasonal terms   : "
            f"{2 * model_data['seasonal_harmonics']}"
        )

    logger.info(
        f"  Chains           : {chains}"
    )

    logger.info(
        f"  Draws / chain    : {draws}"
    )

    logger.info(
        f"  Tune / chain     : {tune}"
    )

    logger.info(
        f"  Target accept    : {target_accept}"
    )

    # =====================================================
    # Validate Expected V2 Variables
    # =====================================================

    expected_vars = {
        "intercept",
        "sigma_obs",
    }

    if model_data["include_trend"]:
        expected_vars.add("trend_beta")

    if model_data["n_controls"] > 0:
        expected_vars.add("beta")

    if model_data["include_seasonality"]:
        expected_vars.add("seasonal_beta")

    model_named_vars = set(
        model.named_vars.keys()
    )

    missing_vars = (
        expected_vars - model_named_vars
    )

    if missing_vars:
        raise ValueError(
            f"{target_dma}: configured v2 model is missing "
            f"expected variables: {sorted(missing_vars)}"
        )

    # Explicitly reject old stochastic-state variables.
    legacy_vars = {
        "level",
        "trend",
        "level_innovation",
        "trend_innovation",
        "sigma_level",
        "sigma_trend",
    }

    legacy_found = (
        legacy_vars & model_named_vars
    )

    if legacy_found:
        raise ValueError(
            f"{target_dma}: legacy stochastic BSTS variables "
            f"detected in v2 model: {sorted(legacy_found)}. "
            "This indicates that the old model configuration "
            "is still being used."
        )

    logger.info(
        f"  Model variables  : "
        f"{sorted(expected_vars)}"
    )

    # =====================================================
    # Sample Posterior
    # =====================================================

    logger.info("=" * 70)
    logger.info(
        f"Starting NUTS sampling for {target_dma}..."
    )
    logger.info("=" * 70)

    with model:

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
        )

    # =====================================================
    # Sampling Diagnostics
    # =====================================================

    logger.info("=" * 70)
    logger.info(
        f"BSTS sampling complete for {target_dma}."
    )
    logger.info("=" * 70)

    posterior = idata.posterior

    n_chains = posterior.sizes.get(
        "chain",
        0,
    )

    n_draws = posterior.sizes.get(
        "draw",
        0,
    )

    total_draws = (
        n_chains *
        n_draws
    )

    divergences = 0

    if (
        hasattr(idata, "sample_stats")
        and
        "diverging" in idata.sample_stats
    ):

        divergences = int(
            idata.sample_stats["diverging"]
            .sum()
            .item()
        )

    divergence_rate = (
        divergences / total_draws
        if total_draws > 0
        else np.nan
    )

    logger.info(
        f"Posterior chains       : {n_chains}"
    )

    logger.info(
        f"Posterior draws/chain  : {n_draws}"
    )

    logger.info(
        f"Total posterior draws  : {total_draws}"
    )

    logger.info(
        f"Posterior divergences  : {divergences}"
    )

    logger.info(
        f"Divergence rate        : "
        f"{divergence_rate:.4%}"
        if np.isfinite(divergence_rate)
        else
        "Divergence rate        : unavailable"
    )

    if divergences > 0:

        logger.warning(
            f"{target_dma}: {divergences} divergent "
            "transitions detected."
        )

    # =====================================================
    # Build Results Object
    # =====================================================

    model_results = {
        "model": model,
        "model_data": model_data,
        "idata": idata,
        "divergences": divergences,
        "divergence_rate": divergence_rate,
        "model_type": model_type,
        "model_version": model_version,
    }


    # =====================================================
    # Return
    # =====================================================

    return model_results

#=========================================================
# Save BSTS Model
#=========================================================
def save_bsts_model(
    idata,
    model_data,
    output_dir=None,
):
    """
    Save a fitted BSTS posterior and the metadata/input data
    required for downstream validation and impact analysis.

    Parameters
    ----------
    idata : arviz.InferenceData
        Fitted posterior returned by fit_bsts_model().

    model_data : dict
        Model configuration and input metadata returned by
        configure_bsts_model().

    output_dir : Path, optional
        Directory in which to save the model artifacts.
        If None, uses MODEL_DIR / "bsts".

    Returns
    -------
    dict
        Paths to the saved model artifacts.
    """

    # =====================================================
    # Validate Inputs
    # =====================================================

    if idata is None:

        raise ValueError(
            "idata is None. Cannot save BSTS model."
        )

    if not hasattr(idata, "posterior"):

        raise ValueError(
            "idata does not contain a posterior group."
        )

    if model_data is None:

        raise ValueError(
            "model_data is None. Cannot save BSTS model."
        )

    required_keys = [
        "target_dma",
        "dates",
        "intervention_date",
        "n_pre",
        "y_mean",
        "y_std",
        "include_trend",
        "include_seasonality",
        "seasonal_period",
        "seasonal_harmonics",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in model_data
    ]

    if missing_keys:

        raise ValueError(
            "model_data is missing required fields: "
            f"{missing_keys}"
        )

    target_dma = model_data["target_dma"]

    if target_dma is None:

        raise ValueError(
            "model_data['target_dma'] is None."
        )

    # =====================================================
    # Determine Output Directory
    # =====================================================

    if output_dir is None:

        output_dir = (
            MODEL_DIR /
            "bsts"
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # Create Filesystem-Safe DMA Name
    # =====================================================

    safe_dma = (
        str(target_dma)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(",", "")
    )

    model_path = (
        output_dir /
        f"{safe_dma}_bsts.nc"
    )

    metadata_path = (
        output_dir /
        f"{safe_dma}_metadata.json"
    )

    inputs_path = (
        output_dir /
        f"{safe_dma}_inputs.npz"
    )

    # =====================================================
    # Save InferenceData
    # =====================================================

    idata.to_netcdf(
        model_path
    )

    if not model_path.exists():

        raise IOError(
            f"BSTS model file was not created: "
            f"{model_path}"
        )

    # =====================================================
    # Prepare Metadata
    # =====================================================

    dates = pd.DatetimeIndex(
        model_data["dates"]
    )

    metadata = {

        "target_dma":
            str(target_dma),

        "dates": [
            pd.Timestamp(
                date
            ).isoformat()
            for date in dates
        ],

        "intervention_date":
            pd.Timestamp(
                model_data["intervention_date"]
            ).isoformat(),

        "n_pre":
            int(
                model_data["n_pre"]
            ),

        "y_mean":
            float(
                model_data["y_mean"]
            ),

        "y_std":
            float(
                model_data["y_std"]
            ),

        "include_trend":
            bool(
                model_data["include_trend"]
            ),

        "include_seasonality":
            bool(
                model_data["include_seasonality"]
            ),

        "seasonal_period":
            int(
                model_data["seasonal_period"]
            ),

        "seasonal_harmonics":
            int(
                model_data["seasonal_harmonics"]
            ),

        "control_dmas":
            list(
                model_data.get(
                    "control_dmas",
                    []
                )
            ),

        "model_type":
            "deterministic_trend_control_fourier_bsts",

        "model_version":
            "v3_regularized",
    }

    # =====================================================
    # Save Metadata
    # =====================================================

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    if not metadata_path.exists():

        raise IOError(
            f"BSTS metadata file was not created: "
            f"{metadata_path}"
        )

    # =====================================================
    # Save Downstream Model Inputs
    #
    # These arrays are required by the validation /
    # counterfactual analysis script.
    # =====================================================

    inputs_to_save = {}

    if "controls_full_scaled" in model_data:

        inputs_to_save[
            "controls_full_scaled"
        ] = np.asarray(
            model_data[
                "controls_full_scaled"
            ],
            dtype=float,
        )

    if "y_pre" in model_data:

        inputs_to_save[
            "y_pre"
        ] = np.asarray(
            model_data[
                "y_pre"
            ],
            dtype=float,
        )

    if not inputs_to_save:

        raise ValueError(
            f"{target_dma}: no downstream model inputs "
            "were found to save."
        )

    np.savez_compressed(
        inputs_path,
        **inputs_to_save,
    )

    if not inputs_path.exists():

        raise IOError(
            f"BSTS input file was not created: "
            f"{inputs_path}"
        )

    # =====================================================
    # Logging
    # =====================================================

    logger.info(
        f"Saved BSTS model for {target_dma} "
        f"to {model_path}"
    )

    logger.info(
        f"Saved BSTS metadata for {target_dma} "
        f"to {metadata_path}"
    )

    logger.info(
        f"Saved BSTS downstream inputs for {target_dma} "
        f"to {inputs_path}"
    )

    # =====================================================
    # Return Artifact Paths
    # =====================================================

    return {
        "model_path":
            model_path,

        "metadata_path":
            metadata_path,

        "inputs_path":
            inputs_path,
    }

def main():

    logger.info("=" * 70)
    logger.info("Begin BSTS / Causal Impact Model Fitting")
    logger.info("=" * 70)

    # =====================================================
    # Configuration
    # =====================================================

    configure_pytensor()

    create_output_folders()

    # -----------------------------------------------------
    # Intervention date
    # -----------------------------------------------------

    intervention_date = pd.Timestamp(
        "2026-01-01"
    )

    # =====================================================
    # Load + Validate Panel
    # =====================================================

    logger.info("=" * 70)
    logger.info("Loading and validating model panel")
    logger.info("=" * 70)

    panel = load_model_panel()

    validate_panel(
        panel
    )

    # =====================================================
    # Prepare DMA Time Series
    # =====================================================

    logger.info("=" * 70)
    logger.info("Preparing DMA time series")
    logger.info("=" * 70)

    dma_timeseries = prepare_dma_timeseries(
        panel=panel,
        intervention_date=intervention_date,
    )

    # =====================================================
    # Identify Treated DMAs
    # =====================================================

    logger.info("=" * 70)
    logger.info("Identifying treated DMAs")
    logger.info("=" * 70)

    treated_dmas, treatment_summary = (
        get_treated_dmas(
            panel=panel,
            intervention_date=intervention_date,
        )
    )

    logger.info(
        f"Treated DMAs identified: "
        f"{len(treated_dmas)}"
    )

    logger.info(
        "%s",
        treated_dmas,
    )

    # =====================================================
    # Identify Candidate Controls
    # =====================================================

    logger.info("=" * 70)
    logger.info("Identifying candidate control DMAs")
    logger.info("=" * 70)

    candidate_controls, control_summary = (
        get_candidate_controls(
            panel,
            treated_dmas,
        )
    )

    logger.info(
        f"Candidate control DMAs: "
        f"{len(candidate_controls)}"
    )

    # =====================================================
    # Build Candidate Control Matrix
    # =====================================================

    logger.info("=" * 70)
    logger.info("Building control matrix")
    logger.info("=" * 70)

    candidate_control_matrix = build_control_matrix(
        dma_timeseries=dma_timeseries,
        candidate_controls=candidate_controls,
        intervention_date=intervention_date,
    )

    # =====================================================
    # Validate Candidate Control Matrix
    # =====================================================

    logger.info("=" * 70)
    logger.info("Validating control matrix")
    logger.info("=" * 70)

    validate_control_matrix(
        candidate_control_matrix
    )

    # =====================================================
    # Select Controls
    # =====================================================

    logger.info("=" * 70)
    logger.info("Selecting control DMAs")
    logger.info("=" * 70)

    selected_controls, control_selection_summary = (
        select_controls(
            treated_dmas,
            dma_timeseries,
            control_matrix=candidate_control_matrix,
        )
    )

    logger.info(
        f"Selected control DMAs: "
        f"{len(selected_controls)}"
    )

    logger.info(
        "%s",
        selected_controls,
    )

    # =====================================================
    # DMA-Level BSTS Model Fitting
    # =====================================================

    for target_dma in treated_dmas:

        logger.info("=" * 70)
        logger.info(
            f"Beginning BSTS model fitting for "
            f"{target_dma}"
        )
        logger.info("=" * 70)

        # -------------------------------------------------
        # Prepare Causal Impact / BSTS Inputs
        # -------------------------------------------------

        prepared_inputs = prepare_causal_impact_inputs(
            dma_timeseries=dma_timeseries,
            target_dma=target_dma,
            selected_controls=selected_controls,
            intervention_date=intervention_date,
        )

        # -------------------------------------------------
        # Extract Prepared BSTS Inputs
        # -------------------------------------------------

        treated_series = pd.Series(
            prepared_inputs["y_full"],
            index=prepared_inputs["dates"],
            name=prepared_inputs["target_dma"],
        )

        control_matrix = pd.DataFrame(
            prepared_inputs["controls_full"],
            index=prepared_inputs["dates"],
            columns=prepared_inputs["control_dmas"],
        )

        validate_bsts_inputs(
            treated_series,
            control_matrix,
            intervention_date,
        )

        # -------------------------------------------------
        # Configure BSTS Model
        # -------------------------------------------------

        model, model_data = configure_bsts_model(
            prepared_inputs=prepared_inputs,
            include_trend=True,
            include_seasonality=False,
        )

        #-------------------------------------------------
        # Validate BSTS with Pre-Treatment Holdout
        #-------------------------------------------------

        holdout_validation, holdout_summary = (
            validate_holdout(
                prepared_inputs=prepared_inputs,
                holdout_periods=3,
            )
        )

        #-------------------------------------------------
        # Fit BSTS Model
        #-------------------------------------------------

        model_results = fit_bsts_model(
            model=model,
            model_data=model_data,
        )

        idata = model_results["idata"]

        #-------------------------------------------------
        # Save Fitted BSTS Model
        #-------------------------------------------------

        model_artifacts = save_bsts_model(
            idata=idata,
            model_data=model_data,
        )

        logger.info(
            f"{target_dma}: BSTS model artifacts saved."
        )

    # =====================================================
    # Final Summary
    # =====================================================

    logger.info("=" * 70)
    logger.info("BSTS Model Fitting Complete")
    logger.info("=" * 70)

    logger.info(
        f"Treated DMAs analyzed: "
        f"{len(treated_dmas)}"
    )

    logger.info(
        f"Selected controls: "
        f"{len(selected_controls)}"
    )


if __name__ == "__main__":
    main()