"""
07a_merchandise_panel_regression.py

Merchandise-Enhanced Panel Regression
-------------------------------------

Purpose
-------
Extend the baseline marketing panel regression framework to
evaluate whether merchandise conditions materially alter the
estimated relationship between marketing spend and store traffic.

This script preserves the existing marketing panel model as the
baseline specification and progressively introduces merchandise
information as robustness controls.

Phase 1
-------
1. Load merchandise-enhanced panel data
2. Validate panel structure and coverage
3. Construct the regression sample
4. Reproduce the baseline panel regression

Merchandise variables are NOT introduced until the baseline
specification is successfully reproduced.

# ==========================================================
# Model Specifications
# ==========================================================


Model progression
-----------------

M0
Traffic ~ Spend
+ DMA Fixed Effects
+ Time Fixed Effects

Purpose:
Reproduce the original marketing baseline.

M1
Traffic ~ Spend
+ Lagged Total Merchandise
+ DMA Fixed Effects
+ Time Fixed Effects

Purpose:
Test whether prior merchandise performance explains
part of the observed Spend-Traffic relationship.

M2
Traffic ~ Spend
+ Lagged Merchandise Groups
+ DMA Fixed Effects
+ Time Fixed Effects

Purpose:
Determine whether specific merchandise categories
provide additional explanatory information.

M3
Traffic ~ Spend
+ Selected Predetermined Controls
+ DMA Fixed Effects
+ Time Fixed Effects

Purpose:
Produce the final controlled marketing specification.
"""


import logging

import numpy as np
import pandas as pd
from statsmodels.api import add_constant, OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor
)
from linearmodels.panel import PanelOLS

from config import (
    PROCESSED_DATA_DIR,
    TABLE_DIR,
    OUTPUT_DIR,
    INTERIM_DATA_DIR,
)


# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(__name__)


# ==========================================================
# Output Directories
# ==========================================================

REPORT_DIR = OUTPUT_DIR / "reports"

MODEL_TABLE_DIR = (
    TABLE_DIR
    / "merchandise_panel_regression"
)


def create_output_folders():

    folders = [
        MODEL_TABLE_DIR,
        REPORT_DIR
    ]

    for folder in folders:
        folder.mkdir(
            parents=True,
            exist_ok=True
        )



# ==========================================================
# Load Data
# ==========================================================
def load_data():

    logger.info(
        "Loading merchandise regression data..."
    )

    panel = pd.read_parquet(
        PROCESSED_DATA_DIR /
        "panel_data.parquet"
    )

    merchandise = pd.read_parquet(
        INTERIM_DATA_DIR /
        "merchandise_dma_quarter.parquet"
    )

    merchandise_group = pd.read_parquet(
        INTERIM_DATA_DIR /
        "merchandise_dma_group_quarter.parquet"
    )

    panel["Month"] = pd.to_datetime(
        panel["Month"]
    )

    merchandise["Quarter_Period"] = pd.to_datetime(
        merchandise["Quarter_Period"]
    )

    merchandise_group[
        "Quarter_Period"
    ] = pd.to_datetime(
        merchandise_group[
            "Quarter_Period"
        ]
    )

    logger.info(
        f"Panel rows       : {len(panel):,}"
    )

    logger.info(
        f"Merch rows       : {len(merchandise):,}"
    )

    logger.info(
        f"Merch group rows : {len(merchandise_group):,}"
    )

    return (
        panel,
        merchandise,
        merchandise_group
    )


# ==========================================================
# Merge Aggregate Merchandise Features
# ==========================================================

def merge_merchandise_features(
    panel,
    merchandise
):
    """
    Merge DMA-quarter merchandise features into the
    monthly modeling panel.

    Monthly observations within the same quarter receive
    the corresponding quarterly merchandise features.

    Grain
    -----
    Left  : DMA x Month
    Right : DMA x Quarter
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("MERGING MERCHANDISE FEATURES")
    logger.info("=" * 70)

    df = panel.copy()
    merch = merchandise.copy()

    # ------------------------------------------------------
    # Standardize dates
    # ------------------------------------------------------

    df["Month"] = pd.to_datetime(
        df["Month"]
    )

    merch["Quarter_Period"] = pd.to_datetime(
        merch["Quarter_Period"]
    )

    # ------------------------------------------------------
    # Create quarter key in monthly panel
    # ------------------------------------------------------

    df["Quarter_Period"] = (
        df["Month"]
        .dt.to_period("Q")
        .dt.to_timestamp()
    )

    # ------------------------------------------------------
    # Validate merchandise grain
    # ------------------------------------------------------

    duplicates = merch.duplicated(
        subset=[
            "DMA",
            "Quarter_Period"
        ]
    ).sum()

    logger.info(
        f"Duplicate merchandise DMA-quarter rows: "
        f"{duplicates:,}"
    )

    if duplicates > 0:

        raise ValueError(
            "Aggregate merchandise data contains "
            "duplicate DMA-quarter rows."
        )

    # ------------------------------------------------------
    # Merge
    # ------------------------------------------------------

    before = len(df)

    df = df.merge(
        merch,
        on=[
            "DMA",
            "Quarter_Period"
        ],
        how="left",
        validate="many_to_one",
        suffixes=("", "_merch")
    )

    if len(df) != before:

        raise ValueError(
            "Merchandise merge changed panel row count."
        )

    # ------------------------------------------------------
    # QA
    # ------------------------------------------------------

    merch_cols = [
        col
        for col in df.columns
        if col.startswith("Merch_")
    ]

    logger.info(
        f"Panel rows             : {len(df):,}"
    )

    logger.info(
        f"Merchandise columns    : {len(merch_cols):,}"
    )

    logger.info("")

    for col in merch_cols:

        missing = df[col].isna().sum()

        logger.info(
            f"{col:<35} "
            f"missing={missing:>4} "
            f"({missing / len(df):6.2%})"
        )

    required = [
        "DMA",
        "Month"
    ]

    missing = [
        col
        for col in required
        if col not in panel.columns
    ]

    if missing:
        raise ValueError(
            f"Panel missing merge keys: {missing}"
        )    

    logger.info("=" * 70)

    return df


# ==========================================================
# Panel Validation
# ==========================================================

def validate_panel(panel):

    logger.info("")
    logger.info("=" * 70)
    logger.info("PANEL VALIDATION")
    logger.info("=" * 70)

    # --------------------------------------------------
    # Grain
    # --------------------------------------------------

    duplicate_rows = (
        panel.duplicated(
            subset=["DMA", "Month"]
        ).sum()
    )

    logger.info(
        f"Duplicate DMA-month rows : "
        f"{duplicate_rows:,}"
    )

    if duplicate_rows > 0:
        raise ValueError(
            "Duplicate DMA-month rows detected."
        )

    # --------------------------------------------------
    # Coverage
    # --------------------------------------------------

    dma_count = panel["DMA"].nunique()
    month_count = panel["Month"].nunique()

    expected_rows = (
        dma_count * month_count
    )

    actual_rows = len(panel)

    logger.info(
        f"DMA count               : "
        f"{dma_count:,}"
    )

    logger.info(
        f"Month count             : "
        f"{month_count:,}"
    )

    logger.info(
        f"Expected balanced rows  : "
        f"{expected_rows:,}"
    )

    logger.info(
        f"Actual rows             : "
        f"{actual_rows:,}"
    )

    logger.info(
        f"Balanced panel          : "
        f"{actual_rows == expected_rows}"
    )

    # --------------------------------------------------
    # Critical Variables
    # --------------------------------------------------

    critical_columns = [
        "DMA",
        "Month",
        "Spend",
        "Traffic",
        "Store_Count"
    ]

    for col in critical_columns:

        if col not in panel.columns:
            raise ValueError(
                f"Required column missing: {col}"
            )

        missing = panel[col].isna().sum()

        logger.info(
            f"Missing {col:<20}: "
            f"{missing:,}"
        )

    # --------------------------------------------------
    # Merchandise Coverage
    # --------------------------------------------------

    merchandise_columns = [
        "Merch_Sales_Dollars",
        "Merch_Sales_Units",
        "Merch_AUR",
        "Merch_Sales_per_Store",
        "Merch_Units_per_Store",
        "Merch_Sales_Lag_1Q",
        "Merch_Units_Lag_1Q",
        "Merch_AUR_Lag_1Q"
    ]

    logger.info("")
    logger.info(
        "Merchandise variable coverage:"
    )

    for col in merchandise_columns:

        if col not in panel.columns:

            logger.warning(
                f"Missing merchandise column: "
                f"{col}"
            )

            continue

        missing = panel[col].isna().sum()

        pct_missing = (
            missing / len(panel)
        )

        logger.info(
            f"{col:<32} "
            f"{missing:>4,} missing "
            f"({pct_missing:.2%})"
        )

    logger.info("=" * 70)


# ==========================================================
# Baseline Regression Sample
# ==========================================================

def create_baseline_sample(panel):
    """
    Construct the baseline regression sample.

    No merchandise variables are used to filter the
    baseline sample.

    The resulting panel is indexed by DMA and Month
    for PanelOLS.
    """

    logger.info(
        "Creating baseline regression sample..."
    )

    df = panel.copy()

    required_columns = [
        "DMA",
        "Month",
        "Spend",
        "Traffic"
    ]

    # --------------------------------------------------
    # Validate Required Columns
    # --------------------------------------------------

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------
    # Drop Missing Baseline Observations
    # --------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=required_columns
    ).copy()

    after = len(df)

    logger.info(
        f"Rows before filtering : {before:,}"
    )

    logger.info(
        f"Rows after filtering  : {after:,}"
    )

    logger.info(
        f"Rows removed          : "
        f"{before - after:,}"
    )

    # --------------------------------------------------
    # Sort and Index
    # --------------------------------------------------

    df = (
        df
        .sort_values(
            ["DMA", "Month"]
        )
        .set_index(
            ["DMA", "Month"]
        )
    )

    logger.info(
        f"Regression rows       : "
        f"{len(df):,}"
    )

    logger.info(
        f"Regression DMAs       : "
        f"{df.index.get_level_values('DMA').nunique():,}"
    )

    logger.info(
        f"Regression months     : "
        f"{df.index.get_level_values('Month').nunique():,}"
    )

    return df

# ==========================================================
# Pooled OLS Regression
# ==========================================================

def run_pooled_ols(panel):
    """
    Reproduce the baseline pooled OLS model from
    07_panel_regression.py.

    Traffic = Spend + Error
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("BASELINE 1: POOLED OLS")
    logger.info("=" * 70)

    model_data = panel.reset_index()

    X = model_data[
        ["Spend"]
    ]

    y = model_data[
        "Traffic"
    ]

    X = add_constant(X)

    model = OLS(
        y,
        X
    ).fit()

    logger.info(
        model.summary()
    )

    return model


# ==========================================================
# DMA Fixed Effects
# ==========================================================

def run_dma_fixed_effects(panel):
    """
    Reproduce the DMA fixed-effects model from
    07_panel_regression.py.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("BASELINE 2: DMA FIXED EFFECTS")
    logger.info("=" * 70)

    model = PanelOLS(
        dependent=panel["Traffic"],
        exog=panel[
            ["Spend"]
        ],
        entity_effects=True
    )

    results = model.fit(
        cov_type="clustered",
        cluster_entity=True
    )

    logger.info(
        results.summary
    )

    return results


# ==========================================================
# DMA + Time Fixed Effects
# ==========================================================

def run_dma_time_fixed_effects(panel):
    """
    Reproduce the primary DMA + Time fixed-effects
    model from 07_panel_regression.py.

    Traffic = Spend + DMA FE + Month FE
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "BASELINE 3: DMA + TIME FIXED EFFECTS"
    )
    logger.info("=" * 70)

    model = PanelOLS(
        dependent=panel["Traffic"],
        exog=panel[
            ["Spend"]
        ],
        entity_effects=True,
        time_effects=True
    )

    results = model.fit(
        cov_type="clustered",
        cluster_entity=True
    )

    logger.info(
        results.summary
    )

    return results


# ==========================================================
# Baseline Model Comparison
# ==========================================================

def create_baseline_comparison(
    pooled_model,
    dma_fe_results,
    dma_time_fe_results
):
    """
    Create a compact comparison of the Spend coefficient
    across the three baseline specifications.
    """

    logger.info(
        "Creating baseline model comparison..."
    )

    rows = []

    # --------------------------------------------------
    # Pooled OLS
    # --------------------------------------------------

    pooled_ci = (
        pooled_model
        .conf_int()
        .loc["Spend"]
    )

    rows.append({
        "Model":
            "Pooled OLS",

        "Spend_Coefficient":
            pooled_model.params["Spend"],

        "Std_Error":
            pooled_model.bse["Spend"],

        "P_Value":
            pooled_model.pvalues["Spend"],

        "CI_Lower":
            pooled_ci.iloc[0],

        "CI_Upper":
            pooled_ci.iloc[1],

        "N":
            int(pooled_model.nobs),

        "R_Squared":
            pooled_model.rsquared
    })

    # --------------------------------------------------
    # DMA Fixed Effects
    # --------------------------------------------------

    dma_ci = (
        dma_fe_results
        .conf_int()
        .loc["Spend"]
    )

    rows.append({
        "Model":
            "DMA FE",

        "Spend_Coefficient":
            dma_fe_results.params["Spend"],

        "Std_Error":
            dma_fe_results.std_errors["Spend"],

        "P_Value":
            dma_fe_results.pvalues["Spend"],

        "CI_Lower":
            dma_ci["lower"],

        "CI_Upper":
            dma_ci["upper"],

        "N":
            int(dma_fe_results.nobs),

        "R_Squared":
            dma_fe_results.rsquared
    })

    # --------------------------------------------------
    # DMA + Time Fixed Effects
    # --------------------------------------------------

    time_ci = (
        dma_time_fe_results
        .conf_int()
        .loc["Spend"]
    )

    rows.append({
        "Model":
            "DMA + Time FE",

        "Spend_Coefficient":
            dma_time_fe_results.params["Spend"],

        "Std_Error":
            dma_time_fe_results.std_errors["Spend"],

        "P_Value":
            dma_time_fe_results.pvalues["Spend"],

        "CI_Lower":
            time_ci["lower"],

        "CI_Upper":
            time_ci["upper"],

        "N":
            int(dma_time_fe_results.nobs),

        "R_Squared":
            dma_time_fe_results.rsquared
    })

    comparison = pd.DataFrame(
        rows
    )

    comparison.to_csv(
        MODEL_TABLE_DIR
        / "baseline_model_comparison.csv",
        index=False
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("BASELINE MODEL COMPARISON")
    logger.info("=" * 70)

    logger.info(
        "\n"
        + comparison.round(4)
        .to_string(index=False)
    )

    logger.info("=" * 70)

    return comparison


# ==========================================================
# Merchandise Feature Inventory
# ==========================================================

def inspect_merchandise_features(panel):
    """
    Identify merchandise variables available in the
    enhanced modeling panel.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("MERCHANDISE FEATURE INVENTORY")
    logger.info("=" * 70)

    merch_columns = [
        col
        for col in panel.columns
        if col.startswith("Merch_")
    ]

    logger.info(
        f"Merchandise columns found: "
        f"{len(merch_columns)}"
    )

    logger.info("")

    for col in merch_columns:

        missing = panel[col].isna().sum()

        logger.info(
            f"{col:<40} "
            f"missing={missing:>4} "
            f"({missing / len(panel):6.2%})"
        )

    logger.info("=" * 70)

    return merch_columns


# ==========================================================
# M1 Merchandise Regression Sample
# ==========================================================

def create_merchandise_regression_sample(panel):
    """
    Construct the regression sample for the first
    merchandise-enhanced specification.

    M1 uses lagged total merchandise performance rather
    than contemporaneous merchandise performance.

    The sample is restricted to observations with complete
    Traffic, Spend, and lagged merchandise information.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("CREATING M1 MERCHANDISE SAMPLE")
    logger.info("=" * 70)

    df = panel.reset_index().copy()

    required_columns = [
        "DMA",
        "Month",
        "Traffic",
        "Spend",
        "Merch_Sales_Lag_1Q",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required M1 columns: "
            f"{missing_columns}"
        )

    before = len(df)

    df = df.dropna(
        subset=required_columns
    ).copy()

    logger.info(
        f"Rows before filtering : {before:,}"
    )

    logger.info(
        f"Rows after filtering  : {len(df):,}"
    )

    logger.info(
        f"Rows removed          : "
        f"{before - len(df):,}"
    )

    logger.info(
        f"DMAs retained         : "
        f"{df['DMA'].nunique():,}"
    )

    logger.info(
        f"Months retained       : "
        f"{df['Month'].nunique():,}"
    )

    logger.info(
        f"Date range            : "
        f"{df['Month'].min().date()} -> "
        f"{df['Month'].max().date()}"
    )

    df = (
        df
        .sort_values(
            ["DMA", "Month"]
        )
        .set_index(
            ["DMA", "Month"]
        )
    )

    logger.info("=" * 70)

    return df


# ==========================================================
# M0 Matched-Sample Baseline
# ==========================================================

def run_matched_baseline(panel):
    """
    Re-estimate the baseline Spend model using exactly
    the same observations available to M1.

    This separates changes caused by sample composition
    from changes caused by merchandise controls.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M0 MATCHED SAMPLE")
    logger.info("=" * 70)

    model = PanelOLS(

        dependent=
        panel["Traffic"],

        exog=
        panel[
            [
                "Spend"
            ]
        ],

        entity_effects=True,

        time_effects=True
    )

    results = model.fit(

        cov_type="clustered",

        cluster_entity=True
    )

    logger.info(
        results.summary
    )

    return results


# ==========================================================
# M1: Lagged Merchandise Model
# ==========================================================

def run_lagged_merchandise_model(panel):
    """
    Estimate the baseline marketing model with prior-quarter
    merchandise sales added as a control.

    Traffic_it =
        beta_1 * Spend_it
        + beta_2 * Merch_Sales_Lag_1Q_it
        + DMA FE
        + Time FE
        + error_it

    Merchandise is lagged to reduce contemporaneous
    simultaneity between traffic and merchandise sales.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M1: LAGGED TOTAL MERCHANDISE")
    logger.info("=" * 70)

    exog_columns = [
        "Spend",
        "Merch_Sales_Lag_1Q"
    ]

    model = PanelOLS(

        dependent=
        panel["Traffic"],

        exog=
        panel[
            exog_columns
        ],

        entity_effects=True,

        time_effects=True
    )

    results = model.fit(

        cov_type="clustered",

        cluster_entity=True
    )

    logger.info(
        results.summary
    )

    return results



# ==========================================================
# Model Result Extraction
# ==========================================================

def extract_model_result(
    results,
    model_name,
    variable="Spend"
):
    """
    Extract comparable statistics from a PanelOLS model.
    """

    ci = (
        results
        .conf_int()
        .loc[variable]
    )

    return {

        "Model":
            model_name,

        "Variable":
            variable,

        "Coefficient":
            results.params[variable],

        "Std_Error":
            results.std_errors[variable],

        "P_Value":
            results.pvalues[variable],

        "CI_Lower":
            ci["lower"],

        "CI_Upper":
            ci["upper"],

        "N":
            int(results.nobs),

        "R2":
            results.rsquared,

        "R2_Within":
            results.rsquared_within,

        "R2_Between":
            results.rsquared_between,

        "R2_Overall":
            results.rsquared_overall
    }


# ==========================================================
# Model Progression Table
# ==========================================================

def create_model_progression_table(
    baseline_full,
    baseline_matched,
    merchandise_model
):
    """
    Compare the Spend coefficient across the baseline
    and first merchandise-enhanced specifications.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("MODEL PROGRESSION")
    logger.info("=" * 70)

    rows = [

        extract_model_result(
            baseline_full,
            "M0 - Full Baseline"
        ),

        extract_model_result(
            baseline_matched,
            "M0 - Matched Sample"
        ),

        extract_model_result(
            merchandise_model,
            "M1 - Lagged Merchandise"
        )

    ]

    comparison = pd.DataFrame(
        rows
    )

    # --------------------------------------------------
    # Spend Coefficient Change
    # --------------------------------------------------

    matched_beta = (
        comparison.loc[
            comparison["Model"]
            == "M0 - Matched Sample",
            "Coefficient"
        ]
        .iloc[0]
    )

    comparison[
        "Change_vs_Matched_Baseline"
    ] = (
        comparison["Coefficient"]
        - matched_beta
    )

    comparison[
        "Pct_Change_vs_Matched_Baseline"
    ] = (
        comparison[
            "Change_vs_Matched_Baseline"
        ]
        / abs(matched_beta)
    )

    comparison.to_csv(

        MODEL_TABLE_DIR /
        "model_progression_m0_m1.csv",

        index=False
    )

    logger.info("")
    logger.info(
        "\n"
        + comparison.round(4)
        .to_string(index=False)
    )

    logger.info("=" * 70)

    return comparison


# ==========================================================
# M1 Coefficient Table
# ==========================================================

def export_m1_coefficients(results):
    """
    Export coefficients from the first merchandise model.
    """

    ci = results.conf_int()

    coefficients = pd.DataFrame({

        "Variable":
            results.params.index,

        "Coefficient":
            results.params.values,

        "Std_Error":
            results.std_errors.values,

        "P_Value":
            results.pvalues.values,

        "CI_Lower":
            ci["lower"].values,

        "CI_Upper":
            ci["upper"].values

    })

    coefficients.to_csv(

        MODEL_TABLE_DIR /
        "m1_lagged_merchandise_coefficients.csv",

        index=False
    )

    logger.info("")
    logger.info("M1 Coefficients")
    logger.info("-" * 70)

    logger.info(
        "\n"
        + coefficients.round(4)
        .to_string(index=False)
    )

    return coefficients


# ==========================================================
# M2: Prepare Lagged Merchandise Group Features
# ==========================================================

def prepare_lagged_merchandise_groups(
    merchandise_group
):
    """
    Validate and prepare the existing DMA x merchandise-group
    x quarter feature dataset for M2 regression screening.

    The group-level merchandise parquet is already feature
    engineered and contains prior-quarter lagged features.

    Primary M2 feature
    ------------------
    Group_Sales_Dollars_Lag_1Q

    Grain
    -----
    DMA x Merch_Group x Quarter_Period
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("PREPARING LAGGED MERCHANDISE GROUP FEATURES")
    logger.info("=" * 70)

    df = merchandise_group.copy()

    # ------------------------------------------------------
    # Required columns
    # ------------------------------------------------------

    required = [
        "DMA",
        "Merch_Group",
        "Quarter_Period",
        "Group_Sales_Dollars",
        "Group_Sales_Dollars_Lag_1Q",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required merchandise-group "
            f"columns: {missing}"
        )

    # ------------------------------------------------------
    # Standardize types
    # ------------------------------------------------------

    df["Quarter_Period"] = pd.to_datetime(
        df["Quarter_Period"]
    )

    df = (
        df
        .sort_values(
            [
                "DMA",
                "Merch_Group",
                "Quarter_Period"
            ]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------
    # Validate grain
    # ------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "DMA",
            "Merch_Group",
            "Quarter_Period"
        ]
    ).sum()

    logger.info(
        f"Rows                  : {len(df):,}"
    )

    logger.info(
        f"DMAs                  : "
        f"{df['DMA'].nunique():,}"
    )

    logger.info(
        f"Merchandise groups    : "
        f"{df['Merch_Group'].nunique():,}"
    )

    logger.info(
        f"Quarters              : "
        f"{df['Quarter_Period'].nunique():,}"
    )

    logger.info(
        f"Duplicate grain rows  : "
        f"{duplicates:,}"
    )

    if duplicates > 0:

        raise ValueError(
            "Duplicate DMA-Merch_Group-Quarter rows "
            "found in merchandise group data."
        )

    # ------------------------------------------------------
    # Lag coverage
    # ------------------------------------------------------

    lag_missing = (
        df["Group_Sales_Dollars_Lag_1Q"]
        .isna()
        .sum()
    )

    logger.info(
        f"Lagged sales present  : "
        f"{len(df) - lag_missing:,}"
    )

    logger.info(
        f"Lagged sales missing  : "
        f"{lag_missing:,} "
        f"({lag_missing / len(df):.2%})"
    )

    logger.info(
        f"Quarter range         : "
        f"{df['Quarter_Period'].min().date()} -> "
        f"{df['Quarter_Period'].max().date()}"
    )

    logger.info("=" * 70)

    return df

# ==========================================================
# M2: Single Group Fixed Effects Model
# ==========================================================

def run_single_group_model(
    model_data,
    group_name
):
    """
    Estimate one M2 merchandise-group model.

    Specification
    -------------
    Traffic =
        Spend
        + Group_Sales_Dollars_Lag_1Q
        + DMA FE
        + Time FE
        + Error

    Standard errors are clustered by DMA.
    """

    required = [
        "Traffic",
        "Spend",
        "Group_Sales_Dollars_Lag_1Q"
    ]

    df = (
        model_data
        .dropna(
            subset=required
        )
        .copy()
    )

    if df.empty:

        logger.warning(
            f"{group_name}: no usable observations."
        )

        return None

    # ------------------------------------------------------
    # Set panel index
    # ------------------------------------------------------

    df = (
        df
        .set_index(
            [
                "DMA",
                "Month"
            ]
        )
        .sort_index()
    )

    # ------------------------------------------------------
    # Estimate model
    # ------------------------------------------------------

    model = PanelOLS(

        dependent=
        df["Traffic"],

        exog=
        df[
            [
                "Spend",
                "Group_Sales_Dollars_Lag_1Q"
            ]
        ],

        entity_effects=True,

        time_effects=True
    )

    results = model.fit(

        cov_type="clustered",

        cluster_entity=True
    )

    return results

# ==========================================================
# M2: Merchandise Group Screening
# ==========================================================

def run_merchandise_group_screening(
    baseline_panel,
    merchandise_group,
    matched_baseline_coefficient
):
    """
    Run one DMA + Time FE regression for every
    merchandise group.

    Each model includes:

        Traffic
            ~ Spend
            + Lagged Group Merchandise Sales
            + DMA FE
            + Time FE

    Returns
    -------
    DataFrame
        Ranked merchandise-group regression results.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M2: MERCHANDISE GROUP SCREENING")
    logger.info("=" * 70)

    # ------------------------------------------------------
    # Prepare lagged group features
    # ------------------------------------------------------

    merch = prepare_lagged_merchandise_groups(
        merchandise_group
    )

    # ------------------------------------------------------
    # Add quarter key to monthly panel
    # ------------------------------------------------------

    panel = baseline_panel.copy()

    # ------------------------------------------------------
    # Normalize panel structure
    # ------------------------------------------------------

    if isinstance(panel.index, pd.MultiIndex):

        index_names = panel.index.names

        if (
            "DMA" in index_names
            and "Month" in index_names
        ):

            panel = panel.reset_index()

    logger.info(
        f"M2 baseline panel rows : {len(panel):,}"
    )

    logger.info(
        f"M2 baseline columns    : "
        f"{panel.columns.tolist()}"
    )

    required_panel_columns = [
        "DMA",
        "Month",
        "Traffic",
        "Spend",
    ]

    missing_panel_columns = [
        col
        for col in required_panel_columns
        if col not in panel.columns
    ]

    if missing_panel_columns:

        raise ValueError(
            "M2 baseline panel missing required "
            f"columns: {missing_panel_columns}"
        )

    panel["Quarter_Period"] = (
        panel["Month"]
        .dt.to_period("Q")
        .dt.to_timestamp()
    )

    groups = sorted(
        merch["Merch_Group"]
        .dropna()
        .unique()
    )

    logger.info(
        f"Groups to estimate: {len(groups)}"
    )

    results_list = []

    # ------------------------------------------------------
    # Estimate one model per merchandise group
    # ------------------------------------------------------

    for i, group in enumerate(
        groups,
        start=1
    ):

        logger.info("")
        logger.info("-" * 70)

        logger.info(
            f"M2 [{i:02d}/{len(groups):02d}] "
            f"{group}"
        )

        # --------------------------------------------------
        # Isolate merchandise group
        # --------------------------------------------------

        group_merch = merch[
            merch["Merch_Group"] == group
        ][
            [
                "DMA",
                "Quarter_Period",
                "Group_Sales_Dollars_Lag_1Q"
            ]
        ].copy()

        # --------------------------------------------------
        # Merge into monthly baseline panel
        # --------------------------------------------------

        model_data = panel.merge(
            group_merch,
            on=[
                "DMA",
                "Quarter_Period"
            ],
            how="left",
            validate="many_to_one"
        )

        # --------------------------------------------------
        # Estimate model
        # --------------------------------------------------

        try:

            model = run_single_group_model(
                model_data,
                group
            )

        except Exception as exc:

            logger.exception(
                f"Model failed for {group}: {exc}"
            )

            continue

        if model is None:
            continue

        # --------------------------------------------------
        # Extract coefficients
        # --------------------------------------------------

        merch_beta = model.params[
            "Group_Sales_Dollars_Lag_1Q"
        ]

        merch_se = model.std_errors[
            "Group_Sales_Dollars_Lag_1Q"
        ]

        merch_p = model.pvalues[
            "Group_Sales_Dollars_Lag_1Q"
        ]

        merch_ci = model.conf_int().loc[
            "Group_Sales_Dollars_Lag_1Q"
        ]

        spend_beta = model.params[
            "Spend"
        ]

        spend_se = model.std_errors[
            "Spend"
        ]

        spend_p = model.pvalues[
            "Spend"
        ]

        # --------------------------------------------------
        # Spend coefficient stability
        # --------------------------------------------------

        spend_change = (
            spend_beta
            - matched_baseline_coefficient
        )

        spend_pct_change = (
            spend_change
            / matched_baseline_coefficient
        )

        # --------------------------------------------------
        # Save result
        # --------------------------------------------------

        results_list.append(
            {
                "Merch_Group":
                    group,

                "Merch_Coefficient":
                    merch_beta,

                "Merch_Std_Error":
                    merch_se,

                "Merch_P_Value":
                    merch_p,

                "Merch_CI_Lower":
                    merch_ci.iloc[0],

                "Merch_CI_Upper":
                    merch_ci.iloc[1],

                "Spend_Coefficient":
                    spend_beta,

                "Spend_Std_Error":
                    spend_se,

                "Spend_P_Value":
                    spend_p,

                "Spend_Change_vs_M0":
                    spend_change,

                "Spend_Pct_Change_vs_M0":
                    spend_pct_change,

                "N":
                    model.nobs,

                "R2":
                    model.rsquared,

                "R2_Within":
                    model.rsquared_within,

                "R2_Between":
                    model.rsquared_between,

                "R2_Overall":
                    model.rsquared_overall,
            }
        )

        logger.info(
            f"Merch β={merch_beta:>10.6f} | "
            f"p={merch_p:>7.4f} | "
            f"Spend β={spend_beta:>7.4f} | "
            f"N={int(model.nobs):>3}"
        )

    # ------------------------------------------------------
    # Final table
    # ------------------------------------------------------

    results_df = pd.DataFrame(
        results_list
    )

    if results_df.empty:

        raise ValueError(
            "No merchandise group models "
            "were successfully estimated."
        )

    results_df = (
        results_df
        .sort_values(
            "Merch_P_Value"
        )
        .reset_index(drop=True)
    )


    # ==========================================================
    # Multiple-Testing Correction
    # ==========================================================

    results_df["FDR_P_Value"] = multipletests(
        results_df["Merch_P_Value"],
        method="fdr_bh"
    )[1]

    results_df["FDR_Significant"] = (
        results_df["FDR_P_Value"] < 0.05
    )

    results_df["Merch_Significant"] = (
        results_df["Merch_P_Value"] < 0.05
    )

    return results_df


# ==========================================================
# M2: Report Group Screening Results
# ==========================================================

def report_merchandise_group_screening(
    results
):
    """
    Log and export M2 merchandise-group results.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M2 MERCHANDISE GROUP RESULTS")
    logger.info("=" * 70)

    display_cols = [
        "Merch_Group",
        "Merch_Coefficient",
        "Merch_Std_Error",
        "Merch_P_Value",
        "Spend_Coefficient",
        "Spend_Pct_Change_vs_M0",
        "N",
        "R2_Within",
    ]

    logger.info(
        "\n%s",
        results[
            display_cols
        ].round(4).to_string(
            index=False
        )
    )

    output_file = (
        TABLE_DIR
        / "merchandise_panel_regression"
        / "m2_group_screening.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    logger.info("")
    logger.info(
        f"Saved {output_file.name}"
    )

    logger.info("=" * 70)


# ==========================================================
# M3 Configuration
# ==========================================================

M3_GROUPS = [
    "Accessories",
    "Basics and Sleepwear",
    "Beauty and Fragrance",
    "Home Essentials",
    "Home Living",
    "Kids and Baby Apparel",
    "Ladies and Juniors Apparel",
    "Mens and Young Mens Apparel",
    "Shoes",
]

M3_FEATURE = "Group_Sales_Dollars_Lag_1Q"


# ==========================================================
# M3: Create Wide Merchandise Feature Matrix
# ==========================================================

def create_merchandise_group_matrix(
    merchandise_group
):
    """
    Create a DMA-quarter feature matrix containing lagged
    merchandise sales for the standard merchandise groups.

    Output grain
    ------------
    DMA x Quarter_Period

    Columns
    -------
    One lagged-sales feature per merchandise group.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M3: CREATING MERCHANDISE GROUP FEATURE MATRIX")
    logger.info("=" * 70)

    df = merchandise_group.copy()

    # ------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------

    required = [
        "DMA",
        "Merch_Group",
        "Quarter_Period",
        M3_FEATURE,
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required M3 columns: "
            f"{missing}"
        )

    # ------------------------------------------------------
    # Keep standard merchandise groups
    # ------------------------------------------------------

    df = df[
        df["Merch_Group"].isin(M3_GROUPS)
    ].copy()

    df["Quarter_Period"] = pd.to_datetime(
        df["Quarter_Period"]
    )

    logger.info(
        f"Rows retained       : {len(df):,}"
    )

    logger.info(
        f"DMAs                : {df['DMA'].nunique():,}"
    )

    logger.info(
        f"Groups              : {df['Merch_Group'].nunique():,}"
    )

    logger.info(
        f"Quarters            : "
        f"{df['Quarter_Period'].nunique():,}"
    )

    # ------------------------------------------------------
    # Pivot wide
    # ------------------------------------------------------

    wide = (
        df
        .pivot(
            index=[
                "DMA",
                "Quarter_Period"
            ],
            columns="Merch_Group",
            values=M3_FEATURE
        )
        .reset_index()
    )

    # ------------------------------------------------------
    # Clean column names
    # ------------------------------------------------------

    wide.columns.name = None

    rename_map = {
        group: (
            "Lag_"
            + group
            .replace(" ", "_")
            .replace("&", "and")
        )
        for group in M3_GROUPS
    }

    wide = wide.rename(
        columns=rename_map
    )

    feature_columns = list(
        rename_map.values()
    )

    # ------------------------------------------------------
    # Coverage QA
    # ------------------------------------------------------

    logger.info("")
    logger.info("Feature Coverage")
    logger.info("-" * 70)

    for col in feature_columns:

        missing_count = wide[col].isna().sum()

        logger.info(
            f"{col:<40} "
            f"missing={missing_count:>5,} "
            f"({missing_count / len(wide):6.2%})"
        )

    logger.info("")
    logger.info(
        f"Wide matrix rows    : {len(wide):,}"
    )

    logger.info(
        f"Feature columns     : {len(feature_columns)}"
    )

    logger.info("=" * 70)

    return wide, feature_columns


# ==========================================================
# M3: Merchandise Group Correlation Diagnostics
# ==========================================================

def analyze_merchandise_group_correlations(
    wide,
    feature_columns
):
    """
    Calculate pairwise correlations among lagged merchandise
    group features.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M3: MERCHANDISE GROUP CORRELATIONS")
    logger.info("=" * 70)

    corr = wide[
        feature_columns
    ].corr()

    corr.to_csv(
        TABLE_DIR /
        "merchandise_panel_regression" /
        "m3_group_correlation_matrix.csv"
    )

    # ------------------------------------------------------
    # Extract unique pairs
    # ------------------------------------------------------

    pairs = []

    for i, col1 in enumerate(feature_columns):

        for col2 in feature_columns[i + 1:]:

            value = corr.loc[
                col1,
                col2
            ]

            pairs.append({
                "Feature_1": col1,
                "Feature_2": col2,
                "Correlation": value,
                "Abs_Correlation": abs(value),
            })

    pairs = (
        pd.DataFrame(pairs)
        .sort_values(
            "Abs_Correlation",
            ascending=False
        )
        .reset_index(drop=True)
    )

    pairs.to_csv(
        TABLE_DIR /
        "merchandise_panel_regression" /
        "m3_group_correlation_pairs.csv",
        index=False
    )

    logger.info("")
    logger.info("Highest Pairwise Correlations")
    logger.info("-" * 70)

    logger.info(
        "\n" +
        pairs.head(15).to_string(
            index=False
        )
    )

    return corr, pairs


# ==========================================================
# M3: Variance Inflation Factors
# ==========================================================

def calculate_merchandise_vif(
    wide,
    feature_columns
):
    """
    Calculate VIF for the lagged merchandise group features.

    Uses complete DMA-quarter observations across all
    selected merchandise groups.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M3: MERCHANDISE MULTICOLLINEARITY - VIF")
    logger.info("=" * 70)

    model_data = (
        wide[
            feature_columns
        ]
        .dropna()
        .copy()
    )

    logger.info(
        f"Complete observations : "
        f"{len(model_data):,}"
    )

    if model_data.empty:

        raise ValueError(
            "No complete observations available "
            "for M3 VIF calculation."
        )

    vif_rows = []

    for i, col in enumerate(feature_columns):

        vif = variance_inflation_factor(
            model_data.values,
            i
        )

        vif_rows.append({
            "Feature": col,
            "VIF": vif,
        })

    vif_table = (
        pd.DataFrame(vif_rows)
        .sort_values(
            "VIF",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------
    # Diagnostic classification
    # ------------------------------------------------------

    vif_table["VIF_Level"] = pd.cut(
        vif_table["VIF"],
        bins=[
            -np.inf,
            2.5,
            5,
            10,
            np.inf
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Severe"
        ]
    )

    vif_table.to_csv(
        TABLE_DIR /
        "merchandise_panel_regression" /
        "m3_group_vif.csv",
        index=False
    )

    logger.info("")
    logger.info(
        "\n" +
        vif_table.to_string(
            index=False
        )
    )

    logger.info("=" * 70)

    return vif_table


# ==========================================================
# M3: Condition Number
# ==========================================================

def calculate_condition_number(
    wide,
    feature_columns
):
    """
    Calculate the condition number of standardized
    merchandise-group regressors.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M3: FEATURE MATRIX CONDITION NUMBER")
    logger.info("=" * 70)

    X = (
        wide[
            feature_columns
        ]
        .dropna()
        .copy()
    )

    # Standardize because merchandise groups differ in scale
    X = (
        X - X.mean()
    ) / X.std()

    condition_number = np.linalg.cond(
        X.values
    )

    logger.info(
        f"Condition number : "
        f"{condition_number:.2f}"
    )

    logger.info("=" * 70)

    return condition_number


# ==========================================================
# M3B: Merchandise Share Diagnostics
# ==========================================================

def prepare_merchandise_share_panel(
    merchandise_group,
    panel_dmas=None
):
    """
    Prepare merchandise-group share data for M3B diagnostics.

    Primary grain:
        DMA x Quarter x Merch_Group

    Focus:
        - Current dollar share
        - Lagged dollar share
        - Current unit share
        - Lagged unit share

    Optionally restricts analysis to DMAs appearing
    in the primary modeling panel.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M3B: PREPARING MERCHANDISE SHARE PANEL")
    logger.info("=" * 70)

    required = [
        "DMA",
        "Merch_Group",
        "Quarter_Period",
        "Group_Dollar_Share",
        "Group_Unit_Share",
        "Group_Dollar_Share_Lag_1Q",
        "Group_Unit_Share_Lag_1Q",
    ]

    missing = [
        col for col in required
        if col not in merchandise_group.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required M3B columns: {missing}"
        )

    df = merchandise_group[
        required
    ].copy()

    df["Quarter_Period"] = pd.to_datetime(
        df["Quarter_Period"]
    )

    # ------------------------------------------------------
    # Restrict to modeling DMAs if supplied
    # ------------------------------------------------------

    if panel_dmas is not None:

        panel_dmas = set(panel_dmas)

        before = len(df)

        df = df[
            df["DMA"].isin(panel_dmas)
        ].copy()

        logger.info(
            f"Restricted to panel DMAs: "
            f"{before:,} -> {len(df):,} rows"
        )

    # ------------------------------------------------------
    # Duplicate grain check
    # ------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "DMA",
            "Quarter_Period",
            "Merch_Group"
        ]
    ).sum()

    logger.info(
        f"Rows                  : {len(df):,}"
    )

    logger.info(
        f"DMAs                  : {df['DMA'].nunique():,}"
    )

    logger.info(
        f"Quarters              : "
        f"{df['Quarter_Period'].nunique():,}"
    )

    logger.info(
        f"Merchandise groups    : "
        f"{df['Merch_Group'].nunique():,}"
    )

    logger.info(
        f"Duplicate grain rows  : {duplicates:,}"
    )

    if duplicates > 0:
        raise ValueError(
            "Duplicate DMA-quarter-group rows found "
            "in merchandise share panel."
        )

    return df


# ==========================================================
# M3B: Merchandise Share Variation
# ==========================================================

def analyze_merchandise_share_variation(
    share_panel
):
    """
    Measure how merchandise dollar share varies:

        1. Overall
        2. Between DMAs
        3. Within DMAs over time

    This determines whether merchandise mix is primarily
    a cross-sectional DMA characteristic or a time-varying
    feature suitable for within-DMA analysis.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M3B: MERCHANDISE SHARE VARIATION")
    logger.info("=" * 70)

    rows = []

    groups = sorted(
        share_panel["Merch_Group"]
        .dropna()
        .unique()
    )

    for group in groups:

        df = share_panel[
            share_panel["Merch_Group"] == group
        ].copy()

        current = df[
            "Group_Dollar_Share"
        ].dropna()

        lagged = df[
            "Group_Dollar_Share_Lag_1Q"
        ].dropna()

        # --------------------------------------------------
        # DMA average merchandise share
        # --------------------------------------------------

        dma_means = (
            df
            .groupby("DMA")[
                "Group_Dollar_Share"
            ]
            .mean()
        )

        between_dma_std = (
            dma_means.std()
        )

        # --------------------------------------------------
        # Within-DMA variation
        #
        # Demean each observation relative to that DMA's
        # average merchandise share.
        # --------------------------------------------------

        df["DMA_Mean_Share"] = (
            df
            .groupby("DMA")[
                "Group_Dollar_Share"
            ]
            .transform("mean")
        )

        df["Within_DMA_Deviation"] = (
            df["Group_Dollar_Share"]
            - df["DMA_Mean_Share"]
        )

        within_dma_std = (
            df["Within_DMA_Deviation"]
            .std()
        )

        # --------------------------------------------------
        # Relative source of variation
        # --------------------------------------------------

        total_component = (
            between_dma_std ** 2
            + within_dma_std ** 2
        )

        if total_component > 0:

            between_share = (
                between_dma_std ** 2
                / total_component
            )

            within_share = (
                within_dma_std ** 2
                / total_component
            )

        else:

            between_share = np.nan
            within_share = np.nan

        rows.append({

            "Merch_Group":
                group,

            "Mean_Share":
                current.mean(),

            "Median_Share":
                current.median(),

            "Std_Share":
                current.std(),

            "Min_Share":
                current.min(),

            "Max_Share":
                current.max(),

            "Between_DMA_Std":
                between_dma_std,

            "Within_DMA_Std":
                within_dma_std,

            "Between_Variance_Share":
                between_share,

            "Within_Variance_Share":
                within_share,

            "Lagged_Mean_Share":
                lagged.mean(),

            "DMAs":
                df["DMA"].nunique(),

            "Quarters":
                df["Quarter_Period"].nunique(),

            "N":
                len(current),

            "Lagged_N":
                len(lagged),

        })

    results = pd.DataFrame(rows)

    results = results.sort_values(
        "Between_DMA_Std",
        ascending=False
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # Export
    # ------------------------------------------------------

    output_path = (
        TABLE_DIR
        / "merchandise_panel_regression"
        / "m3b_merchandise_share_variation.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    logger.info("")
    logger.info(
        results[
            [
                "Merch_Group",
                "Mean_Share",
                "Between_DMA_Std",
                "Within_DMA_Std",
                "Between_Variance_Share",
                "Within_Variance_Share",
                "DMAs",
                "N"
            ]
        ].round(4).to_string(index=False)
    )

    logger.info("")
    logger.info(
        f"Saved: {output_path.name}"
    )

    logger.info("=" * 70)

    return results


# ==========================================================
# M3B: Merchandise Share Correlations
# ==========================================================

def analyze_merchandise_share_correlations(
    share_panel
):
    """
    Analyze pairwise correlations between merchandise
    group dollar shares.

    Converts long merchandise data into a
    DMA-quarter x merchandise-group matrix.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M3B: MERCHANDISE SHARE CORRELATIONS")
    logger.info("=" * 70)

    wide = (
        share_panel
        .pivot_table(
            index=[
                "DMA",
                "Quarter_Period"
            ],
            columns="Merch_Group",
            values="Group_Dollar_Share"
        )
    )

    # ------------------------------------------------------
    # Clean column names
    # ------------------------------------------------------

    wide.columns = [
        "Share_" +
        str(col)
        .replace(" ", "_")
        .replace("&", "and")
        .replace("/", "_")
        .replace("-", "_")
        for col in wide.columns
    ]

    corr = wide.corr()

    # ------------------------------------------------------
    # Convert matrix to unique pairs
    # ------------------------------------------------------

    rows = []

    columns = corr.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            rows.append({

                "Feature_1":
                    columns[i],

                "Feature_2":
                    columns[j],

                "Correlation":
                    corr.iloc[i, j],

                "Abs_Correlation":
                    abs(corr.iloc[i, j])

            })

    pairs = pd.DataFrame(rows)

    pairs = pairs.sort_values(
        "Abs_Correlation",
        ascending=False
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # Export
    # ------------------------------------------------------

    corr.to_csv(
        TABLE_DIR
        / "merchandise_panel_regression"
        / "m3b_share_correlation_matrix.csv"
    )

    pairs.to_csv(
        TABLE_DIR
        / "merchandise_panel_regression"
        / "m3b_share_correlation_pairs.csv",
        index=False
    )

    logger.info("")
    logger.info(
        "Highest Merchandise Share Correlations"
    )

    logger.info("-" * 70)

    logger.info("")
    logger.info(
        pairs.head(15)
        .round(4)
        .to_string(index=False)
    )

    logger.info("=" * 70)

    return wide, corr, pairs


# ==========================================================
# M3B: Quarterly Traffic Outcomes
# ==========================================================

def create_quarterly_traffic_outcomes(panel):
    """
    Create DMA-quarter traffic outcomes for M3B
    merchandise-mix analysis.

    Primary outcomes
    ----------------
    Traffic
        Total quarterly DMA traffic.

    Traffic_per_Store
        Quarterly traffic normalized by average
        DMA store count.

    Traffic_per_1000SqFt
        Quarterly traffic normalized by DMA
        selling square footage.

    Comp_Traffic
        Quarterly comparable-store YoY traffic:

            sum(Comp_Exits)
            ---------------- - 1
            sum(Comp_Exits_LY)

    Supporting comp metrics
    -----------------------
    Comp_Exits
    Comp_Exits_LY
    Comp_Stores
    Comp_Store_Coverage

    Notes
    -----
    Comp traffic is constructed from the comparable-store
    fields created upstream in 02_clean_data.py.

    This function does NOT reconstruct YoY traffic using
    quarterly lags.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M3B: CREATING QUARTERLY TRAFFIC OUTCOMES")
    logger.info("=" * 70)

    # ------------------------------------------------------
    # Prepare panel
    # ------------------------------------------------------

    df = panel.reset_index().copy()

    df["Month"] = pd.to_datetime(
        df["Month"]
    )

    df["Quarter_Period"] = (
        df["Month"]
        .dt.to_period("Q")
        .dt.to_timestamp()
    )

    # ------------------------------------------------------
    # Required columns
    # ------------------------------------------------------

    required_columns = [
        "DMA",
        "Month",
        "Quarter_Period",
        "Traffic",
        "Transactions",
        "Store_Count",
        "Total_Selling_SqFt",
        "Spend",
        "Comp_Exits",
        "Comp_Exits_LY",
        "Comp_Stores",
        "Comp_Store_Coverage",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required M3B traffic columns: "
            f"{missing_columns}"
        )

    # ------------------------------------------------------
    # DMA-quarter aggregation
    # ------------------------------------------------------

    quarterly = (
        df
        .groupby(
            [
                "DMA",
                "Quarter_Period"
            ],
            as_index=False
        )
        .agg(

            Traffic=(
                "Traffic",
                "sum"
            ),

            Transactions=(
                "Transactions",
                "sum"
            ),

            Store_Count=(
                "Store_Count",
                "mean"
            ),

            Total_Selling_SqFt=(
                "Total_Selling_SqFt",
                "mean"
            ),

            Spend=(
                "Spend",
                "sum"
            ),

            Comp_Exits=(
                "Comp_Exits",
                "sum"
            ),

            Comp_Exits_LY=(
                "Comp_Exits_LY",
                "sum"
            ),

            Comp_Stores=(
                "Comp_Stores",
                "mean"
            ),

            Months_Observed=(
                "Month",
                "nunique"
            ),

            Comp_Months_Observed=(
                "Comp_Traffic",
                lambda x: x.notna().sum()
            )
        )
    )

    # ------------------------------------------------------
    # Complete-quarter requirement
    # ------------------------------------------------------

    incomplete_quarters = (
        quarterly[
            quarterly["Months_Observed"] < 3
        ]
    )

    if not incomplete_quarters.empty:

        logger.warning(
            f"Excluding {len(incomplete_quarters):,} "
            "DMA-quarter observations with fewer "
            "than 3 monthly observations."
        )

    quarterly = (
        quarterly[
            quarterly["Months_Observed"] == 3
        ]
        .copy()
    )

    # ------------------------------------------------------
    # Level-based normalized outcomes
    # ------------------------------------------------------

    quarterly["Traffic_per_Store"] = (
        quarterly["Traffic"]
        / quarterly["Store_Count"]
    )

    quarterly["Traffic_per_1000SqFt"] = (
        quarterly["Traffic"]
        / quarterly["Total_Selling_SqFt"]
        * 1000
    )

    quarterly["Spend_per_Store"] = (
        quarterly["Spend"]
        / quarterly["Store_Count"]
    )

    # ------------------------------------------------------
    # Quarterly comparable traffic
    #
    # IMPORTANT:
    # Aggregate current and LY exits first, then calculate
    # the growth rate. Do not average monthly comp rates.
    # ------------------------------------------------------

    valid_comp_base = (
        quarterly["Comp_Exits_LY"].notna()
        & (quarterly["Comp_Exits_LY"] > 0)
    )

    quarterly["Comp_Traffic"] = np.nan

    quarterly.loc[
        valid_comp_base,
        "Comp_Traffic"
    ] = (
        quarterly.loc[
            valid_comp_base,
            "Comp_Exits"
        ]
        / quarterly.loc[
            valid_comp_base,
            "Comp_Exits_LY"
        ]
        - 1
    )

    # ------------------------------------------------------
    # Quarterly comp-store coverage
    # ------------------------------------------------------

    quarterly["Comp_Store_Coverage"] = (
        quarterly["Comp_Stores"]
        / quarterly["Store_Count"]
    )

    # ------------------------------------------------------
    # Sort final panel
    # ------------------------------------------------------

    quarterly = (
        quarterly
        .sort_values(
            [
                "DMA",
                "Quarter_Period"
            ]
        )
        .reset_index(drop=True)
    )

    # ======================================================
    # QA
    # ======================================================

    logger.info("")
    logger.info("-" * 70)
    logger.info("Quarterly Outcome Coverage")
    logger.info("-" * 70)

    logger.info(
        f"Quarterly rows              : "
        f"{len(quarterly):,}"
    )

    logger.info(
        f"DMAs                        : "
        f"{quarterly['DMA'].nunique():,}"
    )

    logger.info(
        f"Quarters                    : "
        f"{quarterly['Quarter_Period'].nunique():,}"
    )

    logger.info(
        f"Quarter range               : "
        f"{quarterly['Quarter_Period'].min().date()} "
        f"-> "
        f"{quarterly['Quarter_Period'].max().date()}"
    )

    logger.info(
        f"Complete Comp Traffic       : "
        f"{quarterly['Comp_Traffic'].notna().sum():,}"
    )

    logger.info(
        f"Comp Traffic coverage       : "
        f"{quarterly['Comp_Traffic'].notna().mean():.2%}"
    )

    # ------------------------------------------------------
    # Comp traffic distribution
    # ------------------------------------------------------

    logger.info("")
    logger.info("-" * 70)
    logger.info("Quarterly Comp Traffic Distribution")
    logger.info("-" * 70)

    logger.info(
        quarterly[
            "Comp_Traffic"
        ]
        .describe()
        .to_string()
    )

    # ------------------------------------------------------
    # Comp-store coverage
    # ------------------------------------------------------

    logger.info("")
    logger.info("-" * 70)
    logger.info("Quarterly Comp Store Coverage")
    logger.info("-" * 70)

    logger.info(
        quarterly[
            "Comp_Store_Coverage"
        ]
        .describe()
        .to_string()
    )

    low_coverage = (
        quarterly[
            "Comp_Store_Coverage"
        ] < 0.80
    ).sum()

    logger.info(
        f"Quarters below 80% coverage : "
        f"{low_coverage:,}"
    )

    # ------------------------------------------------------
    # Duplicate validation
    # ------------------------------------------------------

    duplicates = (
        quarterly
        .duplicated(
            [
                "DMA",
                "Quarter_Period"
            ]
        )
        .sum()
    )

    logger.info(
        f"Duplicate DMA-quarter rows  : "
        f"{duplicates:,}"
    )

    if duplicates > 0:

        raise ValueError(
            "M3B quarterly outcomes contain "
            "duplicate DMA-quarter rows."
        )

    logger.info("=" * 70)

    return quarterly


# ==========================================================
# M3B: Merchandise Mix Analysis Panel
# ==========================================================

def create_merchandise_mix_panel(
    quarterly_outcomes,
    merchandise_group
):
    """
    Create the DMA-quarter-merchandise-group analysis panel
    used for M3B merchandise-mix regression.

    Grain
    -----
    DMA x Quarter x Merchandise Group

    Primary merchandise variable
    ----------------------------
    Group_Dollar_Share

    This measures the proportion of total DMA merchandise
    sales represented by each merchandise group.

    Primary traffic outcome
    -----------------------
    Comp_Traffic

    Secondary outcomes
    ------------------
    Traffic_per_Store
    Traffic_per_1000SqFt
    Traffic

    Notes
    -----
    The purpose of this panel is to evaluate whether
    differences in merchandise mix are associated with
    differences in DMA traffic performance.

    Merchandise dollar levels are retained for diagnostics,
    but group share is the primary M3B explanatory variable.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M3B: CREATING MERCHANDISE MIX PANEL")
    logger.info("=" * 70)

    # ------------------------------------------------------
    # Prepare merchandise data
    # ------------------------------------------------------

    merch = merchandise_group.copy()

    merch["Quarter_Period"] = pd.to_datetime(
        merch["Quarter_Period"]
    )

    # ------------------------------------------------------
    # Required merchandise columns
    # ------------------------------------------------------

    required_merch_columns = [
        "DMA",
        "Quarter_Period",
        "Merch_Group",
        "Group_Sales_Dollars",
        "Group_Sales_Units",
        "Group_Stores",
        "Group_Dollar_Share",
        "Group_Unit_Share",
        "Group_AUR",
    ]

    missing_merch_columns = [
        col
        for col in required_merch_columns
        if col not in merch.columns
    ]

    if missing_merch_columns:

        raise ValueError(
            "Missing required M3B merchandise columns: "
            f"{missing_merch_columns}"
        )

    # ------------------------------------------------------
    # Validate merchandise grain
    # ------------------------------------------------------

    merch_duplicates = (
        merch
        .duplicated(
            [
                "DMA",
                "Quarter_Period",
                "Merch_Group"
            ]
        )
        .sum()
    )

    logger.info(
        f"Merchandise duplicate grain rows : "
        f"{merch_duplicates:,}"
    )

    if merch_duplicates > 0:

        raise ValueError(
            "Merchandise group data contains duplicate "
            "DMA-quarter-group rows."
        )

    # ------------------------------------------------------
    # Prepare quarterly outcomes
    # ------------------------------------------------------

    outcomes = quarterly_outcomes.copy()

    outcomes["Quarter_Period"] = pd.to_datetime(
        outcomes["Quarter_Period"]
    )

    required_outcome_columns = [
        "DMA",
        "Quarter_Period",
        "Traffic",
        "Traffic_per_Store",
        "Traffic_per_1000SqFt",
        "Transactions",
        "Spend",
        "Spend_per_Store",
        "Store_Count",
        "Comp_Traffic",
        "Comp_Exits",
        "Comp_Exits_LY",
        "Comp_Stores",
        "Comp_Store_Coverage",
    ]

    missing_outcome_columns = [
        col
        for col in required_outcome_columns
        if col not in outcomes.columns
    ]

    if missing_outcome_columns:

        raise ValueError(
            "Missing required M3B outcome columns: "
            f"{missing_outcome_columns}"
        )

    # ------------------------------------------------------
    # Validate outcome grain
    # ------------------------------------------------------

    outcome_duplicates = (
        outcomes
        .duplicated(
            [
                "DMA",
                "Quarter_Period"
            ]
        )
        .sum()
    )

    if outcome_duplicates > 0:

        raise ValueError(
            "Quarterly outcomes contain duplicate "
            "DMA-quarter rows."
        )

    # ------------------------------------------------------
    # Restrict merchandise data to relevant outcome sample
    # ------------------------------------------------------

    logger.info("")
    logger.info("-" * 70)
    logger.info("Pre-Merge Coverage")
    logger.info("-" * 70)

    logger.info(
        f"Outcome rows             : "
        f"{len(outcomes):,}"
    )

    logger.info(
        f"Outcome DMAs             : "
        f"{outcomes['DMA'].nunique():,}"
    )

    logger.info(
        f"Outcome quarters         : "
        f"{outcomes['Quarter_Period'].nunique():,}"
    )

    logger.info(
        f"Merchandise rows         : "
        f"{len(merch):,}"
    )

    logger.info(
        f"Merchandise DMAs         : "
        f"{merch['DMA'].nunique():,}"
    )

    logger.info(
        f"Merchandise groups       : "
        f"{merch['Merch_Group'].nunique():,}"
    )

    # ------------------------------------------------------
    # Merge traffic outcomes onto merchandise groups
    #
    # many_to_one:
    # many merchandise groups map to one DMA-quarter
    # traffic observation.
    # ------------------------------------------------------

    mix_panel = merch.merge(
        outcomes,
        on=[
            "DMA",
            "Quarter_Period"
        ],
        how="inner",
        validate="many_to_one"
    )

    # ------------------------------------------------------
    # Sort final analytical panel
    # ------------------------------------------------------

    mix_panel = (
        mix_panel
        .sort_values(
            [
                "DMA",
                "Quarter_Period",
                "Merch_Group"
            ]
        )
        .reset_index(drop=True)
    )

    # ======================================================
    # Post-Merge QA
    # ======================================================

    logger.info("")
    logger.info("-" * 70)
    logger.info("M3B Mix Panel Coverage")
    logger.info("-" * 70)

    logger.info(
        f"Analysis rows            : "
        f"{len(mix_panel):,}"
    )

    logger.info(
        f"DMAs                     : "
        f"{mix_panel['DMA'].nunique():,}"
    )

    logger.info(
        f"Quarters                 : "
        f"{mix_panel['Quarter_Period'].nunique():,}"
    )

    logger.info(
        f"Merchandise groups       : "
        f"{mix_panel['Merch_Group'].nunique():,}"
    )

    logger.info(
        f"Comp traffic observations: "
        f"{mix_panel['Comp_Traffic'].notna().sum():,}"
    )

    # ------------------------------------------------------
    # Group coverage
    # ------------------------------------------------------

    group_coverage = (
        mix_panel
        .groupby(
            "Merch_Group",
            as_index=False
        )
        .agg(
            Observations=(
                "Group_Dollar_Share",
                "count"
            ),
            DMAs=(
                "DMA",
                "nunique"
            ),
            Quarters=(
                "Quarter_Period",
                "nunique"
            ),
            Mean_Dollar_Share=(
                "Group_Dollar_Share",
                "mean"
            ),
            Std_Dollar_Share=(
                "Group_Dollar_Share",
                "std"
            )
        )
        .sort_values(
            "Mean_Dollar_Share",
            ascending=False
        )
    )

    logger.info("")
    logger.info("Merchandise Group Coverage")
    logger.info("-" * 70)

    logger.info(
        "\n%s",
        group_coverage.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Share integrity check
    # ------------------------------------------------------

    share_check = (
        mix_panel
        .groupby(
            [
                "DMA",
                "Quarter_Period"
            ],
            as_index=False
        )
        .agg(
            Dollar_Share_Total=(
                "Group_Dollar_Share",
                "sum"
            ),
            Unit_Share_Total=(
                "Group_Unit_Share",
                "sum"
            ),
            Groups_Observed=(
                "Merch_Group",
                "nunique"
            )
        )
    )

    logger.info("")
    logger.info("-" * 70)
    logger.info("Merchandise Share Integrity")
    logger.info("-" * 70)

    logger.info(
        "Dollar Share Totals:"
    )

    logger.info(
        share_check[
            "Dollar_Share_Total"
        ]
        .describe()
        .to_string()
    )

    logger.info("")
    logger.info(
        "Unit Share Totals:"
    )

    logger.info(
        share_check[
            "Unit_Share_Total"
        ]
        .describe()
        .to_string()
    )

    logger.info("")
    logger.info(
        "Groups Observed:"
    )

    logger.info(
        share_check[
            "Groups_Observed"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ------------------------------------------------------
    # Final grain validation
    # ------------------------------------------------------

    duplicates = (
        mix_panel
        .duplicated(
            [
                "DMA",
                "Quarter_Period",
                "Merch_Group"
            ]
        )
        .sum()
    )

    logger.info("")
    logger.info(
        f"Duplicate DMA-quarter-group rows : "
        f"{duplicates:,}"
    )

    if duplicates > 0:

        raise ValueError(
            "M3B merchandise mix panel contains "
            "duplicate DMA-quarter-group rows."
        )

    # ------------------------------------------------------
    # Export QA tables
    # ------------------------------------------------------

    output_dir = (
        TABLE_DIR
        / "merchandise_panel_regression"
    )

    group_coverage.to_csv(
        output_dir
        / "m3b_group_coverage.csv",
        index=False
    )

    share_check.to_csv(
        output_dir
        / "m3b_share_integrity.csv",
        index=False
    )

    logger.info("")
    logger.info(
        "Saved m3b_group_coverage.csv"
    )

    logger.info(
        "Saved m3b_share_integrity.csv"
    )

    logger.info("=" * 70)

    return mix_panel


# ==========================================================
# M3B: Prepare Regression Sample
# ==========================================================

def prepare_m3b_regression_sample(m3b_panel):
    """
    Prepare the final M3B merchandise-mix regression sample.

    Primary question
    ----------------
    Is a change in merchandise mix associated with a change
    in YoY comp traffic performance?

    Transformations
    ---------------
    Comp_Traffic:
        Decimal -> percentage points.

    Group_Dollar_Share:
        Decimal -> percentage points.

    Group_Unit_Share:
        Decimal -> percentage points.

    Primary sample
    --------------
    Retains the nine core merchandise departments with
    complete DMA-quarter coverage.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M3B: PREPARING REGRESSION SAMPLE")
    logger.info("=" * 70)

    df = m3b_panel.copy()

    # ------------------------------------------------------
    # Core merchandise groups
    # ------------------------------------------------------

    core_groups = [
        "Ladies and Juniors Apparel",
        "Mens and Young Mens Apparel",
        "Shoes",
        "Home Essentials",
        "Home Living",
        "Basics and Sleepwear",
        "Kids and Baby Apparel",
        "Accessories",
        "Beauty and Fragrance",
    ]

    logger.info(
        f"Rows before core-group filter : {len(df):,}"
    )

    df = df[
        df["Merch_Group"].isin(core_groups)
    ].copy()

    logger.info(
        f"Rows after core-group filter  : {len(df):,}"
    )

    # ------------------------------------------------------
    # Required columns
    # ------------------------------------------------------

    required_columns = [
        "DMA",
        "Quarter_Period",
        "Merch_Group",
        "Comp_Traffic",
        "Group_Dollar_Share",
        "Group_Unit_Share",
        "Spend",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required M3B regression columns: "
            f"{missing_columns}"
        )

    # ------------------------------------------------------
    # Percentage-point transformations
    # ------------------------------------------------------

    df["Comp_Traffic_PP"] = (
        df["Comp_Traffic"] * 100
    )

    df["Group_Dollar_Share_PP"] = (
        df["Group_Dollar_Share"] * 100
    )

    df["Group_Unit_Share_PP"] = (
        df["Group_Unit_Share"] * 100
    )

    # ------------------------------------------------------
    # Sort
    # ------------------------------------------------------

    df = df.sort_values(
        [
            "Merch_Group",
            "DMA",
            "Quarter_Period",
        ]
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # Duplicate validation
    # ------------------------------------------------------

    duplicate_count = (
        df.duplicated(
            [
                "DMA",
                "Quarter_Period",
                "Merch_Group",
            ]
        )
        .sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "M3B regression sample contains "
            f"{duplicate_count:,} duplicate "
            "DMA-quarter-group rows."
        )

    # ------------------------------------------------------
    # Missing-value validation
    # ------------------------------------------------------

    logger.info("")
    logger.info("-" * 70)
    logger.info("M3B Required Variable Coverage")
    logger.info("-" * 70)

    for col in required_columns:

        missing = df[col].isna().sum()

        logger.info(
            f"{col:<28} "
            f"missing={missing:>5,} "
            f"({missing / len(df):>6.2%})"
        )

    # ------------------------------------------------------
    # Group balance
    # ------------------------------------------------------

    group_balance = (
        df
        .groupby("Merch_Group")
        .agg(
            Observations=(
                "Comp_Traffic_PP",
                "size"
            ),
            DMAs=(
                "DMA",
                "nunique"
            ),
            Quarters=(
                "Quarter_Period",
                "nunique"
            ),
            Mean_Dollar_Share_PP=(
                "Group_Dollar_Share_PP",
                "mean"
            ),
            Std_Dollar_Share_PP=(
                "Group_Dollar_Share_PP",
                "std"
            ),
            Mean_Unit_Share_PP=(
                "Group_Unit_Share_PP",
                "mean"
            ),
            Std_Unit_Share_PP=(
                "Group_Unit_Share_PP",
                "std"
            ),
        )
        .reset_index()
    )

    logger.info("")
    logger.info("-" * 70)
    logger.info("M3B Core Group Balance")
    logger.info("-" * 70)

    logger.info(
        "\n" +
        group_balance.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Final sample summary
    # ------------------------------------------------------

    logger.info("")
    logger.info("-" * 70)
    logger.info("M3B Final Regression Sample")
    logger.info("-" * 70)

    logger.info(
        f"Rows          : {len(df):,}"
    )

    logger.info(
        f"DMAs          : {df['DMA'].nunique():,}"
    )

    logger.info(
        f"Quarters      : "
        f"{df['Quarter_Period'].nunique():,}"
    )

    logger.info(
        f"Groups        : "
        f"{df['Merch_Group'].nunique():,}"
    )

    logger.info(
        f"Duplicate rows: {duplicate_count:,}"
    )

    expected_rows = (
        df["DMA"].nunique()
        * df["Quarter_Period"].nunique()
        * df["Merch_Group"].nunique()
    )

    logger.info(
        f"Expected balanced rows : {expected_rows:,}"
    )

    logger.info(
        f"Actual rows            : {len(df):,}"
    )

    logger.info("=" * 70)

    # ------------------------------------------------------
    # Save QA
    # ------------------------------------------------------

    group_balance.to_csv(
        TABLE_DIR
        / "merchandise_panel_regression"
        / "m3b_core_group_balance.csv",
        index=False
    )

    return df


# ==========================================================
# M3B: Merchandise Share Screening
# ==========================================================

def run_m3b_share_screening(m3b_sample):
    """
    Screen merchandise-group shares against YoY comp traffic.

    For each core merchandise group, estimate separate
    DMA + quarter fixed-effects models using:

        1. Dollar share
        2. Unit share

    Outcome:
        Comp_Traffic_PP

    Controls:
        Spend

    Fixed effects:
        DMA
        Quarter

    Standard errors:
        Clustered by DMA
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M3B: MERCHANDISE SHARE SCREENING")
    logger.info("=" * 70)

    share_specs = {
        "Dollar_Share":
            "Group_Dollar_Share_PP",

        "Unit_Share":
            "Group_Unit_Share_PP",
    }

    groups = sorted(
        m3b_sample["Merch_Group"]
        .dropna()
        .unique()
    )

    results = []

    total_models = (
        len(groups)
        * len(share_specs)
    )

    model_number = 0

    # ------------------------------------------------------
    # Loop through share specifications
    # ------------------------------------------------------

    for share_name, share_variable in share_specs.items():

        logger.info("")
        logger.info("-" * 70)
        logger.info(
            f"M3B Specification: {share_name}"
        )
        logger.info("-" * 70)

        for group in groups:

            model_number += 1

            logger.info("")
            logger.info(
                f"M3B [{model_number:02d}/{total_models:02d}] "
                f"{group} | {share_name}"
            )

            # --------------------------------------------------
            # Group-specific sample
            # --------------------------------------------------

            model_data = (
                m3b_sample[
                    m3b_sample["Merch_Group"] == group
                ]
                .copy()
            )

            model_data = model_data.dropna(
                subset=[
                    "Comp_Traffic_PP",
                    share_variable,
                    "Spend",
                    "DMA",
                    "Quarter_Period",
                ]
            )

            # --------------------------------------------------
            # Panel index
            # --------------------------------------------------

            model_data = (
                model_data
                .set_index(
                    [
                        "DMA",
                        "Quarter_Period",
                    ]
                )
                .sort_index()
            )

            # --------------------------------------------------
            # Explanatory variables
            # --------------------------------------------------

            X = model_data[
                [
                    share_variable,
                    "Spend",
                ]
            ]

            y = model_data[
                "Comp_Traffic_PP"
            ]

            # --------------------------------------------------
            # Estimate DMA + quarter FE model
            # --------------------------------------------------

            model = PanelOLS(
                dependent=y,
                exog=X,
                entity_effects=True,
                time_effects=True,
                drop_absorbed=True
            )

            fit = model.fit(
                cov_type="clustered",
                cluster_entity=True
            )

            # --------------------------------------------------
            # Extract results
            # --------------------------------------------------

            share_beta = (
                fit.params.get(
                    share_variable,
                    np.nan
                )
            )

            share_se = (
                fit.std_errors.get(
                    share_variable,
                    np.nan
                )
            )

            share_p = (
                fit.pvalues.get(
                    share_variable,
                    np.nan
                )
            )

            share_ci = (
                fit.conf_int()
                .loc[share_variable]
                if share_variable
                in fit.params.index
                else [np.nan, np.nan]
            )

            spend_beta = (
                fit.params.get(
                    "Spend",
                    np.nan
                )
            )

            spend_p = (
                fit.pvalues.get(
                    "Spend",
                    np.nan
                )
            )

            results.append({
                "Merch_Group":
                    group,

                "Share_Type":
                    share_name,

                "Share_Variable":
                    share_variable,

                "Share_Coefficient":
                    share_beta,

                "Share_Std_Error":
                    share_se,

                "Share_P_Value":
                    share_p,

                "Share_CI_Lower":
                    share_ci.iloc[0]
                    if hasattr(share_ci, "iloc")
                    else share_ci[0],

                "Share_CI_Upper":
                    share_ci.iloc[1]
                    if hasattr(share_ci, "iloc")
                    else share_ci[1],

                "Spend_Coefficient":
                    spend_beta,

                "Spend_P_Value":
                    spend_p,

                "N":
                    fit.nobs,

                "R2":
                    fit.rsquared,

                "R2_Within":
                    fit.rsquared_within,

                "R2_Between":
                    fit.rsquared_between,

                "R2_Overall":
                    fit.rsquared_overall,
            })

            logger.info(
                f"Share β={share_beta:>8.4f} | "
                f"p={share_p:>7.4f} | "
                f"Spend β={spend_beta:>8.4f} | "
                f"N={fit.nobs:,}"
            )

    results = pd.DataFrame(results)

    return results


# ==========================================================
# M3B: Finalize Screening Results
# ==========================================================

def finalize_m3b_results(results):
    """
    Apply FDR correction separately to dollar-share
    and unit-share merchandise screening models.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("M3B: FINAL MERCHANDISE MIX RESULTS")
    logger.info("=" * 70)

    df = results.copy()

    df["FDR_P_Value"] = np.nan
    df["FDR_Significant"] = False

    # ------------------------------------------------------
    # Correct within each specification family
    # ------------------------------------------------------

    for share_type in df["Share_Type"].unique():

        mask = (
            df["Share_Type"] == share_type
        )

        valid = (
            mask
            & df["Share_P_Value"].notna()
        )

        reject, corrected_p, _, _ = (
            multipletests(
                df.loc[
                    valid,
                    "Share_P_Value"
                ],
                alpha=0.05,
                method="fdr_bh"
            )
        )

        df.loc[
            valid,
            "FDR_P_Value"
        ] = corrected_p

        df.loc[
            valid,
            "FDR_Significant"
        ] = reject

    # ------------------------------------------------------
    # Sort
    # ------------------------------------------------------

    df = df.sort_values(
        [
            "Share_Type",
            "FDR_P_Value",
            "Share_P_Value",
        ]
    )

    # ------------------------------------------------------
    # Report
    # ------------------------------------------------------

    display_columns = [
        "Merch_Group",
        "Share_Type",
        "Share_Coefficient",
        "Share_Std_Error",
        "Share_P_Value",
        "FDR_P_Value",
        "FDR_Significant",
        "Share_CI_Lower",
        "Share_CI_Upper",
        "Spend_Coefficient",
        "N",
        "R2_Within",
    ]

    logger.info(
        "\n" +
        df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    output_path = (
        TABLE_DIR
        / "merchandise_panel_regression"
        / "m3b_merchandise_share_screening.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    logger.info("")
    logger.info(
        "Saved m3b_merchandise_share_screening.csv"
    )

    logger.info("=" * 70)

    return df



# ==========================================================
# Main
# ==========================================================

def main():

    logger.info("")
    logger.info("=" * 70)
    logger.info("MERCHANDISE PANEL REGRESSION")
    logger.info("=" * 70)

    # ======================================================
    # 1. Setup
    # ======================================================

    create_output_folders()


    # ======================================================
    # 2. Load Source Data
    # ======================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 1: LOAD DATA")
    logger.info("=" * 70)

    (
        panel,
        merchandise,
        merchandise_group
    ) = load_data()


    # ======================================================
    # 3. Validate Original Panel
    # ======================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: VALIDATE BASE PANEL")
    logger.info("=" * 70)

    validate_panel(
        panel
    )


    # ======================================================
    # 4. Create Merchandise-Enhanced Panel
    # ======================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3: MERGE AGGREGATE MERCHANDISE")
    logger.info("=" * 70)

    panel_merch = merge_merchandise_features(
        panel,
        merchandise
    )

    inspect_merchandise_features(
        panel_merch
    )


    # ======================================================
    # 5. Create Baseline Regression Sample
    # ======================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 4: CREATE BASELINE SAMPLE")
    logger.info("=" * 70)

    baseline_panel = create_baseline_sample(
        panel
    )


    # ======================================================
    # 6. Reproduce Baseline Models
    # ======================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 5: REPRODUCE BASELINE MODELS")
    logger.info("=" * 70)


    # ------------------------------------------------------
    # M0a: Pooled OLS
    # ------------------------------------------------------

    pooled_model = run_pooled_ols(
        baseline_panel
    )


    # ------------------------------------------------------
    # M0b: DMA Fixed Effects
    # ------------------------------------------------------

    dma_fe_results = run_dma_fixed_effects(
        baseline_panel
    )


    # ------------------------------------------------------
    # M0c: DMA + Time Fixed Effects
    # ------------------------------------------------------

    dma_time_fe_results = (
        run_dma_time_fixed_effects(
            baseline_panel
        )
    )


    # ------------------------------------------------------
    # Baseline Comparison
    # ------------------------------------------------------

    baseline_comparison = (
        create_baseline_comparison(
            pooled_model,
            dma_fe_results,
            dma_time_fe_results
        )
    )


    # ======================================================
    # 7. Create Matched Merchandise Sample
    # ======================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 6: CREATE MATCHED MERCHANDISE SAMPLE")
    logger.info("=" * 70)

    matched_panel = (
        create_merchandise_regression_sample(
            panel_merch
        )
    )


    # ======================================================
    # 8. M0 Matched-Sample Baseline
    # ======================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 7: M0 MATCHED-SAMPLE BASELINE")
    logger.info("=" * 70)

    matched_baseline = run_matched_baseline(
        matched_panel
    )

    matched_baseline_coefficient = (
        matched_baseline.params["Spend"]
    )


    # ======================================================
    # 9. M1: Aggregate Lagged Merchandise
    # ======================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 8: M1 LAGGED AGGREGATE MERCHANDISE")
    logger.info("=" * 70)

    m1_results = (
        run_lagged_merchandise_model(
            matched_panel
        )
    )

    export_m1_coefficients(
        m1_results
    )


    # ======================================================
    # 10. M0 -> M1 Model Progression
    # ======================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 9: M0 -> M1 MODEL PROGRESSION")
    logger.info("=" * 70)

    progression = (
        create_model_progression_table(
            dma_time_fe_results,
            matched_baseline,
            m1_results
        )
    )


    # ======================================================
    # 11. M2: Lagged Merchandise Group Screening
    # ======================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 10: M2 LAGGED MERCHANDISE GROUP MODELS")
    logger.info("=" * 70)

    m2_results = (
        run_merchandise_group_screening(
            baseline_panel=matched_panel,
            merchandise_group=merchandise_group,
            matched_baseline_coefficient=(
                matched_baseline_coefficient
            )
        )
    )

    report_merchandise_group_screening(
        m2_results
    )


    # --------------------------------------------------
    # STEP 11: M3 Multicollinearity Diagnostics
    # --------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "STEP 11: M3 MERCHANDISE "
        "MULTICOLLINEARITY DIAGNOSTICS"
    )
    logger.info("=" * 70)

    m3_matrix, m3_features = (
        create_merchandise_group_matrix(
            merchandise_group
        )
    )

    m3_corr, m3_corr_pairs = (
        analyze_merchandise_group_correlations(
            m3_matrix,
            m3_features
        )
    )

    m3_vif = (
        calculate_merchandise_vif(
            m3_matrix,
            m3_features
        )
    )

    m3_condition_number = (
        calculate_condition_number(
            m3_matrix,
            m3_features
        )
    )

    # ----------------------------------------------------------
    # STEP 11a: M3B Merchandise Share Diagnostics
    # ----------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 11a: M3B MERCHANDISE SHARE DIAGNOSTICS")
    logger.info("=" * 70)

    share_panel = (
        prepare_merchandise_share_panel(
            merchandise_group=merchandise_group,
            panel_dmas=panel["DMA"].unique()
        )
    )

    share_variation = (
        analyze_merchandise_share_variation(
            share_panel
        )
    )

    share_wide, share_corr, share_pairs = (
        analyze_merchandise_share_correlations(
            share_panel
        )
    )


    # ----------------------------------------------------------
    # STEP 11b: M3B Quarterly Traffic Outcomes
    # ----------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 11b: M3B QUARTERLY TRAFFIC OUTCOMES")
    logger.info("=" * 70)

    quarterly_traffic = (
        create_quarterly_traffic_outcomes(
            panel
        )
    )


    # ----------------------------------------------------------
    # STEP 11c: M3B Merchandise Mix Panel
    # ----------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 11c: M3B MERCHANDISE MIX PANEL")
    logger.info("=" * 70)

    m3b_panel = (
        create_merchandise_mix_panel(
            quarterly_outcomes=quarterly_traffic,
            merchandise_group=merchandise_group
        )
    )


    # ----------------------------------------------------------
    # STEP 11d: Prepare M3B Regression Sample
    # ----------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 11d: PREPARE M3B REGRESSION SAMPLE")
    logger.info("=" * 70)

    m3b_sample = (
        prepare_m3b_regression_sample(
            m3b_panel
        )
    )


    # ----------------------------------------------------------
    # STEP 11e: M3B Merchandise Share Screening
    # ----------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 11e: M3B MERCHANDISE SHARE SCREENING")
    logger.info("=" * 70)

    m3b_results = (
        run_m3b_share_screening(
            m3b_sample
        )
    )


    # ----------------------------------------------------------
    # STEP 11f: Finalize M3B Results
    # ----------------------------------------------------------

    m3b_results = (
        finalize_m3b_results(
            m3b_results
        )
    )

    # ======================================================
    # 12. Completion Summary
    # ======================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("MERCHANDISE REGRESSION PHASE COMPLETE")
    logger.info("=" * 70)

    logger.info(
        f"Baseline observations : "
        f"{len(baseline_panel):,}"
    )

    logger.info(
        f"Matched observations  : "
        f"{len(matched_panel):,}"
    )

    logger.info(
        f"Baseline Spend beta   : "
        f"{dma_time_fe_results.params['Spend']:.4f}"
    )

    logger.info(
        f"Matched Spend beta    : "
        f"{matched_baseline.params['Spend']:.4f}"
    )

    logger.info(
        f"M1 Spend beta         : "
        f"{m1_results.params['Spend']:.4f}"
    )

    logger.info(
        f"M2 groups estimated   : "
        f"{len(m2_results):,}"
    )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()