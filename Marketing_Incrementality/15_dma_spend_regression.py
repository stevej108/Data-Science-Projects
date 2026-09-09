# =============================================================================
# 15_dma_spend_explanatory_power.py
#
# Purpose
# -------
# Estimate the DMA-level explanatory power of marketing spend on traffic
# after controlling for trend and seasonality.
#
# Primary outputs
# ---------------
# For each DMA:
#   - Reduced-model R²
#   - Full-model R²
#   - Incremental R² from adding spend
#   - Partial R² for spend
#   - Spend coefficient
#   - Spend p-value
#   - Spend confidence interval
#   - Residualized spend / traffic correlation
#   - Sample size and model diagnostics
#
# IMPORTANT
# ---------
# This is a DMA-level explanatory / association analysis.
# It should NOT be interpreted as a causal estimate of marketing spend.
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os


# =============================================================================
# CONFIG
# =============================================================================

from config import (
    OUTPUT_DIR,
    DATA_DIR
)


MODEL_PANEL_FILE = (
    DATA_DIR
    / "processed"
    / "model_panel.parquet"
)

TABLE_DIR = (
    OUTPUT_DIR
    / "tables"
    / "dma_spend_explanatory_power"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
    / "dma_spend_explanatory_power"
)

REPORT_DIR = (
    OUTPUT_DIR
    / "reports"
)

BSTS_DMA_IMPACT_FILE = (
    OUTPUT_DIR
    / "tables"
    / "causal_impact"
    / "dma_impact_summary.csv"
)


# Minimum number of monthly observations required to fit a DMA model.
#
# With:
#   intercept
#   Log_Spend
#   Trend
#   Month_Sin
#   Month_Cos
#
# we have 5 estimated parameters in the full model.
#
# 18 months is intentionally conservative enough to avoid fitting
# extremely small DMA-specific regressions.
MIN_OBSERVATIONS = 18


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger(
    "dma_spend_explanatory_power"
)

logger.setLevel(
    logging.INFO
)

if not logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s"
    )

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )


# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

def create_output_directories():
    """
    Create output directories used by the DMA spend analysis.
    """

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Output directories created."
    )


# =============================================================================
# LOAD MODEL PANEL
# =============================================================================

def load_model_panel(
    panel_path=None,
):
    """
    Load the validated DMA-month model panel.
    """

    panel_path = Path(
        panel_path or MODEL_PANEL_FILE
    )

    logger.info("=" * 70)
    logger.info("LOADING MODEL PANEL")
    logger.info("=" * 70)

    logger.info(
        "Loading panel from %s",
        panel_path,
    )

    if not panel_path.exists():

        raise FileNotFoundError(
            f"Model panel does not exist: {panel_path}"
        )

    panel = pd.read_parquet(
        panel_path
    )

    logger.info(
        "Panel rows: %s",
        f"{len(panel):,}",
    )

    logger.info(
        "Panel columns: %s",
        panel.columns.tolist(),
    )

    logger.info(
        "Panel dtypes:\n%s",
        panel.dtypes.to_string(),
    )

    return panel


# =============================================================================
# VALIDATE MODEL PANEL
# =============================================================================

def validate_model_panel(
    panel,
):
    """
    Validate required fields and basic DMA-month panel structure.
    """

    logger.info("=" * 70)
    logger.info("VALIDATING MODEL PANEL")
    logger.info("=" * 70)

    required_columns = {
        "DMA",
        "Month",
        "Traffic",
        "Spend",
        "Log_Spend",
        "Month_Sin",
        "Month_Cos",
    }

    missing_columns = (
        required_columns
        -
        set(panel.columns)
    )

    if missing_columns:

        raise ValueError(
            "Model panel is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if panel.empty:

        raise ValueError(
            "Model panel is empty."
        )

    duplicate_mask = panel.duplicated(
        subset=[
            "DMA",
            "Month",
        ],
        keep=False,
    )

    duplicate_count = (
        duplicate_mask.sum()
    )

    logger.info(
        "Duplicate DMA-month rows: %s",
        f"{duplicate_count:,}",
    )

    if duplicate_count > 0:

        duplicates = (
            panel.loc[
                duplicate_mask
            ]
            .sort_values(
                [
                    "DMA",
                    "Month",
                ]
            )
        )

        duplicates.to_csv(
            TABLE_DIR
            / "duplicate_dma_month_rows.csv",
            index=False,
        )

        raise ValueError(
            "Duplicate DMA-month rows detected."
        )

    logger.info(
        "Unique DMAs: %s",
        f"{panel['DMA'].nunique():,}",
    )

    logger.info(
        "Unique months: %s",
        f"{panel['Month'].nunique():,}",
    )

    logger.info(
        "Date range: %s -> %s",
        panel["Month"].min(),
        panel["Month"].max(),
    )

    logger.info(
        "Missing Traffic: %s",
        f"{panel['Traffic'].isna().sum():,}",
    )

    logger.info(
        "Missing Spend: %s",
        f"{panel['Spend'].isna().sum():,}",
    )

    logger.info(
        "Missing Log_Spend: %s",
        f"{panel['Log_Spend'].isna().sum():,}",
    )


# =============================================================================
# LOAD TREATED DMA UNIVERSE
# =============================================================================

def load_treated_dma_universe(
    impact_path=None,
):
    """
    Load the DMA universe used in the causal-impact / BSTS
    intervention analysis.

    This defines the primary reporting population for the
    DMA spend explanatory-power report.
    """

    impact_path = Path(
        impact_path or BSTS_DMA_IMPACT_FILE
    )

    logger.info("=" * 70)
    logger.info("LOADING TREATED DMA REPORTING UNIVERSE")
    logger.info("=" * 70)

    if not impact_path.exists():

        raise FileNotFoundError(
            "BSTS DMA impact summary does not exist: "
            f"{impact_path}"
        )

    treated = pd.read_csv(
        impact_path
    )

    if "DMA" not in treated.columns:

        raise ValueError(
            "BSTS DMA impact summary is missing required "
            "column 'DMA'."
        )

    treated["DMA"] = (
        treated["DMA"]
        .astype("string")
        .str.strip()
    )

    treated = (
        treated[
            ["DMA"]
        ]
        .dropna()
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    logger.info(
        "Treated / BSTS reporting DMAs: %s",
        f"{len(treated):,}",
    )

    logger.info(
        "Treated DMA names:\n%s",
        treated[
            "DMA"
        ]
        .sort_values()
        .to_string(
            index=False
        ),
    )

    return treated

# =============================================================================
# PREPARE DMA REGRESSION PANEL
# =============================================================================

def prepare_regression_panel(
    panel,
):
    """
    Prepare a lean modeling panel for DMA-specific regressions.

    Existing model-panel features are reused where possible.

    Variables retained
    ------------------
    DMA
    Month
    Traffic
    Spend
    Log_Spend
    Month_Sin
    Month_Cos

    Variables created
    -----------------
    Log_Traffic
    Trend
    """

    logger.info("=" * 70)
    logger.info("PREPARING DMA REGRESSION PANEL")
    logger.info("=" * 70)

    keep_columns = [
        "DMA",
        "Month",
        "Traffic",
        "Spend",
        "Log_Spend",
        "Month_Sin",
        "Month_Cos",
    ]

    model = panel[
        keep_columns
    ].copy()

    # -------------------------------------------------------------------------
    # Normalize types
    # -------------------------------------------------------------------------

    model["DMA"] = (
        model["DMA"]
        .astype("string")
        .str.strip()
    )

    model["Month"] = pd.to_datetime(
        model["Month"],
        errors="coerce",
    )

    numeric_columns = [
        "Traffic",
        "Spend",
        "Log_Spend",
        "Month_Sin",
        "Month_Cos",
    ]

    for column in numeric_columns:

        model[column] = pd.to_numeric(
            model[column],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # Remove unusable rows
    # -------------------------------------------------------------------------

    before = len(
        model
    )

    model = model.loc[
        model["DMA"].notna()
        &
        model["Month"].notna()
        &
        model["Traffic"].notna()
        &
        model["Spend"].notna()
        &
        model["Log_Spend"].notna()
        &
        model["Month_Sin"].notna()
        &
        model["Month_Cos"].notna()
        &
        (
            model["Traffic"] >= 0
        )
    ].copy()

    logger.info(
        "Rows removed during regression-panel validation: %s",
        f"{before - len(model):,}",
    )

    if model.empty:

        raise ValueError(
            "Regression panel is empty after validation."
        )

    # -------------------------------------------------------------------------
    # Log traffic
    #
    # log1p allows zero traffic without generating -inf.
    # -------------------------------------------------------------------------

    model["Log_Traffic"] = np.log1p(
        model["Traffic"]
    )

    # -------------------------------------------------------------------------
    # Global calendar trend
    #
    # All DMAs receive the same trend value for the same calendar month.
    # -------------------------------------------------------------------------

    unique_months = (
        model["Month"]
        .drop_duplicates()
        .sort_values()
        .reset_index(
            drop=True
        )
    )

    trend_map = {
        month: index
        for index, month in enumerate(
            unique_months,
            start=0,
        )
    }

    model["Trend"] = (
        model["Month"]
        .map(
            trend_map
        )
        .astype(float)
    )

    model = (
        model
        .sort_values(
            [
                "DMA",
                "Month",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    logger.info(
        "Prepared regression rows: %s",
        f"{len(model):,}",
    )

    logger.info(
        "DMAs: %s",
        f"{model['DMA'].nunique():,}",
    )

    logger.info(
        "Months: %s",
        f"{model['Month'].nunique():,}",
    )

    logger.info(
        "Date range: %s -> %s",
        model["Month"].min(),
        model["Month"].max(),
    )

    logger.info(
        "Traffic range: %.0f -> %.0f",
        model["Traffic"].min(),
        model["Traffic"].max(),
    )

    logger.info(
        "Spend range: %.0f -> %.0f",
        model["Spend"].min(),
        model["Spend"].max(),
    )

    return model


# =============================================================================
# DMA COVERAGE SUMMARY
# =============================================================================

def build_dma_coverage_summary(
    panel,
):
    """
    Summarize monthly observations and spend variation by DMA.

    Spend variation matters because a DMA with constant spend cannot
    support a meaningful within-DMA spend coefficient.
    """

    logger.info("=" * 70)
    logger.info("BUILDING DMA COVERAGE SUMMARY")
    logger.info("=" * 70)

    coverage = (
        panel
        .groupby(
            "DMA",
            as_index=False,
        )
        .agg(
            Observations=(
                "Month",
                "size",
            ),
            Months=(
                "Month",
                "nunique",
            ),
            First_Month=(
                "Month",
                "min",
            ),
            Last_Month=(
                "Month",
                "max",
            ),
            Mean_Traffic=(
                "Traffic",
                "mean",
            ),
            Mean_Spend=(
                "Spend",
                "mean",
            ),
            Spend_SD=(
                "Spend",
                "std",
            ),
            Spend_Min=(
                "Spend",
                "min",
            ),
            Spend_Max=(
                "Spend",
                "max",
            ),
            Positive_Spend_Months=(
                "Spend",
                lambda x:
                    (x > 0).sum(),
            ),
            Zero_Spend_Months=(
                "Spend",
                lambda x:
                    (x == 0).sum(),
            ),
        )
    )

    coverage[
        "Spend_Varies"
    ] = (
        coverage[
            "Spend_Max"
        ]
        >
        coverage[
            "Spend_Min"
        ]
    )

    coverage[
        "Sufficient_Observations"
    ] = (
        coverage[
            "Observations"
        ]
        >=
        MIN_OBSERVATIONS
    )

    coverage[
        "Eligible_For_Model"
    ] = (
        coverage[
            "Sufficient_Observations"
        ]
        &
        coverage[
            "Spend_Varies"
        ]
    )

    logger.info(
        "DMAs available: %s",
        f"{len(coverage):,}",
    )

    logger.info(
        "DMAs with >= %d observations: %s",
        MIN_OBSERVATIONS,
        f"{coverage['Sufficient_Observations'].sum():,}",
    )

    logger.info(
        "DMAs with spend variation: %s",
        f"{coverage['Spend_Varies'].sum():,}",
    )

    logger.info(
        "DMAs eligible for regression: %s",
        f"{coverage['Eligible_For_Model'].sum():,}",
    )

    coverage.to_csv(
        TABLE_DIR
        / "dma_regression_coverage.csv",
        index=False,
    )

    return coverage


# =============================================================================
# FIT SINGLE DMA MODEL
# =============================================================================

def fit_dma_model(
    dma_panel,
    dma,
):
    """
    Fit reduced and full traffic models for one DMA.

    Reduced model
    -------------
    Log_Traffic ~ Trend + Month_Sin + Month_Cos

    Full model
    ----------
    Log_Traffic ~ Log_Spend + Trend + Month_Sin + Month_Cos

    Returns
    -------
    dict
        DMA-level regression results.
    """

    data = (
        dma_panel
        .sort_values(
            "Month"
        )
        .copy()
    )

    n_obs = len(
        data
    )

    if n_obs < MIN_OBSERVATIONS:

        return {
            "DMA": dma,
            "Model_Status": "Insufficient Observations",
            "Observations": n_obs,
        }

    if (
        data["Log_Spend"].nunique()
        <= 1
    ):

        return {
            "DMA": dma,
            "Model_Status": "No Spend Variation",
            "Observations": n_obs,
        }

    # -------------------------------------------------------------------------
    # Model variables
    # -------------------------------------------------------------------------

    reduced_features = [
        "Trend",
        "Month_Sin",
        "Month_Cos",
    ]

    full_features = [
        "Log_Spend",
        "Trend",
        "Month_Sin",
        "Month_Cos",
    ]

    y = (
        data[
            "Log_Traffic"
        ]
        .astype(float)
    )

    X_reduced = sm.add_constant(
        data[
            reduced_features
        ].astype(float),
        has_constant="add",
    )

    X_full = sm.add_constant(
        data[
            full_features
        ].astype(float),
        has_constant="add",
    )

    # -------------------------------------------------------------------------
    # Fit OLS models
    #
    # Standard OLS residuals are retained for SSE / R² calculations.
    # HC3 covariance is used for coefficient inference.
    # -------------------------------------------------------------------------

    reduced_model = sm.OLS(
        y,
        X_reduced,
    ).fit(
        cov_type="HC3"
    )

    full_model = sm.OLS(
        y,
        X_full,
    ).fit(
        cov_type="HC3"
    )

    # -------------------------------------------------------------------------
    # Model explanatory power
    # -------------------------------------------------------------------------

    reduced_r2 = (
        reduced_model.rsquared
    )

    full_r2 = (
        full_model.rsquared
    )

    delta_r2 = (
        full_r2
        -
        reduced_r2
    )

    reduced_sse = float(
        np.sum(
            np.square(
                reduced_model.resid
            )
        )
    )

    full_sse = float(
        np.sum(
            np.square(
                full_model.resid
            )
        )
    )

    if reduced_sse > 0:

        partial_r2 = (
            (
                reduced_sse
                -
                full_sse
            )
            /
            reduced_sse
        )

    else:

        partial_r2 = np.nan

    # Numerical noise can occasionally create tiny values
    # immediately below zero.
    if (
        pd.notna(partial_r2)
        and
        partial_r2 < 0
        and
        np.isclose(
            partial_r2,
            0,
        )
    ):

        partial_r2 = 0.0

    # -------------------------------------------------------------------------
    # Spend coefficient
    # -------------------------------------------------------------------------

    spend_beta = (
        full_model.params[
            "Log_Spend"
        ]
    )

    spend_pvalue = (
        full_model.pvalues[
            "Log_Spend"
        ]
    )

    spend_ci = (
        full_model
        .conf_int()
        .loc[
            "Log_Spend"
        ]
    )

    spend_ci_lower = (
        spend_ci.iloc[0]
    )

    spend_ci_upper = (
        spend_ci.iloc[1]
    )

    # -------------------------------------------------------------------------
    # Residualized spend / traffic relationship
    #
    # Traffic residuals:
    #   Log_Traffic ~ Trend + seasonality
    #
    # Spend residuals:
    #   Log_Spend ~ Trend + seasonality
    # -------------------------------------------------------------------------

    spend_control_model = sm.OLS(
        data[
            "Log_Spend"
        ].astype(float),
        X_reduced,
    ).fit()

    traffic_residuals = pd.Series(
        reduced_model.resid,
        index=data.index,
    )

    spend_residuals = pd.Series(
        spend_control_model.resid,
        index=data.index,
    )

    if (
        traffic_residuals.std() > 0
        and
        spend_residuals.std() > 0
    ):

        residual_correlation = (
            traffic_residuals.corr(
                spend_residuals
            )
        )

    else:

        residual_correlation = np.nan

    # -------------------------------------------------------------------------
    # Partial-R² sanity check
    #
    # With a single added predictor and identical controls,
    # squared partial correlation should match partial R².
    # -------------------------------------------------------------------------

    if pd.notna(
        residual_correlation
    ):

        residual_corr_squared = (
            residual_correlation ** 2
        )

    else:

        residual_corr_squared = np.nan

    partial_r2_difference = (
        partial_r2
        -
        residual_corr_squared
        if (
            pd.notna(partial_r2)
            and
            pd.notna(
                residual_corr_squared
            )
        )
        else np.nan
    )

    # ---------------------------------------------------------
    # Influence diagnostics
    # ---------------------------------------------------------

    influence = full_model.get_influence()

    cooks_d = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag

    # Observation with maximum Cook's distance
    max_cooks_idx = int(np.nanargmax(cooks_d))

    max_cooks_d = float(
        cooks_d[max_cooks_idx]
    )

    max_cooks_month = (
        data
        .iloc[max_cooks_idx]["Month"]
    )

    max_cooks_traffic = (
        data
        .iloc[max_cooks_idx]["Traffic"]
    )

    max_cooks_spend = (
        data
        .iloc[max_cooks_idx]["Spend"]
    )

    max_cooks_log_spend = (
        data
        .iloc[max_cooks_idx]["Log_Spend"]
    )

    max_leverage = float(
        np.nanmax(leverage)
    )

    # Common Cook's D screening threshold
    cooks_threshold = (
        4
        /
        len(data)
    )

    high_influence_mask = (
        cooks_d
        > cooks_threshold
    )

    high_influence_observations = int(
        np.sum(high_influence_mask)
    )

    high_influence_months = (
        data
        .iloc[
            np.where(
                high_influence_mask
            )[0]
        ]["Month"]
        .dt.strftime("%Y-%m")
        .tolist()
    )

    high_influence_months_text = (
        ", ".join(
            high_influence_months
        )
        if high_influence_months
        else ""
    )

    # -------------------------------------------------------------------------
    # Result
    # -------------------------------------------------------------------------

    return {
        "DMA":
            dma,

        "Model_Status":
            "Success",

        "Observations":
            n_obs,

        "First_Month":
            data["Month"].min(),

        "Last_Month":
            data["Month"].max(),

        "Positive_Spend_Months":
            int(
                (
                    data["Spend"]
                    > 0
                ).sum()
            ),

        "Zero_Spend_Months":
            int(
                (
                    data["Spend"]
                    == 0
                ).sum()
            ),

        "Mean_Traffic":
            data[
                "Traffic"
            ].mean(),

        "Mean_Spend":
            data[
                "Spend"
            ].mean(),

        "Reduced_R2":
            reduced_r2,

        "Full_R2":
            full_r2,

        "Delta_R2":
            delta_r2,

        "Spend_Partial_R2":
            partial_r2,

        "Residual_Spend_Traffic_Correlation":
            residual_correlation,

        "Residual_Correlation_Squared":
            residual_corr_squared,

        "Partial_R2_Check_Difference":
            partial_r2_difference,

        "Spend_Beta":
            spend_beta,

        "Spend_P_Value":
            spend_pvalue,

        "Spend_CI_Lower_95":
            spend_ci_lower,

        "Spend_CI_Upper_95":
            spend_ci_upper,

        "Reduced_SSE":
            reduced_sse,

        "Full_SSE":
            full_sse,

        "Reduced_Adjusted_R2":
            reduced_model.rsquared_adj,

        "Full_Adjusted_R2":
            full_model.rsquared_adj,

        "Max_Cooks_D":
            max_cooks_d,

        "Max_Leverage":
            max_leverage,   

        "Log_Spend_SD":
            data["Log_Spend"].std(),

        "Unique_Log_Spend_Values":
            data["Log_Spend"].nunique(),

        "Spend_Min":
            data["Spend"].min(),

        "Spend_Max":
            data["Spend"].max(),   

        "Max_Cooks_D":
            max_cooks_d,

        "Cooks_D_Threshold":
            cooks_threshold,

        "Max_Cooks_Month":
            max_cooks_month,

        "Max_Cooks_Traffic":
            max_cooks_traffic,

        "Max_Cooks_Spend":
            max_cooks_spend,

        "Max_Cooks_Log_Spend":
            max_cooks_log_spend,

        "High_Influence_Observations":
            high_influence_observations,

        "High_Influence_Months":
            high_influence_months_text,

        "Max_Leverage":
            max_leverage,      
    }


# =============================================================================
# RUN DMA MODELS
# =============================================================================

def run_dma_models(
    panel,
):
    """
    Fit reduced/full regressions independently for each DMA.
    """

    logger.info("=" * 70)
    logger.info("RUNNING DMA-LEVEL REGRESSIONS")
    logger.info("=" * 70)

    results = []

    dmas = sorted(
        panel[
            "DMA"
        ]
        .dropna()
        .unique()
    )

    logger.info(
        "DMAs to evaluate: %s",
        f"{len(dmas):,}",
    )

    for index, dma in enumerate(
        dmas,
        start=1,
    ):

        dma_panel = (
            panel.loc[
                panel[
                    "DMA"
                ]
                ==
                dma
            ]
            .copy()
        )

        logger.info(
            "[%d/%d] Fitting DMA: %s | observations=%d",
            index,
            len(dmas),
            dma,
            len(dma_panel),
        )

        try:

            result = fit_dma_model(
                dma_panel=dma_panel,
                dma=dma,
            )

        except Exception as exc:

            logger.exception(
                "DMA model failed for %s",
                dma,
            )

            result = {
                "DMA":
                    dma,

                "Model_Status":
                    "Failed",

                "Observations":
                    len(dma_panel),

                "Error":
                    str(exc),
            }

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    successful = (
        results_df[
            "Model_Status"
        ]
        ==
        "Success"
    ).sum()

    logger.info(
        "Successful DMA models: %s",
        f"{successful:,}",
    )

    logger.info(
        "Non-successful DMA models: %s",
        f"{len(results_df) - successful:,}",
    )

    return results_df


# =============================================================================
# Classify Relationship
# =============================================================================

def classify_dma_spend_relationship(
    row,
):
    """
    Classify DMA-level spend explanatory power.

    Classification considers:

    1. Spend partial R²
       - incremental explanatory contribution of spend

    2. Spend coefficient and p-value
       - direction and statistical evidence

    3. Influence diagnostics
       - sensitivity to individual observations

    The classification is descriptive and should not be
    interpreted as a causal attribution of traffic to spend.
    """

    partial_r2 = row[
        "Spend_Partial_R2"
    ]

    beta = row[
        "Spend_Beta"
    ]

    p_value = row[
        "Spend_P_Value"
    ]

    max_cooks_d = row[
        "Max_Cooks_D"
    ]

    cooks_threshold = row[
        "Cooks_D_Threshold"
    ]

    high_influence_count = row[
        "High_Influence_Observations"
    ]

    # -----------------------------------------------------
    # Explanatory strength
    # -----------------------------------------------------

    if pd.isna(partial_r2):

        explanatory_strength = (
            "Unavailable"
        )

    elif partial_r2 < 0.05:

        explanatory_strength = (
            "Minimal"
        )

    elif partial_r2 < 0.15:

        explanatory_strength = (
            "Low"
        )

    elif partial_r2 < 0.30:

        explanatory_strength = (
            "Moderate"
        )

    else:

        explanatory_strength = (
            "Strong"
        )

    # -----------------------------------------------------
    # Spend direction
    # -----------------------------------------------------

    if pd.isna(beta):

        spend_direction = (
            "Unavailable"
        )

    elif beta > 0:

        spend_direction = (
            "Positive"
        )

    elif beta < 0:

        spend_direction = (
            "Negative"
        )

    else:

        spend_direction = (
            "Neutral"
        )

    # -----------------------------------------------------
    # Statistical evidence
    # -----------------------------------------------------

    if pd.isna(p_value):

        statistical_evidence = (
            "Unavailable"
        )

    elif p_value < 0.05:

        statistical_evidence = (
            "Strong"
        )

    elif p_value < 0.10:

        statistical_evidence = (
            "Suggestive"
        )

    else:

        statistical_evidence = (
            "Weak"
        )

    # ---------------------------------------------------------
    # Influence severity
    # ---------------------------------------------------------

    if pd.isna(max_cooks_d):

        influence_severity = "Unavailable"

    elif max_cooks_d > 1:

        influence_severity = "High"

    elif max_cooks_d > cooks_threshold:

        influence_severity = "Moderate"

    else:

        influence_severity = "Low"


    # ---------------------------------------------------------
    # Spend estimate stability
    # ---------------------------------------------------------

    if influence_severity == "Unavailable":

        stability = "Unavailable"

    elif influence_severity == "High":

        stability = "Highly Unstable"

    elif influence_severity == "Moderate":

        stability = "Influence Sensitive"

    else:

        stability = "Stable"

    # -----------------------------------------------------
    # Overall interpretation
    # -----------------------------------------------------

    if explanatory_strength == "Minimal":

        classification = (
            "Little Evidence of Spend Explanatory Power"
        )

    elif influence_severity == "High":

        classification = (
            f"{explanatory_strength} Apparent Association "
            "/ Highly Unstable Estimate"
        )

    elif (
        statistical_evidence == "Strong"
        and influence_severity in {"Low", "Moderate"}
    ):

        classification = (
            f"{explanatory_strength} "
            f"{spend_direction} Spend Relationship"
        )

    elif statistical_evidence == "Suggestive":

        classification = (
            f"{explanatory_strength} "
            f"{spend_direction} Spend Relationship "
            "/ Suggestive Evidence"
        )

    else:

        classification = (
            f"{explanatory_strength} Apparent Association "
            "/ Weak Statistical Evidence"
        )

    return pd.Series(
        {
            "Spend_Explanatory_Strength":
                explanatory_strength,

            "Spend_Direction":
                spend_direction,

            "Spend_Statistical_Evidence":
                statistical_evidence,

            "Spend_Estimate_Stability":
                stability,

            "Influence_Severity":
                influence_severity,

            "Spend_Relationship_Classification":
                classification,
        }
    )

# =============================================================================
# SAVE FIRST-PASS MODEL RESULTS
# =============================================================================

def save_model_results(
    results,
):
    """
    Save DMA regression results for inspection before
    adding classification / reporting layers.
    """

    output_file = (
        TABLE_DIR
        / "dma_spend_explanatory_results.csv"
    )

    results.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "Saved DMA regression results: %s",
        output_file,
    )

    if (
        not results.empty
        and
        "Model_Status" in results.columns
    ):

        successful = results.loc[
            results[
                "Model_Status"
            ]
            ==
            "Success"
        ].copy()

        if not successful.empty:

            successful = (
                successful
                .sort_values(
                    "Spend_Partial_R2",
                    ascending=False,
                )
                .reset_index(
                    drop=True
                )
            )

            successful.to_csv(
                TABLE_DIR
                / "dma_spend_explanatory_successful_models.csv",
                index=False,
            )

            logger.info(
                "Top DMA spend partial R² results:\n%s",
                successful[
                    [
                        "DMA",
                        "Observations",
                        "Reduced_R2",
                        "Full_R2",
                        "Delta_R2",
                        "Spend_Partial_R2",
                        "Residual_Spend_Traffic_Correlation",
                        "Spend_Beta",
                        "Spend_P_Value",
                    ]
                ]
                .head(15)
                .to_string(
                    index=False
                ),
            )


# =============================================================================
# LEAVE-ONE-OUT INFLUENCE SENSITIVITY
# =============================================================================

def run_leave_one_out_sensitivity(
    regression_panel,
    primary_results,
):
    """
    Refit each successful DMA model after removing the observation
    with the largest Cook's distance from the primary model.

    Purpose
    -------
    Determine whether the estimated spend relationship is robust
    to the single most influential monthly observation.

    The same model specification is retained:

        Reduced:
            Log_Traffic ~ Trend + Month_Sin + Month_Cos

        Full:
            Log_Traffic ~ Log_Spend + Trend + Month_Sin + Month_Cos

    This is a sensitivity diagnostic, not a replacement for the
    primary DMA-level regression.

    Returns
    -------
    pandas.DataFrame
        One row per DMA containing the original model result,
        leave-one-out result, and change metrics.
    """

    logger.info("=" * 70)
    logger.info("RUNNING LEAVE-ONE-OUT DMA SENSITIVITY ANALYSIS")
    logger.info("=" * 70)

    results = []

    successful = primary_results.loc[
        primary_results["Model_Status"] == "Success"
    ].copy()

    logger.info(
        "Successful DMA models available for sensitivity testing: %s",
        f"{len(successful):,}",
    )

    for index, row in enumerate(
        successful.itertuples(index=False),
        start=1,
    ):

        dma = row.DMA

        influential_month = pd.to_datetime(
            row.Max_Cooks_Month,
            errors="coerce",
        )

        logger.info(
            "[%d/%d] Leave-one-out sensitivity: %s | "
            "removing %s | Cook's D=%.4f",
            index,
            len(successful),
            dma,
            (
                influential_month.strftime("%Y-%m")
                if pd.notna(influential_month)
                else "Unavailable"
            ),
            (
                row.Max_Cooks_D
                if pd.notna(row.Max_Cooks_D)
                else np.nan
            ),
        )

        # ---------------------------------------------------------------------
        # Select DMA observations
        # ---------------------------------------------------------------------

        dma_panel = (
            regression_panel.loc[
                regression_panel["DMA"] == dma
            ]
            .copy()
            .sort_values("Month")
            .reset_index(drop=True)
        )

        if dma_panel.empty:

            logger.warning(
                "No regression-panel observations found for %s.",
                dma,
            )

            continue

        if pd.isna(influential_month):

            logger.warning(
                "No valid influential month found for %s.",
                dma,
            )

            continue

        # ---------------------------------------------------------------------
        # Remove ONLY the maximum-Cook's-D observation
        # ---------------------------------------------------------------------

        sensitivity_panel = (
            dma_panel.loc[
                dma_panel["Month"] != influential_month
            ]
            .copy()
            .reset_index(drop=True)
        )

        observations_before = len(
            dma_panel
        )

        observations_after = len(
            sensitivity_panel
        )

        if (
            observations_after
            !=
            observations_before - 1
        ):

            logger.warning(
                "%s | Expected to remove exactly one observation "
                "but observations changed from %d to %d.",
                dma,
                observations_before,
                observations_after,
            )

        # ---------------------------------------------------------------------
        # We intentionally bypass MIN_OBSERVATIONS here.
        #
        # The primary model required 18 observations.
        # Sensitivity testing necessarily leaves 17.
        # ---------------------------------------------------------------------

        try:

            loo_result = fit_dma_model_sensitivity(
                dma_panel=sensitivity_panel,
                dma=dma,
            )

        except Exception as exc:

            logger.exception(
                "Leave-one-out model failed for %s",
                dma,
            )

            results.append(
                {
                    "DMA":
                        dma,

                    "LOO_Status":
                        "Failed",

                    "Removed_Month":
                        influential_month,

                    "Removed_Cooks_D":
                        row.Max_Cooks_D,

                    "Error":
                        str(exc),
                }
            )

            continue

        # ---------------------------------------------------------------------
        # Original results
        # ---------------------------------------------------------------------

        original_beta = (
            row.Spend_Beta
        )

        original_partial_r2 = (
            row.Spend_Partial_R2
        )

        original_p_value = (
            row.Spend_P_Value
        )

        original_residual_corr = (
            row.Residual_Spend_Traffic_Correlation
        )

        # ---------------------------------------------------------------------
        # Leave-one-out results
        # ---------------------------------------------------------------------

        loo_beta = (
            loo_result[
                "Spend_Beta"
            ]
        )

        loo_partial_r2 = (
            loo_result[
                "Spend_Partial_R2"
            ]
        )

        loo_p_value = (
            loo_result[
                "Spend_P_Value"
            ]
        )

        loo_residual_corr = (
            loo_result[
                "Residual_Spend_Traffic_Correlation"
            ]
        )

        # ---------------------------------------------------------------------
        # Absolute changes
        # ---------------------------------------------------------------------

        beta_change = (
            loo_beta
            -
            original_beta
        )

        partial_r2_change = (
            loo_partial_r2
            -
            original_partial_r2
        )

        p_value_change = (
            loo_p_value
            -
            original_p_value
        )

        residual_corr_change = (
            loo_residual_corr
            -
            original_residual_corr
        )

        # ---------------------------------------------------------------------
        # Relative coefficient change
        #
        # Use absolute original beta in denominator so the magnitude
        # of sensitivity is intuitive regardless of beta direction.
        # ---------------------------------------------------------------------

        if (
            pd.notna(original_beta)
            and
            not np.isclose(
                original_beta,
                0.0,
            )
        ):

            beta_pct_change = (
                beta_change
                /
                abs(original_beta)
            )

        else:

            beta_pct_change = np.nan

        # ---------------------------------------------------------------------
        # Direction stability
        # ---------------------------------------------------------------------

        if (
            pd.notna(original_beta)
            and
            pd.notna(loo_beta)
        ):

            beta_direction_changed = (
                np.sign(original_beta)
                !=
                np.sign(loo_beta)
            )

        else:

            beta_direction_changed = np.nan

        # ---------------------------------------------------------------------
        # Statistical-evidence stability
        # ---------------------------------------------------------------------

        original_significant_05 = (
            original_p_value < 0.05
            if pd.notna(original_p_value)
            else np.nan
        )

        loo_significant_05 = (
            loo_p_value < 0.05
            if pd.notna(loo_p_value)
            else np.nan
        )

        if (
            pd.notna(original_significant_05)
            and
            pd.notna(loo_significant_05)
        ):

            significance_changed = (
                original_significant_05
                !=
                loo_significant_05
            )

        else:

            significance_changed = np.nan

        # ---------------------------------------------------------------------
        # Store result
        # ---------------------------------------------------------------------

        results.append(
            {
                "DMA":
                    dma,

                "LOO_Status":
                    "Success",

                # -------------------------------------------------------------
                # Removed observation
                # -------------------------------------------------------------

                "Removed_Month":
                    influential_month,

                "Removed_Cooks_D":
                    row.Max_Cooks_D,

                "Removed_Traffic":
                    row.Max_Cooks_Traffic,

                "Removed_Spend":
                    row.Max_Cooks_Spend,

                "Original_Influence_Severity":
                    row.Spend_Estimate_Stability,

                # -------------------------------------------------------------
                # Sample sizes
                # -------------------------------------------------------------

                "Original_Observations":
                    observations_before,

                "LOO_Observations":
                    observations_after,

                # -------------------------------------------------------------
                # Original model
                # -------------------------------------------------------------

                "Original_Spend_Beta":
                    original_beta,

                "Original_Spend_Partial_R2":
                    original_partial_r2,

                "Original_Spend_P_Value":
                    original_p_value,

                "Original_Residual_Correlation":
                    original_residual_corr,

                # -------------------------------------------------------------
                # Leave-one-out model
                # -------------------------------------------------------------

                "LOO_Spend_Beta":
                    loo_beta,

                "LOO_Spend_Partial_R2":
                    loo_partial_r2,

                "LOO_Spend_P_Value":
                    loo_p_value,

                "LOO_Residual_Correlation":
                    loo_residual_corr,

                "LOO_Reduced_R2":
                    loo_result[
                        "Reduced_R2"
                    ],

                "LOO_Full_R2":
                    loo_result[
                        "Full_R2"
                    ],

                "LOO_Delta_R2":
                    loo_result[
                        "Delta_R2"
                    ],

                "LOO_Spend_CI_Lower_95":
                    loo_result[
                        "Spend_CI_Lower_95"
                    ],

                "LOO_Spend_CI_Upper_95":
                    loo_result[
                        "Spend_CI_Upper_95"
                    ],

                # -------------------------------------------------------------
                # Sensitivity
                # -------------------------------------------------------------

                "Spend_Beta_Change":
                    beta_change,

                "Spend_Beta_Pct_Change":
                    beta_pct_change,

                "Spend_Partial_R2_Change":
                    partial_r2_change,

                "Spend_P_Value_Change":
                    p_value_change,

                "Residual_Correlation_Change":
                    residual_corr_change,

                "Spend_Beta_Direction_Changed":
                    beta_direction_changed,

                "Significance_05_Changed":
                    significance_changed,
            }
        )

    sensitivity = pd.DataFrame(
        results
    )

    logger.info(
        "Leave-one-out sensitivity tests completed: %s",
        f"{len(sensitivity):,}",
    )

    return sensitivity


# =============================================================================
# FIT DMA MODEL - SENSITIVITY VERSION
# =============================================================================

def fit_dma_model_sensitivity(
    dma_panel,
    dma,
):
    """
    Fit the same reduced/full DMA regression specification used in
    the primary model, but allow one observation to have been removed.

    This function exists so leave-one-out analysis does not interfere
    with the minimum-observation rule used by the production model.
    """

    data = (
        dma_panel
        .sort_values("Month")
        .copy()
        .reset_index(drop=True)
    )

    n_obs = len(
        data
    )

    # We need enough observations to fit:
    #
    # constant
    # Log_Spend
    # Trend
    # Month_Sin
    # Month_Cos
    #
    # Seventeen observations is expected in the current analysis.

    if n_obs < 10:

        raise ValueError(
            f"{dma}: insufficient observations for "
            f"sensitivity regression ({n_obs})."
        )

    if (
        data["Log_Spend"].nunique()
        <= 1
    ):

        raise ValueError(
            f"{dma}: no Log_Spend variation after removing "
            "the influential observation."
        )

    # -------------------------------------------------------------------------
    # Features
    # -------------------------------------------------------------------------

    reduced_features = [
        "Trend",
        "Month_Sin",
        "Month_Cos",
    ]

    full_features = [
        "Log_Spend",
        "Trend",
        "Month_Sin",
        "Month_Cos",
    ]

    y = (
        data["Log_Traffic"]
        .astype(float)
    )

    X_reduced = sm.add_constant(
        data[
            reduced_features
        ].astype(float),
        has_constant="add",
    )

    X_full = sm.add_constant(
        data[
            full_features
        ].astype(float),
        has_constant="add",
    )

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------

    reduced_model = sm.OLS(
        y,
        X_reduced,
    ).fit(
        cov_type="HC3"
    )

    full_model = sm.OLS(
        y,
        X_full,
    ).fit(
        cov_type="HC3"
    )

    # -------------------------------------------------------------------------
    # R²
    # -------------------------------------------------------------------------

    reduced_r2 = (
        reduced_model.rsquared
    )

    full_r2 = (
        full_model.rsquared
    )

    delta_r2 = (
        full_r2
        -
        reduced_r2
    )

    # -------------------------------------------------------------------------
    # SSE / partial R²
    # -------------------------------------------------------------------------

    reduced_sse = float(
        np.sum(
            np.square(
                reduced_model.resid
            )
        )
    )

    full_sse = float(
        np.sum(
            np.square(
                full_model.resid
            )
        )
    )

    if reduced_sse > 0:

        partial_r2 = (
            (
                reduced_sse
                -
                full_sse
            )
            /
            reduced_sse
        )

    else:

        partial_r2 = np.nan

    # -------------------------------------------------------------------------
    # Spend coefficient
    # -------------------------------------------------------------------------

    spend_beta = (
        full_model.params[
            "Log_Spend"
        ]
    )

    spend_p_value = (
        full_model.pvalues[
            "Log_Spend"
        ]
    )

    spend_ci = (
        full_model
        .conf_int()
        .loc[
            "Log_Spend"
        ]
    )

    # -------------------------------------------------------------------------
    # Residualized spend / traffic correlation
    # -------------------------------------------------------------------------

    spend_control_model = sm.OLS(
        data[
            "Log_Spend"
        ].astype(float),
        X_reduced,
    ).fit()

    traffic_residuals = pd.Series(
        reduced_model.resid,
        index=data.index,
    )

    spend_residuals = pd.Series(
        spend_control_model.resid,
        index=data.index,
    )

    if (
        traffic_residuals.std() > 0
        and
        spend_residuals.std() > 0
    ):

        residual_correlation = (
            traffic_residuals.corr(
                spend_residuals
            )
        )

    else:

        residual_correlation = np.nan

    return {
        "DMA":
            dma,

        "Observations":
            n_obs,

        "Reduced_R2":
            reduced_r2,

        "Full_R2":
            full_r2,

        "Delta_R2":
            delta_r2,

        "Spend_Partial_R2":
            partial_r2,

        "Residual_Spend_Traffic_Correlation":
            residual_correlation,

        "Spend_Beta":
            spend_beta,

        "Spend_P_Value":
            spend_p_value,

        "Spend_CI_Lower_95":
            spend_ci.iloc[0],

        "Spend_CI_Upper_95":
            spend_ci.iloc[1],
    }

# =============================================================================
# CLASSIFY LEAVE-ONE-OUT SENSITIVITY
# =============================================================================

def classify_leave_one_out_sensitivity(
    sensitivity,
):
    """
    Classify the sensitivity of each DMA spend estimate to removal
    of its most influential observation.

    The classification considers:

        - coefficient direction reversal
        - magnitude of coefficient change
        - partial R² change
        - statistical-significance change

    These thresholds are diagnostic rather than inferential.
    """

    if sensitivity.empty:

        return sensitivity

    sensitivity = sensitivity.copy()

    classifications = []
    interpretations = []

    for _, row in sensitivity.iterrows():

        if (
            row.get("LOO_Status")
            !=
            "Success"
        ):

            classifications.append(
                "Unavailable"
            )

            interpretations.append(
                "Leave-one-out sensitivity analysis "
                "was not successfully estimated."
            )

            continue

        direction_changed = bool(
            row[
                "Spend_Beta_Direction_Changed"
            ]
        )

        significance_changed = bool(
            row[
                "Significance_05_Changed"
            ]
        )

        beta_pct_change = abs(
            row[
                "Spend_Beta_Pct_Change"
            ]
        ) if pd.notna(
            row[
                "Spend_Beta_Pct_Change"
            ]
        ) else np.nan

        partial_r2_change = abs(
            row[
                "Spend_Partial_R2_Change"
            ]
        ) if pd.notna(
            row[
                "Spend_Partial_R2_Change"
            ]
        ) else np.nan

        # ---------------------------------------------------------------------
        # Classification
        # ---------------------------------------------------------------------

        if direction_changed:

            classification = (
                "Highly Sensitive"
            )

            interpretation = (
                "The estimated spend relationship changes direction "
                "after removing the most influential month."
            )

        elif (
            significance_changed
            or
            (
                pd.notna(beta_pct_change)
                and
                beta_pct_change >= 0.50
            )
            or
            (
                pd.notna(partial_r2_change)
                and
                partial_r2_change >= 0.15
            )
        ):

            classification = (
                "Sensitive"
            )

            interpretation = (
                "The estimated spend relationship changes materially "
                "after removing the most influential month."
            )

        elif (
            (
                pd.notna(beta_pct_change)
                and
                beta_pct_change >= 0.20
            )
            or
            (
                pd.notna(partial_r2_change)
                and
                partial_r2_change >= 0.05
            )
        ):

            classification = (
                "Moderately Sensitive"
            )

            interpretation = (
                "The spend relationship changes somewhat after "
                "removing the most influential month, but the overall "
                "direction remains intact."
            )

        else:

            classification = (
                "Robust"
            )

            interpretation = (
                "The estimated spend relationship remains broadly "
                "stable after removing the most influential month."
            )

        classifications.append(
            classification
        )

        interpretations.append(
            interpretation
        )

    sensitivity[
        "LOO_Sensitivity_Classification"
    ] = classifications

    sensitivity[
        "LOO_Sensitivity_Interpretation"
    ] = interpretations

    return sensitivity

# =============================================================================
# SAVE LEAVE-ONE-OUT RESULTS
# =============================================================================

def save_leave_one_out_results(
    sensitivity,
):
    """
    Save DMA leave-one-out sensitivity results.
    """

    output_file = (
        TABLE_DIR
        / "dma_spend_leave_one_out_sensitivity.csv"
    )

    sensitivity.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "Saved leave-one-out sensitivity results: %s",
        output_file,
    )

    if sensitivity.empty:

        return

    successful = sensitivity.loc[
        sensitivity[
            "LOO_Status"
        ]
        ==
        "Success"
    ].copy()

    if successful.empty:

        return

    # Largest sensitivity first
    successful[
        "Absolute_Partial_R2_Change"
    ] = (
        successful[
            "Spend_Partial_R2_Change"
        ]
        .abs()
    )

    successful = (
        successful
        .sort_values(
            "Absolute_Partial_R2_Change",
            ascending=False,
        )
    )

    logger.info(
        "Most sensitive DMA leave-one-out results:\n%s",
        successful[
            [
                "DMA",
                "Removed_Month",
                "Removed_Cooks_D",
                "Original_Spend_Partial_R2",
                "LOO_Spend_Partial_R2",
                "Spend_Partial_R2_Change",
                "Original_Spend_Beta",
                "LOO_Spend_Beta",
                "Spend_Beta_Direction_Changed",
                "LOO_Sensitivity_Classification",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        ),
    )


# =============================================================================
# BUILD ANALYSIS SUMMARY
# =============================================================================

def build_analysis_summary(
    results,
    loo_sensitivity,
):
    """
    Build portfolio-level summary metrics for the DMA spend
    explanatory-power analysis.
    """

    successful = results.loc[
        results["Model_Status"] == "Success"
    ].copy()

    loo_successful = loo_sensitivity.loc[
        loo_sensitivity["LOO_Status"] == "Success"
    ].copy()

    if successful.empty:

        return {
            "dma_count": 0,
            "mean_partial_r2": np.nan,
            "median_partial_r2": np.nan,
            "mean_beta": np.nan,
            "positive_beta_count": 0,
            "negative_beta_count": 0,
            "significant_positive_count": 0,
            "significant_negative_count": 0,
            "strong_relationship_count": 0,
            "high_instability_count": 0,
            "loo_robust_count": 0,
            "loo_moderate_count": 0,
            "loo_sensitive_count": 0,
            "loo_highly_sensitive_count": 0,
        }

    positive_beta = (
        successful["Spend_Beta"] > 0
    )

    negative_beta = (
        successful["Spend_Beta"] < 0
    )

    significant = (
        successful["Spend_P_Value"] < 0.05
    )

    strong_relationship = (
        successful[
            "Spend_Explanatory_Strength"
        ]
        == "Strong"
    )

    high_instability = (
        successful[
            "Influence_Severity"
        ]
        == "High"
    )

    summary = {
        "dma_count":
            len(successful),

        "mean_partial_r2":
            successful[
                "Spend_Partial_R2"
            ].mean(),

        "median_partial_r2":
            successful[
                "Spend_Partial_R2"
            ].median(),

        "mean_beta":
            successful[
                "Spend_Beta"
            ].mean(),

        "positive_beta_count":
            positive_beta.sum(),

        "negative_beta_count":
            negative_beta.sum(),

        "significant_positive_count":
            (
                positive_beta
                &
                significant
            ).sum(),

        "significant_negative_count":
            (
                negative_beta
                &
                significant
            ).sum(),

        "strong_relationship_count":
            strong_relationship.sum(),

        "high_instability_count":
            high_instability.sum(),

        "loo_robust_count":
            (
                loo_successful[
                    "LOO_Sensitivity_Classification"
                ]
                == "Robust"
            ).sum(),

        "loo_moderate_count":
            (
                loo_successful[
                    "LOO_Sensitivity_Classification"
                ]
                == "Moderately Sensitive"
            ).sum(),

        "loo_sensitive_count":
            (
                loo_successful[
                    "LOO_Sensitivity_Classification"
                ]
                == "Sensitive"
            ).sum(),

        "loo_highly_sensitive_count":
            (
                loo_successful[
                    "LOO_Sensitivity_Classification"
                ]
                == "Highly Sensitive"
            ).sum(),
    }

    logger.info("=" * 70)
    logger.info("DMA SPEND EXPLANATORY SUMMARY")
    logger.info("=" * 70)

    logger.info(
        "Successful DMA models: %d",
        summary["dma_count"],
    )

    logger.info(
        "Mean spend partial R²: %.3f",
        summary["mean_partial_r2"],
    )

    logger.info(
        "Median spend partial R²: %.3f",
        summary["median_partial_r2"],
    )

    logger.info(
        "Positive spend coefficients: %d",
        summary["positive_beta_count"],
    )

    logger.info(
        "Negative spend coefficients: %d",
        summary["negative_beta_count"],
    )

    logger.info(
        "Significant positive relationships: %d",
        summary["significant_positive_count"],
    )

    logger.info(
        "Significant negative relationships: %d",
        summary["significant_negative_count"],
    )

    return summary

# =============================================================================
# BUILD DMA REPORTING TABLE
# =============================================================================

def build_dma_reporting_table(
    results,
    loo_sensitivity,
    treated_dmas,
):
    """
    Combine primary DMA regression results with leave-one-out
    sensitivity diagnostics for treated intervention DMAs only.

    Notes
    -----
    The regression may be estimated across the broader DMA universe,
    but the reporting table is restricted to DMAs included in the
    treatment / BSTS intervention analysis.
    """

    logger.info("=" * 70)
    logger.info("BUILDING TREATED-DMA REPORTING TABLE")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------

    if results.empty:

        logger.warning(
            "Primary DMA regression results are empty."
        )

        return pd.DataFrame()

    if treated_dmas.empty:

        logger.warning(
            "Treated DMA universe is empty."
        )

        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Normalize DMA keys
    # -------------------------------------------------------------------------

    results = results.copy()
    loo_sensitivity = loo_sensitivity.copy()
    treated_dmas = treated_dmas.copy()

    results["DMA"] = (
        results["DMA"]
        .astype("string")
        .str.strip()
    )

    if (
        not loo_sensitivity.empty
        and
        "DMA" in loo_sensitivity.columns
    ):

        loo_sensitivity["DMA"] = (
            loo_sensitivity["DMA"]
            .astype("string")
            .str.strip()
        )

    treated_dmas["DMA"] = (
        treated_dmas["DMA"]
        .astype("string")
        .str.strip()
    )

    treated_dma_set = set(
        treated_dmas[
            "DMA"
        ]
        .dropna()
        .unique()
    )

    # -------------------------------------------------------------------------
    # Filter primary results to treated DMAs
    # -------------------------------------------------------------------------

    primary = results.loc[
        results["DMA"].isin(
            treated_dma_set
        )
    ].copy()

    logger.info(
        "Primary regression DMAs available: %s",
        f"{results['DMA'].nunique():,}",
    )

    logger.info(
        "Treated DMAs requested: %s",
        f"{len(treated_dma_set):,}",
    )

    logger.info(
        "Treated DMAs matched to regression results: %s",
        f"{primary['DMA'].nunique():,}",
    )

    # -------------------------------------------------------------------------
    # Identify missing treated DMAs
    # -------------------------------------------------------------------------

    missing_treated = (
        treated_dma_set
        -
        set(
            primary[
                "DMA"
            ]
            .dropna()
            .unique()
        )
    )

    if missing_treated:

        logger.warning(
            "Treated DMAs missing from DMA regression results: %s",
            f"{len(missing_treated):,}",
        )

        for dma in sorted(
            missing_treated
        ):

            logger.warning(
                "  Missing treated DMA: %s",
                dma,
            )

    # -------------------------------------------------------------------------
    # Primary reporting fields
    # -------------------------------------------------------------------------

    primary_columns = [
        "DMA",
        "Observations",
        "Reduced_R2",
        "Full_R2",
        "Delta_R2",
        "Spend_Partial_R2",
        "Residual_Spend_Traffic_Correlation",
        "Spend_Beta",
        "Spend_P_Value",
        "Spend_CI_Lower_95",
        "Spend_CI_Upper_95",
        "Spend_Explanatory_Strength",
        "Spend_Direction",
        "Spend_Statistical_Evidence",
        "Influence_Severity",
        "Spend_Estimate_Stability",
        "Spend_Relationship_Classification",
    ]

    primary = primary[
        [
            col
            for col in primary_columns
            if col in primary.columns
        ]
    ].copy()

    # -------------------------------------------------------------------------
    # Leave-one-out fields
    # -------------------------------------------------------------------------

    loo_columns = [
        "DMA",
        "Removed_Month",
        "Removed_Cooks_D",
        "LOO_Spend_Beta",
        "LOO_Spend_Partial_R2",
        "LOO_Spend_P_Value",
        "LOO_Residual_Correlation",
        "Spend_Beta_Direction_Changed",
        "LOO_Sensitivity_Classification",
        "LOO_Inference_Stability",
    ]

    if not loo_sensitivity.empty:

        loo = loo_sensitivity[
            [
                col
                for col in loo_columns
                if col in loo_sensitivity.columns
            ]
        ].copy()

        # Restrict LOO table too.
        loo = loo.loc[
            loo["DMA"].isin(
                treated_dma_set
            )
        ].copy()

        report_table = primary.merge(
            loo,
            on="DMA",
            how="left",
            validate="one_to_one",
        )

    else:

        report_table = primary.copy()

    # -------------------------------------------------------------------------
    # Sort by spend explanatory power
    # -------------------------------------------------------------------------

    if (
        "Spend_Partial_R2"
        in report_table.columns
    ):

        report_table = (
            report_table
            .sort_values(
                "Spend_Partial_R2",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    logger.info(
        "Final treated-DMA reporting table: %s rows",
        f"{len(report_table):,}",
    )

    return report_table

# =============================================================================
# FIGURE 1 — PARTIAL R² BY DMA
# =============================================================================

def plot_partial_r2_by_dma(
    results,
):
    """
    Plot DMA-level spend partial R².
    """

    data = results.loc[
        results["Model_Status"] == "Success"
    ].copy()

    data = data.sort_values(
        "Spend_Partial_R2",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(11, 10)
    )

    ax.barh(
        data["DMA"],
        data["Spend_Partial_R2"] * 100,
    )

    ax.set_xlabel(
        "Spend Partial R² (%)"
    )

    ax.set_ylabel(
        "DMA"
    )

    ax.set_title(
        "Incremental Explanatory Power of Marketing Spend by DMA"
    )

    ax.axvline(
        0,
        linewidth=0.8,
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "spend_partial_r2_by_dma.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_file

# =============================================================================
# FIGURE 2 — SPEND COEFFICIENTS
# =============================================================================

def plot_spend_coefficients(
    results,
):
    """
    Plot DMA spend coefficients with 95% confidence intervals.
    """

    data = results.loc[
        results["Model_Status"] == "Success"
    ].copy()

    data = data.sort_values(
        "Spend_Beta",
        ascending=True,
    )

    y = np.arange(
        len(data)
    )

    lower_error = (
        data["Spend_Beta"]
        -
        data["Spend_CI_Lower_95"]
    )

    upper_error = (
        data["Spend_CI_Upper_95"]
        -
        data["Spend_Beta"]
    )

    fig, ax = plt.subplots(
        figsize=(11, 10)
    )

    ax.errorbar(
        data["Spend_Beta"],
        y,
        xerr=[
            lower_error,
            upper_error,
        ],
        fmt="o",
        capsize=3,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        data["DMA"]
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Log Spend Coefficient"
    )

    ax.set_title(
        "DMA-Level Spend / Traffic Association"
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "spend_coefficients_by_dma.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_file

# =============================================================================
# FIGURE 3 — PARTIAL R² VS COEFFICIENT
# =============================================================================

def plot_explanatory_power_vs_beta(
    results,
):
    """
    Compare spend explanatory power with spend coefficient direction.
    """

    data = results.loc[
        results["Model_Status"] == "Success"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.scatter(
        data["Spend_Partial_R2"] * 100,
        data["Spend_Beta"],
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    for _, row in data.iterrows():

        if (
            row["Spend_Partial_R2"] >= 0.30
            or
            row["Influence_Severity"] == "High"
        ):

            ax.annotate(
                row["DMA"],
                (
                    row[
                        "Spend_Partial_R2"
                    ] * 100,
                    row[
                        "Spend_Beta"
                    ],
                ),
                fontsize=8,
            )

    ax.set_xlabel(
        "Spend Partial R² (%)"
    )

    ax.set_ylabel(
        "Spend Coefficient"
    )

    ax.set_title(
        "DMA Spend Explanatory Power vs. Estimated Direction"
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "partial_r2_vs_spend_beta.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_file

# =============================================================================
# FIGURE 4 — ORIGINAL VS LEAVE-ONE-OUT PARTIAL R²
# =============================================================================

def plot_leave_one_out_partial_r2(
    sensitivity,
):
    """
    Compare original and leave-one-out spend partial R².
    """

    data = sensitivity.loc[
        sensitivity["LOO_Status"] == "Success"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.scatter(
        data[
            "Original_Spend_Partial_R2"
        ] * 100,
        data[
            "LOO_Spend_Partial_R2"
        ] * 100,
    )

    max_value = max(
        data[
            "Original_Spend_Partial_R2"
        ].max(),
        data[
            "LOO_Spend_Partial_R2"
        ].max(),
    ) * 100

    ax.plot(
        [
            0,
            max_value,
        ],
        [
            0,
            max_value,
        ],
        linestyle="--",
    )

    for _, row in data.iterrows():

        if (
            row[
                "LOO_Sensitivity_Classification"
            ]
            in {
                "Sensitive",
                "Highly Sensitive",
            }
        ):

            ax.annotate(
                row["DMA"],
                (
                    row[
                        "Original_Spend_Partial_R2"
                    ] * 100,
                    row[
                        "LOO_Spend_Partial_R2"
                    ] * 100,
                ),
                fontsize=8,
            )

    ax.set_xlabel(
        "Original Spend Partial R² (%)"
    )

    ax.set_ylabel(
        "Leave-One-Out Spend Partial R² (%)"
    )

    ax.set_title(
        "Influence Sensitivity of DMA Spend Explanatory Power"
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "leave_one_out_partial_r2_comparison.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_file



# =============================================================================
# GENERATE FIGURES
# =============================================================================

def generate_figures(
    results,
    loo_sensitivity,
    treated_dmas,
):
    """
    Generate DMA spend explanatory-power figures.

    Reporting figures are restricted to DMAs included in the
    treatment / BSTS intervention universe.

    Notes
    -----
    The underlying regression analysis may contain a broader DMA
    population. This function filters only the reporting population
    used in the figures.
    """

    logger.info("=" * 70)
    logger.info("GENERATING TREATED-DMA FIGURES")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Copy inputs
    # -------------------------------------------------------------------------

    results = results.copy()
    loo_sensitivity = loo_sensitivity.copy()
    treated_dmas = treated_dmas.copy()

    # -------------------------------------------------------------------------
    # Normalize DMA names
    # -------------------------------------------------------------------------

    results["DMA"] = (
        results["DMA"]
        .astype("string")
        .str.strip()
    )

    treated_dmas["DMA"] = (
        treated_dmas["DMA"]
        .astype("string")
        .str.strip()
    )

    if (
        not loo_sensitivity.empty
        and
        "DMA" in loo_sensitivity.columns
    ):

        loo_sensitivity["DMA"] = (
            loo_sensitivity["DMA"]
            .astype("string")
            .str.strip()
        )

    # -------------------------------------------------------------------------
    # Build treated DMA universe
    # -------------------------------------------------------------------------

    treated_dma_set = set(
        treated_dmas[
            "DMA"
        ]
        .dropna()
        .unique()
    )

    logger.info(
        "Treated DMA reporting universe: %s",
        f"{len(treated_dma_set):,}",
    )

    # -------------------------------------------------------------------------
    # Filter primary results
    # -------------------------------------------------------------------------

    figure_results = results.loc[
        results["DMA"].isin(
            treated_dma_set
        )
    ].copy()

    # Only successful models should feed the plots.
    if (
        "Model_Status"
        in figure_results.columns
    ):

        figure_results = figure_results.loc[
            figure_results[
                "Model_Status"
            ]
            ==
            "Success"
        ].copy()

    # -------------------------------------------------------------------------
    # Filter leave-one-out results
    # -------------------------------------------------------------------------

    if (
        not loo_sensitivity.empty
        and
        "DMA" in loo_sensitivity.columns
    ):

        figure_loo = loo_sensitivity.loc[
            loo_sensitivity["DMA"].isin(
                treated_dma_set
            )
        ].copy()

    else:

        figure_loo = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    logger.info(
        "All regression DMAs available: %s",
        f"{results['DMA'].nunique():,}",
    )

    logger.info(
        "Treated DMAs matched to primary results: %s",
        f"{figure_results['DMA'].nunique():,}",
    )

    if not figure_loo.empty:

        logger.info(
            "Treated DMAs matched to LOO results: %s",
            f"{figure_loo['DMA'].nunique():,}",
        )

    missing_primary_dmas = (
        treated_dma_set
        -
        set(
            figure_results[
                "DMA"
            ]
            .dropna()
            .unique()
        )
    )

    if missing_primary_dmas:

        logger.warning(
            "Treated DMAs missing from figure results: %s",
            f"{len(missing_primary_dmas):,}",
        )

        for dma in sorted(
            missing_primary_dmas
        ):

            logger.warning(
                "  Missing treated DMA: %s",
                dma,
            )

    # -------------------------------------------------------------------------
    # Validate figure inputs
    # -------------------------------------------------------------------------

    if figure_results.empty:

        raise ValueError(
            "No treated DMA regression results are available "
            "for figure generation."
        )

    # -------------------------------------------------------------------------
    # Generate figures using the EXISTING plotting functions
    # -------------------------------------------------------------------------

    figures = {
        "partial_r2":
            plot_partial_r2_by_dma(
                figure_results
            ),

        "coefficients":
            plot_spend_coefficients(
                figure_results
            ),

        "r2_vs_beta":
            plot_explanatory_power_vs_beta(
                figure_results
            ),
    }

    # Only create LOO figure when matching data exists.
    if not figure_loo.empty:

        figures[
            "loo_partial_r2"
        ] = plot_leave_one_out_partial_r2(
            figure_loo
        )

    logger.info(
        "Generated %s treated-DMA figures.",
        f"{len(figures):,}",
    )

    for name, path in figures.items():

        logger.info(
            "  %s: %s",
            name,
            path,
        )

    return figures


# =============================================================================
# HTML REPORT
# =============================================================================

def generate_html_report(
    summary,
    reporting_table,
    figures,
):
    """
    Generate HTML report for DMA spend explanatory-power analysis.
    """

    logger.info("=" * 70)
    logger.info("GENERATING HTML REPORT")
    logger.info("=" * 70)

    report_table = reporting_table.copy()

    display_columns = [
        "DMA",
        "Spend_Partial_R2",
        "Spend_Beta",
        "Spend_P_Value",
        "Spend_Explanatory_Strength",
        "Influence_Severity",
        "LOO_Sensitivity_Classification",
    ]

    report_table = report_table[
        [
            col
            for col in display_columns
            if col in report_table.columns
        ]
    ].copy()

    if (
        "Spend_Partial_R2"
        in report_table.columns
    ):

        report_table[
            "Spend_Partial_R2"
        ] = (
            report_table[
                "Spend_Partial_R2"
            ]
            * 100
        )

    

    # ---------------------------------------------------------------------
    # Report path
    # ---------------------------------------------------------------------

    report_path = (
        REPORT_DIR
        / "dma_spend_explanatory_power_report.html"
    )

    # ---------------------------------------------------------------------
    # Figure paths relative to HTML report
    # ---------------------------------------------------------------------

    figure_paths = {}

    for key, path in figures.items():

        figure_path = Path(
            path
        ).resolve()

        relative_path = os.path.relpath(
            figure_path,
            start=report_path.parent.resolve(),
        )

        figure_paths[key] = (
            Path(relative_path)
            .as_posix()
        )

    logger.info(
        "HTML figure path | %s -> %s",
        key,
        figure_paths[key],
    )

    positive_pct = (
        summary[
            "positive_beta_count"
        ]
        /
        summary[
            "dma_count"
        ]
        * 100
        if summary[
            "dma_count"
        ] > 0
        else np.nan
    )

    # ---------------------------------------------------------------------
    # Dynamic interpretation
    # ---------------------------------------------------------------------

    if (
        summary[
            "significant_negative_count"
        ]
        == 0
    ):

        directional_interpretation = (
            "No DMA produced a statistically significant "
            "negative spend coefficient under the current "
            "specification. This does not establish causality, "
            "but it provides little evidence of a systematic "
            "negative spend–traffic relationship."
        )

    else:

        directional_interpretation = (
            f"{summary['significant_negative_count']} DMA(s) "
            "produced statistically significant negative "
            "spend coefficients and warrant additional review."
        )

    if (
        summary[
            "median_partial_r2"
        ]
        < 0.10
    ):

        explanatory_interpretation = (
            "Marketing spend generally provides limited "
            "incremental explanatory power after controlling "
            "for trend and seasonality."
        )

    elif (
        summary[
            "median_partial_r2"
        ]
        < 0.30
    ):

        explanatory_interpretation = (
            "Marketing spend provides moderate incremental "
            "explanatory power across the typical DMA, although "
            "the strength of the relationship varies materially "
            "by market."
        )

    else:

        explanatory_interpretation = (
            "Marketing spend provides substantial incremental "
            "explanatory power across many DMAs after controlling "
            "for trend and seasonality."
        )

    html_body = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="utf-8">

<title>
DMA Spend Explanatory Power
</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 0;
    background: #f5f7fa;
    color: #1f2933;
}}

.container {{
    max-width: 1300px;
    margin: auto;
    padding: 32px;
}}

header {{
    background: #102a43;
    color: white;
    padding: 28px;
    border-radius: 12px;
    margin-bottom: 20px;
}}

section {{
    background: white;
    padding: 24px;
    margin-bottom: 20px;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,.08);
}}

.cards {{
    display: grid;
    grid-template-columns:
        repeat(
            auto-fit,
            minmax(180px, 1fr)
        );
    gap: 14px;
}}

.card {{
    background: #eef2f7;
    padding: 16px;
    border-radius: 8px;
}}

.label {{
    font-size: 12px;
    color: #627d98;
}}

.value {{
    font-size: 24px;
    font-weight: bold;
    margin-top: 4px;
}}

.takeaway {{
    background: #eef6ff;
    border-left: 4px solid #3578c6;
    padding: 16px;
    margin-top: 18px;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
}}

th, td {{
    border: 1px solid #ddd;
    padding: 7px;
    text-align: right;
}}

th:first-child,
td:first-child {{
    text-align: left;
}}

th {{
    background: #eef2f7;
}}

img {{
    max-width: 100%;
    display: block;
    margin: 20px auto;
}}

.small {{
    font-size: 12px;
    color: #627d98;
}}

</style>

</head>

<body>

<div class="container">

<header>

<h1>
DMA Spend Explanatory Power
</h1>

<p>
Within-market regression analysis of marketing spend and traffic
after controlling for trend and seasonality.
</p>

</header>


<section>

<h2>
Portfolio Summary
</h2>

<div class="cards">

<div class="card">
<div class="label">
DMAs Modeled
</div>
<div class="value">
{summary["dma_count"]}
</div>
</div>

<div class="card">
<div class="label">
Mean Spend Partial R²
</div>
<div class="value">
{summary["mean_partial_r2"] * 100:.1f}%
</div>
</div>

<div class="card">
<div class="label">
Median Spend Partial R²
</div>
<div class="value">
{summary["median_partial_r2"] * 100:.1f}%
</div>
</div>

<div class="card">
<div class="label">
Positive Spend Coefficients
</div>
<div class="value">
{summary["positive_beta_count"]}
</div>
</div>

<div class="card">
<div class="label">
Significant Positive Relationships
</div>
<div class="value">
{summary["significant_positive_count"]}
</div>
</div>

<div class="card">
<div class="label">
Significant Negative Relationships
</div>
<div class="value">
{summary["significant_negative_count"]}
</div>
</div>

</div>

<div class="takeaway">

<strong>
Portfolio interpretation:
</strong>

<p>
{explanatory_interpretation}
</p>

<p>
{directional_interpretation}
</p>

<p>
Approximately {positive_pct:.1f}% of modeled DMAs produced
positive spend coefficients.
</p>

</div>

</section>


<section>

<h2>
Spend Explanatory Power by DMA
</h2>

<p>
Spend partial R² measures the share of traffic variation remaining
after trend and seasonality controls that becomes explained when
marketing spend is added to the DMA-specific regression.
</p>

<img src="{figure_paths['partial_r2']}">

</section>


<section>

<h2>
Spend / Traffic Relationship
</h2>

<p>
The spend coefficient describes the direction of the modeled
relationship between log marketing spend and log traffic after
controlling for trend and seasonality.
</p>

<img src="{figure_paths['coefficients']}">

<img src="{figure_paths['r2_vs_beta']}">

</section>


<section>

<h2>
Influence & Leave-One-Out Sensitivity
</h2>

<p>
Each DMA model was re-estimated after removing the single
observation with the largest Cook's distance.
This tests whether the estimated spend relationship depends
heavily on one unusually influential month.
</p>

<img src="{figure_paths['loo_partial_r2']}">

<div class="cards">

<div class="card">
<div class="label">
Robust
</div>
<div class="value">
{summary["loo_robust_count"]}
</div>
</div>

<div class="card">
<div class="label">
Moderately Sensitive
</div>
<div class="value">
{summary["loo_moderate_count"]}
</div>
</div>

<div class="card">
<div class="label">
Sensitive
</div>
<div class="value">
{summary["loo_sensitive_count"]}
</div>
</div>

<div class="card">
<div class="label">
Highly Sensitive
</div>
<div class="value">
{summary["loo_highly_sensitive_count"]}
</div>
</div>

</div>

</section>


<section>

<h2>
DMA-Level Results
</h2>

{report_table.to_html(
    index=False,
    float_format=lambda x: f"{x:.3f}",
)}

</section>


<section>

<h2>
Methodology
</h2>

<p>
For each DMA, two monthly traffic regressions were estimated.
</p>

<p>
<strong>
Reduced model:
</strong>
Log Traffic ~ Trend + Month Sin + Month Cos
</p>

<p>
<strong>
Full model:
</strong>
Log Traffic ~ Log Spend + Trend + Month Sin + Month Cos
</p>

<p>
Spend partial R² measures the incremental explanatory contribution
of marketing spend after accounting for trend and annual seasonality.
HC3 robust covariance estimates are used for inference.
</p>

<p>
Cook's distance and leave-one-out sensitivity analysis are used to
identify DMA results that depend materially on individual observations.
</p>

<p class="small">
This analysis measures association and explanatory power.
It does not independently establish that marketing spend caused
changes in traffic.
</p>

</section>


</div>

</body>

</html>
"""

    report_path = (
        REPORT_DIR
        / "dma_spend_explanatory_power_report.html"
    )

    report_path.write_text(
        html_body,
        encoding="utf-8",
    )

    logger.info(
        "HTML report written to %s",
        report_path,
    )

    return report_path

def save_reporting_outputs(
    summary,
    reporting_table,
):
    """
    Save consolidated reporting outputs.
    """

    reporting_table.to_csv(
        TABLE_DIR
        / "dma_spend_explanatory_reporting_table.csv",
        index=False,
    )

    summary_df = pd.DataFrame(
        [
            summary
        ]
    )

    summary_df.to_csv(
        TABLE_DIR
        / "dma_spend_explanatory_summary.csv",
        index=False,
    )

    logger.info(
        "Reporting summary datasets saved."
    )

# =============================================================================
# MAIN
# =============================================================================

def main():

    logger.info("")
    logger.info("=" * 80)
    logger.info(
        "DMA SPEND EXPLANATORY POWER ANALYSIS"
    )
    logger.info("=" * 80)

    try:

        # =====================================================================
        # 0. INITIALIZATION
        # =====================================================================

        create_output_directories()

        # =====================================================================
        # 1. LOAD MODEL PANEL
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 1: LOADING MODEL PANEL"
        )
        logger.info("-" * 80)

        panel = load_model_panel()

        logger.info(
            "Model panel loaded successfully: %s rows",
            f"{len(panel):,}",
        )

        # =====================================================================
        # 2. VALIDATE MODEL PANEL
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 2: VALIDATING MODEL PANEL"
        )
        logger.info("-" * 80)

        validate_model_panel(
            panel
        )

        logger.info(
            "Model panel validation complete."
        )

        # =====================================================================
        # 3. PREPARE DMA REGRESSION PANEL
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 3: PREPARING DMA REGRESSION PANEL"
        )
        logger.info("-" * 80)

        regression_panel = (
            prepare_regression_panel(
                panel
            )
        )

        logger.info(
            "Regression panel prepared: %s rows",
            f"{len(regression_panel):,}",
        )

        logger.info(
            "Regression-panel DMAs: %s",
            f"{regression_panel['DMA'].nunique():,}",
        )

        # =====================================================================
        # 4. DMA COVERAGE / ELIGIBILITY
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 4: BUILDING DMA COVERAGE SUMMARY"
        )
        logger.info("-" * 80)

        coverage = (
            build_dma_coverage_summary(
                regression_panel
            )
        )

        logger.info(
            "Eligible DMAs: %s",
            f"{coverage['Eligible_For_Model'].sum():,}",
        )

        # =====================================================================
        # 5. PRIMARY DMA REGRESSIONS
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 5: RUNNING DMA-LEVEL REGRESSIONS"
        )
        logger.info("=" * 80)

        results = run_dma_models(
            regression_panel
        )

        logger.info(
            "Primary DMA models complete: %s result rows",
            f"{len(results):,}",
        )

        # =====================================================================
        # 6. CLASSIFY PRIMARY DMA RESULTS
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 6: CLASSIFYING DMA SPEND RELATIONSHIPS"
        )
        logger.info("-" * 80)

        successful_mask = (
            results["Model_Status"]
            ==
            "Success"
        )

        if successful_mask.any():

            classification = (
                results.loc[
                    successful_mask
                ]
                .apply(
                    classify_dma_spend_relationship,
                    axis=1,
                )
            )

            for column in classification.columns:

                results.loc[
                    successful_mask,
                    column,
                ] = classification[
                    column
                ].values

        logger.info(
            "DMA relationship classifications complete."
        )

        # =====================================================================
        # 7. SAVE PRIMARY MODEL RESULTS
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 7: SAVING PRIMARY DMA MODEL RESULTS"
        )
        logger.info("-" * 80)

        save_model_results(
            results
        )

        logger.info(
            "Primary DMA model outputs saved."
        )

        # =====================================================================
        # 8. LEAVE-ONE-OUT SENSITIVITY
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 8: RUNNING LEAVE-ONE-OUT SENSITIVITY"
        )
        logger.info("=" * 80)

        loo_sensitivity = (
            run_leave_one_out_sensitivity(
                regression_panel=regression_panel,
                primary_results=results,
            )
        )

        logger.info(
            "Leave-one-out tests completed: %s",
            f"{len(loo_sensitivity):,}",
        )

        # =====================================================================
        # 9. CLASSIFY LEAVE-ONE-OUT SENSITIVITY
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 9: CLASSIFYING LEAVE-ONE-OUT SENSITIVITY"
        )
        logger.info("-" * 80)

        loo_sensitivity = (
            classify_leave_one_out_sensitivity(
                loo_sensitivity
            )
        )

        logger.info(
            "Leave-one-out classifications complete."
        )

        # =====================================================================
        # 10. SAVE LEAVE-ONE-OUT RESULTS
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 10: SAVING LEAVE-ONE-OUT RESULTS"
        )
        logger.info("-" * 80)

        save_leave_one_out_results(
            loo_sensitivity
        )

        logger.info(
            "Leave-one-out outputs saved."
        )

        # =====================================================================
        # 11. BUILD PORTFOLIO SUMMARY
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 11: BUILDING ANALYSIS SUMMARY"
        )
        logger.info("=" * 80)

        summary = build_analysis_summary(
            results=results,
            loo_sensitivity=loo_sensitivity,
        )

        logger.info(
            "Portfolio summary created."
        )

        # =====================================================================
        # 11A. LOAD TREATED DMA REPORTING UNIVERSE
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 11A: LOADING TREATED DMA REPORTING UNIVERSE"
        )
        logger.info("-" * 80)

        treated_dmas = (
            load_treated_dma_universe()
        )

        logger.info(
            "Treated DMA reporting universe: %s",
            f"{len(treated_dmas):,}",
        )

        # =====================================================================
        # 12. BUILD DMA REPORTING TABLE
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 12: BUILDING DMA REPORTING TABLE"
        )
        logger.info("-" * 80)

        reporting_table = (
            build_dma_reporting_table(
                results=results,
                loo_sensitivity=loo_sensitivity,
                treated_dmas=treated_dmas,
            )
        )

        logger.info(
            "DMA reporting table created: %s rows",
            f"{len(reporting_table):,}",
        )

        # =====================================================================
        # 13. SAVE REPORTING DATASETS
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 13: SAVING REPORTING DATASETS"
        )
        logger.info("-" * 80)

        save_reporting_outputs(
            summary=summary,
            reporting_table=reporting_table,
        )

        logger.info(
            "Reporting datasets saved."
        )

        # =====================================================================
        # 14. GENERATE FIGURES
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 14: GENERATING FIGURES"
        )
        logger.info("=" * 80)

        figures = generate_figures(
            results=results,
            loo_sensitivity=loo_sensitivity,
            treated_dmas=treated_dmas,
        )

        logger.info(
            "Figures generated: %s",
            len(figures),
        )

        for figure_name, figure_path in figures.items():

            logger.info(
                "  %s: %s",
                figure_name,
                figure_path,
            )

        # =====================================================================
        # 15. GENERATE HTML REPORT
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 15: GENERATING HTML REPORT"
        )
        logger.info("=" * 80)

        report_path = generate_html_report(
            summary=summary,
            reporting_table=reporting_table,
            figures=figures,
        )

        logger.info(
            "HTML report generated: %s",
            report_path,
        )

        # =====================================================================
        # 16. FINAL SUMMARY
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "DMA SPEND EXPLANATORY POWER ANALYSIS COMPLETE"
        )
        logger.info("=" * 80)

        successful_models = (
            results[
                "Model_Status"
            ]
            ==
            "Success"
        ).sum()

        logger.info(
            "Regression-panel rows: %s",
            f"{len(regression_panel):,}",
        )

        logger.info(
            "DMAs in regression panel: %s",
            f"{regression_panel['DMA'].nunique():,}",
        )

        logger.info(
            "Eligible DMAs: %s",
            f"{coverage['Eligible_For_Model'].sum():,}",
        )

        logger.info(
            "Successful DMA models: %s",
            f"{successful_models:,}",
        )

        logger.info(
            "Mean spend partial R²: %.2f%%",
            (
                summary[
                    "mean_partial_r2"
                ]
                * 100
            ),
        )

        logger.info(
            "Median spend partial R²: %.2f%%",
            (
                summary[
                    "median_partial_r2"
                ]
                * 100
            ),
        )

        logger.info(
            "Positive spend coefficients: %s",
            f"{summary['positive_beta_count']:,}",
        )

        logger.info(
            "Negative spend coefficients: %s",
            f"{summary['negative_beta_count']:,}",
        )

        logger.info(
            "Significant positive relationships: %s",
            f"{summary['significant_positive_count']:,}",
        )

        logger.info(
            "Significant negative relationships: %s",
            f"{summary['significant_negative_count']:,}",
        )

        logger.info(
            "High-instability primary models: %s",
            f"{summary['high_instability_count']:,}",
        )

        logger.info(
            "LOO Robust: %s",
            f"{summary['loo_robust_count']:,}",
        )

        logger.info(
            "LOO Moderately Sensitive: %s",
            f"{summary['loo_moderate_count']:,}",
        )

        logger.info(
            "LOO Sensitive: %s",
            f"{summary['loo_sensitive_count']:,}",
        )

        logger.info(
            "LOO Highly Sensitive: %s",
            f"{summary['loo_highly_sensitive_count']:,}",
        )

        logger.info(
            "Report: %s",
            report_path,
        )

        logger.info("=" * 80)
        logger.info("SUCCESS")
        logger.info("=" * 80)

    except Exception:

        logger.exception(
            "DMA SPEND EXPLANATORY POWER ANALYSIS FAILED."
        )

        raise


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()
# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()