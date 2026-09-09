# 16_dma_opportunity_analysis.py

# Purpose
# -------
# Estimate the DMA-level growth opportunities and directed spend.
#
# IMPORTANT
# ---------
# This is a DMA-level explanatory / association analysis.
# It should NOT be interpreted as a causal estimate of marketing spend.
# =============================================================================



# =============================================================================
# DMA OPPORTUNITY ANALYSIS
# =============================================================================

from pathlib import Path
import logging

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    OUTPUT_DIR,
    DATA_DIR
)


# =============================================================================
# CONFIG
# =============================================================================

TABLE_DIR = (
    OUTPUT_DIR
    / "tables"
    / "dma_opportunity"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
    / "dma_opportunity"
)

REPORT_DIR = (
    OUTPUT_DIR
    / "reports"
)


BSTS_COMP_FILE = (
    OUTPUT_DIR
    / "tables"
    / "bsts_comp_comparison"
    / "bsts_comp_dma_comparison.csv"
)


DMA_REGRESSION_FILE = (
    OUTPUT_DIR
    / "tables"
    / "dma_spend_explanatory_power"
    / "dma_spend_explanatory_reporting_table.csv"
)

DMA_REGRESSION_ALL_FILE = (
    OUTPUT_DIR
    / "tables"
    / "dma_spend_explanatory_power"
    / "dma_spend_explanatory_results.csv"
)


MODEL_PANEL_FILE = (
    DATA_DIR
    / "processed"
    / "model_panel.parquet"
)


# =============================================================================
# SCORING CONFIGURATION
# =============================================================================

# Overall opportunity-score weights.
#
# These should remain configurable because they represent
# business decision weights rather than statistical laws.

PERFORMANCE_WEIGHT = 0.30
RESPONSIVENESS_WEIGHT = 0.30
CONFIDENCE_WEIGHT = 0.25
MARKET_SCALE_WEIGHT = 0.15

# =============================================================================
# ALL-DMA OPPORTUNITY CONFIGURATION
# =============================================================================

ALL_DMA_RESPONSIVENESS_WEIGHT = 0.45
ALL_DMA_CONFIDENCE_WEIGHT = 0.30
ALL_DMA_MARKET_SCALE_WEIGHT = 0.25


# Spend-testing ranges.
#
# These are strategic testing recommendations, NOT model-estimated
# optimal spend levels.

SPEND_BUCKETS = {

    "Scale":
        (0.15, 0.25),

    "Test Increase":
        (0.05, 0.15),

    "Maintain / Optimize":
        (-0.05, 0.05),

    "Hold & Diagnose":
        (0.00, 0.00),

    "Reallocate / Reduce":
        (-0.25, -0.10),
}

ALL_DMA_SPEND_BUCKETS = {

    "Priority Test":
        (0.10, 0.20),

    "Explore Incremental Test":
        (0.05, 0.10),

    "Maintain / Monitor":
        (0.00, 0.05),

    "Diagnostic":
        (0.00, 0.00),

    "Low Priority":
        (0.00, 0.00),
}


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
    Create output directories used by the DMA opportunity analysis.
    """

    for path in [
        TABLE_DIR,
        FIGURE_DIR,
        REPORT_DIR,
    ]:

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    logger.info(
        "DMA opportunity output directories ready."
    )


# =============================================================================
# DMA NORMALIZATION
# =============================================================================

def normalize_dma_key(
    series,
):
    """
    Normalize DMA names to the simplified marketing-analysis
    convention used across downstream reporting.

    This is intentionally aligned with the DMA mappings used
    elsewhere in the project.
    """

    dma_map = {

        "Atlanta, GA":
            "Atlanta",

        "Baltimore, MD":
            "Baltimore",

        "Boston, MA (Manchester, NH)":
            "Boston",

        "Buffalo, NY":
            "Buffalo",

        "Charlotte, NC":
            "Charlotte",

        "Chicago, IL":
            "Chicago",

        "Cincinnati, OH":
            "Cincinnati",

        "Cleveland-Akron (Canton), OH":
            "Cleveland",

        "Columbus, OH":
            "Columbus",

        "Dallas-Ft. Worth, TX":
            "Dallas",

        "Detroit, MI":
            "Detroit",

        "El Paso, TX (Las Cruces, NM)":
            "El Paso",

        "Ft. Myers-Naples, FL":
            "Fort Myers",

        "Harlingen-Weslaco-Brownsville-McAllen, TX":
            "Harlingen/McAllen",

        "Hartford & New Haven, CT":
            "Hartford",

        "Houston, TX":
            "Houston",

        "Las Vegas, NV":
            "Las Vegas",

        "Los Angeles, CA":
            "Los Angeles",

        "Miami-Ft. Lauderdale, FL":
            "Miami",

        "Minneapolis-St. Paul, MN":
            "Minneapolis",

        "New York, NY":
            "New York",

        "Oklahoma City, OK":
            "Oklahoma City",

        "Orlando-Daytona Beach-Melbourne, FL":
            "Orlando",

        "Philadelphia, PA":
            "Philadelphia",

        "Phoenix (Prescott), AZ":
            "Phoenix",

        "Providence, RI-New Bedford, MA":
            "Providence",

        "Sacramento-Stockton-Modesto, CA":
            "Sacramento",

        "San Antonio, TX":
            "San Antonio",

        "San Diego, CA":
            "San Diego",

        "Seattle-Tacoma, WA":
            "Seattle",

        "Tampa-St. Petersburg (Sarasota), FL":
            "Tampa",

        "Washington, DC (Hagerstown, MD)":
            "Washington DC",

        "West Palm Beach-Ft. Pierce, FL":
            "West Palm",

        "Puerto Rico":
            "Puerto Rico",
    }

    normalized = (
        series
        .astype("string")
        .str.strip()
        .replace(
            dma_map
        )
    )

    return normalized


# =============================================================================
# LOAD BSTS / COMP RESULTS
# =============================================================================

def load_bsts_comp_results(
    path=None,
):
    """
    Load DMA-level BSTS / comp translation results.
    """

    path = Path(
        path or BSTS_COMP_FILE
    )

    logger.info("=" * 70)
    logger.info(
        "LOADING BSTS / COMP RESULTS"
    )
    logger.info("=" * 70)

    if not path.exists():

        raise FileNotFoundError(
            f"BSTS / comp comparison file does not exist: {path}"
        )

    data = pd.read_csv(
        path
    )

    required_columns = {

        "DMA",
        "BSTS_Relative_Lift",
        "BSTS_Probability_Positive",
        "Implied_Incremental_Comp",
    }

    missing = (
        required_columns
        -
        set(
            data.columns
        )
    )

    if missing:

        raise ValueError(
            "BSTS / comp results are missing required columns: "
            f"{sorted(missing)}"
        )

    data["DMA_Key"] = (
        normalize_dma_key(
            data["DMA"]
        )
    )

    numeric_columns = [

        "BSTS_Relative_Lift",
        "BSTS_Probability_Positive",
        "Implied_Incremental_Comp",
        "BSTS_Incremental_Traffic",
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    logger.info(
        "Loaded BSTS / comp results: %s DMAs",
        f"{data['DMA_Key'].nunique():,}",
    )

    return data


# =============================================================================
# LOAD DMA REGRESSION RESULTS
# =============================================================================

def load_dma_regression_results(
    path=None,
):
    """
    Load treated-DMA spend explanatory-power results.
    """

    path = Path(
        path or DMA_REGRESSION_FILE
    )

    logger.info("=" * 70)
    logger.info(
        "LOADING DMA SPEND REGRESSION RESULTS"
    )
    logger.info("=" * 70)

    if not path.exists():

        raise FileNotFoundError(
            f"DMA regression file does not exist: {path}"
        )

    data = pd.read_csv(
        path
    )

    required_columns = {

        "DMA",
        "Spend_Partial_R2",
        "Spend_Beta",
        "Spend_P_Value",
    }

    missing = (
        required_columns
        -
        set(
            data.columns
        )
    )

    if missing:

        raise ValueError(
            "DMA regression results are missing required columns: "
            f"{sorted(missing)}"
        )

    data["DMA_Key"] = (
        normalize_dma_key(
            data["DMA"]
        )
    )

    numeric_columns = [

        "Spend_Partial_R2",
        "Spend_Beta",
        "Spend_P_Value",
        "Removed_Cooks_D",
        "LOO_Spend_Beta",
        "LOO_Spend_Partial_R2",
        "LOO_Spend_P_Value",
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    logger.info(
        "Loaded DMA regression results: %s DMAs",
        f"{data['DMA_Key'].nunique():,}",
    )

    return data


def load_all_dma_regression_results(
    path=None,
):
    """
    Load spend explanatory-power results for the complete DMA universe.
    """

    path = Path(
        path or DMA_REGRESSION_ALL_FILE
    )

    logger.info("=" * 70)
    logger.info(
        "LOADING ALL-DMA SPEND REGRESSION RESULTS"
    )
    logger.info("=" * 70)

    if not path.exists():

        raise FileNotFoundError(
            f"DMA regression file does not exist: {path}"
        )

    data = pd.read_csv(
        path
    )

    required = {
        "DMA",
        "Model_Status",
        "Spend_Partial_R2",
        "Spend_Beta",
        "Spend_P_Value",
        "Influence_Severity",
        "Spend_Estimate_Stability",
    }

    missing = (
        required
        -
        set(data.columns)
    )

    if missing:

        raise ValueError(
            "All-DMA regression results missing columns: "
            f"{sorted(missing)}"
        )

    data = data.loc[
        data["Model_Status"]
        ==
        "Success"
    ].copy()

    data["DMA_Key"] = (
        normalize_dma_key(
            data["DMA"]
        )
    )

    numeric_columns = [
        "Spend_Partial_R2",
        "Spend_Beta",
        "Spend_P_Value",
        "Mean_Spend",
        "Mean_Traffic",
        "Max_Cooks_D",
    ]

    for col in numeric_columns:

        if col in data.columns:

            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            )

    logger.info(
        "Successful all-DMA regression models: %s",
        f"{len(data):,}",
    )

    return data


# =============================================================================
# LOAD MODEL PANEL
# =============================================================================

def load_model_panel(
    path=None,
):
    """
    Load model panel for current investment context.
    """

    path = Path(
        path or MODEL_PANEL_FILE
    )

    logger.info("=" * 70)
    logger.info(
        "LOADING MODEL PANEL"
    )
    logger.info("=" * 70)

    if not path.exists():

        raise FileNotFoundError(
            f"Model panel does not exist: {path}"
        )

    panel = pd.read_parquet(
        path
    )

    required_columns = {

        "DMA",
        "Month",
        "Spend",
        "Traffic",
        "Store_Count",
    }

    missing = (
        required_columns
        -
        set(
            panel.columns
        )
    )

    if missing:

        raise ValueError(
            "Model panel is missing required columns: "
            f"{sorted(missing)}"
        )

    panel["DMA_Key"] = (
        normalize_dma_key(
            panel["DMA"]
        )
    )

    panel["Month"] = pd.to_datetime(
        panel["Month"],
        errors="coerce",
    )

    panel["Spend"] = pd.to_numeric(
        panel["Spend"],
        errors="coerce",
    )

    panel["Traffic"] = pd.to_numeric(
        panel["Traffic"],
        errors="coerce",
    )

    logger.info(
        "Loaded model panel: %s rows",
        f"{len(panel):,}",
    )

    logger.info(
        "DMAs: %s",
        f"{panel['DMA_Key'].nunique():,}",
    )

    return panel



# =============================================================================
# SOURCE DMA COVERAGE DIAGNOSTIC
# =============================================================================

def diagnose_dma_source_coverage(
    bsts,
    regression,
):
    """
    Compare DMA coverage between BSTS / comp results and
    DMA spend-regression results.
    """

    logger.info("=" * 70)
    logger.info("DMA SOURCE COVERAGE DIAGNOSTIC")
    logger.info("=" * 70)

    bsts_dmas = set(
        bsts[
            "DMA_Key"
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    regression_dmas = set(
        regression[
            "DMA_Key"
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    common_dmas = (
        bsts_dmas
        &
        regression_dmas
    )

    missing_from_bsts = (
        regression_dmas
        -
        bsts_dmas
    )

    missing_from_regression = (
        bsts_dmas
        -
        regression_dmas
    )

    logger.info(
        "BSTS / comp DMAs: %s",
        f"{len(bsts_dmas):,}",
    )

    logger.info(
        "DMA regression DMAs: %s",
        f"{len(regression_dmas):,}",
    )

    logger.info(
        "Common DMAs: %s",
        f"{len(common_dmas):,}",
    )

    if missing_from_bsts:

        logger.warning(
            "Regression DMAs missing from BSTS / comp results: %s",
            f"{len(missing_from_bsts):,}",
        )

        for dma in sorted(
            missing_from_bsts
        ):

            logger.warning(
                "  Missing from BSTS / comp: %s",
                dma,
            )

    if missing_from_regression:

        logger.warning(
            "BSTS / comp DMAs missing from regression results: %s",
            f"{len(missing_from_regression):,}",
        )

        for dma in sorted(
            missing_from_regression
        ):

            logger.warning(
                "  Missing from regression: %s",
                dma,
            )

    return {
        "bsts_dma_count":
            len(bsts_dmas),

        "regression_dma_count":
            len(regression_dmas),

        "common_dma_count":
            len(common_dmas),

        "missing_from_bsts":
            sorted(
                missing_from_bsts
            ),

        "missing_from_regression":
            sorted(
                missing_from_regression
            ),
    }

# =============================================================================
# BUILD INVESTMENT PROFILE
# =============================================================================

def build_investment_profile(
    panel,
    recent_months=3,
):
    """
    Build DMA-level investment context.

    Metrics include:

        Current / latest monthly spend
        Recent average spend
        Historical average spend
        Total observed spend
        Recent traffic
        Store count
        Spend per store

    Recent average spend is used as the initial baseline for
    translating strategic spend buckets into dollar ranges.
    """

    logger.info("=" * 70)
    logger.info(
        "BUILDING DMA INVESTMENT PROFILE"
    )
    logger.info("=" * 70)

    results = []

    for dma_key, dma_data in panel.groupby(
        "DMA_Key"
    ):

        dma_data = (
            dma_data
            .sort_values(
                "Month"
            )
            .copy()
        )

        latest = (
            dma_data
            .iloc[-1]
        )

        recent = (
            dma_data
            .tail(
                recent_months
            )
        )

        latest_spend = (
            latest[
                "Spend"
            ]
        )

        recent_average_spend = (
            recent[
                "Spend"
            ]
            .mean()
        )

        historical_average_spend = (
            dma_data[
                "Spend"
            ]
            .mean()
        )

        total_spend = (
            dma_data[
                "Spend"
            ]
            .sum()
        )

        recent_average_traffic = (
            recent[
                "Traffic"
            ]
            .mean()
        )

        latest_store_count = (
            latest[
                "Store_Count"
            ]
        )

        if (
            pd.notna(
                recent_average_spend
            )
            and
            pd.notna(
                latest_store_count
            )
            and
            latest_store_count > 0
        ):

            spend_per_store = (
                recent_average_spend
                /
                latest_store_count
            )

        else:

            spend_per_store = np.nan

        results.append(
            {
                "DMA_Key":
                    dma_key,

                "Latest_Month":
                    latest[
                        "Month"
                    ],

                "Latest_Spend":
                    latest_spend,

                "Recent_Average_Spend":
                    recent_average_spend,

                "Historical_Average_Spend":
                    historical_average_spend,

                "Total_Observed_Spend":
                    total_spend,

                "Recent_Average_Traffic":
                    recent_average_traffic,

                "Store_Count":
                    latest_store_count,

                "Recent_Spend_Per_Store":
                    spend_per_store,
            }
        )

    profile = pd.DataFrame(
        results
    )

    logger.info(
        "Investment profiles created: %s",
        f"{len(profile):,}",
    )

    return profile


# =============================================================================
# MERGE ANALYTICAL SOURCES
# =============================================================================

def build_opportunity_panel(
    bsts,
    regression,
    investment,
):
    """
    Merge BSTS, DMA regression, and current investment information
    into the opportunity-analysis panel.
    """

    logger.info("=" * 70)
    logger.info(
        "BUILDING DMA OPPORTUNITY PANEL"
    )
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Keep only required / useful fields
    # -------------------------------------------------------------------------

    bsts_columns = [

        "DMA",
        "DMA_Key",
        "BSTS_Relative_Lift",
        "BSTS_Probability_Positive",
        "BSTS_Incremental_Traffic",
        "Actual_Comp_Traffic",
        "Implied_Counterfactual_Comp",
        "Implied_Incremental_Comp",
    ]

    bsts_panel = bsts[
        [
            col
            for col in bsts_columns
            if col in bsts.columns
        ]
    ].copy()

    regression_columns = [

        "DMA_Key",
        "Spend_Partial_R2",
        "Spend_Beta",
        "Spend_P_Value",
        "Spend_Explanatory_Strength",
        "Influence_Severity",
        "Spend_Estimate_Stability",
        "Spend_Relationship_Classification",
        "LOO_Sensitivity_Classification",
        "LOO_Inference_Stability",
        "LOO_Spend_Beta",
        "LOO_Spend_Partial_R2",
    ]

    regression_panel = regression[
        [
            col
            for col in regression_columns
            if col in regression.columns
        ]
    ].copy()

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    panel = bsts_panel.merge(
        regression_panel,
        on="DMA_Key",
        how="left",
        validate="one_to_one",
    )

    panel = panel.merge(
        investment,
        on="DMA_Key",
        how="left",
        validate="one_to_one",
    )

    logger.info(
        "Opportunity panel rows: %s",
        f"{len(panel):,}",
    )

    logger.info(
        "BSTS DMAs with regression results: %s",
        f"{panel['Spend_Partial_R2'].notna().sum():,}",
    )

    logger.info(
        "BSTS DMAs with investment profiles: %s",
        f"{panel['Recent_Average_Spend'].notna().sum():,}",
    )

    return panel


# =============================================================================
# HELPER: CLIP SCORE
# =============================================================================

def clip_score(
    value,
):
    """
    Restrict score to 0-100.
    """

    if pd.isna(
        value
    ):

        return np.nan

    return float(
        np.clip(
            value,
            0,
            100,
        )
    )


# =============================================================================
# MARKET SCALE SCORE
# =============================================================================

def calculate_market_scale_score(
    panel,
):
    """
    Calculate relative DMA market scale from store count and traffic.

    Market size is intentionally normalized across the opportunity
    population so that large DMAs receive credit for commercial upside
    without automatically dominating the opportunity ranking.

    Components
    ----------
    Store_Count
        Represents physical market footprint.

    Recent_Average_Traffic
        Represents current customer / traffic volume.

    Each component is converted to a percentile rank from 0-100.

    Final Market_Scale_Score:
        50% Store Count
        50% Recent Average Traffic
    """

    logger.info("=" * 70)
    logger.info(
        "CALCULATING DMA MARKET SCALE SCORE"
    )
    logger.info("=" * 70)

    panel = panel.copy()

    # -------------------------------------------------------------------------
    # Store-count percentile
    # -------------------------------------------------------------------------

    panel[
        "Store_Count_Score"
    ] = (
        panel[
            "Store_Count"
        ]
        .rank(
            pct=True,
            method="average",
        )
        *
        100
    )

    # -------------------------------------------------------------------------
    # Traffic-volume percentile
    # -------------------------------------------------------------------------

    panel[
        "Traffic_Scale_Score"
    ] = (
        panel[
            "Recent_Average_Traffic"
        ]
        .rank(
            pct=True,
            method="average",
        )
        *
        100
    )

    # -------------------------------------------------------------------------
    # Combined market scale
    # -------------------------------------------------------------------------

    panel[
        "Market_Scale_Score"
    ] = (
        panel[
            "Store_Count_Score"
        ]
        *
        0.50
        +
        panel[
            "Traffic_Scale_Score"
        ]
        *
        0.50
    )

    logger.info(
        "Mean market scale score: %.2f",
        panel[
            "Market_Scale_Score"
        ].mean(),
    )

    logger.info(
        "Largest DMA scale scores:\n%s",
        panel[
            [
                "DMA",
                "Store_Count",
                "Recent_Average_Traffic",
                "Store_Count_Score",
                "Traffic_Scale_Score",
                "Market_Scale_Score",
            ]
        ]
        .sort_values(
            "Market_Scale_Score",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False
        ),
    )

    return panel

# =============================================================================
# PERFORMANCE SCORE
# =============================================================================

def calculate_performance_score(
    row,
):
    """
    Score intervention-period performance from 0-100.

    Uses:

        Implied Incremental Comp
        BSTS Probability Positive

    Interpretation
    --------------
    Higher score:
        stronger evidence of positive intervention-period performance.

    Lower score:
        stronger evidence of intervention-period underperformance.

    The score is intentionally directional and capped to avoid one
    extreme DMA dominating the ranking.
    """

    implied_comp = row.get(
        "Implied_Incremental_Comp"
    )

    prob_positive = row.get(
        "BSTS_Probability_Positive"
    )

    # -------------------------------------------------------------------------
    # Implied comp component
    #
    # +/- 10 percentage points roughly maps to the full 0-100 range.
    # Because the source value is stored as a decimal:
    #
    # +0.10 -> 100
    #  0.00 -> 50
    # -0.10 -> 0
    # -------------------------------------------------------------------------

    if pd.notna(
        implied_comp
    ):

        comp_component = (
            50
            +
            (
                implied_comp
                /
                0.10
                *
                50
            )
        )

        comp_component = clip_score(
            comp_component
        )

    else:

        comp_component = np.nan

    # -------------------------------------------------------------------------
    # Probability component
    # -------------------------------------------------------------------------

    if pd.notna(
        prob_positive
    ):

        probability_component = (
            prob_positive
            *
            100
        )

        probability_component = clip_score(
            probability_component
        )

    else:

        probability_component = np.nan

    # -------------------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------------------

    available = [
        value
        for value in [
            comp_component,
            probability_component,
        ]
        if pd.notna(
            value
        )
    ]

    if not available:

        return np.nan

    return float(
        np.mean(
            available
        )
    )


# =============================================================================
# RESPONSIVENESS SCORE
# =============================================================================

def calculate_responsiveness_score(
    row,
):
    """
    Score evidence that spend and traffic move together.

    Uses:

        Spend Partial R²
        Spend Beta direction

    Partial R² drives the magnitude of the responsiveness score.
    A positive beta receives full directional credit.
    A negative beta heavily penalizes the result.
    """

    partial_r2 = row.get(
        "Spend_Partial_R2"
    )

    beta = row.get(
        "Spend_Beta"
    )

    if pd.isna(
        partial_r2
    ):

        return np.nan

    # -------------------------------------------------------------------------
    # Partial R² component
    #
    # 0.00 -> 0
    # 0.50 -> 100
    #
    # Values above 0.50 are capped.
    # -------------------------------------------------------------------------

    explanatory_component = clip_score(
        (
            partial_r2
            /
            0.50
        )
        *
        100
    )

    # -------------------------------------------------------------------------
    # Direction modifier
    # -------------------------------------------------------------------------

    if pd.isna(
        beta
    ):

        direction_modifier = 0.50

    elif beta > 0:

        direction_modifier = 1.00

    elif beta < 0:

        direction_modifier = 0.20

    else:

        direction_modifier = 0.50

    responsiveness_score = (
        explanatory_component
        *
        direction_modifier
    )

    return clip_score(
        responsiveness_score
    )


# =============================================================================
# CONFIDENCE SCORE
# =============================================================================

def calculate_confidence_score(
    row,
):
    """
    Score how confidently the opportunity signal should be acted upon.

    Uses:

        BSTS probability strength
        regression statistical evidence
        influence severity
        leave-one-out stability

    Confidence is deliberately separate from opportunity.
    """

    scores = []

    # -------------------------------------------------------------------------
    # BSTS directional confidence
    #
    # Confidence should be high whether the estimated effect is
    # confidently positive OR confidently negative.
    #
    # 0.50 probability = weak directional confidence.
    # Near 0 or 1 = strong directional confidence.
    # -------------------------------------------------------------------------

    prob_positive = row.get(
        "BSTS_Probability_Positive"
    )

    if pd.notna(
        prob_positive
    ):

        directional_probability = abs(
            prob_positive
            -
            0.50
        ) * 2

        scores.append(
            directional_probability
            *
            100
        )

    # -------------------------------------------------------------------------
    # Regression statistical evidence
    # -------------------------------------------------------------------------

    p_value = row.get(
        "Spend_P_Value"
    )

    if pd.notna(
        p_value
    ):

        if p_value < 0.05:

            regression_score = 100

        elif p_value < 0.10:

            regression_score = 75

        elif p_value < 0.20:

            regression_score = 50

        else:

            regression_score = 25

        scores.append(
            regression_score
        )

    # -------------------------------------------------------------------------
    # Influence severity
    # -------------------------------------------------------------------------

    influence = row.get(
        "Influence_Severity"
    )

    influence_scores = {

        "Low":
            100,

        "Moderate":
            65,

        "High":
            30,

        "Unavailable":
            40,
    }

    if pd.notna(
        influence
    ):

        scores.append(
            influence_scores.get(
                influence,
                40,
            )
        )

    # -------------------------------------------------------------------------
    # Leave-one-out stability
    # -------------------------------------------------------------------------

    loo = row.get(
        "LOO_Sensitivity_Classification"
    )

    loo_scores = {

        "Robust":
            100,

        "Moderately Sensitive":
            70,

        "Sensitive":
            40,

        "Highly Sensitive":
            15,

        "Unavailable":
            40,
    }

    if pd.notna(
        loo
    ):

        scores.append(
            loo_scores.get(
                loo,
                40,
            )
        )

    if not scores:

        return np.nan

    return float(
        np.mean(
            scores
        )
    )


# =============================================================================
# ALL-DMA REGRESSION CONFIDENCE SCORE
# =============================================================================

def calculate_regression_confidence_score(
    row,
):
    """
    Calculate confidence in the DMA spend-response estimate
    without using BSTS information.

    Components
    ----------
    Statistical evidence
    Influence severity
    Estimate stability

    Score range
    -----------
    0-100
    """

    scores = []

    # -------------------------------------------------------------------------
    # Statistical evidence
    # -------------------------------------------------------------------------

    p_value = row.get(
        "Spend_P_Value"
    )

    if pd.notna(
        p_value
    ):

        if p_value < 0.05:

            statistical_score = 100

        elif p_value < 0.10:

            statistical_score = 75

        elif p_value < 0.20:

            statistical_score = 50

        else:

            statistical_score = 25

        scores.append(
            statistical_score
        )

    # -------------------------------------------------------------------------
    # Influence severity
    # -------------------------------------------------------------------------

    influence = row.get(
        "Influence_Severity"
    )

    influence_scores = {

        "Low":
            100,

        "Moderate":
            65,

        "High":
            30,

        "Unavailable":
            40,
    }

    if pd.notna(
        influence
    ):

        scores.append(
            influence_scores.get(
                influence,
                40,
            )
        )

    # -------------------------------------------------------------------------
    # Estimate stability
    # -------------------------------------------------------------------------

    stability = row.get(
        "Spend_Estimate_Stability"
    )

    stability_scores = {

        "Stable":
            100,

        "Influence Sensitive":
            60,

        "Highly Unstable":
            25,

        "Unavailable":
            40,
    }

    if pd.notna(
        stability
    ):

        scores.append(
            stability_scores.get(
                stability,
                40,
            )
        )

    if not scores:

        return np.nan

    return float(
        np.mean(
            scores
        )
    )


# =============================================================================
# BUILD COMPONENT SCORES
# =============================================================================

def calculate_component_scores(
    panel,
):
    """
    Calculate performance, responsiveness, and confidence scores.
    """

    logger.info("=" * 70)
    logger.info(
        "CALCULATING DMA OPPORTUNITY COMPONENT SCORES"
    )
    logger.info("=" * 70)

    panel = panel.copy()

    panel[
        "Performance_Score"
    ] = panel.apply(
        calculate_performance_score,
        axis=1,
    )

    panel[
        "Responsiveness_Score"
    ] = panel.apply(
        calculate_responsiveness_score,
        axis=1,
    )

    panel[
        "Confidence_Score"
    ] = panel.apply(
        calculate_confidence_score,
        axis=1,
    )

    return panel


# =============================================================================
# OPPORTUNITY SCORE
# =============================================================================

def calculate_opportunity_score(
    panel,
):
    """
    Calculate overall DMA opportunity score.

    Components
    ----------
    Performance_Score
        Intervention-period counterfactual performance.

    Responsiveness_Score
        Evidence that marketing spend helps explain traffic variation.

    Confidence_Score
        Reliability / robustness of the analytical evidence.

    Market_Scale_Score
        Relative commercial scale based upon traffic and store footprint.
    """

    logger.info("=" * 70)
    logger.info(
        "CALCULATING DMA OPPORTUNITY SCORE"
    )
    logger.info("=" * 70)

    panel = panel.copy()

    total_weight = (
        PERFORMANCE_WEIGHT
        +
        RESPONSIVENESS_WEIGHT
        +
        CONFIDENCE_WEIGHT
        +
        MARKET_SCALE_WEIGHT
    )

    if not np.isclose(
        total_weight,
        1.0,
    ):

        raise ValueError(
            "Opportunity-score weights must sum to 1.0. "
            f"Current total: {total_weight:.4f}"
        )

    panel[
        "Opportunity_Score"
    ] = (
        panel[
            "Performance_Score"
        ]
        *
        PERFORMANCE_WEIGHT

        +

        panel[
            "Responsiveness_Score"
        ]
        *
        RESPONSIVENESS_WEIGHT

        +

        panel[
            "Confidence_Score"
        ]
        *
        CONFIDENCE_WEIGHT

        +

        panel[
            "Market_Scale_Score"
        ]
        *
        MARKET_SCALE_WEIGHT
    )

    panel[
        "Opportunity_Score"
    ] = (
        panel[
            "Opportunity_Score"
        ]
        .clip(
            lower=0,
            upper=100,
        )
    )

    return panel
# =============================================================================
# CLASSIFY OPPORTUNITY
# =============================================================================

# =============================================================================
# EVIDENCE ALIGNMENT
# =============================================================================

def classify_evidence_alignment(
    row,
):
    """
    Classify agreement between BSTS intervention-period performance
    and the DMA-level spend-response model.

    Regression direction is treated as informative only when spend
    explains a meaningful amount of residual traffic variation.

    This prevents very small / effectively zero spend relationships
    from being classified as genuine directional conflicts.
    """

    bsts_lift = row.get(
        "BSTS_Relative_Lift"
    )

    beta = row.get(
        "Spend_Beta"
    )

    partial_r2 = row.get(
        "Spend_Partial_R2"
    )

    p_value = row.get(
        "Spend_P_Value"
    )

    # -------------------------------------------------------------------------
    # Missing evidence
    # -------------------------------------------------------------------------

    if (
        pd.isna(bsts_lift)
        or
        pd.isna(beta)
        or
        pd.isna(partial_r2)
    ):

        return (
            "Insufficient Evidence"
        )

    # -------------------------------------------------------------------------
    # Spend relationship is essentially uninformative
    #
    # Partial R² below 5% means spend contributes very little
    # incremental explanatory power in the DMA-level regression.
    # -------------------------------------------------------------------------

    if partial_r2 < 0.05:

        if bsts_lift > 0:

            return (
                "Positive BSTS / Spend Uninformative"
            )

        elif bsts_lift < 0:

            return (
                "Negative BSTS / Spend Uninformative"
            )

        else:

            return (
                "Spend Relationship Uninformative"
            )

    # -------------------------------------------------------------------------
    # Meaningful directional agreement
    # -------------------------------------------------------------------------

    if (
        bsts_lift > 0
        and
        beta > 0
    ):

        return (
            "Convergent Positive"
        )

    if (
        bsts_lift < 0
        and
        beta < 0
    ):

        return (
            "Convergent Negative"
        )

    # -------------------------------------------------------------------------
    # Meaningful directional disagreement
    # -------------------------------------------------------------------------

    if (
        (
            bsts_lift > 0
            and
            beta < 0
        )
        or
        (
            bsts_lift < 0
            and
            beta > 0
        )
    ):

        return (
            "Attribution Conflict"
        )

    return (
        "Neutral / Mixed"
    )

def classify_dma_opportunity(
    row,
):
    """
    Translate DMA evidence into an opportunity tier while keeping
    evidence alignment separate from opportunity magnitude.
    """

    score = row.get(
        "Opportunity_Score"
    )

    confidence = row.get(
        "Confidence_Score"
    )

    alignment = (
        classify_evidence_alignment(
            row
        )
    )

    if pd.isna(
        score
    ):

        return pd.Series(
            {
                "Opportunity_Tier":
                    "Unavailable",

                "Evidence_Alignment":
                    alignment,
            }
        )

    # -------------------------------------------------------------------------
    # Genuine model conflict is diagnostic by definition.
    # -------------------------------------------------------------------------

    if (
        alignment
        ==
        "Attribution Conflict"
    ):

        tier = (
            "Diagnostic"
        )

    # -------------------------------------------------------------------------
    # Strong negative convergence = low investment opportunity.
    # -------------------------------------------------------------------------

    elif (
        alignment
        ==
        "Convergent Negative"
        and
        confidence >= 60
    ):

        tier = (
            "Low"
        )

    # -------------------------------------------------------------------------
    # Otherwise opportunity follows the composite score.
    # -------------------------------------------------------------------------

    elif score >= 75:

        tier = (
            "High"
        )

    elif score >= 55:

        tier = (
            "Moderate"
        )

    else:

        tier = (
            "Low"
        )

    return pd.Series(
        {
            "Opportunity_Tier":
                tier,

            "Evidence_Alignment":
                alignment,
        }
    )


# =============================================================================
# STRATEGIC ACTION
# =============================================================================

def assign_strategic_action(
    row,
):
    """
    Translate opportunity, evidence alignment, and confidence
    into a recommended strategic action.

    Philosophy
    ----------
    Scale
        Requires both strong opportunity and relatively strong confidence.

    Test Increase
        Used when opportunity is attractive and evidence is directionally
        supportive, but uncertainty remains.

    Maintain / Optimize
        Used for promising but less compelling markets.

    Hold & Diagnose
        Used when models materially disagree.

    Reallocate / Reduce
        Reserved for meaningful negative convergence with adequate confidence.
    """

    tier = row.get(
        "Opportunity_Tier"
    )

    confidence = row.get(
        "Confidence_Score"
    )

    alignment = row.get(
        "Evidence_Alignment"
    )

    score = row.get(
        "Opportunity_Score"
    )

    # -------------------------------------------------------------------------
    # Genuine attribution conflict
    # -------------------------------------------------------------------------

    if (
        alignment
        ==
        "Attribution Conflict"
    ):

        return (
            "Hold & Diagnose"
        )

    # -------------------------------------------------------------------------
    # Strong negative convergence
    # -------------------------------------------------------------------------

    if (
        alignment
        ==
        "Convergent Negative"
        and
        confidence >= 60
    ):

        return (
            "Reallocate / Reduce"
        )

    # -------------------------------------------------------------------------
    # Strong positive convergence
    # -------------------------------------------------------------------------

    if (
        alignment
        ==
        "Convergent Positive"
    ):

        # Strong enough to scale only when both score and
        # evidence confidence are unusually compelling.

        if (
            score >= 75
            and
            confidence >= 65
        ):

            return (
                "Scale"
            )

        # Controlled incremental test.
        if (
            score >= 55
            and
            confidence >= 35
        ):

            return (
                "Test Increase"
            )

        # Directionally encouraging but still weak.
        if score >= 40:

            return (
                "Maintain / Optimize"
            )

        return (
            "Hold & Diagnose"
        )

    # -------------------------------------------------------------------------
    # Positive BSTS but regression is uninformative
    # -------------------------------------------------------------------------

    if (
        alignment
        ==
        "Positive BSTS / Spend Uninformative"
    ):

        if (
            score >= 55
            and
            confidence >= 40
        ):

            return (
                "Maintain / Optimize"
            )

        return (
            "Hold & Diagnose"
        )

    # -------------------------------------------------------------------------
    # Negative BSTS but regression is uninformative
    # -------------------------------------------------------------------------

    if (
        alignment
        ==
        "Negative BSTS / Spend Uninformative"
    ):

        return (
            "Hold & Diagnose"
        )

    # -------------------------------------------------------------------------
    # Remaining opportunities
    # -------------------------------------------------------------------------

    if (
        tier == "High"
        and
        confidence >= 60
    ):

        return (
            "Test Increase"
        )

    if (
        tier == "Moderate"
        and
        confidence >= 40
    ):

        return (
            "Maintain / Optimize"
        )

    if tier == "Low":

        return (
            "Maintain / Optimize"
        )

    return (
        "Hold & Diagnose"
    )

# =============================================================================
# SPEND BUCKETS
# =============================================================================

def assign_spend_bucket(
    row,
):
    """
    Assign strategic spend-testing bucket based upon recommended action.

    Returns decimal changes rather than formatted strings.
    """

    action = row.get(
        "Strategic_Action"
    )

    lower, upper = SPEND_BUCKETS.get(
        action,
        (
            0.00,
            0.00,
        )
    )

    return pd.Series(
        {
            "Spend_Change_Lower":
                lower,

            "Spend_Change_Upper":
                upper,
        }
    )


# =============================================================================
# CALCULATE DOLLAR RANGE
# =============================================================================

def calculate_spend_range(
    panel,
):
    """
    Convert spend-testing percentages into dollar ranges.

    Recent_Average_Spend is used as the baseline investment level.
    """

    panel = panel.copy()

    panel[
        "Recommended_Incremental_Spend_Lower"
    ] = (

        panel[
            "Recent_Average_Spend"
        ]
        *
        panel[
            "Spend_Change_Lower"
        ]
    )

    panel[
        "Recommended_Incremental_Spend_Upper"
    ] = (

        panel[
            "Recent_Average_Spend"
        ]
        *
        panel[
            "Spend_Change_Upper"
        ]
    )

    panel[
        "Recommended_Total_Spend_Lower"
    ] = (

        panel[
            "Recent_Average_Spend"
        ]
        +
        panel[
            "Recommended_Incremental_Spend_Lower"
        ]
    )

    panel[
        "Recommended_Total_Spend_Upper"
    ] = (

        panel[
            "Recent_Average_Spend"
        ]
        +
        panel[
            "Recommended_Incremental_Spend_Upper"
        ]
    )

    return panel


# =============================================================================
# BUILD BUSINESS INTERPRETATION
# =============================================================================

def build_dma_interpretation(
    row,
):
    """
    Generate concise business interpretation for each DMA.
    """

    dma = row.get(
        "DMA"
    )

    alignment = row.get(
        "Evidence_Alignment"
    )

    action = row.get(
        "Strategic_Action"
    )

    score = row.get(
        "Opportunity_Score"
    )

    if (
        alignment
        ==
        "Convergent Positive"
    ):

        interpretation = (
            f"{dma} shows directionally consistent positive evidence "
            "across the BSTS intervention analysis and DMA spend-response "
            "model. This market represents a comparatively stronger "
            "candidate for controlled incremental investment."
        )

    elif (
        alignment
        ==
        "Convergent Negative"
    ):

        interpretation = (
            f"{dma} shows directionally consistent negative evidence "
            "across both the intervention and spend-response analyses. "
            "Additional investment should be approached cautiously and "
            "existing allocation should be reassessed."
        )

    elif (
        alignment
        ==
        "Attribution Conflict"
    ):

        interpretation = (
            f"{dma} shows conflicting evidence between intervention-period "
            "performance and the estimated spend-response relationship. "
            "The counterfactual gap should not be attributed directly to "
            "marketing spend without additional market-level diagnosis."
        )

    else:

        interpretation = (
            f"{dma} produces mixed or limited evidence regarding incremental "
            "marketing opportunity. Maintain disciplined testing while "
            "additional evidence is developed."
        )

    interpretation += (
        f" Current opportunity score is {score:.1f}/100 "
        f"with a recommended action of {action}."
        if pd.notna(
            score
        )
        else ""
    )

    return interpretation


# =============================================================================
# BUILD FINAL OPPORTUNITY TABLE
# =============================================================================

def build_dma_opportunity_table(
    panel,
):
    """
    Apply scoring, classifications, strategic recommendations,
    and spend-testing ranges.
    """

    logger.info("=" * 70)
    logger.info(
        "BUILDING FINAL DMA OPPORTUNITY TABLE"
    )
    logger.info("=" * 70)

    opportunity = (
        calculate_component_scores(
            panel
        )
    )

    opportunity = (
        calculate_market_scale_score(
            opportunity
        )
    )

    opportunity = (
        calculate_opportunity_score(
            opportunity
        )
    )

    classification = (
        opportunity
        .apply(
            classify_dma_opportunity,
            axis=1,
        )
    )

    opportunity = pd.concat(
        [
            opportunity,
            classification,
        ],
        axis=1,
    )

    opportunity[
        "Strategic_Action"
    ] = (
        opportunity
        .apply(
            assign_strategic_action,
            axis=1,
        )
    )

    spend_bucket = (
        opportunity
        .apply(
            assign_spend_bucket,
            axis=1,
        )
    )

    opportunity = pd.concat(
        [
            opportunity,
            spend_bucket,
        ],
        axis=1,
    )

    opportunity = (
        calculate_spend_range(
            opportunity
        )
    )

    opportunity[
        "Strategic_Interpretation"
    ] = (
        opportunity
        .apply(
            build_dma_interpretation,
            axis=1,
        )
    )

    opportunity = (
        opportunity
        .sort_values(
            "Opportunity_Score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    opportunity[
        "Opportunity_Rank"
    ] = (
        np.arange(
            1,
            len(opportunity) + 1,
        )
    )

    # Put rank near the beginning.
    columns = (
        opportunity
        .columns
        .tolist()
    )

    columns.remove(
        "Opportunity_Rank"
    )

    columns.insert(
        0,
        "Opportunity_Rank",
    )

    opportunity = opportunity[
        columns
    ]

    logger.info(
        "DMA opportunities scored: %s",
        f"{len(opportunity):,}",
    )

    logger.info(
        "Top DMA opportunities:\n%s",
        opportunity[
            [
                "Opportunity_Rank",
                "DMA",
                "Performance_Score",
                "Responsiveness_Score",
                "Confidence_Score",
                "Opportunity_Score",
                "Evidence_Alignment",
                "Strategic_Action",
            ]
        ]
        .head(
            10
        )
        .to_string(
            index=False
        ),
    )

    return opportunity


# =============================================================================
# BUILD All-DMA OPPORTUNITY TABLE
# =============================================================================
def build_all_dma_opportunity_panel(
    regression,
    investment,
):
    """
    Merge all-DMA regression evidence with DMA investment
    and market-scale information.
    """

    logger.info("=" * 70)
    logger.info(
        "BUILDING ALL-DMA OPPORTUNITY PANEL"
    )
    logger.info("=" * 70)

    regression_columns = [

        "DMA",
        "DMA_Key",

        "Spend_Partial_R2",
        "Spend_Beta",
        "Spend_P_Value",

        "Spend_Explanatory_Strength",
        "Spend_Direction",
        "Spend_Statistical_Evidence",
        "Spend_Estimate_Stability",
        "Influence_Severity",
        "Spend_Relationship_Classification",
    ]

    regression_panel = regression[
        [
            col
            for col in regression_columns
            if col in regression.columns
        ]
    ].copy()

    panel = regression_panel.merge(
        investment,
        on="DMA_Key",
        how="left",
        validate="one_to_one",
    )

    logger.info(
        "All-DMA opportunity rows: %s",
        f"{len(panel):,}",
    )

    return panel

def classify_all_dma_opportunity(
    row,
):
    """
    Classify all-DMA analytical opportunity.
    """

    score = row.get(
        "All_DMA_Opportunity_Score"
    )

    confidence = row.get(
        "Regression_Confidence_Score"
    )

    beta = row.get(
        "Spend_Beta"
    )

    partial_r2 = row.get(
        "Spend_Partial_R2"
    )

    if pd.isna(
        score
    ):

        return "Unavailable"

    # Meaningfully negative relationship.
    if (
        pd.notna(beta)
        and beta < 0
        and
        pd.notna(partial_r2)
        and partial_r2 >= 0.05
    ):

        return "Diagnostic"

    if score >= 75:

        return "High"

    if score >= 55:

        return "Moderate"

    if score >= 40:

        return "Low"

    return "Minimal"

def assign_all_dma_action(
    row,
):
    """
    Translate all-DMA opportunity evidence into a testing recommendation.

    This is not a causal investment recommendation because BSTS
    intervention evidence is unavailable for these markets.
    """

    tier = row.get(
        "All_DMA_Opportunity_Tier"
    )

    confidence = row.get(
        "Regression_Confidence_Score"
    )

    beta = row.get(
        "Spend_Beta"
    )

    if tier == "Diagnostic":

        return "Diagnostic"

    if (
        tier == "High"
        and
        confidence >= 65
        and
        beta > 0
    ):

        return "Priority Test"

    if (
        tier in {
            "High",
            "Moderate",
        }
        and
        confidence >= 45
        and
        beta > 0
    ):

        return "Explore Incremental Test"

    if tier in {
        "Moderate",
        "Low",
    }:

        return "Maintain / Monitor"

    return "Low Priority"

def add_bsts_opportunity_context(
    all_dma,
    treated_opportunity,
):
    """
    Add richer BSTS opportunity evidence where available.

    All_DMA_Opportunity_Score remains the common comparable score
    across the entire DMA population.
    """

    bsts_columns = [

        "DMA_Key",

        "BSTS_Relative_Lift",
        "BSTS_Probability_Positive",
        "Implied_Incremental_Comp",

        "Performance_Score",

        "Opportunity_Score",

        "Evidence_Alignment",
        "Opportunity_Tier",
        "Strategic_Action",
    ]

    bsts_context = treated_opportunity[
        [
            col
            for col in bsts_columns
            if col in treated_opportunity.columns
        ]
    ].copy()

    output = all_dma.merge(
        bsts_context,
        on="DMA_Key",
        how="left",
        validate="one_to_one",
    )

    output[
        "BSTS_Evidence_Available"
    ] = (
        output[
            "BSTS_Relative_Lift"
        ]
        .notna()
    )

    return output

# =============================================================================
# CALCULATE ALL-DMA OPPORTUNITY SCORES
# =============================================================================

def calculate_all_dma_scores(
    panel,
):
    """
    Calculate opportunity scores for the complete DMA universe.

    Unlike the BSTS-enhanced opportunity score, this score is based only on
    dimensions available consistently across all DMAs:

        45% Spend Responsiveness
        30% Regression Confidence
        25% Market Scale

    The resulting score is intended for comparative opportunity screening
    across all DMAs and should not be interpreted as a causal impact estimate.
    """

    logger.info("=" * 70)
    logger.info(
        "CALCULATING ALL-DMA OPPORTUNITY SCORES"
    )
    logger.info("=" * 70)

    panel = panel.copy()

    # -------------------------------------------------------------------------
    # Validate weights
    # -------------------------------------------------------------------------

    total_weight = (
        ALL_DMA_RESPONSIVENESS_WEIGHT
        +
        ALL_DMA_CONFIDENCE_WEIGHT
        +
        ALL_DMA_MARKET_SCALE_WEIGHT
    )

    if not np.isclose(
        total_weight,
        1.0,
    ):

        raise ValueError(
            "All-DMA opportunity weights must sum to 1.0."
        )

    # -------------------------------------------------------------------------
    # Spend responsiveness
    # -------------------------------------------------------------------------

    panel[
        "Responsiveness_Score"
    ] = panel.apply(
        calculate_responsiveness_score,
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Regression-only confidence
    # -------------------------------------------------------------------------

    panel[
        "Regression_Confidence_Score"
    ] = panel.apply(
        calculate_regression_confidence_score,
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Market scale
    # -------------------------------------------------------------------------

    panel = (
        calculate_market_scale_score(
            panel
        )
    )

    # -------------------------------------------------------------------------
    # Composite all-DMA score
    # -------------------------------------------------------------------------

    panel[
        "All_DMA_Opportunity_Score"
    ] = (

        panel[
            "Responsiveness_Score"
        ]
        *
        ALL_DMA_RESPONSIVENESS_WEIGHT

        +

        panel[
            "Regression_Confidence_Score"
        ]
        *
        ALL_DMA_CONFIDENCE_WEIGHT

        +

        panel[
            "Market_Scale_Score"
        ]
        *
        ALL_DMA_MARKET_SCALE_WEIGHT
    )

    panel[
        "All_DMA_Opportunity_Score"
    ] = (
        panel[
            "All_DMA_Opportunity_Score"
        ]
        .clip(
            lower=0,
            upper=100,
        )
    )

    # -------------------------------------------------------------------------
    # Opportunity tier
    # -------------------------------------------------------------------------

    panel[
        "All_DMA_Opportunity_Tier"
    ] = panel.apply(
        classify_all_dma_opportunity,
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Strategic action
    # -------------------------------------------------------------------------

    panel[
        "All_DMA_Strategic_Action"
    ] = panel.apply(
        assign_all_dma_action,
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Rank
    # -------------------------------------------------------------------------

    panel = (
        panel
        .sort_values(
            "All_DMA_Opportunity_Score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    panel[
        "All_DMA_Opportunity_Rank"
    ] = (
        np.arange(
            1,
            len(panel) + 1,
        )
    )

    logger.info(
        "All-DMA opportunities scored: %s",
        f"{len(panel):,}",
    )

    logger.info(
        "Mean opportunity score: %.2f",
        panel[
            "All_DMA_Opportunity_Score"
        ].mean(),
    )

    logger.info(
        "Median opportunity score: %.2f",
        panel[
            "All_DMA_Opportunity_Score"
        ].median(),
    )

    return panel


# =============================================================================
# ASSIGN ALL-DMA SPEND TEST RANGES
# =============================================================================

def assign_all_dma_spend_ranges(
    panel,
):
    """
    Assign prospective controlled-testing ranges to all-DMA opportunities.

    These ranges are exploratory testing parameters, not estimated
    optimal spend levels.
    """

    panel = panel.copy()

    panel[
        "All_DMA_Spend_Change_Lower"
    ] = 0.0

    panel[
        "All_DMA_Spend_Change_Upper"
    ] = 0.0

    for action, (
        lower,
        upper,
    ) in ALL_DMA_SPEND_BUCKETS.items():

        mask = (
            panel[
                "All_DMA_Strategic_Action"
            ]
            ==
            action
        )

        panel.loc[
            mask,
            "All_DMA_Spend_Change_Lower",
        ] = lower

        panel.loc[
            mask,
            "All_DMA_Spend_Change_Upper",
        ] = upper

    # -------------------------------------------------------------------------
    # Dollar recommendations
    # -------------------------------------------------------------------------

    panel[
        "All_DMA_Recommended_Incremental_Spend_Lower"
    ] = (
        panel[
            "Recent_Average_Spend"
        ]
        *
        panel[
            "All_DMA_Spend_Change_Lower"
        ]
    )

    panel[
        "All_DMA_Recommended_Incremental_Spend_Upper"
    ] = (
        panel[
            "Recent_Average_Spend"
        ]
        *
        panel[
            "All_DMA_Spend_Change_Upper"
        ]
    )

    panel[
        "All_DMA_Recommended_Total_Spend_Lower"
    ] = (
        panel[
            "Recent_Average_Spend"
        ]
        +
        panel[
            "All_DMA_Recommended_Incremental_Spend_Lower"
        ]
    )

    panel[
        "All_DMA_Recommended_Total_Spend_Upper"
    ] = (
        panel[
            "Recent_Average_Spend"
        ]
        +
        panel[
            "All_DMA_Recommended_Incremental_Spend_Upper"
        ]
    )

    return panel

# =============================================================================
# BUILD ALL-DMA REPORTING TABLE
# =============================================================================

def build_all_dma_reporting_table(
    panel,
):
    """
    Build concise all-DMA opportunity reporting table.
    """

    logger.info("=" * 70)
    logger.info(
        "BUILDING ALL-DMA OPPORTUNITY REPORTING TABLE"
    )
    logger.info("=" * 70)

    table = (
        panel
        .sort_values(
            "All_DMA_Opportunity_Score",
            ascending=False,
        )
        .copy()
    )

    # -------------------------------------------------------------------------
    # Spend test label
    # -------------------------------------------------------------------------

    def spend_test_range(
        row,
    ):

        lower = row.get(
            "All_DMA_Spend_Change_Lower"
        )

        upper = row.get(
            "All_DMA_Spend_Change_Upper"
        )

        if (
            pd.isna(lower)
            or
            pd.isna(upper)
        ):

            return "Unavailable"

        if (
            np.isclose(lower, 0)
            and
            np.isclose(upper, 0)
        ):

            return "Hold"

        return (
            f"{lower:+.0%} to "
            f"{upper:+.0%}"
        )

    table[
        "All_DMA_Spend_Test_Range"
    ] = table.apply(
        spend_test_range,
        axis=1,
    )

    columns = [

        "All_DMA_Opportunity_Rank",
        "DMA",

        "Spend_Partial_R2",
        "Spend_Beta",
        "Spend_P_Value",

        "Spend_Explanatory_Strength",
        "Spend_Statistical_Evidence",
        "Spend_Estimate_Stability",
        "Influence_Severity",

        "Responsiveness_Score",
        "Regression_Confidence_Score",
        "Market_Scale_Score",

        "All_DMA_Opportunity_Score",
        "All_DMA_Opportunity_Tier",
        "All_DMA_Strategic_Action",

        "Recent_Average_Spend",
        "Recent_Average_Traffic",
        "Store_Count",

        "All_DMA_Spend_Test_Range",

        "All_DMA_Recommended_Incremental_Spend_Lower",
        "All_DMA_Recommended_Incremental_Spend_Upper",
    ]

    columns = [
        column
        for column in columns
        if column in table.columns
    ]

    return table[
        columns
    ].copy()


# =============================================================================
# SAVE ANALYTICAL OUTPUTS
# =============================================================================

def save_opportunity_outputs(
    opportunity,
):
    """
    Save DMA opportunity-analysis table.
    """

    output_path = (
        TABLE_DIR
        / "dma_opportunity_analysis.csv"
    )

    opportunity.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Saved DMA opportunity analysis: %s",
        output_path,
    )

    return output_path


# =============================================================================
# BUILD ANALYSIS SUMMARY
# =============================================================================

def build_analysis_summary(
    opportunity,
):
    """
    Build executive-level summary metrics for the DMA opportunity report.

    The summary is based entirely on analytical outputs.
    No business-priority overrides are applied.
    """

    logger.info("=" * 70)
    logger.info(
        "BUILDING DMA OPPORTUNITY SUMMARY"
    )
    logger.info("=" * 70)

    if opportunity.empty:

        return {
            "dma_count": 0,
            "top_dma": None,
            "top_score": np.nan,
            "high_count": 0,
            "moderate_count": 0,
            "low_count": 0,
            "diagnostic_count": 0,
            "scale_count": 0,
            "test_increase_count": 0,
            "maintain_count": 0,
            "diagnose_count": 0,
            "reduce_count": 0,
            "convergent_positive_count": 0,
            "convergent_negative_count": 0,
            "attribution_conflict_count": 0,
            "spend_uninformative_count": 0,
            "mean_opportunity_score": np.nan,
            "median_opportunity_score": np.nan,
            "test_markets": [],
            "diagnostic_markets": [],
            "proposed_incremental_lower": 0.0,
            "proposed_incremental_upper": 0.0,
        }

    ranked = (
        opportunity
        .sort_values(
            "Opportunity_Score",
            ascending=False,
        )
        .copy()
    )

    # -------------------------------------------------------------------------
    # Opportunity tiers
    # -------------------------------------------------------------------------

    high_count = (
        ranked[
            "Opportunity_Tier"
        ]
        .eq(
            "High"
        )
        .sum()
    )

    moderate_count = (
        ranked[
            "Opportunity_Tier"
        ]
        .eq(
            "Moderate"
        )
        .sum()
    )

    low_count = (
        ranked[
            "Opportunity_Tier"
        ]
        .eq(
            "Low"
        )
        .sum()
    )

    diagnostic_count = (
        ranked[
            "Opportunity_Tier"
        ]
        .eq(
            "Diagnostic"
        )
        .sum()
    )

    # -------------------------------------------------------------------------
    # Strategic actions
    # -------------------------------------------------------------------------

    scale_count = (
        ranked[
            "Strategic_Action"
        ]
        .eq(
            "Scale"
        )
        .sum()
    )

    test_increase_count = (
        ranked[
            "Strategic_Action"
        ]
        .eq(
            "Test Increase"
        )
        .sum()
    )

    maintain_count = (
        ranked[
            "Strategic_Action"
        ]
        .eq(
            "Maintain / Optimize"
        )
        .sum()
    )

    diagnose_count = (
        ranked[
            "Strategic_Action"
        ]
        .eq(
            "Hold & Diagnose"
        )
        .sum()
    )

    reduce_count = (
        ranked[
            "Strategic_Action"
        ]
        .eq(
            "Reallocate / Reduce"
        )
        .sum()
    )

    # -------------------------------------------------------------------------
    # Evidence alignment
    # -------------------------------------------------------------------------

    convergent_positive_count = (
        ranked[
            "Evidence_Alignment"
        ]
        .eq(
            "Convergent Positive"
        )
        .sum()
    )

    convergent_negative_count = (
        ranked[
            "Evidence_Alignment"
        ]
        .eq(
            "Convergent Negative"
        )
        .sum()
    )

    attribution_conflict_count = (
        ranked[
            "Evidence_Alignment"
        ]
        .eq(
            "Attribution Conflict"
        )
        .sum()
    )

    spend_uninformative_count = (
        ranked[
            "Evidence_Alignment"
        ]
        .astype("string")
        .str.contains(
            "Spend Uninformative",
            na=False,
        )
        .sum()
    )

    # -------------------------------------------------------------------------
    # Test / diagnostic populations
    # -------------------------------------------------------------------------

    test_markets = (
        ranked.loc[
            ranked[
                "Strategic_Action"
            ]
            .isin(
                [
                    "Scale",
                    "Test Increase",
                ]
            ),
            "DMA",
        ]
        .tolist()
    )

    diagnostic_markets = (
        ranked.loc[
            ranked[
                "Strategic_Action"
            ]
            ==
            "Hold & Diagnose",
            "DMA",
        ]
        .tolist()
    )

    # -------------------------------------------------------------------------
    # Incremental testing dollars
    # -------------------------------------------------------------------------

    test_population = ranked.loc[
        ranked[
            "Strategic_Action"
        ]
        .isin(
            [
                "Scale",
                "Test Increase",
            ]
        )
    ].copy()

    proposed_incremental_lower = (
        test_population[
            "Recommended_Incremental_Spend_Lower"
        ]
        .clip(
            lower=0
        )
        .sum()
    )

    proposed_incremental_upper = (
        test_population[
            "Recommended_Incremental_Spend_Upper"
        ]
        .clip(
            lower=0
        )
        .sum()
    )

    summary = {

        "dma_count":
            ranked[
                "DMA"
            ].nunique(),

        "top_dma":
            ranked.iloc[0][
                "DMA"
            ],

        "top_score":
            ranked.iloc[0][
                "Opportunity_Score"
            ],

        "high_count":
            int(
                high_count
            ),

        "moderate_count":
            int(
                moderate_count
            ),

        "low_count":
            int(
                low_count
            ),

        "diagnostic_count":
            int(
                diagnostic_count
            ),

        "scale_count":
            int(
                scale_count
            ),

        "test_increase_count":
            int(
                test_increase_count
            ),

        "maintain_count":
            int(
                maintain_count
            ),

        "diagnose_count":
            int(
                diagnose_count
            ),

        "reduce_count":
            int(
                reduce_count
            ),

        "convergent_positive_count":
            int(
                convergent_positive_count
            ),

        "convergent_negative_count":
            int(
                convergent_negative_count
            ),

        "attribution_conflict_count":
            int(
                attribution_conflict_count
            ),

        "spend_uninformative_count":
            int(
                spend_uninformative_count
            ),

        "mean_opportunity_score":
            ranked[
                "Opportunity_Score"
            ].mean(),

        "median_opportunity_score":
            ranked[
                "Opportunity_Score"
            ].median(),

        "test_markets":
            test_markets,

        "diagnostic_markets":
            diagnostic_markets,

        "proposed_incremental_lower":
            proposed_incremental_lower,

        "proposed_incremental_upper":
            proposed_incremental_upper,
    }

    logger.info(
        "DMAs summarized: %s",
        f"{summary['dma_count']:,}",
    )

    logger.info(
        "Top analytical opportunity: %s | score %.2f",
        summary[
            "top_dma"
        ],
        summary[
            "top_score"
        ],
    )

    logger.info(
        "Test Increase / Scale markets: %s",
        f"{len(test_markets):,}",
    )

    logger.info(
        "Hold & Diagnose markets: %s",
        f"{len(diagnostic_markets):,}",
    )

    return summary


# =============================================================================
# BUILD DMA REPORTING TABLE
# =============================================================================

def build_dma_reporting_table(
    opportunity,
):
    """
    Build concise leadership-facing DMA opportunity table.

    Analytical scores and recommendations are retained exactly as
    generated by the opportunity framework.
    """

    logger.info("=" * 70)
    logger.info(
        "BUILDING DMA OPPORTUNITY REPORTING TABLE"
    )
    logger.info("=" * 70)

    table = (
        opportunity
        .sort_values(
            "Opportunity_Score",
            ascending=False,
        )
        .copy()
    )

    # -------------------------------------------------------------------------
    # Spend test range
    # -------------------------------------------------------------------------

    def format_spend_test(
        row,
    ):

        lower = row.get(
            "Spend_Change_Lower"
        )

        upper = row.get(
            "Spend_Change_Upper"
        )

        if (
            pd.isna(lower)
            or
            pd.isna(upper)
        ):

            return (
                "Unavailable"
            )

        if (
            np.isclose(
                lower,
                0
            )
            and
            np.isclose(
                upper,
                0
            )
        ):

            return (
                "Hold"
            )

        return (
            f"{lower:+.0%} to "
            f"{upper:+.0%}"
        )

    table[
        "Spend_Test_Range"
    ] = table.apply(
        format_spend_test,
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Management interpretation
    # -------------------------------------------------------------------------

    def management_interpretation(
        row,
    ):

        alignment = row.get(
            "Evidence_Alignment"
        )

        action = row.get(
            "Strategic_Action"
        )

        if (
            alignment
            ==
            "Convergent Positive"
        ):

            if (
                action
                in {
                    "Scale",
                    "Test Increase",
                }
            ):

                return (
                    "BSTS intervention performance and DMA spend-response "
                    "evidence are directionally positive. The market is a "
                    "candidate for controlled incremental investment testing."
                )

            return (
                "BSTS and spend-response evidence are directionally positive, "
                "but overall evidence strength supports maintaining or "
                "optimizing investment rather than increasing it."
            )

        if (
            alignment
            ==
            "Convergent Negative"
        ):

            return (
                "BSTS intervention performance and spend-response evidence "
                "are directionally negative. Existing investment warrants "
                "additional scrutiny before further allocation."
            )

        if (
            alignment
            ==
            "Attribution Conflict"
        ):

            return (
                "Intervention-period performance and the DMA spend-response "
                "relationship point in different directions. The BSTS "
                "counterfactual gap should not be attributed directly to "
                "incremental spend without additional diagnosis."
            )

        if (
            alignment
            ==
            "Positive BSTS / Spend Uninformative"
        ):

            return (
                "Intervention-period performance is positive, but spend "
                "explains little DMA traffic variation. The positive outcome "
                "cannot be confidently attributed to spend magnitude."
            )

        if (
            alignment
            ==
            "Negative BSTS / Spend Uninformative"
        ):

            return (
                "The DMA underperformed its BSTS counterfactual, while the "
                "spend-response model provides little evidence regarding "
                "whether spend contributed to the shortfall."
            )

        return (
            "Available evidence is mixed or insufficient for a stronger "
            "investment conclusion."
        )

    table[
        "Management_Interpretation"
    ] = (
        table
        .apply(
            management_interpretation,
            axis=1,
        )
    )

    reporting_columns = [

        "Opportunity_Rank",
        "DMA",

        "BSTS_Relative_Lift",
        "BSTS_Probability_Positive",
        "Implied_Incremental_Comp",

        "Spend_Partial_R2",
        "Spend_Beta",

        "Performance_Score",
        "Responsiveness_Score",
        "Confidence_Score",
        "Market_Scale_Score",
        "Opportunity_Score",

        "Evidence_Alignment",
        "Opportunity_Tier",
        "Strategic_Action",
        "Spend_Test_Range",

        "Recent_Average_Spend",
        "Recommended_Incremental_Spend_Lower",
        "Recommended_Incremental_Spend_Upper",

        "Management_Interpretation",
    ]

    available_columns = [
        col
        for col in reporting_columns
        if col in table.columns
    ]

    table = table[
        available_columns
    ].copy()

    logger.info(
        "Reporting table created: %s rows",
        f"{len(table):,}",
    )

    return table

# =============================================================================
# Plotting Functions
# =============================================================================

def plot_dma_opportunity_scores(
    opportunity,
):
    """
    Plot DMA opportunity score ranking.
    """

    logger.info(
        "Generating DMA opportunity score figure."
    )

    data = (
        opportunity
        .sort_values(
            "Opportunity_Score",
            ascending=True,
        )
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            7,
        )
    )

    ax.barh(
        data[
            "DMA_Key"
        ],
        data[
            "Opportunity_Score"
        ],
    )

    ax.axvline(
        55,
        linestyle="--",
        linewidth=1,
    )

    ax.axvline(
        75,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Opportunity Score"
    )

    ax.set_ylabel(
        "DMA"
    )

    ax.set_title(
        "DMA Marketing Opportunity Ranking"
    )

    ax.set_xlim(
        0,
        100,
    )

    fig.tight_layout()

    path = (
        FIGURE_DIR
        / "dma_opportunity_scores.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path

def plot_opportunity_components(
    opportunity,
):
    """
    Compare the four components of the DMA opportunity score.
    """

    logger.info(
        "Generating DMA opportunity component figure."
    )

    data = (
        opportunity
        .sort_values(
            "Opportunity_Score",
            ascending=False,
        )
        .copy()
    )

    x = np.arange(
        len(data)
    )

    width = 0.20

    fig, ax = plt.subplots(
        figsize=(
            14,
            7,
        )
    )

    ax.bar(
        x - 1.5 * width,
        data[
            "Performance_Score"
        ],
        width,
        label="Performance",
    )

    ax.bar(
        x - 0.5 * width,
        data[
            "Responsiveness_Score"
        ],
        width,
        label="Spend Responsiveness",
    )

    ax.bar(
        x + 0.5 * width,
        data[
            "Confidence_Score"
        ],
        width,
        label="Confidence",
    )

    ax.bar(
        x + 1.5 * width,
        data[
            "Market_Scale_Score"
        ],
        width,
        label="Market Scale",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        data[
            "DMA_Key"
        ],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "Component Score"
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_title(
        "DMA Opportunity Score Components"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        FIGURE_DIR
        / "dma_opportunity_components.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path

def plot_performance_vs_responsiveness(
    opportunity,
):
    """
    Compare intervention-period performance and spend responsiveness.

    Marker size represents relative market scale.
    """

    logger.info(
        "Generating performance vs responsiveness figure."
    )

    data = opportunity.copy()

    sizes = (
        60
        +
        data[
            "Market_Scale_Score"
        ]
        *
        4
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            8,
        )
    )

    ax.scatter(
        data[
            "Responsiveness_Score"
        ],
        data[
            "Performance_Score"
        ],
        s=sizes,
        alpha=0.7,
    )

    ax.axvline(
        50,
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        50,
        linestyle="--",
        linewidth=1,
    )

    for _, row in (
        data.iterrows()
    ):

        ax.annotate(
            row[
                "DMA_Key"
            ],
            (
                row[
                    "Responsiveness_Score"
                ],
                row[
                    "Performance_Score"
                ],
            ),
            xytext=(
                5,
                4,
            ),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(
        "Spend Responsiveness Score"
    )

    ax.set_ylabel(
        "BSTS Intervention Performance Score"
    )

    ax.set_title(
        "DMA Performance vs. Spend Responsiveness"
    )

    ax.set_xlim(
        0,
        100,
    )

    ax.set_ylim(
        0,
        100,
    )

    fig.tight_layout()

    path = (
        FIGURE_DIR
        / "performance_vs_responsiveness.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path


# =============================================================================
# FIGURE: ALL-DMA OPPORTUNITY RANKING
# =============================================================================

def plot_all_dma_opportunity_scores(
    panel,
):
    """
    Plot opportunity scores across the complete DMA universe.

    Higher scores indicate stronger combinations of:
        - spend responsiveness,
        - regression confidence,
        - commercial market scale.

    BSTS performance is intentionally excluded so the score remains
    directly comparable across all DMAs.
    """

    logger.info(
        "Generating all-DMA opportunity ranking figure..."
    )

    data = (
        panel
        .dropna(
            subset=[
                "All_DMA_Opportunity_Score"
            ]
        )
        .sort_values(
            "All_DMA_Opportunity_Score",
            ascending=True,
        )
        .copy()
    )

    # Dynamic height keeps all DMA labels readable.
    figure_height = max(
        8,
        len(data) * 0.38,
    )

    fig, ax = plt.subplots(
        figsize=(
            12,
            figure_height,
        )
    )

    ax.barh(
        data[
            "DMA"
        ],
        data[
            "All_DMA_Opportunity_Score"
        ],
    )

    # -------------------------------------------------------------------------
    # Opportunity thresholds
    # -------------------------------------------------------------------------

    ax.axvline(
        40,
        linestyle="--",
        linewidth=1,
    )

    ax.axvline(
        55,
        linestyle="--",
        linewidth=1,
    )

    ax.axvline(
        75,
        linestyle="--",
        linewidth=1,
    )

    # -------------------------------------------------------------------------
    # Score labels
    # -------------------------------------------------------------------------

    for index, value in enumerate(
        data[
            "All_DMA_Opportunity_Score"
        ]
    ):

        ax.text(
            value + 1,
            index,
            f"{value:.1f}",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel(
        "All-DMA Opportunity Score"
    )

    ax.set_ylabel(
        "DMA"
    )

    ax.set_title(
        "Marketing Opportunity Across All DMAs"
    )

    ax.set_xlim(
        0,
        105,
    )

    fig.tight_layout()

    output_path = (
        FIGURE_DIR
        /
        "all_dma_opportunity_scores.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    logger.info(
        "Saved all-DMA opportunity figure: %s",
        output_path,
    )

    return output_path


# =============================================================================
# FIGURE: RESPONSIVENESS VS CONFIDENCE
# =============================================================================

def plot_all_dma_responsiveness_vs_confidence(
    panel,
):
    """
    Plot spend responsiveness against regression confidence.

    Marker size reflects relative market scale.
    """

    logger.info(
        "Generating all-DMA responsiveness vs confidence figure..."
    )

    data = panel.copy()

    sizes = (
        50
        +
        data[
            "Market_Scale_Score"
        ]
        *
        4
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            8,
        )
    )

    ax.scatter(
        data[
            "Responsiveness_Score"
        ],
        data[
            "Regression_Confidence_Score"
        ],
        s=sizes,
        alpha=0.7,
    )

    ax.axvline(
        50,
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        50,
        linestyle="--",
        linewidth=1,
    )

    for _, row in (
        data.iterrows()
    ):

        ax.annotate(
            row[
                "DMA_Key"
            ],
            (
                row[
                    "Responsiveness_Score"
                ],
                row[
                    "Regression_Confidence_Score"
                ],
            ),
            xytext=(
                4,
                4,
            ),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel(
        "Spend Responsiveness Score"
    )

    ax.set_ylabel(
        "Regression Confidence Score"
    )

    ax.set_title(
        "All-DMA Opportunity: Spend Responsiveness vs. Evidence Confidence"
    )

    ax.set_xlim(
        0,
        100,
    )

    ax.set_ylim(
        0,
        100,
    )

    fig.tight_layout()

    output_path = (
        FIGURE_DIR
        /
        "all_dma_responsiveness_vs_confidence.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


def generate_figures(
    opportunity,
):
    """
    Generate DMA opportunity report figures.
    """

    logger.info("=" * 70)
    logger.info(
        "GENERATING DMA OPPORTUNITY FIGURES"
    )
    logger.info("=" * 70)

    figures = {

        "opportunity_scores":
            plot_dma_opportunity_scores(
                opportunity
            ),

        "components":
            plot_opportunity_components(
                opportunity
            ),

        "performance_vs_responsiveness":
            plot_performance_vs_responsiveness(
                opportunity
            ),
    }

    logger.info(
        "Opportunity figures generated: %s",
        f"{len(figures):,}",
    )

    return figures

# =============================================================================
# GENERATE ALL-DMA FIGURES
# =============================================================================

def generate_all_dma_figures(
    panel,
):
    """
    Generate figures for the complete DMA opportunity analysis.
    """

    logger.info("=" * 70)
    logger.info(
        "GENERATING ALL-DMA OPPORTUNITY FIGURES"
    )
    logger.info("=" * 70)

    figures = {

        "opportunity_ranking":
            plot_all_dma_opportunity_scores(
                panel
            ),

        "responsiveness_vs_confidence":
            plot_all_dma_responsiveness_vs_confidence(
                panel
            ),
    }

    logger.info(
        "All-DMA figures generated: %s",
        f"{len(figures):,}",
    )

    return figures


# =============================================================================
# GENERATE HTML REPORT
# =============================================================================

def generate_html_report(
    opportunity,
    summary,
    reporting_table,
    figures,
):
    """
    Generate leadership-facing DMA marketing opportunity report.

    The report reflects analytical results only. Business initiatives
    and market-priority decisions are intentionally left outside the
    scoring and recommendation framework.
    """

    logger.info("=" * 70)
    logger.info(
        "GENERATING DMA OPPORTUNITY HTML REPORT"
    )
    logger.info("=" * 70)

    #---------------------------------------------------------------------
    # Report path
    # ---------------------------------------------------------------------

    report_path = (
        REPORT_DIR
        / "DMA_Marketing_Opportunity_Report.html"
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

    # -------------------------------------------------------------------------
    # Strategic groups
    # -------------------------------------------------------------------------

    test_candidates = (
        opportunity.loc[
            opportunity[
                "Strategic_Action"
            ]
            .isin(
                [
                    "Scale",
                    "Test Increase",
                ]
            )
        ]
        .sort_values(
            "Opportunity_Score",
            ascending=False,
        )
    )

    maintain_markets = (
        opportunity.loc[
            opportunity[
                "Strategic_Action"
            ]
            ==
            "Maintain / Optimize"
        ]
        .sort_values(
            "Opportunity_Score",
            ascending=False,
        )
    )

    diagnostic_markets = (
        opportunity.loc[
            opportunity[
                "Strategic_Action"
            ]
            ==
            "Hold & Diagnose"
        ]
        .sort_values(
            "Opportunity_Score",
            ascending=False,
        )
    )

    # -------------------------------------------------------------------------
    # Executive narrative
    # -------------------------------------------------------------------------

    if summary[
        "test_markets"
    ]:

        test_market_text = (
            ", ".join(
                summary[
                    "test_markets"
                ]
            )
        )

        executive_takeaway = (
            f"The analytical framework identifies "
            f"{test_market_text} as the strongest current candidates "
            "for controlled incremental marketing investment testing. "
            "These recommendations reflect the combined evidence from "
            "BSTS intervention performance, DMA-level spend responsiveness, "
            "model confidence, and market scale."
        )

    else:

        executive_takeaway = (
            "No DMA currently meets the analytical threshold for "
            "controlled incremental spend testing. The current results "
            "favor maintaining, optimizing, or diagnosing existing "
            "market investments."
        )

    # -------------------------------------------------------------------------
    # Display table
    # -------------------------------------------------------------------------

    display = (
        reporting_table
        .copy()
    )

    pct_columns = [

        "BSTS_Relative_Lift",
        "BSTS_Probability_Positive",
        "Implied_Incremental_Comp",
        "Spend_Partial_R2",
    ]

    for column in pct_columns:

        if column in display.columns:

            display[
                column
            ] = (
                display[
                    column
                ]
                *
                100
            )

    display = display.rename(
        columns={

            "BSTS_Relative_Lift":
                "BSTS Lift (%)",

            "BSTS_Probability_Positive":
                "Probability Positive (%)",

            "Implied_Incremental_Comp":
                "Implied Incremental Comp (p.p.)",

            "Spend_Partial_R2":
                "Spend Partial R² (%)",

            "Spend_Beta":
                "Spend Beta",

            "Performance_Score":
                "Performance",

            "Responsiveness_Score":
                "Responsiveness",

            "Confidence_Score":
                "Confidence",

            "Market_Scale_Score":
                "Market Scale",

            "Opportunity_Score":
                "Opportunity",

            "Evidence_Alignment":
                "Evidence Alignment",

            "Opportunity_Tier":
                "Opportunity Tier",

            "Strategic_Action":
                "Model Recommendation",

            "Spend_Test_Range":
                "Suggested Spend Test",
        }
    )

    # -------------------------------------------------------------------------
    # HTML
    # -------------------------------------------------------------------------

    body = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<title>
DMA Marketing Opportunity Analysis
</title>

<style>

body {{
    font-family:
        Arial,
        Helvetica,
        sans-serif;

    margin: 0;

    background:
        #f4f6f8;

    color:
        #1f2933;

    line-height:
        1.5;
}}

.container {{
    max-width:
        1300px;

    margin:
        0 auto;

    padding:
        30px;
}}

header {{
    background:
        #102a43;

    color:
        white;

    padding:
        30px;

    border-radius:
        12px;

    margin-bottom:
        20px;
}}

h1 {{
    margin:
        0 0 8px 0;
}}

h2 {{
    margin-top:
        0;

    color:
        #243b53;
}}

section {{
    background:
        white;

    padding:
        24px;

    border-radius:
        12px;

    margin-bottom:
        20px;

    box-shadow:
        0 1px 4px rgba(0,0,0,.08);
}}

.cards {{
    display:
        grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                190px,
                1fr
            )
        );

    gap:
        12px;
}}

.card {{
    background:
        #eef2f7;

    padding:
        16px;

    border-radius:
        10px;
}}

.label {{
    font-size:
        12px;

    color:
        #627d98;
}}

.value {{
    margin-top:
        4px;

    font-size:
        24px;

    font-weight:
        700;

    color:
        #102a43;
}}

.takeaway {{
    background:
        #eef6ff;

    border-left:
        4px solid #3578c6;

    padding:
        16px 18px;

    border-radius:
        6px;

    margin-top:
        18px;
}}

table {{
    border-collapse:
        collapse;

    width:
        100%;

    font-size:
        11px;
}}

th {{
    background:
        #e9eef3;
}}

th,
td {{
    border:
        1px solid #d9e2ec;

    padding:
        6px;

    vertical-align:
        top;
}}

img {{
    max-width:
        100%;

    display:
        block;

    margin:
        20px auto;
}}

.small {{
    font-size:
        12px;

    color:
        #627d98;
}}

.methodology {{
    font-size:
        13px;

    color:
        #486581;
}}

</style>

</head>

<body>

<div class="container">

<header>

<h1>
DMA Marketing Opportunity Analysis
</h1>

<div>
Analytical decision framework combining counterfactual performance,
spend responsiveness, evidence confidence, and market scale
</div>

</header>


<section>

<h2>
Executive Summary
</h2>

<div class="cards">

<div class="card">

<div class="label">
DMAs Evaluated
</div>

<div class="value">
{summary["dma_count"]}
</div>

</div>


<div class="card">

<div class="label">
Top Analytical Opportunity
</div>

<div class="value">
{summary["top_dma"]}
</div>

</div>


<div class="card">

<div class="label">
Top Opportunity Score
</div>

<div class="value">
{summary["top_score"]:.1f}
</div>

</div>


<div class="card">

<div class="label">
Incremental Test Candidates
</div>

<div class="value">
{summary["test_increase_count"] + summary["scale_count"]}
</div>

</div>


<div class="card">

<div class="label">
Diagnostic Markets
</div>

<div class="value">
{summary["diagnose_count"]}
</div>

</div>

</div>


<div class="takeaway">

<strong>
Executive takeaway:
</strong>

{executive_takeaway}

</div>

</section>


<section>

<h2>
Opportunity Ranking
</h2>

<p>
The opportunity score combines four distinct dimensions:
intervention-period performance, spend responsiveness,
model confidence, and market scale.
</p>

<img
    src="{figure_paths['opportunity_scores']}"
>

</section>


<section>

<h2>
Opportunity Drivers
</h2>

<p>
The component view illustrates why markets with similar BSTS results
may receive different opportunity rankings. A strong counterfactual
result alone is insufficient if spend responsiveness, confidence,
or commercial scale is limited.
</p>

<img
    src="{figure_paths['components']}"
>

</section>


<section>

<h2>
Performance vs. Spend Responsiveness
</h2>

<p>
The vertical axis reflects intervention-period performance relative
to the BSTS counterfactual. The horizontal axis reflects evidence
that marketing spend explains traffic variation within each DMA.
Marker size reflects relative market scale.
</p>

<img
    src="{figure_paths['performance_vs_responsiveness']}"
>

</section>


<section>

<h2>
Controlled Investment Test Candidates
</h2>

<p>
Markets listed below meet the current analytical criteria for
either a controlled incremental investment test or scaling.
Suggested ranges are testing parameters rather than model-estimated
optimal budgets.
</p>

{test_candidates[
    [
        "DMA",
        "Opportunity_Score",
        "Confidence_Score",
        "Strategic_Action",
        "Recent_Average_Spend",
        "Recommended_Incremental_Spend_Lower",
        "Recommended_Incremental_Spend_Upper",
    ]
].to_html(
    index=False,
    float_format=lambda x: f"{x:,.1f}",
)}

<p class="small">

Combined suggested incremental monthly testing range:
<strong>
${summary["proposed_incremental_lower"]:,.0f}
to
${summary["proposed_incremental_upper"]:,.0f}
</strong>.

</p>

</section>


<section>

<h2>
Maintain / Optimize Markets
</h2>

<p>
These markets show some favorable evidence but do not currently meet
the analytical threshold for incremental investment testing.
</p>

{maintain_markets[
    [
        "DMA",
        "Opportunity_Score",
        "Evidence_Alignment",
        "Strategic_Action",
    ]
].to_html(
    index=False,
    float_format=lambda x: f"{x:.1f}",
)}

</section>


<section>

<h2>
Markets Requiring Additional Diagnosis
</h2>

<p>
A diagnostic classification does not imply that incremental marketing
spend caused underperformance. In many cases, BSTS intervention-period
performance conflicts with a positive historical spend-response
relationship. These markets warrant investigation before changing
allocation solely on the basis of the counterfactual estimate.
</p>

{diagnostic_markets[
    [
        "DMA",
        "BSTS_Relative_Lift",
        "Implied_Incremental_Comp",
        "Spend_Partial_R2",
        "Spend_Beta",
        "Evidence_Alignment",
    ]
].to_html(
    index=False,
    float_format=lambda x: f"{x:.3f}",
)}

</section>


<section>

<h2>
DMA-Level Analytical Results
</h2>

{display.to_html(
    index=False,
    float_format=lambda x: f"{x:.1f}",
)}

</section>


<section class="methodology">

<h2>
How to Interpret the Framework
</h2>

<p>
<strong>
BSTS intervention performance:
</strong>

Measures whether observed traffic performed above or below the
estimated no-intervention counterfactual. The result is an
intervention-period performance estimate and should not automatically
be interpreted as the direct effect of marketing spend alone.
</p>

<p>
<strong>
Spend responsiveness:
</strong>

Measures the incremental explanatory contribution and direction of
marketing spend within the DMA-level regression. Spend Partial R²
is an explanatory-power statistic and is not an ROI measure.
</p>

<p>
<strong>
Confidence:
</strong>

Reflects the strength and robustness of the underlying evidence,
including BSTS directional probability and DMA regression stability.
</p>

<p>
<strong>
Market scale:
</strong>

Accounts for relative traffic volume and store footprint so that
commercial upside is considered without allowing the largest DMAs
to dominate the ranking solely because of size.
</p>

<p>
<strong>
Opportunity score:
</strong>

Combines the four dimensions into a transparent decision-support
metric. It is not itself a causal estimate.
</p>

<p>
<strong>
Suggested spend ranges:
</strong>

Represent controlled testing ranges intended to generate additional
evidence. They should not be interpreted as statistically optimized
budget levels.
</p>

</section>


</div>

</body>

</html>
"""

    report_path = (
        REPORT_DIR
        / "DMA_Marketing_Opportunity_Report.html"
    )

    report_path.write_text(
        body,
        encoding="utf-8",
    )

    logger.info(
        "DMA opportunity report written to: %s",
        report_path,
    )

    return report_path

# =============================================================================
# MAIN
# =============================================================================

def main():

    logger.info("")
    logger.info("=" * 80)
    logger.info(
        "DMA MARKETING OPPORTUNITY ANALYSIS"
    )
    logger.info("=" * 80)

    try:

        # =====================================================================
        # 0. INITIALIZATION
        # =====================================================================

        create_output_directories()

        logger.info(
            "Performance weight: %.0f%%",
            PERFORMANCE_WEIGHT * 100,
        )

        logger.info(
            "Responsiveness weight: %.0f%%",
            RESPONSIVENESS_WEIGHT * 100,
        )

        logger.info(
            "Confidence weight: %.0f%%",
            CONFIDENCE_WEIGHT * 100,
        )

        logger.info(
            "Market scale weight: %.0f%%",
            MARKET_SCALE_WEIGHT * 100,
        )

        # =====================================================================
        # 1. LOAD BSTS / COMP RESULTS
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 1: LOADING BSTS / COMP RESULTS"
        )
        logger.info("-" * 80)

        bsts = load_bsts_comp_results()

        logger.info(
            "BSTS / comp DMAs loaded: %s",
            f"{bsts['DMA_Key'].nunique():,}",
        )

        # =====================================================================
        # 2. LOAD DMA SPEND REGRESSION RESULTS
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 2: LOADING DMA SPEND REGRESSION RESULTS"
        )
        logger.info("-" * 80)

        regression = (
            load_dma_regression_results()
        )

        logger.info(
            "DMA regression DMAs loaded: %s",
            f"{regression['DMA_Key'].nunique():,}",
        )

        # =====================================================================
        # 2a. Check DMA Coverage
        # =====================================================================
        
        dma_coverage = diagnose_dma_source_coverage(
            bsts=bsts,
            regression=regression,
        )

        # =====================================================================
        # 3. LOAD MODEL PANEL
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 3: LOADING MODEL PANEL"
        )
        logger.info("-" * 80)

        model_panel = (
            load_model_panel()
        )

        logger.info(
            "Model-panel rows: %s",
            f"{len(model_panel):,}",
        )

        logger.info(
            "Model-panel DMAs: %s",
            f"{model_panel['DMA_Key'].nunique():,}",
        )

        # =====================================================================
        # 4. BUILD INVESTMENT PROFILE
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 4: BUILDING DMA INVESTMENT PROFILE"
        )
        logger.info("=" * 80)

        investment = (
            build_investment_profile(
                model_panel,
                recent_months=3,
            )
        )

        logger.info(
            "Investment profiles created: %s",
            f"{len(investment):,}",
        )

        # =====================================================================
        # 5. BUILD ANALYTICAL OPPORTUNITY PANEL
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 5: BUILDING DMA OPPORTUNITY PANEL"
        )
        logger.info("=" * 80)

        opportunity_panel = (
            build_opportunity_panel(
                bsts=bsts,
                regression=regression,
                investment=investment,
            )
        )

        if opportunity_panel.empty:

            raise ValueError(
                "DMA opportunity panel is empty."
            )

        logger.info(
            "Opportunity-panel rows: %s",
            f"{len(opportunity_panel):,}",
        )

        # =====================================================================
        # 6. SOURCE COVERAGE DIAGNOSTICS
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 6: SOURCE COVERAGE DIAGNOSTICS"
        )
        logger.info("-" * 80)

        missing_regression = (
            opportunity_panel[
                "Spend_Partial_R2"
            ]
            .isna()
            .sum()
        )

        missing_investment = (
            opportunity_panel[
                "Recent_Average_Spend"
            ]
            .isna()
            .sum()
        )

        missing_traffic = (
            opportunity_panel[
                "Recent_Average_Traffic"
            ]
            .isna()
            .sum()
        )

        missing_store_count = (
            opportunity_panel[
                "Store_Count"
            ]
            .isna()
            .sum()
        )

        logger.info(
            "Missing regression results: %s",
            f"{missing_regression:,}",
        )

        logger.info(
            "Missing investment profiles: %s",
            f"{missing_investment:,}",
        )

        logger.info(
            "Missing recent traffic: %s",
            f"{missing_traffic:,}",
        )

        logger.info(
            "Missing store count: %s",
            f"{missing_store_count:,}",
        )

        # ---------------------------------------------------------------------
        # Do not silently score incomplete DMAs.
        # ---------------------------------------------------------------------

        required_score_columns = [
            "BSTS_Relative_Lift",
            "BSTS_Probability_Positive",
            "Implied_Incremental_Comp",
            "Spend_Partial_R2",
            "Spend_Beta",
            "Recent_Average_Spend",
            "Recent_Average_Traffic",
            "Store_Count",
        ]

        opportunity_panel[
            "Opportunity_Input_Complete"
        ] = (
            opportunity_panel[
                required_score_columns
            ]
            .notna()
            .all(
                axis=1
            )
        )

        incomplete = (
            opportunity_panel.loc[
                ~opportunity_panel[
                    "Opportunity_Input_Complete"
                ]
            ]
        )

        if not incomplete.empty:

            logger.warning(
                "DMAs with incomplete opportunity inputs: %s",
                f"{len(incomplete):,}",
            )

            logger.warning(
                "\n%s",
                incomplete[
                    [
                        "DMA",
                        *required_score_columns,
                    ]
                ]
                .to_string(
                    index=False
                ),
            )

        scoring_panel = (
            opportunity_panel.loc[
                opportunity_panel[
                    "Opportunity_Input_Complete"
                ]
            ]
            .copy()
        )

        if scoring_panel.empty:

            raise ValueError(
                "No DMAs have complete inputs for "
                "opportunity scoring."
            )

        logger.info(
            "DMAs eligible for opportunity scoring: %s",
            f"{len(scoring_panel):,}",
        )

        # =====================================================================
        # 7. BUILD FINAL DMA OPPORTUNITY TABLE
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 7: SCORING DMA OPPORTUNITIES"
        )
        logger.info("=" * 80)

        opportunity = (
            build_dma_opportunity_table(
                scoring_panel
            )
        )

        if opportunity.empty:

            raise ValueError(
                "DMA opportunity scoring returned no results."
            )

        # =====================================================================
        # 8. RANKING DIAGNOSTICS
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 8: REVIEWING OPPORTUNITY RANKINGS"
        )
        logger.info("=" * 80)

        diagnostic_columns = [
            "Opportunity_Rank",
            "DMA",
            "Performance_Score",
            "Responsiveness_Score",
            "Confidence_Score",
            "Market_Scale_Score",
            "Opportunity_Score",
            "Evidence_Alignment",
            "Opportunity_Tier",
            "Strategic_Action",
            "Recent_Average_Spend",
            "Spend_Change_Lower",
            "Spend_Change_Upper",
        ]

        logger.info(
            "DMA opportunity ranking:\n%s",
            opportunity[
                diagnostic_columns
            ]
            .to_string(
                index=False
            ),
        )

        # =====================================================================
        # 9. STRATEGIC ACTION SUMMARY
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 9: STRATEGIC ACTION SUMMARY"
        )
        logger.info("-" * 80)

        action_summary = (
            opportunity
            .groupby(
                "Strategic_Action",
                as_index=False,
            )
            .agg(
                DMAs=(
                    "DMA",
                    "nunique",
                ),
                Mean_Opportunity_Score=(
                    "Opportunity_Score",
                    "mean",
                ),
                Current_Monthly_Spend=(
                    "Recent_Average_Spend",
                    "sum",
                ),
                Recommended_Incremental_Lower=(
                    "Recommended_Incremental_Spend_Lower",
                    "sum",
                ),
                Recommended_Incremental_Upper=(
                    "Recommended_Incremental_Spend_Upper",
                    "sum",
                ),
            )
            .sort_values(
                "Mean_Opportunity_Score",
                ascending=False,
            )
        )

        logger.info(
            "Strategic action summary:\n%s",
            action_summary.to_string(
                index=False
            ),
        )

        # =====================================================================
        # 10. EVIDENCE ALIGNMENT SUMMARY
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 10: EVIDENCE ALIGNMENT SUMMARY"
        )
        logger.info("-" * 80)

        alignment_summary = (
            opportunity
            .groupby(
                "Evidence_Alignment",
                as_index=False,
            )
            .agg(
                DMAs=(
                    "DMA",
                    "nunique",
                ),
                Mean_Opportunity_Score=(
                    "Opportunity_Score",
                    "mean",
                ),
                Mean_Performance_Score=(
                    "Performance_Score",
                    "mean",
                ),
                Mean_Responsiveness_Score=(
                    "Responsiveness_Score",
                    "mean",
                ),
                Mean_Confidence_Score=(
                    "Confidence_Score",
                    "mean",
                ),
            )
            .sort_values(
                "Mean_Opportunity_Score",
                ascending=False,
            )
        )

        logger.info(
            "Evidence alignment summary:\n%s",
            alignment_summary.to_string(
                index=False
            ),
        )

        # =====================================================================
        # 11. SAVE OUTPUTS
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 11: SAVING DMA OPPORTUNITY OUTPUTS"
        )
        logger.info("=" * 80)

        opportunity_path = (
            save_opportunity_outputs(
                opportunity
            )
        )

        action_summary.to_csv(
            TABLE_DIR
            / "dma_opportunity_action_summary.csv",
            index=False,
        )

        alignment_summary.to_csv(
            TABLE_DIR
            / "dma_opportunity_alignment_summary.csv",
            index=False,
        )

        investment.to_csv(
            TABLE_DIR
            / "dma_investment_profile.csv",
            index=False,
        )

        opportunity_panel.to_csv(
            TABLE_DIR
            / "dma_opportunity_input_panel.csv",
            index=False,
        )

        logger.info(
            "Opportunity outputs saved."
        )


        # =====================================================================
        # 12. BUILD REPORT SUMMARY
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 12: BUILDING ANALYSIS SUMMARY"
        )
        logger.info("=" * 80)

        summary = (
            build_analysis_summary(
                opportunity
            )
        )


        # =====================================================================
        # 13. BUILD REPORTING TABLE
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 13: BUILDING DMA REPORTING TABLE"
        )
        logger.info("=" * 80)

        reporting_table = (
            build_dma_reporting_table(
                opportunity
            )
        )

        reporting_table.to_csv(
            TABLE_DIR
            / "dma_opportunity_reporting_table.csv",
            index=False,
        )


        # =====================================================================
        # 14. GENERATE FIGURES
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 14: GENERATING OPPORTUNITY FIGURES"
        )
        logger.info("=" * 80)

        figures = (
            generate_figures(
                opportunity
            )
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

        report_path = (
            generate_html_report(
                opportunity=opportunity,
                summary=summary,
                reporting_table=reporting_table,
                figures=figures,
            )
        )

        logger.info(
            "HTML report generated: %s",
            report_path,
        )

        # =====================================================================
        # 16. LOAD COMPLETE DMA REGRESSION RESULTS
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 16: LOADING COMPLETE DMA REGRESSION RESULTS"
        )
        logger.info("=" * 80)

        all_dma_regression = (
            load_all_dma_regression_results()
        )

        logger.info(
            "Successful DMA regression models loaded: %s",
            f"{len(all_dma_regression):,}",
        )


        # =====================================================================
        # 17. BUILD ALL-DMA OPPORTUNITY PANEL
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 17: BUILDING ALL-DMA OPPORTUNITY PANEL"
        )
        logger.info("=" * 80)

        all_dma_opportunity = (
            build_all_dma_opportunity_panel(
                regression=all_dma_regression,
                investment=investment,
            )
        )

        logger.info(
            "All-DMA opportunity panel rows: %s",
            f"{len(all_dma_opportunity):,}",
        )


        # =====================================================================
        # 18. CALCULATE ALL-DMA OPPORTUNITY SCORES
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 18: CALCULATING ALL-DMA OPPORTUNITY SCORES"
        )
        logger.info("=" * 80)

        all_dma_opportunity = (
            calculate_all_dma_scores(
                all_dma_opportunity
            )
        )


        # =====================================================================
        # 19. ASSIGN ALL-DMA TEST RANGES
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 19: ASSIGNING ALL-DMA TEST RANGES"
        )
        logger.info("=" * 80)

        all_dma_opportunity = (
            assign_all_dma_spend_ranges(
                all_dma_opportunity
            )
        )


        # =====================================================================
        # 20. ADD BSTS CONTEXT WHERE AVAILABLE
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 20: ADDING BSTS CONTEXT TO ALL-DMA RESULTS"
        )
        logger.info("=" * 80)

        all_dma_opportunity = (
            add_bsts_opportunity_context(
                all_dma=all_dma_opportunity,
                treated_opportunity=opportunity,
            )
        )

        logger.info(
            "DMAs with BSTS evidence: %s",
            f"{all_dma_opportunity['BSTS_Evidence_Available'].sum():,}",
        )

        logger.info(
            "DMAs without BSTS evidence: %s",
            f"{(~all_dma_opportunity['BSTS_Evidence_Available']).sum():,}",
        )


        # =====================================================================
        # 21. BUILD ALL-DMA REPORTING TABLE
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 21: BUILDING ALL-DMA REPORTING TABLE"
        )
        logger.info("=" * 80)

        all_dma_reporting_table = (
            build_all_dma_reporting_table(
                all_dma_opportunity
            )
        )


        # =====================================================================
        # 22. GENERATE ALL-DMA FIGURES
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 22: GENERATING ALL-DMA OPPORTUNITY FIGURES"
        )
        logger.info("=" * 80)

        all_dma_figures = (
            generate_all_dma_figures(
                all_dma_opportunity
            )
        )


        # =====================================================================
        # 23. EXPORT ALL-DMA RESULTS
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 23: EXPORTING ALL-DMA OPPORTUNITY RESULTS"
        )
        logger.info("=" * 80)

        all_dma_output_path = (
            TABLE_DIR
            /
            "all_dma_opportunity_analysis.csv"
        )

        all_dma_opportunity.to_csv(
            all_dma_output_path,
            index=False,
        )

        logger.info(
            "Saved full all-DMA opportunity analysis: %s",
            all_dma_output_path,
        )


        all_dma_reporting_path = (
            TABLE_DIR
            /
            "all_dma_opportunity_reporting_table.csv"
        )

        all_dma_reporting_table.to_csv(
            all_dma_reporting_path,
            index=False,
        )

        logger.info(
            "Saved all-DMA reporting table: %s",
            all_dma_reporting_path,
        )

        # =====================================================================
        # 16. FINAL SUMMARY
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "DMA MARKETING OPPORTUNITY ANALYSIS COMPLETE"
        )
        logger.info("=" * 80)

        logger.info(
            "BSTS intervention DMAs: %s",
            f"{bsts['DMA_Key'].nunique():,}",
        )

        logger.info(
            "DMAs scored: %s",
            f"{len(opportunity):,}",
        )

        logger.info(
            "Mean opportunity score: %.2f",
            opportunity[
                "Opportunity_Score"
            ].mean(),
        )

        logger.info(
            "Median opportunity score: %.2f",
            opportunity[
                "Opportunity_Score"
            ].median(),
        )

        logger.info(
            "High opportunities: %s",
            f"{(opportunity['Opportunity_Tier'] == 'High').sum():,}",
        )

        logger.info(
            "Moderate opportunities: %s",
            f"{(opportunity['Opportunity_Tier'] == 'Moderate').sum():,}",
        )

        logger.info(
            "Low opportunities: %s",
            f"{(opportunity['Opportunity_Tier'] == 'Low').sum():,}",
        )

        logger.info(
            "Diagnostic markets: %s",
            f"{(opportunity['Opportunity_Tier'] == 'Diagnostic').sum():,}",
        )

        logger.info(
            "Convergent positive markets: %s",
            f"{(opportunity['Evidence_Alignment'] == 'Convergent Positive').sum():,}",
        )

        logger.info(
            "Attribution-conflict markets: %s",
            f"{(opportunity['Evidence_Alignment'] == 'Attribution Conflict').sum():,}",
        )

        logger.info(
            "Opportunity output: %s",
            opportunity_path,
        )

        logger.info("=" * 80)
        logger.info("SUCCESS")
        logger.info("=" * 80)

    except Exception:

        logger.exception(
            "DMA MARKETING OPPORTUNITY ANALYSIS FAILED."
        )

        raise


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()

