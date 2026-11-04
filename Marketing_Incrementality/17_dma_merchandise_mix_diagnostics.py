"""
17_dma_merchandise_mix_diagnostics.py

DMA Merchandise Mix Diagnostics
--------------------------------

Purpose
-------
Evaluate whether differences in merchandise composition
provide a plausible explanation for heterogeneous traffic
and marketing performance across DMAs.

This analysis is intended as a diagnostic extension of the
marketing incrementality workflow rather than an additional
causal model.

Primary Questions
-----------------
1. How materially does merchandise mix vary across DMAs?
2. Which DMAs have similar merchandise profiles?
3. Do DMAs with similar merchandise mixes exhibit materially
   different traffic and marketing outcomes?
4. Can merchandise composition plausibly explain observed
   DMA-level performance heterogeneity?

Primary Universe
----------------
Treated / BSTS DMAs used in the downstream marketing analysis.

Reference Universe
------------------
All DMAs with sufficient merchandise coverage.

Primary Grain
-------------
DMA x Merchandise Group

Outputs
-------
DMA merchandise profiles
Network merchandise benchmarks
DMA merchandise deviations
DMA similarity diagnostics
Management-facing figures and tables
"""


# =============================================================================
# IMPORTS
# =============================================================================

import logging

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    DATA_DIR,
    INTERIM_DATA_DIR,
    OUTPUT_DIR,
    STYLE,
)


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

TABLE_DIR = (
    OUTPUT_DIR
    / "tables"
    / "dma_merchandise_mix"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
    / "dma_merchandise_mix"
)

REPORT_DIR = (
    OUTPUT_DIR
    / "reports"
)


# -----------------------------------------------------------------------------
# Core Merchandise Source
# -----------------------------------------------------------------------------

MERCHANDISE_FILE = (
    INTERIM_DATA_DIR
    / "merchandise_dma_group_quarter.parquet"
)


# -----------------------------------------------------------------------------
# Downstream Marketing Results
# -----------------------------------------------------------------------------

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

DMA_OPPORTUNITY_FILE = (
    OUTPUT_DIR
    / "tables"
    / "dma_opportunity"
    / "dma_opportunity_analysis.csv"
)

ALL_DMA_OPPORTUNITY_FILE = (
    OUTPUT_DIR
    / "tables"
    / "dma_opportunity"
    / "all_dma_opportunity_analysis.csv"
)

MODEL_PANEL_FILE = (
    DATA_DIR
    / "processed"
    / "model_panel.parquet"
)


# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

CORE_MERCHANDISE_GROUPS = [

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

REFERENCE_MERCHANDISE_GROUPS = [

    "Central Balance",
    "Leased",
    "Unassigned dB Group",

]

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
# LOAD MERCHANDISE DATA
# =============================================================================

def load_merchandise_data(
    path=None,
):
    """
    Load quarterly DMA x merchandise-group data.

    DMA names are normalized to the downstream marketing
    analysis convention using DMA_Key.
    """

    path = Path(
        path or MERCHANDISE_FILE
    )

    logger.info("=" * 70)
    logger.info(
        "LOADING MERCHANDISE DATA"
    )
    logger.info("=" * 70)

    if not path.exists():

        raise FileNotFoundError(
            f"Merchandise file does not exist: {path}"
        )

    data = pd.read_parquet(
        path
    )

    required_columns = {

        "DMA",
        "Merch_Group",
        "Quarter_Period",
        "Group_Sales_Dollars",
        "Group_Sales_Units",
        "Group_Dollar_Share",
        "Group_Unit_Share",

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
            "Merchandise data is missing required columns: "
            f"{sorted(missing)}"
        )

    # -------------------------------------------------------------------------
    # Normalize DMA
    # -------------------------------------------------------------------------

    data["DMA_Key"] = (
        normalize_dma_key(
            data["DMA"]
        )
    )

    # -------------------------------------------------------------------------
    # Dates
    # -------------------------------------------------------------------------

    data["Quarter_Period"] = pd.to_datetime(
        data["Quarter_Period"],
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Numeric Fields
    # -------------------------------------------------------------------------

    numeric_columns = [

        "Group_Sales_Dollars",
        "Group_Sales_Units",
        "Group_Dollar_Share",
        "Group_Unit_Share",
        "Group_Stores",
        "Group_AUR",
        "Group_Sales_per_Store",
        "Group_Units_per_Store",

    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    # -------------------------------------------------------------------------
    # Basic QA
    # -------------------------------------------------------------------------

    logger.info(
        "Merchandise rows: %s",
        f"{len(data):,}",
    )

    logger.info(
        "DMAs: %s",
        f"{data['DMA_Key'].nunique():,}",
    )

    logger.info(
        "Merchandise groups: %s",
        f"{data['Merch_Group'].nunique():,}",
    )

    logger.info(
        "Quarters: %s",
        f"{data['Quarter_Period'].nunique():,}",
    )

    logger.info(
        "Quarter range: %s -> %s",
        data["Quarter_Period"].min(),
        data["Quarter_Period"].max(),
    )

    duplicate_rows = (
        data.duplicated(
            [
                "DMA_Key",
                "Quarter_Period",
                "Merch_Group",
            ],
            keep=False,
        )
        .sum()
    )

    logger.info(
        "Duplicate DMA-quarter-group rows: %s",
        f"{duplicate_rows:,}",
    )

    if duplicate_rows > 0:

        raise ValueError(
            "Merchandise data contains duplicate "
            "DMA-quarter-group observations."
        )

    missing_dma = (
        data["DMA_Key"]
        .isna()
        .sum()
    )

    logger.info(
        "Rows missing normalized DMA: %s",
        f"{missing_dma:,}",
    )

    logger.info("=" * 70)

    return data

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
# DMA SOURCE COVERAGE
# =============================================================================

def diagnose_dma_coverage(
    merchandise,
    bsts,
):
    """
    Compare merchandise coverage with the treated BSTS universe.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "DMA SOURCE COVERAGE"
    )
    logger.info("=" * 70)

    merch_dmas = set(
        merchandise[
            "DMA_Key"
        ]
        .dropna()
        .unique()
    )

    treated_dmas = set(
        bsts[
            "DMA_Key"
        ]
        .dropna()
        .unique()
    )

    matched = (
        merch_dmas
        &
        treated_dmas
    )

    missing_merchandise = (
        treated_dmas
        -
        merch_dmas
    )

    reference_only = (
        merch_dmas
        -
        treated_dmas
    )

    logger.info(
        "Merchandise DMAs        : %s",
        f"{len(merch_dmas):,}",
    )

    logger.info(
        "Treated BSTS DMAs       : %s",
        f"{len(treated_dmas):,}",
    )

    logger.info(
        "Matched treated DMAs    : %s",
        f"{len(matched):,}",
    )

    logger.info(
        "Treated missing merch   : %s",
        f"{len(missing_merchandise):,}",
    )

    logger.info(
        "Reference-only DMAs     : %s",
        f"{len(reference_only):,}",
    )

    if missing_merchandise:

        logger.warning(
            "Treated DMAs without merchandise data:\n%s",
            "\n".join(
                sorted(
                    missing_merchandise
                )
            ),
        )

    logger.info("=" * 70)

    coverage = pd.DataFrame({

        "DMA_Key":
            sorted(
                merch_dmas
                |
                treated_dmas
            )

    })

    coverage[
        "Has_Merchandise"
    ] = (
        coverage["DMA_Key"]
        .isin(
            merch_dmas
        )
    )

    coverage[
        "Is_Treated_DMA"
    ] = (
        coverage["DMA_Key"]
        .isin(
            treated_dmas
        )
    )

    coverage[
        "Primary_Analysis_Eligible"
    ] = (
        coverage[
            "Has_Merchandise"
        ]
        &
        coverage[
            "Is_Treated_DMA"
        ]
    )

    return coverage


# =============================================================================
# BUILD DMA MERCHANDISE PROFILES
# =============================================================================

def build_dma_merchandise_profiles(
    merchandise,
    treated_dmas=None,
):
    """
    Build stable DMA-level merchandise-mix profiles.

    Grain
    -----
    DMA x Merchandise Group

    For each DMA / merchandise group, calculate:

        - Average dollar share
        - Median dollar share
        - Dollar-share standard deviation
        - Minimum / maximum dollar share
        - Average unit share
        - Average AUR
        - Average sales per store
        - Number of quarters observed

    chain benchmarks are calculated using the complete
    merchandise DMA universe.

    Dollar-share deviations are expressed in percentage
    points for business-friendly interpretation.

    Parameters
    ----------
    merchandise : pd.DataFrame
        Quarterly DMA x merchandise-group dataset.

    treated_dmas : iterable, optional
        DMA_Key values belonging to the primary treated
        analysis universe.

    Returns
    -------
    profiles : pd.DataFrame
        DMA x merchandise-group profile table.

    benchmarks : pd.DataFrame
        chain-level merchandise-group benchmarks.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "BUILDING DMA MERCHANDISE PROFILES"
    )
    logger.info("=" * 70)

    df = merchandise.copy()

    # -------------------------------------------------------------------------
    # Required Columns
    # -------------------------------------------------------------------------

    required_columns = {

        "DMA_Key",
        "Merch_Group",
        "Quarter_Period",
        "Group_Dollar_Share",
        "Group_Unit_Share",

    }

    missing = (
        required_columns
        -
        set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            "Merchandise data is missing profile columns: "
            f"{sorted(missing)}"
        )

    # -------------------------------------------------------------------------
    # chain Benchmarks
    #
    # First average each DMA across time.
    #
    # Then calculate the chain benchmark across DMAs.
    #
    # This prevents DMAs with more observations from receiving
    # disproportionate weight.
    # -------------------------------------------------------------------------

    dma_group_means = (
        df
        .groupby(
            [
                "DMA_Key",
                "Merch_Group",
            ],
            as_index=False,
        )
        .agg(

            DMA_Avg_Dollar_Share=(
                "Group_Dollar_Share",
                "mean",
            ),

            DMA_Avg_Unit_Share=(
                "Group_Unit_Share",
                "mean",
            ),

        )
    )

    benchmarks = (
        dma_group_means
        .groupby(
            "Merch_Group",
            as_index=False,
        )
        .agg(

            Network_Avg_Dollar_Share=(
                "DMA_Avg_Dollar_Share",
                "mean",
            ),

            Network_Median_Dollar_Share=(
                "DMA_Avg_Dollar_Share",
                "median",
            ),

            Network_Std_Dollar_Share=(
                "DMA_Avg_Dollar_Share",
                "std",
            ),

            Network_Avg_Unit_Share=(
                "DMA_Avg_Unit_Share",
                "mean",
            ),

            Network_DMA_Count=(
                "DMA_Key",
                "nunique",
            ),

        )
    )

    # -------------------------------------------------------------------------
    # DMA Profiles
    # -------------------------------------------------------------------------

    agg_dict = {

        "Avg_Dollar_Share": (
            "Group_Dollar_Share",
            "mean",
        ),

        "Median_Dollar_Share": (
            "Group_Dollar_Share",
            "median",
        ),

        "Std_Dollar_Share": (
            "Group_Dollar_Share",
            "std",
        ),

        "Min_Dollar_Share": (
            "Group_Dollar_Share",
            "min",
        ),

        "Max_Dollar_Share": (
            "Group_Dollar_Share",
            "max",
        ),

        "Avg_Unit_Share": (
            "Group_Unit_Share",
            "mean",
        ),

        "Quarters_Observed": (
            "Quarter_Period",
            "nunique",
        ),

    }

    # Add optional descriptive metrics when available.

    if "Group_AUR" in df.columns:

        agg_dict[
            "Avg_AUR"
        ] = (
            "Group_AUR",
            "mean",
        )

    if "Group_Sales_per_Store" in df.columns:

        agg_dict[
            "Avg_Sales_per_Store"
        ] = (
            "Group_Sales_per_Store",
            "mean",
        )

    if "Group_Units_per_Store" in df.columns:

        agg_dict[
            "Avg_Units_per_Store"
        ] = (
            "Group_Units_per_Store",
            "mean",
        )

    profiles = (
        df
        .groupby(
            [
                "DMA_Key",
                "Merch_Group",
            ],
            as_index=False,
        )
        .agg(
            **agg_dict
        )
    )

    # -------------------------------------------------------------------------
    # Merge Network Benchmarks
    # -------------------------------------------------------------------------

    profiles = profiles.merge(
        benchmarks,
        on="Merch_Group",
        how="left",
        validate="many_to_one",
    )

    # -------------------------------------------------------------------------
    # Business-Friendly Percentage-Point Deviations
    # -------------------------------------------------------------------------

    profiles[
        "Dollar_Share_Pct"
    ] = (
        profiles[
            "Avg_Dollar_Share"
        ]
        * 100
    )

    profiles[
        "Network_Dollar_Share_Pct"
    ] = (
        profiles[
            "Network_Avg_Dollar_Share"
        ]
        * 100
    )

    profiles[
        "Dollar_Share_Deviation_PP"
    ] = (
        (
            profiles[
                "Avg_Dollar_Share"
            ]
            -
            profiles[
                "Network_Avg_Dollar_Share"
            ]
        )
        * 100
    )

    profiles[
        "Unit_Share_Deviation_PP"
    ] = (
        (
            profiles[
                "Avg_Unit_Share"
            ]
            -
            profiles[
                "Network_Avg_Unit_Share"
            ]
        )
        * 100
    )

    # -------------------------------------------------------------------------
    # Standardized Deviation
    #
    # Useful analytically even though PP deviation should remain
    # the primary management-facing metric.
    # -------------------------------------------------------------------------

    profiles[
        "Dollar_Share_Z"
    ] = np.where(

        profiles[
            "Network_Std_Dollar_Share"
        ]
        > 0,

        (
            profiles[
                "Avg_Dollar_Share"
            ]
            -
            profiles[
                "Network_Avg_Dollar_Share"
            ]
        )
        /
        profiles[
            "Network_Std_Dollar_Share"
        ],

        np.nan,

    )

    # -------------------------------------------------------------------------
    # Stability
    # -------------------------------------------------------------------------

    profiles[
        "Dollar_Share_Range_PP"
    ] = (
        (
            profiles[
                "Max_Dollar_Share"
            ]
            -
            profiles[
                "Min_Dollar_Share"
            ]
        )
        * 100
    )

    profiles[
        "Dollar_Share_Std_PP"
    ] = (
        profiles[
            "Std_Dollar_Share"
        ]
        * 100
    )

    # -------------------------------------------------------------------------
    # Treated Flag
    # -------------------------------------------------------------------------

    if treated_dmas is not None:

        treated_set = set(
            pd.Series(
                treated_dmas
            )
            .dropna()
            .astype(str)
        )

        profiles[
            "Is_Treated_DMA"
        ] = (
            profiles[
                "DMA_Key"
            ]
            .astype(str)
            .isin(
                treated_set
            )
        )

    else:

        profiles[
            "Is_Treated_DMA"
        ] = False

    # -------------------------------------------------------------------------
    # Core / Reference Merchandise Classification
    # -------------------------------------------------------------------------

    profiles[
        "Merchandise_Type"
    ] = np.select(

        [

            profiles[
                "Merch_Group"
            ]
            .isin(
                CORE_MERCHANDISE_GROUPS
            ),

            profiles[
                "Merch_Group"
            ]
            .isin(
                REFERENCE_MERCHANDISE_GROUPS
            ),

        ],

        [
            "Core",
            "Reference",
        ],

        default="Other",

    )

    

    # -------------------------------------------------------------------------
    # QA
    # -------------------------------------------------------------------------

    logger.info(
        "Profile rows             : %s",
        f"{len(profiles):,}",
    )

    logger.info(
        "Profile DMAs             : %s",
        f"{profiles['DMA_Key'].nunique():,}",
    )

    logger.info(
        "Merchandise groups       : %s",
        f"{profiles['Merch_Group'].nunique():,}",
    )

    logger.info(
        "Network benchmarks       : %s",
        f"{len(benchmarks):,}",
    )

    logger.info(
        "Treated profile DMAs     : %s",
        f"{profiles.loc[profiles['Is_Treated_DMA'], 'DMA_Key'].nunique():,}",
    )

    logger.info("")
    logger.info(
        "Quarter coverage:"
    )

    logger.info(
        profiles[
            "Quarters_Observed"
        ]
        .describe()
        .to_string()
    )

    logger.info("")
    logger.info(
        "Largest absolute core merchandise deviations:"
    )

    largest_deviations = (
        profiles[
            profiles[
                "Merchandise_Type"
            ]
            ==
            "Core"
        ]
        .assign(
            Abs_Deviation=lambda x:
                x[
                    "Dollar_Share_Deviation_PP"
                ]
                .abs()
        )
        .sort_values(
            "Abs_Deviation",
            ascending=False,
        )
        [
            [
                "DMA_Key",
                "Merch_Group",
                "Dollar_Share_Pct",
                "Network_Dollar_Share_Pct",
                "Dollar_Share_Deviation_PP",
                "Dollar_Share_Z",
                "Quarters_Observed",
                "Is_Treated_DMA",
            ]
        ]
        .head(20)
    )

    logger.info(
        "\n%s",
        largest_deviations.to_string(
            index=False
        ),
    )

    logger.info("=" * 70)

    return profiles, benchmarks


# =============================================================================
# BUILD DMA PROFILE SUMMARY
# =============================================================================

def build_dma_profile_summary(
    profiles,
):
    """
    Create one-row-per-DMA merchandise profile summary.

    Focuses on core merchandise groups so negligible reference
    categories do not distort merchandise-mix differentiation.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "BUILDING DMA PROFILE SUMMARY"
    )
    logger.info("=" * 70)

    core = profiles[
        profiles[
            "Merchandise_Type"
        ]
        ==
        "Core"
    ].copy()

    # -------------------------------------------------------------------------
    # Overall Mix Differentiation
    # -------------------------------------------------------------------------

    summary = (
        core
        .groupby(
            "DMA_Key",
            as_index=False,
        )
        .agg(

            Mean_Abs_Mix_Deviation_PP=(
                "Dollar_Share_Deviation_PP",
                lambda x:
                    x.abs().mean(),
            ),

            Max_Abs_Mix_Deviation_PP=(
                "Dollar_Share_Deviation_PP",
                lambda x:
                    x.abs().max(),
            ),

            Mix_Deviation_RMS_PP=(
                "Dollar_Share_Deviation_PP",
                lambda x:
                    np.sqrt(
                        np.mean(
                            np.square(x)
                        )
                    ),
            ),

            Mean_Abs_Z=(
                "Dollar_Share_Z",
                lambda x:
                    x.abs().mean(),
            ),

            Max_Abs_Z=(
                "Dollar_Share_Z",
                lambda x:
                    x.abs().max(),
            ),

            Avg_Share_Stability_PP=(
                "Dollar_Share_Std_PP",
                "mean",
            ),

            Core_Groups_Observed=(
                "Merch_Group",
                "nunique",
            ),

        )
    )

    # -------------------------------------------------------------------------
    # Most Over-Indexed Department
    # -------------------------------------------------------------------------

    over_idx = (
        core
        .sort_values(
            [
                "DMA_Key",
                "Dollar_Share_Deviation_PP",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "DMA_Key",
            as_index=False,
        )
        .first()
        [
            [
                "DMA_Key",
                "Merch_Group",
                "Dollar_Share_Deviation_PP",
            ]
        ]
        .rename(
            columns={

                "Merch_Group":
                    "Most_Overindexed_Group",

                "Dollar_Share_Deviation_PP":
                    "Most_Overindexed_PP",

            }
        )
    )

    # -------------------------------------------------------------------------
    # Most Under-Indexed Department
    # -------------------------------------------------------------------------

    under_idx = (
        core
        .sort_values(
            [
                "DMA_Key",
                "Dollar_Share_Deviation_PP",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .groupby(
            "DMA_Key",
            as_index=False,
        )
        .first()
        [
            [
                "DMA_Key",
                "Merch_Group",
                "Dollar_Share_Deviation_PP",
            ]
        ]
        .rename(
            columns={

                "Merch_Group":
                    "Most_Underindexed_Group",

                "Dollar_Share_Deviation_PP":
                    "Most_Underindexed_PP",

            }
        )
    )

    # -------------------------------------------------------------------------
    # Treated Flag
    # -------------------------------------------------------------------------

    treated = (
        profiles
        .groupby(
            "DMA_Key",
            as_index=False,
        )
        .agg(
            Is_Treated_DMA=(
                "Is_Treated_DMA",
                "max",
            )
        )
    )

    summary = (
        summary
        .merge(
            over_idx,
            on="DMA_Key",
            how="left",
            validate="one_to_one",
        )
        .merge(
            under_idx,
            on="DMA_Key",
            how="left",
            validate="one_to_one",
        )
        .merge(
            treated,
            on="DMA_Key",
            how="left",
            validate="one_to_one",
        )
    )

    # -------------------------------------------------------------------------
    # Rank Mix Differentiation
    # -------------------------------------------------------------------------

    summary[
        "Mix_Differentiation_Rank"
    ] = (
        summary[
            "Mean_Abs_Mix_Deviation_PP"
        ]
        .rank(
            ascending=False,
            method="min",
        )
    )

    logger.info(
        "DMA summaries created: %s",
        f"{len(summary):,}",
    )

    logger.info("")
    logger.info(
        "Most differentiated treated DMAs:"
    )

    logger.info(
        "\n%s",
        summary[
            summary[
                "Is_Treated_DMA"
            ]
        ]
        .sort_values(
            "Mean_Abs_Mix_Deviation_PP",
            ascending=False,
        )
        .to_string(
            index=False
        ),
    )

    logger.info("=" * 70)

    return summary


# =============================================================================
# BUILD DMA MERCHANDISE MIX MATRIX
# =============================================================================

def build_dma_merchandise_mix_matrix(
    profiles,
    treated_dmas,
):
    """
    Build a DMA x merchandise-group matrix using each DMA's
    dollar-share deviation from the network benchmark.

    Each matrix cell represents the percentage-point deviation
    between a DMA's average merchandise-group dollar share and
    the network average for that merchandise group.

    Positive values:
        DMA over-indexes versus the network.

    Negative values:
        DMA under-indexes versus the network.

    Zero:
        DMA is aligned with the network benchmark.

    The resulting matrix is intended for downstream DMA
    similarity, distance, clustering, and management-facing
    merchandise-mix diagnostics.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "BUILDING DMA MERCHANDISE MIX MATRIX"
    )
    logger.info("=" * 70)

    df = profiles.copy()

    # -------------------------------------------------------------------------
    # Validate required columns
    # -------------------------------------------------------------------------

    required_columns = {
        "DMA_Key",
        "Merch_Group",
        "Dollar_Share_Deviation_PP",
        "Is_Treated_DMA",
    }

    missing_columns = (
        required_columns
        -
        set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Merchandise profiles are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # -------------------------------------------------------------------------
    # Normalize treated DMA universe
    # -------------------------------------------------------------------------

    treated_dma_set = set(
        pd.Series(
            treated_dmas
        )
        .dropna()
        .astype(str)
        .str.strip()
    )

    logger.info(
        "Profile rows            : %s",
        f"{len(df):,}",
    )

    logger.info(
        "Profile DMAs            : %s",
        f"{df['DMA_Key'].nunique():,}",
    )

    logger.info(
        "Merchandise groups      : %s",
        f"{df['Merch_Group'].nunique():,}",
    )

    logger.info(
        "Requested treated DMAs  : %s",
        f"{len(treated_dma_set):,}",
    )

    # -------------------------------------------------------------------------
    # Validate grain
    # -------------------------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=[
            "DMA_Key",
            "Merch_Group",
        ],
        keep=False,
    )

    duplicate_count = (
        duplicate_mask.sum()
    )

    logger.info(
        "Duplicate DMA-group rows: %s",
        f"{duplicate_count:,}",
    )

    if duplicate_count > 0:

        duplicate_rows = (
            df.loc[
                duplicate_mask,
                [
                    "DMA_Key",
                    "Merch_Group",
                    "Dollar_Share_Deviation_PP",
                ],
            ]
            .sort_values(
                [
                    "DMA_Key",
                    "Merch_Group",
                ]
            )
        )

        logger.error(
            "Duplicate DMA-group rows detected:\n%s",
            duplicate_rows.to_string(
                index=False
            ),
        )

        raise ValueError(
            "Merchandise profiles must contain one row "
            "per DMA x merchandise group."
        )

    # -------------------------------------------------------------------------
    # Core merchandise groups
    #
    # Keep the primary merchandise departments for the similarity matrix.
    # Non-core / accounting-style groups can be analyzed separately.
    # -------------------------------------------------------------------------

    core_profiles = df.loc[
        df["Merchandise_Type"]
        ==
        "Core"
    ].copy()

    logger.info(
        "Core profile rows       : %s",
        f"{len(core_profiles):,}",
    )

    logger.info(
        "Core merchandise groups : %s",
        f"{core_profiles['Merch_Group'].nunique():,}",
    )

    # -------------------------------------------------------------------------
    # Build DMA x merchandise-group matrix
    # -------------------------------------------------------------------------

    mix_matrix = (
        core_profiles
        .pivot(
            index="DMA_Key",
            columns="Merch_Group",
            values="Dollar_Share_Deviation_PP",
        )
        .sort_index()
    )

    mix_matrix.columns.name = None

    # -------------------------------------------------------------------------
    # Matrix coverage
    # -------------------------------------------------------------------------

    mix_matrix["Groups_Observed"] = (
        mix_matrix
        .notna()
        .sum(
            axis=1
        )
    )

    expected_groups = (
        core_profiles[
            "Merch_Group"
        ]
        .nunique()
    )

    mix_matrix["Complete_Mix_Profile"] = (
        mix_matrix[
            "Groups_Observed"
        ]
        ==
        expected_groups
    )

    mix_matrix["Is_Treated_DMA"] = (
        mix_matrix.index
        .astype(str)
        .isin(
            treated_dma_set
        )
    )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    logger.info("")
    logger.info("-" * 70)
    logger.info(
        "DMA Merchandise Mix Matrix"
    )
    logger.info("-" * 70)

    logger.info(
        "Matrix DMAs             : %s",
        f"{len(mix_matrix):,}",
    )

    logger.info(
        "Matrix merchandise groups: %s",
        f"{expected_groups:,}",
    )

    logger.info(
        "Complete DMA profiles   : %s",
        f"{mix_matrix['Complete_Mix_Profile'].sum():,}",
    )

    logger.info(
        "Treated DMAs in matrix  : %s",
        f"{mix_matrix['Is_Treated_DMA'].sum():,}",
    )

    # -------------------------------------------------------------------------
    # Validate treated DMA coverage
    # -------------------------------------------------------------------------

    matrix_dma_set = set(
        mix_matrix.index
        .astype(str)
    )

    missing_treated_dmas = (
        treated_dma_set
        -
        matrix_dma_set
    )

    if missing_treated_dmas:

        logger.warning(
            "Treated DMAs missing from merchandise matrix: %s",
            len(
                missing_treated_dmas
            ),
        )

        for dma in sorted(
            missing_treated_dmas
        ):

            logger.warning(
                "  Missing treated DMA: %s",
                dma,
            )

    else:

        logger.info(
            "All treated DMAs are represented "
            "in the merchandise mix matrix."
        )

    logger.info("=" * 70)

    return mix_matrix


# =============================================================================
# BUILD DMA-TO-DMA MERCHANDISE DISTANCE MATRIX
# =============================================================================

def build_dma_merchandise_distance_matrix(
    mix_matrix,
):
    """
    Calculate pairwise merchandise-mix distances between DMAs.

    Distance is calculated using Euclidean distance across the
    core merchandise-group dollar-share deviations.

    Each merchandise feature is expressed in percentage points,
    so all dimensions are already on the same interpretable scale.

    Smaller distance:
        DMAs have more similar merchandise mixes.

    Larger distance:
        DMAs have more different merchandise mixes.

    Returns
    -------
    distance_matrix : DataFrame
        Square DMA x DMA Euclidean distance matrix.

    pairwise : DataFrame
        Long-form table containing each unique DMA pair and
        its merchandise-mix distance.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "BUILDING DMA-TO-DMA MERCHANDISE DISTANCE MATRIX"
    )
    logger.info("=" * 70)

    df = mix_matrix.copy()

    # -------------------------------------------------------------------------
    # Identify merchandise feature columns
    # -------------------------------------------------------------------------

    metadata_columns = {
        "Groups_Observed",
        "Complete_Mix_Profile",
        "Is_Treated_DMA",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns
    ]

    logger.info(
        "Merchandise features used: %s",
        f"{len(feature_columns):,}",
    )

    for feature in feature_columns:

        logger.info(
            "  %s",
            feature,
        )

    # -------------------------------------------------------------------------
    # Use complete profiles for pairwise distance
    # -------------------------------------------------------------------------

    complete = df.loc[
        df["Complete_Mix_Profile"]
    ].copy()

    logger.info(
        "Complete DMA profiles: %s",
        f"{len(complete):,}",
    )

    if complete.empty:

        raise ValueError(
            "No complete DMA merchandise profiles are available "
            "for distance analysis."
        )

    # -------------------------------------------------------------------------
    # Feature matrix
    # -------------------------------------------------------------------------

    X = (
        complete[
            feature_columns
        ]
        .astype(float)
    )

    if X.isna().any().any():

        raise ValueError(
            "Merchandise feature matrix contains missing values "
            "after filtering to complete profiles."
        )

    # -------------------------------------------------------------------------
    # Euclidean distance
    # -------------------------------------------------------------------------

    from scipy.spatial.distance import pdist, squareform

    distances = squareform(
        pdist(
            X.values,
            metric="euclidean",
        )
    )

    distance_matrix = pd.DataFrame(
        distances,
        index=X.index,
        columns=X.index,
    )

    distance_matrix.index.name = "DMA_Key"

    # -------------------------------------------------------------------------
    # Build long-form unique DMA pairs
    # -------------------------------------------------------------------------

    pair_records = []

    dma_names = (
        distance_matrix.index
        .tolist()
    )

    for i in range(
        len(dma_names)
    ):

        for j in range(
            i + 1,
            len(dma_names)
        ):

            dma_1 = dma_names[i]
            dma_2 = dma_names[j]

            pair_records.append(
                {
                    "DMA_1": dma_1,
                    "DMA_2": dma_2,
                    "Mix_Distance": (
                        distance_matrix.loc[
                            dma_1,
                            dma_2,
                        ]
                    ),
                }
            )

    pairwise = pd.DataFrame(
        pair_records
    )

    # -------------------------------------------------------------------------
    # Add treated status
    # -------------------------------------------------------------------------

    treated_lookup = (
        complete[
            "Is_Treated_DMA"
        ]
        .astype(bool)
        .to_dict()
    )

    pairwise["DMA_1_Treated"] = (
        pairwise["DMA_1"]
        .map(
            treated_lookup
        )
    )

    pairwise["DMA_2_Treated"] = (
        pairwise["DMA_2"]
        .map(
            treated_lookup
        )
    )

    pairwise["Pair_Type"] = np.select(
        [
            (
                pairwise["DMA_1_Treated"]
                &
                pairwise["DMA_2_Treated"]
            ),

            (
                pairwise["DMA_1_Treated"]
                |
                pairwise["DMA_2_Treated"]
            ),
        ],
        [
            "Treated vs Treated",
            "Treated vs Reference",
        ],
        default="Reference vs Reference",
    )

    # -------------------------------------------------------------------------
    # Sort most similar -> least similar
    # -------------------------------------------------------------------------

    pairwise = (
        pairwise
        .sort_values(
            "Mix_Distance"
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------------------
    # Percentile rank
    #
    # Lower distance = more similar.
    # -------------------------------------------------------------------------

    pairwise[
        "Mix_Distance_Percentile"
    ] = (
        pairwise[
            "Mix_Distance"
        ]
        .rank(
            pct=True
        )
    )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    logger.info("")
    logger.info("-" * 70)
    logger.info(
        "DMA Merchandise Distance Summary"
    )
    logger.info("-" * 70)

    logger.info(
        "DMAs compared          : %s",
        f"{len(distance_matrix):,}",
    )

    logger.info(
        "Unique DMA pairs       : %s",
        f"{len(pairwise):,}",
    )

    logger.info(
        "Treated vs Treated     : %s",
        f"{(pairwise['Pair_Type'] == 'Treated vs Treated').sum():,}",
    )

    logger.info(
        "Treated vs Reference   : %s",
        f"{(pairwise['Pair_Type'] == 'Treated vs Reference').sum():,}",
    )

    logger.info(
        "Reference vs Reference : %s",
        f"{(pairwise['Pair_Type'] == 'Reference vs Reference').sum():,}",
    )

    logger.info("")
    logger.info(
        "Distance distribution:"
    )

    logger.info(
        "\n%s",
        pairwise[
            "Mix_Distance"
        ]
        .describe()
        .to_string(),
    )

    # -------------------------------------------------------------------------
    # Most similar treated-DMA pairs
    # -------------------------------------------------------------------------

    treated_pairs = (
        pairwise.loc[
            pairwise["Pair_Type"]
            ==
            "Treated vs Treated"
        ]
        .copy()
    )

    logger.info("")
    logger.info("-" * 70)
    logger.info(
        "MOST SIMILAR TREATED DMA PAIRS"
    )
    logger.info("-" * 70)

    if not treated_pairs.empty:

        logger.info(
            "\n%s",
            treated_pairs[
                [
                    "DMA_1",
                    "DMA_2",
                    "Mix_Distance",
                    "Mix_Distance_Percentile",
                ]
            ]
            .head(15)
            .to_string(
                index=False
            ),
        )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    distance_matrix.to_csv(
        TABLE_DIR
        / "dma_merchandise_distance_matrix.csv"
    )

    pairwise.to_csv(
        TABLE_DIR
        / "dma_merchandise_pairwise_distance.csv",
        index=False,
    )

    logger.info("")
    logger.info(
        "Saved dma_merchandise_distance_matrix.csv"
    )

    logger.info(
        "Saved dma_merchandise_pairwise_distance.csv"
    )

    logger.info("=" * 70)

    return distance_matrix, pairwise


# =============================================================================
# BUILD TREATED DMA MERCHANDISE-PERFORMANCE PAIRS
# =============================================================================

def build_treated_dma_performance_pairs(
    pairwise_distances,
    bsts,
):
    """
    Combine treated-DMA merchandise similarity with observed
    marketing / traffic performance differences.

    The purpose is to identify cases where DMAs with very similar
    merchandise mixes nevertheless exhibit materially different
    traffic outcomes.

    This provides a direct diagnostic of whether merchandise mix
    appears capable of explaining heterogeneous DMA performance.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "BUILDING TREATED DMA MERCHANDISE-PERFORMANCE PAIRS"
    )
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Keep treated-to-treated comparisons
    # -------------------------------------------------------------------------

    pairs = (
        pairwise_distances.loc[
            pairwise_distances["Pair_Type"]
            ==
            "Treated vs Treated"
        ]
        .copy()
    )

    logger.info(
        "Treated DMA pairs: %s",
        f"{len(pairs):,}",
    )

    # -------------------------------------------------------------------------
    # Prepare BSTS lookup
    # -------------------------------------------------------------------------

    required_columns = {
        "DMA_Key",
        "BSTS_Relative_Lift",
        "BSTS_Probability_Positive",
        "Implied_Incremental_Comp",
    }

    missing = (
        required_columns
        -
        set(bsts.columns)
    )

    if missing:

        raise ValueError(
            "BSTS results are missing required columns: "
            f"{sorted(missing)}"
        )

    performance = (
        bsts[
            [
                "DMA_Key",
                "BSTS_Relative_Lift",
                "BSTS_Probability_Positive",
                "Implied_Incremental_Comp",
            ]
        ]
        .drop_duplicates(
            subset="DMA_Key"
        )
        .copy()
    )

    # -------------------------------------------------------------------------
    # Merge DMA 1 performance
    # -------------------------------------------------------------------------

    dma_1 = (
        performance
        .rename(
            columns={
                "DMA_Key":
                    "DMA_1",

                "BSTS_Relative_Lift":
                    "DMA_1_BSTS_Lift",

                "BSTS_Probability_Positive":
                    "DMA_1_Probability_Positive",

                "Implied_Incremental_Comp":
                    "DMA_1_Incremental_Comp",
            }
        )
    )

    pairs = pairs.merge(
        dma_1,
        on="DMA_1",
        how="left",
        validate="many_to_one",
    )

    # -------------------------------------------------------------------------
    # Merge DMA 2 performance
    # -------------------------------------------------------------------------

    dma_2 = (
        performance
        .rename(
            columns={
                "DMA_Key":
                    "DMA_2",

                "BSTS_Relative_Lift":
                    "DMA_2_BSTS_Lift",

                "BSTS_Probability_Positive":
                    "DMA_2_Probability_Positive",

                "Implied_Incremental_Comp":
                    "DMA_2_Incremental_Comp",
            }
        )
    )

    pairs = pairs.merge(
        dma_2,
        on="DMA_2",
        how="left",
        validate="many_to_one",
    )

    # -------------------------------------------------------------------------
    # Calculate performance gaps
    # -------------------------------------------------------------------------

    pairs["BSTS_Lift_Gap"] = (
        (
            pairs["DMA_1_BSTS_Lift"]
            -
            pairs["DMA_2_BSTS_Lift"]
        )
        .abs()
    )

    pairs["Incremental_Comp_Gap"] = (
        (
            pairs["DMA_1_Incremental_Comp"]
            -
            pairs["DMA_2_Incremental_Comp"]
        )
        .abs()
    )

    pairs["Probability_Positive_Gap"] = (
        (
            pairs["DMA_1_Probability_Positive"]
            -
            pairs["DMA_2_Probability_Positive"]
        )
        .abs()
    )

    # -------------------------------------------------------------------------
    # Rank merchandise similarity
    #
    # Lower merchandise distance = more similar.
    # -------------------------------------------------------------------------

    pairs["Treated_Mix_Similarity_Rank"] = (
        pairs["Mix_Distance"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    pairs["Treated_Mix_Distance_Percentile"] = (
        pairs["Mix_Distance"]
        .rank(
            pct=True
        )
    )

    # -------------------------------------------------------------------------
    # Rank performance divergence
    #
    # Higher gap = greater performance difference.
    # -------------------------------------------------------------------------

    pairs["Performance_Gap_Rank"] = (
        pairs["Incremental_Comp_Gap"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    # -------------------------------------------------------------------------
    # Key diagnostic:
    #
    # Large performance gap despite small merchandise distance.
    #
    # Higher score = more interesting counterexample to a
    # merchandise-mix explanation.
    # -------------------------------------------------------------------------

    pairs["Mix_Performance_Divergence_Score"] = (
        pairs["Incremental_Comp_Gap"]
        /
        pairs["Mix_Distance"]
    )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    logger.info("")
    logger.info("-" * 70)
    logger.info(
        "MOST SIMILAR TREATED DMA PAIRS"
    )
    logger.info("-" * 70)

    display_columns = [
        "DMA_1",
        "DMA_2",
        "Mix_Distance",
        "DMA_1_Incremental_Comp",
        "DMA_2_Incremental_Comp",
        "Incremental_Comp_Gap",
        "DMA_1_BSTS_Lift",
        "DMA_2_BSTS_Lift",
        "BSTS_Lift_Gap",
    ]

    logger.info(
        "\n%s",
        pairs[
            display_columns
        ]
        .sort_values(
            "Mix_Distance"
        )
        .head(15)
        .to_string(
            index=False
        ),
    )

    logger.info("")
    logger.info("-" * 70)
    logger.info(
        "SIMILAR MIX / DIFFERENT PERFORMANCE CASES"
    )
    logger.info("-" * 70)

    logger.info(
        "\n%s",
        pairs[
            display_columns
            +
            [
                "Mix_Performance_Divergence_Score"
            ]
        ]
        .sort_values(
            "Mix_Performance_Divergence_Score",
            ascending=False,
        )
        .head(15)
        .to_string(
            index=False
        ),
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    pairs.to_csv(
        TABLE_DIR
        / "treated_dma_merchandise_performance_pairs.csv",
        index=False,
    )

    logger.info("")
    logger.info(
        "Saved treated_dma_merchandise_performance_pairs.csv"
    )

    logger.info("=" * 70)

    return pairs


# =============================================================================
# ANALYZE MERCHANDISE DISTANCE VS PERFORMANCE GAP
# =============================================================================

def analyze_mix_performance_relationship(
    performance_pairs,
):
    """
    Test whether treated DMAs with more different merchandise mixes
    also exhibit more different traffic / marketing performance.

    This is a descriptive diagnostic, not a causal model.

    If merchandise mix helps explain heterogeneous DMA performance,
    we would generally expect larger merchandise distances to be
    associated with larger performance gaps.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "ANALYZING MERCHANDISE DISTANCE VS PERFORMANCE GAP"
    )
    logger.info("=" * 70)

    df = performance_pairs.copy()

    metrics = [
        (
            "Incremental Comp",
            "Incremental_Comp_Gap",
        ),
        (
            "BSTS Relative Lift",
            "BSTS_Lift_Gap",
        ),
    ]

    results = []

    for metric_name, metric_column in metrics:

        analysis = (
            df[
                [
                    "Mix_Distance",
                    metric_column,
                ]
            ]
            .dropna()
        )

        if len(analysis) < 3:

            logger.warning(
                "Insufficient observations for %s.",
                metric_name,
            )

            continue

        # -------------------------------------------------------------
        # Pearson correlation
        # -------------------------------------------------------------

        pearson_r = (
            analysis[
                "Mix_Distance"
            ]
            .corr(
                analysis[
                    metric_column
                ],
                method="pearson",
            )
        )

        # -------------------------------------------------------------
        # Spearman correlation
        # -------------------------------------------------------------

        spearman_r = (
            analysis[
                "Mix_Distance"
            ]
            .corr(
                analysis[
                    metric_column
                ],
                method="spearman",
            )
        )

        results.append(
            {
                "Performance_Metric":
                    metric_name,

                "Pearson_Correlation":
                    pearson_r,

                "Spearman_Correlation":
                    spearman_r,

                "N_Pairs":
                    len(analysis),
            }
        )

    results = pd.DataFrame(
        results
    )

    logger.info("")
    logger.info("-" * 70)
    logger.info(
        "MIX DISTANCE VS PERFORMANCE GAP"
    )
    logger.info("-" * 70)

    logger.info(
        "\n%s",
        results.to_string(
            index=False
        ),
    )

    # -------------------------------------------------------------------------
    # Performance gaps among the most merchandise-similar pairs
    # -------------------------------------------------------------------------

    similar_pairs = (
        df.sort_values(
            "Mix_Distance"
        )
        .head(10)
        .copy()
    )

    logger.info("")
    logger.info("-" * 70)
    logger.info(
        "TOP 10 MOST MERCHANDISE-SIMILAR TREATED PAIRS"
    )
    logger.info("-" * 70)

    logger.info(
        "\n%s",
        similar_pairs[
            [
                "DMA_1",
                "DMA_2",
                "Mix_Distance",
                "Incremental_Comp_Gap",
                "BSTS_Lift_Gap",
            ]
        ]
        .to_string(
            index=False
        ),
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    results.to_csv(
        TABLE_DIR
        / "mix_distance_performance_relationship.csv",
        index=False,
    )

    logger.info("")
    logger.info(
        "Saved mix_distance_performance_relationship.csv"
    )

    logger.info("=" * 70)

    return results


# =============================================================================
# TREATED DMA MERCHANDISE COMPARISON TABLE
# =============================================================================

def build_treated_dma_comparison_table(
    profiles,
    treated_pairs,
    bsts,
):
    """
    Build a leadership-facing treated-DMA merchandise comparison table.

    One row per treated DMA containing:
        - Core merchandise mix shares
        - Closest treated DMA by merchandise distance
        - Merchandise distance
        - Merchandise-distance percentile, where available
        - DMA performance
        - Closest-DMA performance
        - Absolute performance gaps

    This table is descriptive/diagnostic. It is intended to show whether
    merchandise-similar treated markets also exhibit similar traffic
    performance.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "BUILDING TREATED DMA MERCHANDISE COMPARISON TABLE"
    )
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Restrict profiles to treated DMAs and core merchandise groups
    # -------------------------------------------------------------------------

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

    treated_profiles = (
        profiles.loc[
            (
                profiles["Is_Treated_DMA"]
            )
            &
            (
                profiles["Merch_Group"].isin(
                    core_groups
                )
            )
        ]
        .copy()
    )

    # -------------------------------------------------------------------------
    # 2. Convert merchandise profiles to one row per DMA
    # -------------------------------------------------------------------------

    mix_wide = (
        treated_profiles
        .pivot_table(
            index="DMA_Key",
            columns="Merch_Group",
            values="Avg_Dollar_Share",
            aggfunc="mean",
        )
        .reset_index()
    )

    # Convert shares to percentage points for reporting.

    for group in core_groups:

        if group in mix_wide.columns:

            mix_wide[group] = (
                mix_wide[group]
                * 100
            )

    # -------------------------------------------------------------------------
    # 3. Convert pair table into directional DMA -> comparison DMA records
    #
    # Every pair currently exists once:
    #
    #     DMA_1 | DMA_2
    #
    # We need both:
    #
    #     DMA_1 -> DMA_2
    #     DMA_2 -> DMA_1
    #
    # so that every treated DMA can independently identify its nearest peer.
    # -------------------------------------------------------------------------

    pair_columns = [
        "DMA_1",
        "DMA_2",
        "Mix_Distance",
        "Incremental_Comp_Gap",
        "BSTS_Lift_Gap",
    ]

    optional_pair_columns = [
        "Mix_Distance_Percentile",
        "DMA_1_Incremental_Comp",
        "DMA_2_Incremental_Comp",
        "DMA_1_BSTS_Lift",
        "DMA_2_BSTS_Lift",
    ]

    available_pair_columns = (
        pair_columns
        +
        [
            col
            for col in optional_pair_columns
            if col in treated_pairs.columns
        ]
    )

    pairs = (
        treated_pairs[
            available_pair_columns
        ]
        .copy()
    )

    forward = pd.DataFrame({

        "DMA_Key":
            pairs["DMA_1"],

        "Closest_Treated_DMA":
            pairs["DMA_2"],

        "Mix_Distance":
            pairs["Mix_Distance"],

        "Incremental_Comp_Gap":
            pairs["Incremental_Comp_Gap"],

        "BSTS_Lift_Gap":
            pairs["BSTS_Lift_Gap"],
    })

    reverse = pd.DataFrame({

        "DMA_Key":
            pairs["DMA_2"],

        "Closest_Treated_DMA":
            pairs["DMA_1"],

        "Mix_Distance":
            pairs["Mix_Distance"],

        "Incremental_Comp_Gap":
            pairs["Incremental_Comp_Gap"],

        "BSTS_Lift_Gap":
            pairs["BSTS_Lift_Gap"],
    })

    if "Mix_Distance_Percentile" in pairs.columns:

        forward["Mix_Distance_Percentile"] = (
            pairs["Mix_Distance_Percentile"]
        )

        reverse["Mix_Distance_Percentile"] = (
            pairs["Mix_Distance_Percentile"]
        )

    directional_pairs = pd.concat(
        [
            forward,
            reverse,
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # 4. Identify nearest treated merchandise peer
    # -------------------------------------------------------------------------

    directional_pairs = (
        directional_pairs
        .sort_values(
            [
                "DMA_Key",
                "Mix_Distance",
            ]
        )
    )

    closest_peer = (
        directional_pairs
        .groupby(
            "DMA_Key",
            as_index=False,
        )
        .first()
    )

    # -------------------------------------------------------------------------
    # 5. Prepare performance lookup
    # -------------------------------------------------------------------------

    performance = (
        bsts[
            [
                "DMA_Key",
                "Implied_Incremental_Comp",
                "BSTS_Relative_Lift",
            ]
        ]
        .drop_duplicates(
            "DMA_Key"
        )
        .copy()
    )

    performance = performance.rename(
        columns={
            "Implied_Incremental_Comp":
                "Incremental_Comp",

            "BSTS_Relative_Lift":
                "BSTS_Lift",
        }
    )

    # -------------------------------------------------------------------------
    # 6. Attach primary DMA performance
    # -------------------------------------------------------------------------

    result = (
        mix_wide
        .merge(
            closest_peer,
            on="DMA_Key",
            how="left",
        )
        .merge(
            performance,
            on="DMA_Key",
            how="left",
        )
    )

    # -------------------------------------------------------------------------
    # 7. Attach closest-peer performance
    # -------------------------------------------------------------------------

    peer_performance = (
        performance
        .rename(
            columns={
                "DMA_Key":
                    "Closest_Treated_DMA",

                "Incremental_Comp":
                    "Closest_DMA_Incremental_Comp",

                "BSTS_Lift":
                    "Closest_DMA_BSTS_Lift",
            }
        )
    )

    result = (
        result
        .merge(
            peer_performance,
            on="Closest_Treated_DMA",
            how="left",
        )
    )

    # -------------------------------------------------------------------------
    # 8. Recalculate performance gaps directly
    #
    # This makes the final table self-contained and avoids depending on
    # orientation in the pairwise table.
    # -------------------------------------------------------------------------

    result["Incremental_Comp_Gap"] = (
        (
            result["Incremental_Comp"]
            -
            result["Closest_DMA_Incremental_Comp"]
        )
        .abs()
    )

    result["BSTS_Lift_Gap"] = (
        (
            result["BSTS_Lift"]
            -
            result["Closest_DMA_BSTS_Lift"]
        )
        .abs()
    )

    result["Performance_Gap_PP"] = (
        result["Incremental_Comp_Gap"]
        * 100
    )

    result["BSTS_Lift_Gap_PP"] = (
        result["BSTS_Lift_Gap"]
        * 100
    )

    result["Incremental_Comp_Pct"] = (
        result["Incremental_Comp"]
        * 100
    )

    result["Closest_DMA_Incremental_Comp_Pct"] = (
        result["Closest_DMA_Incremental_Comp"]
        * 100
    )

    # -------------------------------------------------------------------------
    # 9. Friendly column names
    # -------------------------------------------------------------------------

    rename_map = {

        "Ladies and Juniors Apparel":
            "Ladies_Juniors_Share_Pct",

        "Mens and Young Mens Apparel":
            "Mens_Share_Pct",

        "Shoes":
            "Shoes_Share_Pct",

        "Home Essentials":
            "Home_Essentials_Share_Pct",

        "Home Living":
            "Home_Living_Share_Pct",

        "Basics and Sleepwear":
            "Basics_Sleepwear_Share_Pct",

        "Kids and Baby Apparel":
            "Kids_Baby_Share_Pct",

        "Accessories":
            "Accessories_Share_Pct",

        "Beauty and Fragrance":
            "Beauty_Fragrance_Share_Pct",
    }

    result = result.rename(
        columns=rename_map
    )

    # -------------------------------------------------------------------------
    # 10. Final ordering
    # -------------------------------------------------------------------------

    share_columns = list(
        rename_map.values()
    )

    final_columns = [

        "DMA_Key",

        *share_columns,

        "Closest_Treated_DMA",
        "Mix_Distance",

    ]

    if "Mix_Distance_Percentile" in result.columns:

        final_columns.append(
            "Mix_Distance_Percentile"
        )

    final_columns += [

        "Incremental_Comp",
        "Incremental_Comp_Pct",
        "Closest_DMA_Incremental_Comp",
        "Closest_DMA_Incremental_Comp_Pct",
        "Incremental_Comp_Gap",

        "BSTS_Lift",
        "Closest_DMA_BSTS_Lift",
        "BSTS_Lift_Gap",
    ]

    result = (
        result[
            final_columns
        ]
        .sort_values(
            "Incremental_Comp_Pct",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------------------
    # 11. QA
    # -------------------------------------------------------------------------

    logger.info(
        "Treated DMAs represented : %s",
        f"{len(result):,}",
    )

    logger.info(
        "Unique treated DMAs      : %s",
        f"{result['DMA_Key'].nunique():,}",
    )

    logger.info("")
    logger.info(
        "Closest Treated DMA Relationships"
    )
    logger.info("-" * 70)

    display_columns = [

        "DMA_Key",
        "Closest_Treated_DMA",
        "Mix_Distance",
        "Incremental_Comp",
        "Incremental_Comp_Pct",
        "Closest_DMA_Incremental_Comp",
        "Closest_DMA_Incremental_Comp_Pct",
        "Incremental_Comp_Gap",
    ]

    logger.info(
        "\n%s",
        result[
            display_columns
        ]
        .to_string(
            index=False
        ),
    )

    logger.info("=" * 70)

    return result


# =============================================================================
# FIGURE 1: TREATED DMA MERCHANDISE MIX HEATMAP
# =============================================================================

def plot_treated_dma_mix_heatmap(
    profiles,
    bsts,
):
    """
    Plot treated-DMA merchandise mix deviations from the network average.

    Rows:
        Treated DMAs, ordered from highest to lowest incremental comp.

    Columns:
        Core merchandise groups.

    Values:
        Dollar-share deviation from the network average,
        expressed in percentage points.

    Interpretation:
        Positive values indicate that the DMA over-indexes in the
        merchandise group relative to the network average.

        Negative values indicate that the DMA under-indexes.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "PLOTTING TREATED DMA MERCHANDISE MIX HEATMAP"
    )
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Restrict to treated/core merchandise
    # -------------------------------------------------------------------------

    data = (
        profiles.loc[
            (
                profiles["Is_Treated_DMA"]
            )
            &
            (
                profiles["Merchandise_Type"]
                .eq("Core")
            )
        ]
        .copy()
    )

    if data.empty:

        raise ValueError(
            "No treated DMA core merchandise profiles "
            "available for heatmap."
        )

    # -------------------------------------------------------------------------
    # Performance ordering
    # -------------------------------------------------------------------------

    performance = (
        bsts[
            [
                "DMA_Key",
                "Implied_Incremental_Comp",
            ]
        ]
        .drop_duplicates(
            "DMA_Key"
        )
        .copy()
    )

    performance[
        "Incremental_Comp_Pct"
    ] = (
        performance[
            "Implied_Incremental_Comp"
        ]
        * 100
    )

    dma_order = (
        performance
        .sort_values(
            "Implied_Incremental_Comp",
            ascending=False,
        )[
            "DMA_Key"
        ]
        .tolist()
    )

    # -------------------------------------------------------------------------
    # Build heatmap matrix
    # -------------------------------------------------------------------------

    heatmap = (
        data
        .pivot_table(
            index="DMA_Key",
            columns="Merch_Group",
            values="Dollar_Share_Deviation_PP",
            aggfunc="mean",
        )
    )

    # Only retain DMAs actually represented.

    dma_order = [
        dma
        for dma in dma_order
        if dma in heatmap.index
    ]

    heatmap = heatmap.loc[
        dma_order
    ]

    # -------------------------------------------------------------------------
    # Friendly merchandise labels
    # -------------------------------------------------------------------------

    label_map = {

        "Ladies and Juniors Apparel":
            "Ladies & Juniors",

        "Mens and Young Mens Apparel":
            "Mens",

        "Basics and Sleepwear":
            "Basics & Sleepwear",

        "Kids and Baby Apparel":
            "Kids & Baby",

        "Beauty and Fragrance":
            "Beauty & Fragrance",

        "Home Essentials":
            "Home Essentials",

        "Home Living":
            "Home Living",

        "Accessories":
            "Accessories",

        "Shoes":
            "Shoes",
    }

    heatmap = heatmap.rename(
        columns=label_map
    )

    # -------------------------------------------------------------------------
    # Add performance to row labels
    # -------------------------------------------------------------------------

    performance_lookup = (
        performance
        .set_index("DMA_Key")[
            "Incremental_Comp_Pct"
        ]
        .to_dict()
    )

    row_labels = [

        (
            f"{dma}  "
            f"({performance_lookup.get(dma, np.nan):+.1f}%)"
        )

        for dma in heatmap.index
    ]

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    sns.heatmap(
        heatmap,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={
            "label":
                "Dollar-share deviation from chain average (pp)"
        },
        ax=ax,
    )

    ax.set_yticklabels(
        row_labels,
        rotation=0,
    )

    ax.set_xlabel(
        "Merchandise Group"
    )

    ax.set_ylabel(
        "Treated DMA (Incremental Comp)"
    )

    ax.set_title(
        "Treated DMA Merchandise Mix vs. Chain Average",
        fontsize=16,
        pad=30,
    )

    ax.text(
        0,
        1.01,
        (
            "Positive values indicate over-indexing; "
            "negative values indicate under-indexing. "
            "DMAs are ordered by incremental comp performance."
        ),
        transform=ax.transAxes,
        fontsize=10,
    )

    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "treated_dma_merchandise_mix_heatmap.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Saved %s",
        output_path.name,
    )

    return output_path


# =============================================================================
# FIGURE 2: MIX DISTANCE VS PERFORMANCE GAP
# =============================================================================

def plot_mix_distance_vs_performance_gap(
    treated_pairs,
):
    """
    Plot merchandise mix distance against the absolute
    incremental-comp performance gap for treated DMA pairs.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "PLOTTING MIX DISTANCE VS PERFORMANCE GAP"
    )
    logger.info("=" * 70)

    required = {

        "Mix_Distance",
        "Incremental_Comp_Gap",
    }

    missing = (
        required
        -
        set(
            treated_pairs.columns
        )
    )

    if missing:

        raise ValueError(
            "Treated pair data missing required columns: "
            f"{sorted(missing)}"
        )

    data = (
        treated_pairs[
            [
                "DMA_1",
                "DMA_2",
                "Mix_Distance",
                "Incremental_Comp_Gap",
            ]
        ]
        .dropna()
        .copy()
    )

    data[
        "Performance_Gap_PP"
    ] = (
        data[
            "Incremental_Comp_Gap"
        ]
        * 100
    )

    # -------------------------------------------------------------------------
    # Correlations
    # -------------------------------------------------------------------------

    pearson = (
        data[
            [
                "Mix_Distance",
                "Performance_Gap_PP",
            ]
        ]
        .corr(
            method="pearson"
        )
        .iloc[0, 1]
    )

    spearman = (
        data[
            [
                "Mix_Distance",
                "Performance_Gap_PP",
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[0, 1]
    )

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.scatter(
        data["Mix_Distance"],
        data["Performance_Gap_PP"],
        alpha=0.75,
        s=70,
    )

    # -------------------------------------------------------------------------
    # Simple fitted line
    # -------------------------------------------------------------------------

    x = data[
        "Mix_Distance"
    ].to_numpy()

    y = data[
        "Performance_Gap_PP"
    ].to_numpy()

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    x_line = np.linspace(
        x.min(),
        x.max(),
        100,
    )

    y_line = (
        slope
        * x_line
        +
        intercept
    )

    ax.plot(
        x_line,
        y_line,
        linewidth=2,
    )

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------

    ax.set_xlabel(
        "Merchandise Mix Distance"
    )

    ax.set_ylabel(
        "Incremental Comp Performance Gap (pp)"
    )

    ax.set_title(
        "Merchandise Mix Differences Weakly Explain DMA Performance Gaps",
        fontsize=15,
        pad=18,
    )

    ax.text(
        0.02,
        0.96,
        (
            f"Pearson r = {pearson:.2f}\n"
            f"Spearman ρ = {spearman:.2f}\n"
            f"N = {len(data)} DMA pairs"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "mix_distance_vs_performance_gap.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Pearson correlation : %.4f",
        pearson,
    )

    logger.info(
        "Spearman correlation: %.4f",
        spearman,
    )

    logger.info(
        "Saved %s",
        output_path.name,
    )

    return output_path


# =============================================================================
# FIGURE 3: CLOSEST MERCHANDISE PEER PERFORMANCE GAP
# =============================================================================

def plot_closest_peer_performance_gap(
    comparison,
):
    """
    Show each treated DMA's performance gap relative to its
    closest treated merchandise-mix peer.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "PLOTTING CLOSEST MERCHANDISE PEER PERFORMANCE GAPS"
    )
    logger.info("=" * 70)

    data = (
        comparison[
            [
                "DMA_Key",
                "Closest_Treated_DMA",
                "Mix_Distance",
                "Incremental_Comp",
                "Closest_DMA_Incremental_Comp",
                "Incremental_Comp_Gap",
            ]
        ]
        .copy()
    )

    data[
        "Performance_Gap_PP"
    ] = (
        data[
            "Incremental_Comp_Gap"
        ]
        * 100
    )

    data = (
        data
        .sort_values(
            "Performance_Gap_PP",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    bars = ax.barh(
        data["DMA_Key"],
        data["Performance_Gap_PP"],
    )

    # -------------------------------------------------------------------------
    # Peer labels
    # -------------------------------------------------------------------------

    max_gap = (
        data[
            "Performance_Gap_PP"
        ]
        .max()
    )

    for bar, (_, row) in zip(
        bars,
        data.iterrows(),
    ):

        value = (
            row[
                "Performance_Gap_PP"
            ]
        )

        label = (
            f"{row['Closest_Treated_DMA']}  |  "
            f"mix distance {row['Mix_Distance']:.2f}"
        )

        ax.text(
            value
            +
            max_gap * 0.015,
            bar.get_y()
            +
            bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
        )

    # -------------------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------------------

    ax.set_xlabel(
        "Absolute Incremental Comp Gap (percentage points)"
    )

    ax.set_ylabel(
        "Treated DMA"
    )

    ax.set_title(
        "Performance Can Differ Substantially Between Merchandise-Similar DMAs",
        fontsize=15,
        pad=30,
    )

    ax.text(
        0,
        1.01,
        (
            "Each DMA is compared with its closest treated-market "
            "peer based on merchandise mix."
        ),
        transform=ax.transAxes,
        fontsize=10,
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    # Give peer labels room.

    ax.set_xlim(
        0,
        max_gap * 1.55,
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "closest_merchandise_peer_performance_gap.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Saved %s",
        output_path.name,
    )

    return output_path


# =============================================================================
# GENERATE FIGURES
# =============================================================================

def generate_figures(
    profiles,
    bsts,
    treated_pairs,
    treated_dma_comparison,
):
    """
    Generate final DMA merchandise diagnostic figures.
    """

    logger.info("")
    logger.info("=" * 80)
    logger.info(
        "GENERATING DMA MERCHANDISE DIAGNOSTIC FIGURES"
    )
    logger.info("=" * 80)

    figures = {}

    figures[
        "merchandise_mix_heatmap"
    ] = (
        plot_treated_dma_mix_heatmap(
            profiles=profiles,
            bsts=bsts,
        )
    )

    figures[
        "mix_distance_vs_performance"
    ] = (
        plot_mix_distance_vs_performance_gap(
            treated_pairs=treated_pairs,
        )
    )

    figures[
        "closest_peer_performance"
    ] = (
        plot_closest_peer_performance_gap(
            comparison=treated_dma_comparison,
        )
    )

    logger.info("")
    logger.info(
        "Figures generated: %s",
        len(figures),
    )

    for name, path in figures.items():

        logger.info(
            "  %s: %s",
            name,
            path,
        )

    logger.info("=" * 80)

    return figures


# =============================================================================
# HTML REPORT
# =============================================================================

def generate_merchandise_diagnostics_report(
    dma_comparison_table,
    mix_performance_relationship,
    figures,
):
    """
    Generate the final management-facing HTML report for the
    DMA merchandise mix diagnostic analysis.

    The report summarizes:
        1. Treated-DMA merchandise composition
        2. Merchandise similarity across treated markets
        3. Merchandise distance vs. performance distance
        4. Closest merchandise-peer comparisons
        5. Key conclusions and analytical limitations
    """

    logger.info("")
    logger.info("=" * 80)
    logger.info(
        "GENERATING DMA MERCHANDISE MIX DIAGNOSTIC REPORT"
    )
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def image_to_base64(path):
        """
        Encode an image so the HTML report is fully self-contained.
        """

        import base64

        with open(path, "rb") as image_file:

            encoded = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return encoded

    def format_pct(value):

        if pd.isna(value):
            return ""

        return f"{value * 100:+.1f}%"

    def format_pp(value):

        if pd.isna(value):
            return ""

        return f"{value * 100:.1f} pp"

    # -------------------------------------------------------------------------
    # Extract figures
    # -------------------------------------------------------------------------

    heatmap_b64 = image_to_base64(
        figures["merchandise_mix_heatmap"]
    )

    scatter_b64 = image_to_base64(
        figures["mix_distance_vs_performance"]
    )

    peer_b64 = image_to_base64(
        figures["closest_peer_performance"]
    )

    # -------------------------------------------------------------------------
    # Relationship statistics
    # -------------------------------------------------------------------------

    relationship = (
        mix_performance_relationship
        .set_index("Performance_Metric")
    )

    comp_pearson = (
        relationship.loc[
            "Incremental Comp",
            "Pearson_Correlation",
        ]
    )

    comp_spearman = (
        relationship.loc[
            "Incremental Comp",
            "Spearman_Correlation",
        ]
    )

    bsts_pearson = (
        relationship.loc[
            "BSTS Relative Lift",
            "Pearson_Correlation",
        ]
    )

    bsts_spearman = (
        relationship.loc[
            "BSTS Relative Lift",
            "Spearman_Correlation",
        ]
    )

    n_pairs = int(
        relationship.loc[
            "Incremental Comp",
            "N_Pairs",
        ]
    )

    # -------------------------------------------------------------------------
    # Leadership table
    # -------------------------------------------------------------------------

    table = (
        dma_comparison_table
        .copy()
    )

    # Business-readable display fields

    table[
        "Incremental Comp"
    ] = (
        table[
            "Incremental_Comp"
        ]
        .apply(format_pct)
    )

    table[
        "Closest DMA Comp"
    ] = (
        table[
            "Closest_DMA_Incremental_Comp"
        ]
        .apply(format_pct)
    )

    table[
        "Performance Gap"
    ] = (
        table[
            "Incremental_Comp_Gap"
        ]
        .apply(format_pp)
    )

    table[
        "BSTS Lift"
    ] = (
        table[
            "BSTS_Lift"
        ]
        .apply(format_pct)
    )

    table[
        "BSTS Gap"
    ] = (
        table[
            "BSTS_Lift_Gap"
        ]
        .apply(format_pp)
    )

    # -------------------------------------------------------------------------
    # Rename merchandise columns for display
    # -------------------------------------------------------------------------

    display_map = {

        "DMA_Key":
            "DMA",

        "Ladies_Juniors_Share_Pct":
            "Ladies & Juniors",

        "Mens_Share_Pct":
            "Mens",

        "Shoes_Share_Pct":
            "Shoes",

        "Home_Essentials_Share_Pct":
            "Home Essentials",

        "Home_Living_Share_Pct":
            "Home Living",

        "Basics_Sleepwear_Share_Pct":
            "Basics",

        "Kids_Baby_Share_Pct":
            "Kids & Baby",

        "Accessories_Share_Pct":
            "Accessories",

        "Beauty_Fragrance_Share_Pct":
            "Beauty",

        "Closest_Treated_DMA":
            "Closest Mix Peer",

        "Mix_Distance":
            "Mix Distance",
    }

    table = table.rename(
        columns=display_map
    )

    merchandise_columns = [

        "Ladies & Juniors",
        "Mens",
        "Shoes",
        "Home Essentials",
        "Home Living",
        "Basics",
        "Kids & Baby",
        "Accessories",
        "Beauty",
    ]

    # These fields were already converted to percentage points
    # when the comparison table was constructed.

    for column in merchandise_columns:

        if column in table.columns:

            table[column] = (
                table[column]
                .map(
                    lambda x:
                    ""
                    if pd.isna(x)
                    else f"{x:.1f}%"
                )
            )

    leadership_columns = [

        "DMA",

        *merchandise_columns,

        "Closest Mix Peer",
        "Mix Distance",

        "Incremental Comp",
        "Closest DMA Comp",
        "Performance Gap",

        "BSTS Lift",
        "BSTS Gap",
    ]

    table = table[
        [
            col
            for col in leadership_columns
            if col in table.columns
        ]
    ]

    # -------------------------------------------------------------------------
    # Table HTML
    # -------------------------------------------------------------------------

    table_html = table.to_html(
        index=False,
        border=0,
        classes="data-table",
        escape=False,
    )

    # -------------------------------------------------------------------------
    # HTML
    # -------------------------------------------------------------------------

    html = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
DMA Merchandise Mix Diagnostics
</title>

<style>

    * {{
        box-sizing: border-box;
    }}

    body {{
        margin: 0;
        background: #f5f6f8;
        color: #222;
        font-family:
            Arial,
            Helvetica,
            sans-serif;
        line-height: 1.55;
    }}

    .container {{
        width: 94%;
        max-width: 1500px;
        margin: 0 auto;
        padding: 40px 0 80px 0;
    }}

    .header {{
        background: #172033;
        color: white;
        padding: 42px 48px;
        border-radius: 10px;
        margin-bottom: 28px;
    }}

    .header h1 {{
        margin: 0 0 8px 0;
        font-size: 32px;
        font-weight: 600;
    }}

    .header p {{
        margin: 0;
        color: #d5dbe7;
        font-size: 16px;
    }}

    .section {{
        background: white;
        padding: 32px 36px;
        margin-bottom: 24px;
        border-radius: 10px;
        box-shadow:
            0 1px 4px
            rgba(0, 0, 0, 0.08);
    }}

    h2 {{
        margin-top: 0;
        color: #172033;
        font-size: 24px;
    }}

    h3 {{
        color: #374151;
        margin-top: 28px;
    }}

    .executive-box {{
        border-left: 5px solid #415a77;
        background: #f4f7fb;
        padding: 18px 22px;
        margin: 20px 0;
    }}

    .metric-grid {{
        display: grid;
        grid-template-columns:
            repeat(
                auto-fit,
                minmax(200px, 1fr)
            );
        gap: 16px;
        margin: 24px 0;
    }}

    .metric-card {{
        background: #f7f8fa;
        border: 1px solid #e2e5e9;
        border-radius: 8px;
        padding: 18px;
    }}

    .metric-value {{
        font-size: 27px;
        font-weight: 600;
        color: #172033;
    }}

    .metric-label {{
        font-size: 13px;
        color: #68707d;
        margin-top: 5px;
    }}

    .figure {{
        margin: 28px auto;
        text-align: center;
    }}

    .figure img {{
        width: 100%;
        max-width: 1250px;
        height: auto;
    }}

    .figure-caption {{
        color: #606773;
        font-size: 14px;
        margin-top: 10px;
    }}

    .table-wrapper {{
        overflow-x: auto;
        margin-top: 24px;
    }}

    .data-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
        white-space: nowrap;
    }}

    .data-table th {{
        background: #172033;
        color: white;
        padding: 10px 8px;
        text-align: right;
        position: sticky;
        top: 0;
    }}

    .data-table th:first-child {{
        text-align: left;
    }}

    .data-table td {{
        padding: 9px 8px;
        border-bottom: 1px solid #e5e7eb;
        text-align: right;
    }}

    .data-table td:first-child {{
        text-align: left;
        font-weight: 600;
    }}

    .data-table tr:nth-child(even) {{
        background: #f8f9fb;
    }}

    .takeaway {{
        padding: 15px 18px;
        margin: 12px 0;
        background: #f6f8fa;
        border-left: 4px solid #687b96;
    }}

    .limitation {{
        color: #59616d;
        font-size: 14px;
    }}

    .footer {{
        color: #777;
        font-size: 12px;
        text-align: center;
        margin-top: 30px;
    }}

</style>

</head>

<body>

<div class="container">

    <div class="header">

        <h1>
            DMA Merchandise Mix Diagnostics
        </h1>

        <p>
            Evaluating whether merchandise composition helps explain
            heterogeneous traffic and marketing performance across
            treated DMAs.
        </p>

    </div>


    <!-- ============================================================= -->
    <!-- EXECUTIVE SUMMARY -->
    <!-- ============================================================= -->

    <div class="section">

        <h2>
            Executive Summary
        </h2>

        <div class="executive-box">

            Merchandise composition varies across treated markets,
            but differences in merchandise mix show only a weak
            relationship with differences in traffic performance.

            Markets with highly similar merchandise profiles can
            produce materially different traffic outcomes, indicating
            that observed merchandise mix alone is insufficient to
            explain the substantial performance heterogeneity across
            treated DMAs.

        </div>

        <div class="metric-grid">

            <div class="metric-card">

                <div class="metric-value">
                    {len(dma_comparison_table)}
                </div>

                <div class="metric-label">
                    Treated DMAs
                </div>

            </div>

            <div class="metric-card">

                <div class="metric-value">
                    {n_pairs}
                </div>

                <div class="metric-label">
                    Treated DMA pair comparisons
                </div>

            </div>

            <div class="metric-card">

                <div class="metric-value">
                    {comp_pearson:.2f}
                </div>

                <div class="metric-label">
                    Pearson correlation:
                    merchandise distance vs.
                    incremental comp gap
                </div>

            </div>

            <div class="metric-card">

                <div class="metric-value">
                    {comp_spearman:.2f}
                </div>

                <div class="metric-label">
                    Spearman correlation:
                    merchandise distance vs.
                    incremental comp gap
                </div>

            </div>

        </div>

        <p>
            The relationship remains weak when performance is measured
            using BSTS relative lift:
            Pearson r = <strong>{bsts_pearson:.2f}</strong>
            and Spearman ρ =
            <strong>{bsts_spearman:.2f}</strong>.
        </p>

    </div>


    <!-- ============================================================= -->
    <!-- MERCHANDISE MIX -->
    <!-- ============================================================= -->

    <div class="section">

        <h2>
            Merchandise Mix Across Treated DMAs
        </h2>

        <p>
            The figure below compares each treated DMA's merchandise
            dollar-share mix with the broader chain average.
            Positive values represent over-indexing in a merchandise
            group, while negative values represent under-indexing.
            Markets are ordered by incremental comp performance.
        </p>

        <div class="figure">

            <img
                src="data:image/png;base64,{heatmap_b64}"
                alt="Treated DMA merchandise mix heatmap"
            >

            <div class="figure-caption">

                Merchandise dollar-share deviation from the chain
                average, expressed in percentage points.

            </div>

        </div>

    </div>


    <!-- ============================================================= -->
    <!-- DISTANCE VS PERFORMANCE -->
    <!-- ============================================================= -->

    <div class="section">

        <h2>
            Does Merchandise Mix Explain Performance Differences?
        </h2>

        <p>
            Across all {n_pairs} treated-DMA pairs, merchandise mix
            distance has only a weak positive relationship with the
            absolute difference in incremental comp performance.
        </p>

        <div class="figure">

            <img
                src="data:image/png;base64,{scatter_b64}"
                alt="Merchandise mix distance versus performance gap"
            >

            <div class="figure-caption">

                Each point represents one unique pair of treated DMAs.
                Larger merchandise distance indicates more different
                merchandise composition.

            </div>

        </div>

        <div class="takeaway">

            <strong>Interpretation:</strong>

            Increasing merchandise dissimilarity does not reliably
            correspond to increasing performance dissimilarity.
            Merchandise composition may contribute to local market
            performance, but it does not appear sufficient to explain
            the large differences observed across treated markets.

        </div>

    </div>


    <!-- ============================================================= -->
    <!-- CLOSEST PEERS -->
    <!-- ============================================================= -->

    <div class="section">

        <h2>
            Closest Merchandise Peer Comparisons
        </h2>

        <p>
            Each treated DMA is matched to the treated market with the
            most similar merchandise composition. If merchandise mix
            were a primary explanation for cross-market performance
            differences, these nearest merchandise peers would
            generally be expected to exhibit relatively similar
            outcomes.
        </p>

        <div class="figure">

            <img
                src="data:image/png;base64,{peer_b64}"
                alt="Closest merchandise peer performance gaps"
            >

            <div class="figure-caption">

                Absolute incremental comp difference between each
                treated DMA and its closest treated merchandise-mix
                peer.

            </div>

        </div>

    </div>


    <!-- ============================================================= -->
    <!-- LEADERSHIP TABLE -->
    <!-- ============================================================= -->

    <div class="section">

        <h2>
            Treated DMA Merchandise & Performance Summary
        </h2>

        <p>
            This table combines merchandise composition, nearest
            merchandise peer, merchandise distance, and traffic
            performance into a single treated-market view.
        </p>

        <div class="table-wrapper">

            {table_html}

        </div>

    </div>


    <!-- ============================================================= -->
    <!-- CONCLUSIONS -->
    <!-- ============================================================= -->

    <div class="section">

        <h2>
            Key Takeaways
        </h2>

        <div class="takeaway">

            <strong>1. Merchandise mix differs across markets.</strong>

            Treated DMAs exhibit identifiable over- and under-indexing
            across major merchandise groups.

        </div>

        <div class="takeaway">

            <strong>2. Similar merchandise does not consistently imply
            similar traffic performance.</strong>

            Several highly merchandise-similar treated markets exhibit
            materially different incremental comp and BSTS outcomes.

        </div>

        <div class="takeaway">

            <strong>3. Merchandise distance has weak explanatory
            relationship with performance distance.</strong>

            Pairwise correlations are low for both incremental comp and
            BSTS relative lift.

        </div>

        <div class="takeaway">

            <strong>4. Merchandise mix therefore appears insufficient
            to explain observed DMA performance heterogeneity.</strong>

            The remaining variation is consistent with the presence of
            additional market-level factors not captured by the
            variables currently available in the analysis.

        </div>

        <h3>
            Analytical Limitations
        </h3>

        <p class="limitation">

            This analysis is descriptive and diagnostic rather than a
            causal test of merchandise assortment. Merchandise sales
            can themselves respond to traffic and marketing activity,
            creating simultaneity between merchandise performance and
            traffic. Pairwise DMA observations are also not independent,
            because each treated DMA appears in multiple comparisons.

        </p>

        <p class="limitation">

            Results should therefore be interpreted as evidence about
            whether observed merchandise composition can explain the
            cross-market pattern in traffic outcomes, rather than as an
            estimate of the causal effect of merchandise assortment on
            traffic.

        </p>

    </div>


    <div class="footer">

        Marketing Incrementality |
        DMA Merchandise Mix Diagnostics

    </div>

</div>

</body>

</html>
"""

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    output_path = (
        REPORT_DIR
        / "dma_merchandise_mix_diagnostics.html"
    )

    output_path.write_text(
        html,
        encoding="utf-8",
    )

    logger.info("")
    logger.info(
        "Saved report: %s",
        output_path
    )

    logger.info("=" * 80)

    return output_path



# =============================================================================
# MAIN
# =============================================================================

def main():

    logger.info("")
    logger.info("=" * 80)
    logger.info(
        "DMA MERCHANDISE MIX DIAGNOSTICS"
    )
    logger.info("=" * 80)

    try:

        # =====================================================================
        # 0. INITIALIZATION
        # =====================================================================

        create_output_directories()

        # =====================================================================
        # 1. LOAD MERCHANDISE DATA
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 1: LOADING MERCHANDISE DATA"
        )
        logger.info("-" * 80)

        merchandise = (
            load_merchandise_data()
        )

        # =====================================================================
        # 2. LOAD TREATED DMA UNIVERSE
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 2: LOADING TREATED DMA UNIVERSE"
        )
        logger.info("-" * 80)

        bsts = (
            load_bsts_comp_results()
        )

        # =====================================================================
        # 3. DIAGNOSE DMA COVERAGE
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 3: DIAGNOSING DMA COVERAGE"
        )
        logger.info("-" * 80)

        dma_coverage = (
            diagnose_dma_coverage(
                merchandise=merchandise,
                bsts=bsts,
            )
        )

        # =====================================================================
        # 4. BUILD DMA MERCHANDISE PROFILES
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 4: BUILDING DMA MERCHANDISE PROFILES"
        )
        logger.info("-" * 80)

        treated_dma_keys = (
            bsts[
                "DMA_Key"
            ]
            .dropna()
            .unique()
        )

        profiles, benchmarks = (
            build_dma_merchandise_profiles(
                merchandise=merchandise,
                treated_dmas=treated_dma_keys,
            )
        )

        
        # =====================================================================
        # 5. BUILD DMA PROFILE SUMMARY
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 5: BUILDING DMA PROFILE SUMMARY"
        )
        logger.info("-" * 80)

        dma_profile_summary = (
            build_dma_profile_summary(
                profiles
            )
        )

       # =====================================================================
        # 6. BUILD DMA MERCHANDISE MIX MATRIX
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 6: BUILDING DMA MERCHANDISE MIX MATRIX"
        )
        logger.info("-" * 80)

        mix_matrix = (
            build_dma_merchandise_mix_matrix(
                profiles=profiles,
                treated_dmas=treated_dma_keys,
            )
        )

        # =====================================================================
        # 7. BUILD DMA-TO-DMA MERCHANDISE DISTANCE MATRIX
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 7: BUILDING DMA-TO-DMA MERCHANDISE DISTANCE MATRIX"
        )
        logger.info("-" * 80)

        distance_matrix, pairwise_distances = (
            build_dma_merchandise_distance_matrix(
                mix_matrix=mix_matrix,
            )
        )

        # =====================================================================
        # 8. COMPARE MERCHANDISE SIMILARITY WITH DMA PERFORMANCE
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 8: COMPARING MERCHANDISE MIX WITH DMA PERFORMANCE"
        )
        logger.info("-" * 80)

        performance_pairs = (
            build_treated_dma_performance_pairs(
                pairwise_distances=pairwise_distances,
                bsts=bsts,
            )
        )

        # =====================================================================
        # 9. ANALYZE MIX DISTANCE VS PERFORMANCE GAP
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 9: ANALYZING MIX DISTANCE VS PERFORMANCE GAP"
        )
        logger.info("-" * 80)

        mix_performance_relationship = (
            analyze_mix_performance_relationship(
                performance_pairs
            )
        )

        # =====================================================================
        # 10. BUILD TREATED DMA MERCHANDISE COMPARISON TABLE
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)

        logger.info(
            "STEP 10: BUILDING TREATED DMA MERCHANDISE COMPARISON TABLE"
        )
        logger.info("-" * 80)

        dma_comparison_table = (
            build_treated_dma_comparison_table(
                profiles=profiles,
                treated_pairs=performance_pairs,
                bsts=bsts,
            )
        )

        # =====================================================================
        # 11. GENERATE VISUALIZATIONS
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 11: GENERATING MERCHANDISE DIAGNOSTIC VISUALIZATIONS"
        )
        logger.info("=" * 80)

        figures = (
            generate_figures(
                profiles=profiles,
                bsts=bsts,
                treated_pairs=performance_pairs,
                treated_dma_comparison=dma_comparison_table,
            )
        )

        # =====================================================================
        # 12. GENERATE HTML REPORT
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "STEP 12: GENERATING FINAL HTML REPORT"
        )
        logger.info("=" * 80)

        report_path = (
            generate_merchandise_diagnostics_report(
                dma_comparison_table=dma_comparison_table,
                mix_performance_relationship=mix_performance_relationship,
                figures=figures,
            )
        )


        # =====================================================================
        # 12. SAVE PROFILE OUTPUTS
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "STEP 12: SAVING MERCHANDISE PROFILE OUTPUTS"
        )
        logger.info("-" * 80)

        profiles.to_csv(
            TABLE_DIR
            / "dma_merchandise_profiles.csv",
            index=False,
        )

        benchmarks.to_csv(
            TABLE_DIR
            / "merchandise_network_benchmarks.csv",
            index=False,
        )

        dma_profile_summary.to_csv(
            TABLE_DIR
            / "dma_merchandise_profile_summary.csv",
            index=False,
        )

        logger.info(
            "Merchandise profile outputs saved."
        )

        # =====================================================================
        # 13. SAVE QA OUTPUT
        # =====================================================================

        dma_coverage.to_csv(
            TABLE_DIR
            / "dma_source_coverage.csv",
            index=False,
        )

        # =====================================================================
        # 14. FIRST-PASS SUMMARY
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "DMA MERCHANDISE MIX INITIALIZATION COMPLETE"
        )
        logger.info("=" * 80)

        logger.info(
            "Merchandise rows: %s",
            f"{len(merchandise):,}",
        )

        logger.info(
            "Merchandise DMAs: %s",
            f"{merchandise['DMA_Key'].nunique():,}",
        )

        logger.info(
            "Treated DMAs: %s",
            f"{bsts['DMA_Key'].nunique():,}",
        )

        logger.info(
            "Primary eligible DMAs: %s",
            f"{dma_coverage['Primary_Analysis_Eligible'].sum():,}",
        )

        logger.info("=" * 80)
        logger.info("SUCCESS")
        logger.info("=" * 80)

    except Exception:

        logger.exception(
            "DMA MERCHANDISE MIX DIAGNOSTICS FAILED."
        )

        raise


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()


