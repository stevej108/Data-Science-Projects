"""
13a_comp_traffic_analysis.py

Comp Traffic Analysis
=====================

Purpose
-------
Build a clean store-level and DMA-level comp traffic dataset from
weekly store traffic.

Core methodology
----------------
1. Load weekly store traffic.
2. Validate Store, WeekEndDate, Exits, and Exits_LY.
3. Calculate weekly store-level Comp Traffic:

       Comp Traffic = ((Exits - Exits_LY) / Exits_LY) * 100

4. Aggregate weekly traffic to store-month.
5. Calculate monthly comp traffic using:

       Monthly Comp Traffic =
           ((Monthly Exits - Monthly Exits LY)
            / Monthly Exits LY) * 100

   This is intentionally calculated from summed Exits and Exits LY,
   rather than averaging weekly comp percentages.

6. Load stores.csv.
7. Map stores to DMA.
8. Aggregate store-month traffic to DMA-month.
9. Produce national and DMA-level diagnostics.
10. Export clean datasets and diagnostic tables.
11. Generate figures.
12. Generate an internal HTML report.

Important
---------
Comp traffic is calculated BEFORE DMA aggregation.

This preserves the store-level comparison and prevents stores with
different traffic volumes from receiving equal weight simply because
they contribute one observation to an aggregated percentage.

The monthly DMA comp traffic measure is therefore:

    Sum(Exits) - Sum(Exits_LY)
    --------------------------- * 100
         Sum(Exits_LY)

This is equivalent to weighting store-level monthly comp traffic by
prior-year traffic volume.

Author
------
Marketing Incrementality Project
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


# =============================================================================
# CONFIGURATION
# =============================================================================

try:
    from config import (
        DATA_DIR,
        OUTPUT_DIR,
        FIGURE_DIR,
        TABLE_DIR,
        REPORT_DIR,
    )
except ImportError:
    # -------------------------------------------------------------------------
    # Fallback paths.
    #
    # These are only used if config.py does not expose the expected variables.
    # -------------------------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parent

    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    FIGURE_DIR = OUTPUT_DIR / "figures"
    TABLE_DIR = OUTPUT_DIR / "tables"
    REPORT_DIR = OUTPUT_DIR / "reports"


# =============================================================================
# FILE PATHS
# =============================================================================

TRAFFIC_FILE = DATA_DIR / "traffic.csv"
STORES_FILE = DATA_DIR / "interim" / "stores_clean.parquet"
SPEND_FILE = DATA_DIR / "spend.csv"


# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

COMP_OUTPUT_DIR = OUTPUT_DIR / "comp_traffic"

COMP_FIGURE_DIR = FIGURE_DIR / "comp_traffic"

COMP_TABLE_DIR = TABLE_DIR / "comp_traffic"

COMP_REPORT_DIR = REPORT_DIR / "comp_traffic"


# =============================================================================
# LOGGING
# =============================================================================

# =============================================================================
# LOGGING
# =============================================================================

import logging
import sys


def configure_logging():
    """
    Configure console logging for the analysis.

    INFO-level messages are written directly to stdout so that the
    workflow progress is visible during execution.
    """

    logger = logging.getLogger()

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(
        sys.stdout
    )

    console_handler.setLevel(
        logging.INFO
    )

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    return logger


logger = configure_logging()


# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)


# =============================================================================
# DIRECTORY SETUP
# =============================================================================

def create_output_directories():
    """
    Create all output directories used by the analysis.
    """

    directories = [
        COMP_OUTPUT_DIR,
        COMP_FIGURE_DIR,
        COMP_TABLE_DIR,
        COMP_REPORT_DIR,

        COMP_FIGURE_DIR / "traffic",
        COMP_FIGURE_DIR / "comp",
        COMP_FIGURE_DIR / "coverage",
        COMP_FIGURE_DIR / "dma",
        COMP_FIGURE_DIR / "diagnostics",

        COMP_TABLE_DIR / "raw",
        COMP_TABLE_DIR / "store",
        COMP_TABLE_DIR / "dma",
        COMP_TABLE_DIR / "diagnostics",
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# COLUMN NORMALIZATION
# =============================================================================

def normalize_columns(df):
    """
    Normalize column names.

    Examples
    --------
    "Week End Date" -> "WeekEndDate"
    "Exits LY"      -> "Exits_LY"
    """

    df = df.copy()

    normalized = {}

    for column in df.columns:

        clean = (
            str(column)
            .strip()
            .replace(" ", "")
            .replace("-", "")
            .replace("/", "")
            .replace("#", "")
        )

        normalized[column] = clean

    df = df.rename(columns=normalized)

    # Explicit aliases
    aliases = {
        "WeekEndDate": "WeekEndDate",
        "WeekendDate": "WeekEndDate",
        "WeekEnd": "WeekEndDate",

        "Exits": "Exits",
        "ExitsLY": "Exits_LY",

        # "Store": "Store",
        # "StoreNumber": "Store",
        # "StoreID": "Store",
        # "StoreLocation": "Store",

        "DMA": "DMA",
    }

    rename_map = {}

    for column in df.columns:

        if column in aliases:
            rename_map[column] = aliases[column]

    df = df.rename(columns=rename_map)

    return df


# =============================================================================
# STORE ID NORMALIZATION
# =============================================================================

def normalize_store_ids(series):
    """
    Normalize store identifiers.

    Numeric store IDs are converted to strings without trailing .0.
    """

    result = (
        series
        .astype("string")
        .str.strip()
    )

    result = (
        result
        .str.replace(r"\.0$", "", regex=True)
    )

    result = result.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "NULL": pd.NA,
        }
    )

    return result


def normalize_dma_names(
    df,
    dma_col="DMA",
):
    """
    Normalize DMA names to the canonical marketing-analysis
    naming convention.

    The canonical names correspond to the DMA naming used
    in the marketing spend data.
    """

    df = df.copy()

    df[dma_col] = (
        df[dma_col]
        .astype("string")
        .str.strip()
    )

    dma_map = {
        # -------------------------------------------------------------
        # Major DMAs
        # -------------------------------------------------------------
        "Atlanta, GA": "Atlanta",
        "Baltimore, MD": "Baltimore",
        "Boston, MA (Manchester, NH)": "Boston",
        "Buffalo, NY": "Buffalo",
        "Charlotte, NC": "Charlotte",
        "Chicago, IL": "Chicago",
        "Cincinnati, OH": "Cincinnati",
        "Cleveland-Akron (Canton), OH": "Cleveland",
        "Columbus, OH": "Columbus",
        "Dallas-Ft. Worth, TX": "Dallas",
        "Detroit, MI": "Detroit",
        "El Paso, TX (Las Cruces, NM)": "El Paso",
        "Ft. Myers-Naples, FL": "Fort Myers",
        "Harlingen-Weslaco-Brownsville-McAllen, TX":
            "Harlingen/McAllen",
        "Hartford & New Haven, CT": "Hartford",
        "Houston, TX": "Houston",
        "Las Vegas, NV": "Las Vegas",
        "Los Angeles, CA": "Los Angeles",
        "Miami-Ft. Lauderdale, FL": "Miami",
        "Minneapolis-St. Paul, MN": "Minneapolis",
        "New York, NY": "New York",
        "Oklahoma City, OK": "Oklahoma City",
        "Orlando-Daytona Beach-Melbourne, FL": "Orlando",
        "Philadelphia, PA": "Philadelphia",
        "Phoenix (Prescott), AZ": "Phoenix",
        "Providence, RI-New Bedford, MA": "Providence",
        "Sacramento-Stockton-Modesto, CA": "Sacramento",
        "San Antonio, TX": "San Antonio",
        "San Diego, CA": "San Diego",
        "Seattle-Tacoma, WA": "Seattle",
        "Tampa-St. Petersburg (Sarasota), FL": "Tampa",
        "Washington, DC (Hagerstown, MD)": "Washington DC",
        "West Palm Beach-Ft. Pierce, FL": "West Palm",

        # Already canonical / special
        "Puerto Rico": "Puerto Rico",
        "National (Other)": "National (Other)",
    }

    df[dma_col] = (
        df[dma_col]
        .replace(dma_map)
    )

    return df

def parse_numeric_column(
    series,
):
    """
    Convert numeric-like values to float.

    Handles values such as:
        4651
        "4,651"
        " 4,651 "
        ""
        "-"
        NaN
    """

    cleaned = (
        series
        .astype("string")
        .str.strip()
        .str.replace(
            ",",
            "",
            regex=False,
        )
    )

    cleaned = cleaned.replace(
        {
            "": pd.NA,
            "-": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
        }
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    )

# =============================================================================
# LOAD TRAFFIC
# =============================================================================

def load_store_traffic(
    traffic_path=None,
):
    """
    Load raw weekly store traffic.

    Required columns
    ----------------
    Store
    WeekEndDate
    Exits
    Exits_LY

    Exits_LY is required because comp traffic is calculated directly
    from weekly store-level traffic.
    """

    traffic_path = Path(
        traffic_path or TRAFFIC_FILE
    )

    if not traffic_path.exists():
        raise FileNotFoundError(
            f"Traffic file does not exist: {traffic_path}"
        )

    logger.info("=" * 70)
    logger.info("LOADING STORE-LEVEL TRAFFIC")
    logger.info("=" * 70)

    logger.info(
        "Loading traffic from %s",
        traffic_path,
    )

    logger.info(
        "Resolved traffic path: %s",
        traffic_path.resolve(),
    )

    # -------------------------------------------------------------------------
    # Load raw CSV
    #
    # low_memory=False prevents pandas from inferring column types
    # independently across chunks for mixed-type columns.
    # -------------------------------------------------------------------------

    traffic = pd.read_csv(
        traffic_path,
        low_memory=False,
    )

    logger.info(
        "Original traffic columns: %s",
        traffic.columns.tolist(),
    )

    # -------------------------------------------------------------------------
    # Normalize COLUMN NAMES before diagnostics.
    #
    # This does not alter row values or filter the dataset.
    # It simply converts raw names such as:
    #
    #     " Exits "
    #     " Exits LY "
    #
    # into the canonical names used throughout this script:
    #
    #     Exits
    #     Exits_LY
    # -------------------------------------------------------------------------

    traffic = normalize_columns(
        traffic
    )

    logger.info(
        "Normalized traffic columns: %s",
        traffic.columns.tolist(),
    )

    # -------------------------------------------------------------------------
    # Validate required columns BEFORE diagnostics
    # -------------------------------------------------------------------------

    required_columns = {
        "Store",
        "WeekEndDate",
        "Exits",
        "Exits_LY",
    }

    missing_columns = (
        required_columns
        -
        set(traffic.columns)
    )

    if missing_columns:

        raise ValueError(
            "Traffic file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # =========================================================================
    # DIAGNOSTIC 1:
    # RAW MONTHLY COVERAGE BEFORE VALUE CONVERSION / FILTERING
    # =========================================================================

    raw_diag = traffic.copy()

    raw_diag[
        "WeekEndDate_Diagnostic"
    ] = pd.to_datetime(
        raw_diag["WeekEndDate"],
        errors="coerce",
    )

    raw_diag[
        "Traffic_Month_Diagnostic"
    ] = (
        raw_diag[
            "WeekEndDate_Diagnostic"
        ]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    raw_monthly_coverage = (
        raw_diag
        .groupby(
            "Traffic_Month_Diagnostic",
            dropna=False,
        )
        .agg(
            Rows=(
                "Store",
                "size",
            ),
            Stores=(
                "Store",
                "nunique",
            ),
            Weeks=(
                "WeekEndDate_Diagnostic",
                "nunique",
            ),
            Missing_Store=(
                "Store",
                lambda x:
                    x.isna().sum(),
            ),
            Missing_WeekEndDate=(
                "WeekEndDate_Diagnostic",
                lambda x:
                    x.isna().sum(),
            ),
            Missing_Exits=(
                "Exits",
                lambda x:
                    x.isna().sum(),
            ),
            Missing_Exits_LY=(
                "Exits_LY",
                lambda x:
                    x.isna().sum(),
            ),
        )
        .reset_index()
    )

    logger.info(
        "RAW MONTHLY TRAFFIC COVERAGE "
        "BEFORE CONVERSION/FILTERING:\n%s",
        raw_monthly_coverage.to_string(
            index=False
        ),
    )

    raw_monthly_coverage.to_csv(
        COMP_TABLE_DIR
        / "diagnostics"
        / "raw_monthly_coverage_before_filtering.csv",
        index=False,
    )

    # =========================================================================
    # NORMALIZE DATA TYPES
    # =========================================================================

    traffic["Store"] = normalize_store_ids(
        traffic["Store"]
    )

    traffic["WeekEndDate"] = pd.to_datetime(
        traffic["WeekEndDate"],
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Preserve raw values temporarily so we can determine whether numeric
    # coercion is causing the observed loss of later-period traffic.
    # -------------------------------------------------------------------------

    traffic["Exits_Raw"] = (
        traffic["Exits"]
    )

    traffic["Exits_LY_Raw"] = (
        traffic["Exits_LY"]
    )

    traffic["Exits"] = parse_numeric_column(
        traffic["Exits"]
    )

    traffic["Exits_LY"] = parse_numeric_column(
        traffic["Exits_LY"]
    )

    traffic["SaleTxns"] = parse_numeric_column(
        traffic["SaleTxns"]
    )

    traffic["SaleTxnsLY"] = parse_numeric_column(
        traffic["SaleTxnsLY"]
    )

    # =========================================================================
    # DIAGNOSTIC 2:
    # COVERAGE AFTER TYPE CONVERSION, BEFORE ROW REMOVAL
    # =========================================================================

    pre_filter_coverage = (
        traffic
        .assign(
            Traffic_Month=(
                traffic[
                    "WeekEndDate"
                ]
                .dt.to_period("M")
                .dt.to_timestamp()
            )
        )
        .groupby(
            "Traffic_Month",
            dropna=False,
        )
        .agg(
            Rows=(
                "Store",
                "size",
            ),
            Stores=(
                "Store",
                "nunique",
            ),
            Weeks=(
                "WeekEndDate",
                "nunique",
            ),
            Valid_Exits=(
                "Exits",
                lambda x:
                    x.notna().sum(),
            ),
            Missing_Exits=(
                "Exits",
                lambda x:
                    x.isna().sum(),
            ),
            Valid_Exits_LY=(
                "Exits_LY",
                lambda x:
                    x.notna().sum(),
            ),
            Missing_Exits_LY=(
                "Exits_LY",
                lambda x:
                    x.isna().sum(),
            ),
        )
        .reset_index()
    )

    pre_filter_coverage[
        "Pct_Valid_Exits"
    ] = (
        pre_filter_coverage[
            "Valid_Exits"
        ]
        /
        pre_filter_coverage[
            "Rows"
        ]
    )

    pre_filter_coverage[
        "Pct_Valid_Exits_LY"
    ] = (
        pre_filter_coverage[
            "Valid_Exits_LY"
        ]
        /
        pre_filter_coverage[
            "Rows"
        ]
    )

    logger.info(
        "MONTHLY TRAFFIC COVERAGE "
        "AFTER TYPE CONVERSION / BEFORE ROW FILTER:\n%s",
        pre_filter_coverage.to_string(
            index=False
        ),
    )

    pre_filter_coverage.to_csv(
        COMP_TABLE_DIR
        / "diagnostics"
        / "monthly_coverage_before_row_filter.csv",
        index=False,
    )

    # =========================================================================
    # NUMERIC CONVERSION DIAGNOSTICS
    # =========================================================================

    failed_exits_conversion = (
        traffic["Exits"].isna()
        &
        traffic["Exits_Raw"].notna()
    )

    failed_ly_conversion = (
        traffic["Exits_LY"].isna()
        &
        traffic["Exits_LY_Raw"].notna()
    )

    logger.info(
        "Exits values failing numeric conversion: %s",
        f"{failed_exits_conversion.sum():,}",
    )

    logger.info(
        "Exits_LY values failing numeric conversion: %s",
        f"{failed_ly_conversion.sum():,}",
    )

    if failed_exits_conversion.any():

        failed = traffic.loc[
            failed_exits_conversion,
            [
                "Store",
                "WeekEndDate",
                "Exits_Raw",
            ],
        ].copy()

        logger.warning(
            "Sample Exits values failing numeric conversion:\n%s",
            failed.head(50).to_string(
                index=False
            ),
        )

        failed.to_csv(
            COMP_TABLE_DIR
            / "diagnostics"
            / "failed_exits_numeric_conversion.csv",
            index=False,
        )

    if failed_ly_conversion.any():

        failed = traffic.loc[
            failed_ly_conversion,
            [
                "Store",
                "WeekEndDate",
                "Exits_LY_Raw",
            ],
        ].copy()

        logger.warning(
            "Sample Exits_LY values failing numeric conversion:\n%s",
            failed.head(50).to_string(
                index=False
            ),
        )

        failed.to_csv(
            COMP_TABLE_DIR
            / "diagnostics"
            / "failed_exits_ly_numeric_conversion.csv",
            index=False,
        )

    # =========================================================================
    # MISSINGNESS DIAGNOSTICS
    # =========================================================================

    logger.info(
        "Invalid Store rows: %s",
        f"{traffic['Store'].isna().sum():,}",
    )

    logger.info(
        "Invalid WeekEndDate rows: %s",
        f"{traffic['WeekEndDate'].isna().sum():,}",
    )

    logger.info(
        "Invalid Exits rows: %s",
        f"{traffic['Exits'].isna().sum():,}",
    )

    logger.info(
        "Invalid Exits_LY rows: %s",
        f"{traffic['Exits_LY'].isna().sum():,}",
    )

    # =========================================================================
    # REMOVE UNUSABLE ROWS
    # =========================================================================

    rows_before_filter = len(
        traffic
    )

    traffic = traffic.loc[
        traffic["Store"].notna()
        &
        traffic["WeekEndDate"].notna()
        &
        traffic["Exits"].notna()
        &
        traffic["Exits_LY"].notna()
    ].copy()

    logger.info(
        "Rows removed during validation filter: %s",
        f"{rows_before_filter - len(traffic):,}",
    )

    if traffic.empty:

        raise ValueError(
            "Traffic file is empty after validation."
        )

    # =========================================================================
    # DIAGNOSTIC 3:
    # COVERAGE AFTER VALIDATION / FILTERING
    # =========================================================================

    post_filter_coverage = (
        traffic
        .assign(
            Traffic_Month=(
                traffic[
                    "WeekEndDate"
                ]
                .dt.to_period("M")
                .dt.to_timestamp()
            )
        )
        .groupby(
            "Traffic_Month"
        )
        .agg(
            Rows=(
                "Store",
                "size",
            ),
            Stores=(
                "Store",
                "nunique",
            ),
            Weeks=(
                "WeekEndDate",
                "nunique",
            ),
            Exits=(
                "Exits",
                "sum",
            ),
            Exits_LY=(
                "Exits_LY",
                "sum",
            ),
        )
        .reset_index()
    )

    logger.info(
        "MONTHLY TRAFFIC COVERAGE "
        "AFTER ROW FILTER:\n%s",
        post_filter_coverage.to_string(
            index=False
        ),
    )

    post_filter_coverage.to_csv(
        COMP_TABLE_DIR
        / "diagnostics"
        / "monthly_coverage_after_row_filter.csv",
        index=False,
    )

    # =========================================================================
    # CALENDAR FIELDS
    # =========================================================================

    traffic["Traffic_Year"] = (
        traffic[
            "WeekEndDate"
        ]
        .dt.year
    )

    traffic["Traffic_Month"] = (
        traffic[
            "WeekEndDate"
        ]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    traffic[
        "Traffic_Month_Number"
    ] = (
        traffic[
            "WeekEndDate"
        ]
        .dt.month
    )

    # -------------------------------------------------------------------------
    # Remove diagnostic-only raw columns before returning
    # -------------------------------------------------------------------------

    traffic = traffic.drop(
        columns=[
            "Exits_Raw",
            "Exits_LY_Raw",
        ],
        errors="ignore",
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    logger.info(
        "Loaded %s store-week rows.",
        f"{len(traffic):,}",
    )

    logger.info(
        "Traffic coverage: %s to %s",
        traffic["WeekEndDate"].min().date(),
        traffic["WeekEndDate"].max().date(),
    )

    logger.info(
        "Unique stores: %s",
        f"{traffic['Store'].nunique():,}",
    )

    logger.info(
        "Unique weeks: %s",
        f"{traffic['WeekEndDate'].nunique():,}",
    )

    logger.info(
        "SaleTxns dtype after normalization: %s",
        traffic["SaleTxns"].dtype,
    )

    return traffic


# =============================================================================
# LOAD SPEND
# =============================================================================



def load_spend(
    spend_path=None,
):
    """
    Load and reshape the wide-format DMA marketing spend file.

    Expected raw structure:

        DMA | 25-Feb | 25-Mar | ... | Q125 | ... | FY25 | 26-Feb | ...

    Only monthly spend columns are retained.

    Returns
    -------
    pandas.DataFrame
        Long-format spend data with:

            DMA
            Month
            Spend
    """

    spend_path = Path(
        spend_path or SPEND_FILE
    )

    if not spend_path.exists():
        raise FileNotFoundError(
            f"Spend file does not exist: {spend_path}"
        )

    logger.info("=" * 70)
    logger.info("LOADING MARKETING SPEND")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Load raw file
    # -------------------------------------------------------------------------

    spend_raw = pd.read_csv(
        spend_path,
        
    )

    logger.info(
        "Raw spend shape: %s rows x %s columns",
        f"{spend_raw.shape[0]:,}",
        f"{spend_raw.shape[1]:,}",
    )

    logger.info(
        "Raw spend columns: %s",
        spend_raw.columns.tolist(),
    )

    # -------------------------------------------------------------------------
    # Normalize column names
    # -------------------------------------------------------------------------

    spend_raw.columns = (
        spend_raw.columns
        .astype(str)
        .str.strip()
    )

    # -------------------------------------------------------------------------
    # Validate DMA
    # -------------------------------------------------------------------------

    if "DMA" not in spend_raw.columns:

        raise ValueError(
            "Spend file is missing required column: 'DMA'. "
            f"Available columns: {spend_raw.columns.tolist()}"
        )    
    
    # Identify monthly columns
    #
    # Expected pattern:
    #
    #   25-Feb
    #   25-Mar
    #   ...
    #   26-Jul
    #
    # This intentionally excludes:
    #
    #   Q125
    #   Q225
    #   Q325
    #   Q425
    #   FY25
    #   Q126
    #   Q226
    #   Unnamed: 26
    # -------------------------------------------------------------------------

    monthly_columns = []

    for column in spend_raw.columns:

        if re.fullmatch(
            r"\d{2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
            column,
            flags=re.IGNORECASE,
        ):
            monthly_columns.append(
                column
            )

    if not monthly_columns:

        raise ValueError(
            "No monthly spend columns were identified. "
            f"Available columns: {spend_raw.columns.tolist()}"
        )

    logger.info(
        "Monthly spend columns identified: %s",
        f"{len(monthly_columns):,}",
    )

    logger.info(
        "Monthly columns: %s",
        monthly_columns,
    )

    # -------------------------------------------------------------------------
    # Keep only DMA + monthly columns
    # -------------------------------------------------------------------------

    spend = spend_raw[
        [
            "DMA",
            *monthly_columns,
        ]
    ].copy()

    # -------------------------------------------------------------------------
    # Normalize DMA strings
    # -------------------------------------------------------------------------

    spend["DMA"] = (
        spend["DMA"]
        .astype("string")
        .str.strip()
    )

    spend = normalize_dma_names(
        spend,
        dma_col="DMA",
    )

    spend.loc[
        spend["DMA"].isin(
            [
                "",
                "nan",
                "None",
                "<NA>",
            ]
        ),
        "DMA",
    ] = pd.NA


    # -------------------------------------------------------------------------
    # Remove non-DMA summary rows
    # -------------------------------------------------------------------------

    summary_rows = (
        spend["DMA"]
        .str.upper()
        .isin(
            [
                "TOTALS",
                "TOTAL",
            ]
        )
    )

    if summary_rows.any():

        logger.info(
            "Removing %s non-DMA summary rows from spend.",
            f"{summary_rows.sum():,}",
        )

        spend = spend.loc[
            ~summary_rows
        ].copy()

    # -------------------------------------------------------------------------
    # Remove rows without DMA
    # -------------------------------------------------------------------------

    before = len(spend)

    spend = spend.loc[
        spend["DMA"].notna()
    ].copy()

    logger.info(
        "Removed %s spend rows without DMA.",
        f"{before - len(spend):,}",
    )

    # -------------------------------------------------------------------------
    # Convert wide -> long
    # -------------------------------------------------------------------------

    spend = spend.melt(
        id_vars="DMA",
        value_vars=monthly_columns,
        var_name="Month",
        value_name="Spend",
    )

    # -------------------------------------------------------------------------
    # Convert Month
    #
    # Example:
    #
    #   25-Feb -> 2025-02-01
    #   26-Jul -> 2026-07-01
    # -------------------------------------------------------------------------

    spend["Month"] = pd.to_datetime(
        spend["Month"],
        format="%y-%b",
        errors="coerce",
    )

    invalid_months = spend["Month"].isna()

    if invalid_months.any():

        logger.warning(
            "Found %s rows with invalid spend months.",
            f"{invalid_months.sum():,}",
        )

        logger.warning(
            "Invalid month values: %s",
            spend.loc[
                invalid_months,
                "Month",
            ].unique().tolist(),
        )

        spend = spend.loc[
            ~invalid_months
        ].copy()

    # -------------------------------------------------------------------------
    # Convert Spend
    #
    # Raw values look like:
    #
    #   $138K
    #   $215K
    #   -$6K
    #   -
    #   NaN
    #
    # Convert these to numeric dollar amounts.
    # -------------------------------------------------------------------------

    def parse_spend_value(value):

        if pd.isna(value):
            return np.nan

        value = str(value).strip()

        if value in {
            "",
            "-",
            "nan",
            "None",
            "<NA>",
        }:
            return np.nan

        # Remove dollar sign and commas
        value = (
            value
            .replace("$", "")
            .replace(",", "")
            .strip()
        )

        # Handle negative values such as:
        #
        #   -$6K
        #
        # which after removing $ becomes:
        #
        #   -6K
        #
        multiplier = 1.0

        if value.upper().endswith("K"):

            multiplier = 1_000.0

            value = value[:-1]

        elif value.upper().endswith("M"):

            multiplier = 1_000_000.0

            value = value[:-1]

        elif value.upper().endswith("B"):

            multiplier = 1_000_000_000.0

            value = value[:-1]

        try:

            return float(value) * multiplier

        except (TypeError, ValueError):

            return np.nan

    spend["Spend"] = (
        spend["Spend"]
        .apply(parse_spend_value)
        .astype(float)
    )

    # -------------------------------------------------------------------------
    # Sort
    # -------------------------------------------------------------------------

    spend = (
        spend
        .sort_values(
            [
                "DMA",
                "Month",
            ]
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    logger.info(
        "Long-format spend rows: %s",
        f"{len(spend):,}",
    )

    logger.info(
        "Unique spend DMAs: %s",
        f"{spend['DMA'].nunique():,}",
    )

    logger.info(
        "Spend date range: %s to %s",
        spend["Month"].min().strftime("%Y-%m-%d"),
        spend["Month"].max().strftime("%Y-%m-%d"),
    )

    logger.info(
        "Non-zero spend observations: %s",
        f"{(spend['Spend'].fillna(0) > 0).sum():,}",
    )

    logger.info(
        "Total spend: $%s",
        f"{spend['Spend'].sum():,.0f}",
    )

    # -------------------------------------------------------------------------
    # Monthly spend coverage
    # -------------------------------------------------------------------------

    monthly_summary = (
        spend
        .groupby("Month", as_index=False)
        .agg(
            DMAs=("DMA", "nunique"),
            Total_Spend=("Spend", "sum"),
            Nonzero_DMA_Count=(
                "Spend",
                lambda x: (x.fillna(0) > 0).sum(),
            ),
        )
        .sort_values("Month")
    )

    logger.info(
        "Monthly spend coverage:\n%s",
        monthly_summary.to_string(index=False),
    )

    return spend

# =============================================================================
# PREPARE TRAFFIC
# =============================================================================

def prepare_traffic(
    traffic,
):
    """
    Prepare weekly store traffic and calculate weekly comp traffic.

    Formula
    -------

        ((Exits - Exits_LY) / Exits_LY) * 100

    Rows where Exits_LY <= 0 cannot produce a valid comp percentage
    and are retained with NaN Comp_Traffic rather than silently removed.
    """

    traffic = traffic.copy()

    logger.info("=" * 70)
    logger.info("PREPARING WEEKLY STORE TRAFFIC")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Duplicate Store-Date validation
    # -------------------------------------------------------------------------

    duplicate_mask = traffic.duplicated(
        subset=[
            "Store",
            "WeekEndDate",
        ],
        keep=False,
    )

    duplicate_count = duplicate_mask.sum()

    if duplicate_count > 0:

        logger.warning(
            "Found %s duplicate Store-Date rows.",
            f"{duplicate_count:,}",
        )

        duplicate_rows = (
            traffic.loc[
                duplicate_mask
            ]
            .sort_values(
                [
                    "Store",
                    "WeekEndDate",
                ]
            )
        )

        duplicate_rows.to_csv(
            COMP_TABLE_DIR
            / "diagnostics"
            / "duplicate_store_week_rows.csv",
            index=False,
        )

        raise ValueError(
            "Duplicate Store-Week rows detected. "
            "Resolve duplicates before calculating comp traffic."
        )

    logger.info(
        "No duplicate Store-Date rows detected."
    )

    # -------------------------------------------------------------------------
    # Comp traffic
    # -------------------------------------------------------------------------

    valid_ly = (
        traffic["Exits_LY"] > 0
    )

    traffic["Comp_Traffic"] = np.nan

    traffic.loc[
        valid_ly,
        "Comp_Traffic",
    ] = (
        (
            traffic.loc[
                valid_ly,
                "Exits"
            ]
            -
            traffic.loc[
                valid_ly,
                "Exits_LY"
            ]
        )
        /
        traffic.loc[
            valid_ly,
            "Exits_LY"
        ]
        *
        100
    )

    invalid_ly = (
        ~valid_ly
    )

    logger.info(
        "Rows with Exits_LY <= 0: %s",
        f"{invalid_ly.sum():,}",
    )

    logger.info(
        "Valid weekly comp rows: %s",
        f"{traffic['Comp_Traffic'].notna().sum():,}",
    )

    logger.info(
        "Weekly comp traffic mean: %.3f%%",
        traffic["Comp_Traffic"].mean(),
    )

    logger.info(
        "Weekly comp traffic median: %.3f%%",
        traffic["Comp_Traffic"].median(),
    )

    return traffic


# =============================================================================
# RAW WEEKLY DIAGNOSTICS
# =============================================================================

def log_weekly_coverage(
    traffic,
):
    """
    Log weekly store coverage.
    """

    logger.info("=" * 70)
    logger.info("WEEKLY STORE COVERAGE DIAGNOSTICS")
    logger.info("=" * 70)

    coverage = (
        traffic
        .groupby("WeekEndDate")
        .agg(
            Stores=("Store", "nunique"),
            Rows=("Store", "size"),
            Exits=("Exits", "sum"),
            Exits_LY=("Exits_LY", "sum"),
        )
        .sort_index()
    )

    for year in sorted(
        traffic["Traffic_Year"].unique()
    ):

        logger.info(
            "%s weekly coverage:",
            year,
        )

        year_coverage = coverage.loc[
            coverage.index.year == year
        ]

        for date, row in year_coverage.iterrows():

            logger.info(
                "  %s | Stores: %s | Rows: %s | "
                "Exits: %s | Exits LY: %s",
                date.date(),
                f"{int(row['Stores']):,}",
                f"{int(row['Rows']):,}",
                f"{int(row['Exits']):,}",
                f"{int(row['Exits_LY']):,}",
            )

    coverage.to_csv(
        COMP_TABLE_DIR
        / "diagnostics"
        / "weekly_store_coverage.csv"
    )

    return coverage


# =============================================================================
# STORE YEAR COVERAGE
# =============================================================================

def log_store_year_coverage(
    traffic,
):
    """
    Summarize the number of weeks observed for each store by year.
    """

    logger.info("=" * 70)
    logger.info("STORE-LEVEL YEAR COVERAGE")
    logger.info("=" * 70)

    store_year = (
        traffic
        .groupby(
            [
                "Traffic_Year",
                "Store",
            ]
        )
        ["WeekEndDate"]
        .nunique()
        .rename("Weeks")
        .reset_index()
    )

    summary_rows = []

    for year in sorted(
        store_year["Traffic_Year"].unique()
    ):

        year_data = store_year.loc[
            store_year["Traffic_Year"] == year,
            "Weeks",
        ]

        logger.info(
            "%s stores: %s",
            year,
            f"{len(year_data):,}",
        )

        logger.info(
            "%s mean weeks/store: %.2f",
            year,
            year_data.mean(),
        )

        logger.info(
            "%s median weeks/store: %.2f",
            year,
            year_data.median(),
        )

        for threshold in [
            4,
            8,
            12,
            16,
            17,
            20,
            26,
            39,
            48,
        ]:

            count = (
                year_data >= threshold
            ).sum()

            logger.info(
                "%s stores with %s+ weeks: %s",
                year,
                threshold,
                f"{count:,}",
            )

        summary_rows.append(
            {
                "Year": year,
                "Stores": len(year_data),
                "Mean_Weeks": year_data.mean(),
                "Median_Weeks": year_data.median(),
                "Stores_16Plus": (
                    year_data >= 16
                ).sum(),
                "Stores_26Plus": (
                    year_data >= 26
                ).sum(),
                "Stores_39Plus": (
                    year_data >= 39
                ).sum(),
                "Stores_48Plus": (
                    year_data >= 48
                ).sum(),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        COMP_TABLE_DIR
        / "diagnostics"
        / "store_year_coverage.csv",
        index=False,
    )

    return store_year, summary


# =============================================================================
# STORE-MONTH AGGREGATION
# =============================================================================

def aggregate_store_month(
    traffic,
):
    """
    Aggregate weekly traffic to store-month.

    IMPORTANT
    ---------
    Monthly comp traffic is calculated from the monthly totals:

        (sum Exits - sum Exits_LY)
        -------------------------- * 100
             sum Exits_LY

    We do NOT average weekly Comp_Traffic.

    This prevents a low-volume week from receiving the same weight
    as a high-volume week.
    """

    logger.info("=" * 70)
    logger.info("AGGREGATING WEEKLY TRAFFIC TO STORE-MONTH")
    logger.info("=" * 70)

    store_month = (
        traffic
        .groupby(
            [
                "Store",
                "Traffic_Month",
            ],
            as_index=False,
        )
        .agg(
            Exits=(
                "Exits",
                "sum",
            ),
            Exits_LY=(
                "Exits_LY",
                "sum",
            ),
            Weeks_Observed=(
                "WeekEndDate",
                "nunique",
            ),
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Monthly comp
    # -------------------------------------------------------------------------

    valid_ly = (
        store_month["Exits_LY"] > 0
    )

    store_month["Comp_Traffic"] = np.nan

    store_month.loc[
        valid_ly,
        "Comp_Traffic",
    ] = (
        (
            store_month.loc[
                valid_ly,
                "Exits"
            ]
            -
            store_month.loc[
                valid_ly,
                "Exits_LY"
            ]
        )
        /
        store_month.loc[
            valid_ly,
            "Exits_LY"
        ]
        *
        100
    )

    store_month["Traffic_Year"] = (
        store_month["Traffic_Month"]
        .dt.year
    )

    store_month["Traffic_Month_Number"] = (
        store_month["Traffic_Month"]
        .dt.month
    )

    logger.info(
        "Store-month rows: %s",
        f"{len(store_month):,}",
    )

    logger.info(
        "Unique stores: %s",
        f"{store_month['Store'].nunique():,}",
    )

    logger.info(
        "Months: %s",
        f"{store_month['Traffic_Month'].nunique():,}",
    )

    logger.info(
        "Store-month comp mean: %.3f%%",
        store_month["Comp_Traffic"].mean(),
    )

    logger.info(
        "Store-month comp median: %.3f%%",
        store_month["Comp_Traffic"].median(),
    )

    return store_month


# =============================================================================
# MONTHLY STORE COVERAGE
# =============================================================================

def log_monthly_store_diagnostics(
    store_month,
):
    """
    Diagnose store-month coverage.
    """

    logger.info("=" * 70)
    logger.info("STORE-MONTH TRAFFIC DIAGNOSTICS")
    logger.info("=" * 70)

    monthly = (
        store_month
        .groupby("Traffic_Month")
        .agg(
            Stores=("Store", "nunique"),
            Rows=("Store", "size"),
            Exits=("Exits", "sum"),
            Exits_LY=("Exits_LY", "sum"),
            Mean_Comp=("Comp_Traffic", "mean"),
            Median_Comp=("Comp_Traffic", "median"),
            Min_Weeks=("Weeks_Observed", "min"),
            Mean_Weeks=("Weeks_Observed", "mean"),
            Max_Weeks=("Weeks_Observed", "max"),
        )
        .sort_index()
    )

    for date, row in monthly.iterrows():

        logger.info(
            "%s | Stores: %s | Rows: %s | "
            "Exits: %s | Exits LY: %s | "
            "Comp: %.2f%% | Mean Weeks: %.2f",
            date.strftime("%Y-%m"),
            f"{int(row['Stores']):,}",
            f"{int(row['Rows']):,}",
            f"{int(row['Exits']):,}",
            f"{int(row['Exits_LY']):,}",
            row["Mean_Comp"],
            row["Mean_Weeks"],
        )

    monthly.to_csv(
        COMP_TABLE_DIR
        / "diagnostics"
        / "store_month_coverage.csv"
    )

    return monthly


# =============================================================================
# LOAD STORES
# =============================================================================

def load_stores(
    stores_path=None,
):
    """
    Load store master data.

    The store master must contain:
        Store
        DMA

    Additional columns are retained for possible future analysis.
    """

    stores_path = Path(
        stores_path or STORES_FILE
    )

    if not stores_path.exists():
        raise FileNotFoundError(
            f"Stores file does not exist: {stores_path}"
        )

    logger.info("=" * 70)
    logger.info("LOADING STORE MASTER")
    logger.info("=" * 70)

    stores = pd.read_parquet(
        stores_path,
        
    )

    stores = normalize_columns(
        stores
    )

    required_columns = {
        "Store",
        "DMA",
    }

    missing_columns = (
        required_columns
        -
        set(stores.columns)
    )

    if missing_columns:

        raise ValueError(
            "Stores file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    stores["Store"] = normalize_store_ids(
        stores["Store"]
    )

    stores["DMA"] = (
        stores["DMA"]
        .astype("string")
        .str.strip()
    )

    stores = normalize_dma_names(
        stores,
        dma_col="DMA",
    )

    stores.loc[
        stores["DMA"].isin(
            [
                "",
                "nan",
                "None",
            ]
        ),
        "DMA",
    ] = pd.NA

    stores = stores.loc[
        stores["Store"].notna()
    ].copy()

    # -------------------------------------------------------------------------
    # Duplicate store mapping
    # -------------------------------------------------------------------------

    duplicate_store = (
        stores
        .groupby("Store")["DMA"]
        .nunique(dropna=False)
    )

    ambiguous_stores = (
        duplicate_store[
            duplicate_store > 1
        ]
        .index
    )

    if len(ambiguous_stores) > 0:

        logger.warning(
            "Found %s stores mapped to multiple DMAs.",
            f"{len(ambiguous_stores):,}",
        )

        ambiguous = stores.loc[
            stores["Store"].isin(
                ambiguous_stores
            )
        ].sort_values("Store")

        ambiguous.to_csv(
            COMP_TABLE_DIR
            / "diagnostics"
            / "ambiguous_store_dma_mapping.csv",
            index=False,
        )

        raise ValueError(
            "Ambiguous Store -> DMA mappings detected."
        )

    stores = (
        stores
        .drop_duplicates(
            subset=["Store"]
        )
    )

    logger.info(
        "Store master rows: %s",
        f"{len(stores):,}",
    )

    logger.info(
        "Unique stores: %s",
        f"{stores['Store'].nunique():,}",
    )

    logger.info(
        "Unique DMAs: %s",
        f"{stores['DMA'].nunique():,}",
    )

    return stores


# =============================================================================
# MAP STORE TO DMA
# =============================================================================

def map_store_to_dma(
    store_month,
    stores,
):
    """
    Join store-month traffic to the store master and attach DMA.
    """

    logger.info("=" * 70)
    logger.info("MAPPING STORE TRAFFIC TO DMA")
    logger.info("=" * 70)

    traffic_stores = set(
        store_month["Store"].dropna()
    )

    master_stores = set(
        stores["Store"].dropna()
    )

    missing_from_master = (
        traffic_stores
        -
        master_stores
    )

    logger.info(
        "Traffic stores: %s",
        f"{len(traffic_stores):,}",
    )

    logger.info(
        "Master stores: %s",
        f"{len(master_stores):,}",
    )

    logger.info(
        "Traffic stores missing from master: %s",
        f"{len(missing_from_master):,}",
    )

    if missing_from_master:

        missing_df = pd.DataFrame(
            {
                "Store": sorted(
                    missing_from_master
                )
            }
        )

        missing_df.to_csv(
            COMP_TABLE_DIR
            / "diagnostics"
            / "traffic_stores_missing_from_master.csv",
            index=False,
        )

    mapped = store_month.merge(
        stores[
            [
                "Store",
                "DMA",
            ]
        ],
        on="Store",
        how="left",
        validate="many_to_one",
    )

    

    missing_dma = mapped["DMA"].isna()

    logger.info(
        "Store-month rows missing DMA: %s",
        f"{missing_dma.sum():,}",
    )

    logger.info(
        "Store-month rows with DMA: %s",
        f"{(~missing_dma).sum():,}",
    )

    return mapped


# =============================================================================
# DMA-MONTH AGGREGATION
# =============================================================================

def aggregate_dma_month(
    mapped,
):
    """
    Aggregate store-month traffic to DMA-month.

    DMA comp traffic is calculated from aggregate traffic totals:

        (DMA Exits - DMA Exits LY)
        --------------------------- * 100
              DMA Exits LY

    This provides a traffic-volume-weighted comp measure.
    """

    logger.info("=" * 70)
    logger.info("AGGREGATING STORE-MONTH TRAFFIC TO DMA-MONTH")
    logger.info("=" * 70)

    valid = mapped.loc[
        mapped["DMA"].notna()
    ].copy()

    dma_month = (
        valid
        .groupby(
            [
                "DMA",
                "Traffic_Month",
            ],
            as_index=False,
        )
        .agg(
            Exits=(
                "Exits",
                "sum",
            ),
            Exits_LY=(
                "Exits_LY",
                "sum",
            ),
            Stores=(
                "Store",
                "nunique",
            ),
            Store_Month_Rows=(
                "Store",
                "size",
            ),
            Mean_Store_Comp=(
                "Comp_Traffic",
                "mean",
            ),
            Median_Store_Comp=(
                "Comp_Traffic",
                "median",
            ),
            Mean_Weeks_Observed=(
                "Weeks_Observed",
                "mean",
            ),
        )
        .reset_index(drop=True)
    )

    valid_ly = (
        dma_month["Exits_LY"] > 0
    )

    dma_month["Comp_Traffic"] = np.nan

    dma_month.loc[
        valid_ly,
        "Comp_Traffic",
    ] = (
        (
            dma_month.loc[
                valid_ly,
                "Exits"
            ]
            -
            dma_month.loc[
                valid_ly,
                "Exits_LY"
            ]
        )
        /
        dma_month.loc[
            valid_ly,
            "Exits_LY"
        ]
        *
        100
    )

    dma_month["Traffic_Year"] = (
        dma_month["Traffic_Month"]
        .dt.year
    )

    dma_month["Traffic_Month_Number"] = (
        dma_month["Traffic_Month"]
        .dt.month
    )

    logger.info(
        "DMA-month rows: %s",
        f"{len(dma_month):,}",
    )

    logger.info(
        "Unique DMAs: %s",
        f"{dma_month['DMA'].nunique():,}",
    )

    logger.info(
        "DMA-month comp mean: %.3f%%",
        dma_month["Comp_Traffic"].mean(),
    )

    logger.info(
        "DMA-month comp median: %.3f%%",
        dma_month["Comp_Traffic"].median(),
    )

    return dma_month


# =============================================================================
# NATIONAL MONTHLY AGGREGATION
# =============================================================================

def aggregate_national_month(
    mapped,
):
    """
    Aggregate mapped store-month traffic to a national monthly series.
    """

    logger.info("=" * 70)
    logger.info("BUILDING NATIONAL MONTHLY TRAFFIC")
    logger.info("=" * 70)

    valid = mapped.loc[
        mapped["DMA"].notna()
    ].copy()

    national = (
        valid
        .groupby(
            "Traffic_Month",
            as_index=False,
        )
        .agg(
            Exits=(
                "Exits",
                "sum",
            ),
            Exits_LY=(
                "Exits_LY",
                "sum",
            ),
            Stores=(
                "Store",
                "nunique",
            ),
            DMAs=(
                "DMA",
                "nunique",
            )
        
        )
        .reset_index(drop=True)
    )

    valid_ly = (
        national["Exits_LY"] > 0
    )

    national["Comp_Traffic"] = np.nan

    national.loc[
        valid_ly,
        "Comp_Traffic",
    ] = (
        (
            national.loc[
                valid_ly,
                "Exits"
            ]
            -
            national.loc[
                valid_ly,
                "Exits_LY"
            ]
        )
        /
        national.loc[
            valid_ly,
            "Exits_LY"
        ]
        *
        100
    )

    national["Traffic_Year"] = (
        national["Traffic_Month"]
        .dt.year
    )

    logger.info(
        "National monthly rows: %s",
        f"{len(national):,}",
    )

    logger.info(
        "National traffic coverage: %s to %s",
        national["Traffic_Month"].min().date(),
        national["Traffic_Month"].max().date(),
    )

    return national


# =============================================================================
# Identify Treated DMA's
# =============================================================================
def identify_treated_dmas(
    spend,
):
    """
    Identify DMAs that received incremental marketing spend.

    A DMA is considered treated if it has positive marketing spend
    during the incremental-spend analysis period.
    """

    logger.info("=" * 70)
    logger.info("IDENTIFYING TREATED DMAs")
    logger.info("=" * 70)

    treated = (
        spend.loc[
            spend["Spend"] > 0
        ]
        .groupby("DMA", as_index=False)
        .agg(
            Total_Spend=("Spend", "sum"),
            Spend_Months=("Month", "nunique"),
            First_Spend_Month=("Month", "min"),
            Last_Spend_Month=("Month", "max"),
        )
    )

    treated["Treated"] = True

    treated = treated.loc[
        treated["DMA"] != "TOTALS"
    ].copy()

    treated = treated.sort_values(
        "Total_Spend",
        ascending=False,
    )

    logger.info(
        "Treated DMAs: %s",
        f"{len(treated):,}",
    )

    logger.info(
        "\n%s",
        treated.to_string(index=False),
    )

    treated.to_csv(
        COMP_TABLE_DIR
        / "treatment"
        / "treated_dmas.csv",
        index=False,
    )

    return treated

# =============================================================================
# Filter Treated DMA's
# =============================================================================
def filter_treated_dma_month(
    dma_month,
    treated_dmas,
):
    """
    Restrict DMA-month traffic to DMAs receiving incremental
    marketing spend.
    """

    logger.info("=" * 70)
    logger.info("FILTERING TRAFFIC TO TREATED DMAs")
    logger.info("=" * 70)

    treated_dma_set = set(
        treated_dmas["DMA"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    dma_month = dma_month.copy()

    dma_month["DMA"] = (
        dma_month["DMA"]
        .astype(str)
        .str.strip()
    )

    treated = dma_month.loc[
        dma_month["DMA"].isin(treated_dma_set)
    ].copy()

    logger.info(
        "All DMA-month rows: %s",
        f"{len(dma_month):,}",
    )

    logger.info(
        "Treated DMA-month rows: %s",
        f"{len(treated):,}",
    )

    logger.info(
        "All DMAs: %s",
        f"{dma_month['DMA'].nunique():,}",
    )

    logger.info(
        "Treated DMAs: %s",
        f"{treated['DMA'].nunique():,}",
    )

    treated.to_csv(
        COMP_TABLE_DIR
        / "treatment"
        / "treated_dma_month.csv",
        index=False,
    )

    return treated

# =============================================================================
# Build Treated DMA Panel
# =============================================================================
def build_treated_dma_panel(
    treated_dma_month,
    spend,
):
    """
    Build the primary treated DMA-month analytical panel.

    Combines:
        DMA traffic
        Comp traffic
        Marketing spend
    """

    logger.info("=" * 70)
    logger.info("BUILDING TREATED DMA-MONTH PANEL")
    logger.info("=" * 70)

    panel = treated_dma_month.merge(
        spend,
        left_on=[
            "DMA",
            "Traffic_Month",
        ],
        right_on=[
            "DMA",
            "Month",
        ],
        how="left",
        validate="one_to_one",
    )

    panel = panel.drop(
        columns=["Month"]
    )

    panel["Spend"] = (
        panel["Spend"]
        .fillna(0)
    )

    panel["Spend_Flag"] = (
        panel["Spend"] > 0
    )

    panel = panel.sort_values(
        [
            "DMA",
            "Traffic_Month",
        ]
    )

    logger.info(
        "Treated DMA panel rows: %s",
        f"{len(panel):,}",
    )

    logger.info(
        "DMAs: %s",
        f"{panel['DMA'].nunique():,}",
    )

    logger.info(
        "Months: %s",
        f"{panel['Traffic_Month'].nunique():,}",
    )

    logger.info(
        "Months with positive spend: %s",
        f"{panel.loc[panel['Spend'] > 0, 'Traffic_Month'].nunique():,}",
    )

    return panel

# =============================================================================
# Aggregate Treated DMA's
# =============================================================================
def aggregate_treated_dma_month(
    treated_dma_month,
):
    """
    Aggregate all treated DMAs into a single monthly treated-market
    traffic series.

    The resulting dataset is intentionally no longer DMA-level.

    Returns
    -------
    pandas.DataFrame
        One row per Traffic_Month containing aggregate treated-market
        traffic and comp metrics.
    """

    logger.info("=" * 70)
    logger.info("AGGREGATING TREATED DMAs INTO MONTHLY MARKET SERIES")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Input diagnostics
    # -------------------------------------------------------------------------

    logger.info(
        "Input DMA-month rows: %s",
        f"{len(treated_dma_month):,}",
    )

    logger.info(
        "Input treated DMAs: %s",
        f"{treated_dma_month['DMA'].nunique():,}",
    )

    logger.info(
        "Input months: %s",
        f"{treated_dma_month['Traffic_Month'].nunique():,}",
    )

    # -------------------------------------------------------------------------
    # Aggregate all treated DMAs by month
    # -------------------------------------------------------------------------

    treated = (
        treated_dma_month
        .groupby(
            "Traffic_Month",
            as_index=False,
        )
        .agg(
            Exits=("Exits", "sum"),
            Exits_LY=("Exits_LY", "sum"),
            Stores=("Stores", "sum"),
            DMAs=("DMA", "nunique"),
        )
    )

    # -------------------------------------------------------------------------
    # Calculate aggregate comp traffic
    # -------------------------------------------------------------------------

    valid = treated["Exits_LY"] > 0

    treated["Comp_Traffic"] = np.nan

    treated.loc[valid, "Comp_Traffic"] = (
        (
            treated.loc[valid, "Exits"]
            - treated.loc[valid, "Exits_LY"]
        )
        /
        treated.loc[valid, "Exits_LY"]
        *
        100
    )

    # -------------------------------------------------------------------------
    # Sort
    # -------------------------------------------------------------------------

    treated = (
        treated
        .sort_values("Traffic_Month")
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Output diagnostics
    # -------------------------------------------------------------------------

    logger.info(
        "Aggregated treated-market rows: %s",
        f"{len(treated):,}",
    )

    logger.info(
        "Aggregated months: %s",
        f"{treated['Traffic_Month'].nunique():,}",
    )

    logger.info(
        "Date range: %s -> %s",
        treated["Traffic_Month"].min(),
        treated["Traffic_Month"].max(),
    )

    logger.info(
        "Average DMAs represented per month: %.2f",
        treated["DMAs"].mean(),
    )

    logger.info(
        "Minimum DMAs represented in a month: %s",
        f"{treated['DMAs'].min():,}",
    )

    logger.info(
        "Maximum DMAs represented in a month: %s",
        f"{treated['DMAs'].max():,}",
    )

    logger.info(
        "Aggregate comp mean: %.3f%%",
        treated["Comp_Traffic"].mean(),
    )

    logger.info(
        "Aggregate comp median: %.3f%%",
        treated["Comp_Traffic"].median(),
    )

    return treated# =============================================================================
# COMP TRAFFIC SUMMARY
# =============================================================================

def create_comp_summary(
    store_month,
    dma_month,
    national,
):
    """
    Create high-level comp traffic summary tables.
    """

    summary = pd.DataFrame(
        [
            {
                "Level": "Store-Month",
                "Rows": len(store_month),
                "Entities": store_month["Store"].nunique(),
                "Mean_Comp_Traffic": store_month[
                    "Comp_Traffic"
                ].mean(),
                "Median_Comp_Traffic": store_month[
                    "Comp_Traffic"
                ].median(),
                "P10": store_month[
                    "Comp_Traffic"
                ].quantile(0.10),
                "P90": store_month[
                    "Comp_Traffic"
                ].quantile(0.90),
            },
            {
                "Level": "DMA-Month",
                "Rows": len(dma_month),
                "Entities": dma_month["DMA"].nunique(),
                "Mean_Comp_Traffic": dma_month[
                    "Comp_Traffic"
                ].mean(),
                "Median_Comp_Traffic": dma_month[
                    "Comp_Traffic"
                ].median(),
                "P10": dma_month[
                    "Comp_Traffic"
                ].quantile(0.10),
                "P90": dma_month[
                    "Comp_Traffic"
                ].quantile(0.90),
            },
            {
                "Level": "National-Month",
                "Rows": len(national),
                "Entities": 1,
                "Mean_Comp_Traffic": national[
                    "Comp_Traffic"
                ].mean(),
                "Median_Comp_Traffic": national[
                    "Comp_Traffic"
                ].median(),
                "P10": national[
                    "Comp_Traffic"
                ].quantile(0.10),
                "P90": national[
                    "Comp_Traffic"
                ].quantile(0.90),
            },
        ]
    )

    summary.to_csv(
        COMP_TABLE_DIR
        / "diagnostics"
        / "comp_traffic_summary.csv",
        index=False,
    )

    return summary


# =============================================================================
# DATA QUALITY DIAGNOSTICS
# =============================================================================

def run_data_quality_diagnostics(
    traffic,
    store_month,
    mapped,
    dma_month,
    national,
):
    """
    Run high-level data quality checks.
    """

    logger.info("=" * 70)
    logger.info("DATA QUALITY DIAGNOSTICS")
    logger.info("=" * 70)

    diagnostics = []

    # -------------------------------------------------------------------------
    # Weekly traffic
    # -------------------------------------------------------------------------

    diagnostics.append(
        {
            "Metric": "Raw store-week rows",
            "Value": len(traffic),
        }
    )

    diagnostics.append(
        {
            "Metric": "Raw stores",
            "Value": traffic["Store"].nunique(),
        }
    )

    diagnostics.append(
        {
            "Metric": "Raw weeks",
            "Value": traffic["WeekEndDate"].nunique(),
        }
    )

    # -------------------------------------------------------------------------
    # Store-month
    # -------------------------------------------------------------------------

    diagnostics.append(
        {
            "Metric": "Store-month rows",
            "Value": len(store_month),
        }
    )

    diagnostics.append(
        {
            "Metric": "Store-months with valid comp",
            "Value": store_month[
                "Comp_Traffic"
            ].notna().sum(),
        }
    )

    # -------------------------------------------------------------------------
    # DMA
    # -------------------------------------------------------------------------

    diagnostics.append(
        {
            "Metric": "Store-month rows",
            "Value": len(mapped),
        }
    )

    diagnostics.append(
        {
            "Metric": "Store-month rows with DMA",
            "Value": mapped[
                "DMA"
            ].notna().sum(),
        }
    )

    diagnostics.append(
        {
            "Metric": "DMA-month rows",
            "Value": len(dma_month),
        }
    )

    diagnostics.append(
        {
            "Metric": "Unique DMAs",
            "Value": dma_month[
                "DMA"
            ].nunique(),
        }
    )

    # -------------------------------------------------------------------------
    # National
    # -------------------------------------------------------------------------

    diagnostics.append(
        {
            "Metric": "National months",
            "Value": len(national),
        }
    )

    diagnostics_df = pd.DataFrame(
        diagnostics
    )

    diagnostics_df.to_csv(
        COMP_TABLE_DIR
        / "diagnostics"
        / "data_quality_summary.csv",
        index=False,
    )

    logger.info(
        "\n%s",
        diagnostics_df.to_string(
            index=False
        ),
    )

    return diagnostics_df


# =============================================================================
# FIGURES
# =============================================================================

def plot_national_traffic(
    national,
):
    """
    Plot national Exits and Exits LY.
    """

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        national["Traffic_Month"],
        national["Exits"],
        label="Exits",
    )

    ax.plot(
        national["Traffic_Month"],
        national["Exits_LY"],
        label="Exits LY",
    )

    ax.set_title(
        "National Monthly Traffic"
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "Exits"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    output = (
        COMP_FIGURE_DIR
        / "traffic"
        / "national_traffic.png"
    )

    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_national_comp(
    national,
):
    """
    Plot national monthly comp traffic.
    """

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        national["Traffic_Month"],
        national["Comp_Traffic"],
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(
        "National Monthly Comp Traffic"
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "Comp Traffic (%)"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    output = (
        COMP_FIGURE_DIR
        / "comp"
        / "national_comp_traffic.png"
    )

    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_store_comp_distribution(
    store_month,
):
    """
    Plot distribution of store-month comp traffic.
    """

    values = (
        store_month["Comp_Traffic"]
        .dropna()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        values,
        bins=50,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(
        "Distribution of Store-Month Comp Traffic"
    )

    ax.set_xlabel(
        "Comp Traffic (%)"
    )

    ax.set_ylabel(
        "Store-Month Observations"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    output = (
        COMP_FIGURE_DIR
        / "comp"
        / "store_month_comp_distribution.png"
    )

    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_dma_comp_heatmap(
    dma_month,
):
    """
    Plot DMA-month comp traffic heatmap.

    DMAs are limited to the 30 largest by average traffic to keep
    the visualization readable.
    """

    top_dmas = (
        dma_month
        .groupby("DMA")["Exits"]
        .sum()
        .nlargest(30)
        .index
    )

    data = (
        dma_month.loc[
            dma_month["DMA"].isin(
                top_dmas
            )
        ]
        .pivot(
            index="DMA",
            columns="Traffic_Month",
            values="Comp_Traffic",
        )
    )

    if data.empty:
        return

    fig, ax = plt.subplots(
        figsize=(16, 10)
    )

    image = ax.imshow(
        data.values,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_yticks(
        np.arange(len(data.index))
    )

    ax.set_yticklabels(
        data.index
    )

    # Show only selected x labels
    x_positions = np.arange(
        len(data.columns)
    )

    selected = (
        np.linspace(
            0,
            len(data.columns) - 1,
            min(
                12,
                len(data.columns)
            ),
        )
        .astype(int)
    )

    ax.set_xticks(
        selected
    )

    ax.set_xticklabels(
        [
            data.columns[i].strftime("%Y-%m")
            for i in selected
        ],
        rotation=45,
        ha="right",
    )

    ax.set_title(
        "DMA Monthly Comp Traffic — Top 30 DMAs by Traffic"
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "DMA"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Comp Traffic (%)",
    )

    fig.tight_layout()

    output = (
        COMP_FIGURE_DIR
        / "dma"
        / "dma_comp_heatmap.png"
    )

    fig.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# TOP / BOTTOM DMA SUMMARY
# =============================================================================

def create_dma_rankings(
    dma_month,
):
    """
    Rank DMAs by average monthly comp traffic.

    Also provide traffic-volume-weighted aggregate performance.
    """

    rankings = (
        dma_month
        .groupby("DMA")
        .agg(
            Months=("Traffic_Month", "nunique"),
            Mean_Comp_Traffic=(
                "Comp_Traffic",
                "mean",
            ),
            Median_Comp_Traffic=(
                "Comp_Traffic",
                "median",
            ),
            Total_Exits=(
                "Exits",
                "sum",
            ),
            Total_Exits_LY=(
                "Exits_LY",
                "sum",
            ),
            Mean_Stores=(
                "Stores",
                "mean",
            ),
        )
        .reset_index()
    )

    valid = (
        rankings["Total_Exits_LY"] > 0
    )

    rankings["Aggregate_Comp_Traffic"] = np.nan

    rankings.loc[
        valid,
        "Aggregate_Comp_Traffic",
    ] = (
        (
            rankings.loc[
                valid,
                "Total_Exits"
            ]
            -
            rankings.loc[
                valid,
                "Total_Exits_LY"
            ]
        )
        /
        rankings.loc[
            valid,
            "Total_Exits_LY"
        ]
        *
        100
    )

    rankings = rankings.sort_values(
        "Aggregate_Comp_Traffic",
        ascending=False,
    )

    rankings.to_csv(
        COMP_TABLE_DIR
        / "dma"
        / "dma_comp_rankings.csv",
        index=False,
    )

    return rankings

# =============================================================================
# Treated DMA SUMMARY
# =============================================================================
def create_treated_dma_summary(
    treated_dma_month,
):
    """
    Summarize comp traffic performance across treated DMAs.
    """

    required_columns = {
        "DMA",
        "Traffic_Month",
        "Comp_Traffic",
        "Exits",
        "Exits_LY",
        "Stores",
    }

    missing_columns = (
        required_columns
        - set(treated_dma_month.columns)
    )

    if missing_columns:
        raise ValueError(
            "create_treated_dma_summary() is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: {treated_dma_month.columns.tolist()}"
        )


    summary = (
        treated_dma_month
        .groupby("DMA")
        .agg(
            Months=(
                "Traffic_Month",
                "nunique",
            ),
            Mean_Comp_Traffic=(
                "Comp_Traffic",
                "mean",
            ),
            Median_Comp_Traffic=(
                "Comp_Traffic",
                "median",
            ),
            Total_Exits=(
                "Exits",
                "sum",
            ),
            Total_Exits_LY=(
                "Exits_LY",
                "sum",
            ),
            Mean_Stores=(
                "Stores",
                "mean",
            ),
        )
        .reset_index()
    )

    valid = (
        summary["Total_Exits_LY"] > 0
    )

    summary["Aggregate_Comp_Traffic"] = np.nan

    summary.loc[
        valid,
        "Aggregate_Comp_Traffic",
    ] = (
        (
            summary.loc[
                valid,
                "Total_Exits"
            ]
            -
            summary.loc[
                valid,
                "Total_Exits_LY"
            ]
        )
        /
        summary.loc[
            valid,
            "Total_Exits_LY"
        ]
        *
        100
    )

    summary = summary.sort_values(
        "Aggregate_Comp_Traffic",
        ascending=False,
    )

    summary.to_csv(
        COMP_TABLE_DIR
        / "treatment"
        / "treated_dma_comp_summary.csv",
        index=False,
    )

    return summary

# =============================================================================
# EXPORT DATASETS
# =============================================================================

def export_datasets(
    traffic,
    store_month,
    mapped,
    dma_month,
    national,
):
    """
    Export all major analytical datasets.
    """

    logger.info("=" * 70)
    logger.info("EXPORTING ANALYTICAL DATASETS")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Raw / prepared weekly
    # -------------------------------------------------------------------------

    logger.info(
        "Traffic dtypes before Parquet export:\n%s",
        traffic.dtypes.to_string(),
    )

    logger.info(
        "SaleTxns dtype: %s",
        traffic["SaleTxns"].dtype,
    )

    logger.info(
        "SaleTxns sample values:\n%s",
        traffic["SaleTxns"].head(10).to_string(),
    )

    logger.info(
        "SaleTxns Python types:\n%s",
        traffic["SaleTxns"]
        .map(type)
        .value_counts()
        .to_string(),
    )

    logger.info(
        "Object-column Python types before Parquet export:"
    )

    for col in traffic.select_dtypes(include=["object"]).columns:
        logger.info(
            "  %s:\n%s",
            col,
            traffic[col]
            .dropna()
            .map(type)
            .value_counts()
            .to_string(),
        )
    
    traffic.to_parquet(
        COMP_OUTPUT_DIR
        / "traffic_store_week.parquet",
        index=False,
    )

    traffic.to_csv(
        COMP_TABLE_DIR
        / "raw"
        / "traffic_store_week.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Store-month
    # -------------------------------------------------------------------------

    store_month.to_parquet(
        COMP_OUTPUT_DIR
        / "traffic_store_month.parquet",
        index=False,
    )

    store_month.to_csv(
        COMP_TABLE_DIR
        / "store"
        / "traffic_store_month.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Store-month mapped to DMA
    # -------------------------------------------------------------------------

    mapped.to_parquet(
        COMP_OUTPUT_DIR
        / "traffic_store_month_dma.parquet",
        index=False,
    )

    mapped.to_csv(
        COMP_TABLE_DIR
        / "store"
        / "traffic_store_month_dma.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # DMA-month
    # -------------------------------------------------------------------------

    dma_month.to_parquet(
        COMP_OUTPUT_DIR
        / "traffic_dma_month.parquet",
        index=False,
    )

    dma_month.to_csv(
        COMP_TABLE_DIR
        / "dma"
        / "traffic_dma_month.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # National
    # -------------------------------------------------------------------------

    national.to_parquet(
        COMP_OUTPUT_DIR
        / "traffic_national_month.parquet",
        index=False,
    )

    national.to_csv(
        COMP_TABLE_DIR
        / "dma"
        / "traffic_national_month.csv",
        index=False,
    )

    logger.info(
        "Dataset export complete."
    )


# =============================================================================
# HTML REPORT
# =============================================================================

def generate_html_report(
    traffic,
    store_month,
    mapped,
    dma_month,
    national,
    treated_dma_month,
    treated_summary,
    dma_rankings,
):
    """
    Generate an internal HTML report focused on treated DMAs.

    Primary analytical population:
        Treated DMAs identified by the treatment definition.

    Primary outcome:
        Aggregate comp traffic across treated DMAs, calculated from
        aggregate Exits and Exits LY.

    Supporting diagnostics:
        - Treated-DMA monthly performance
        - Treated-DMA summary table
        - All-DMA rankings
        - National traffic context
        - National comp traffic context
    """

    logger.info("=" * 70)
    logger.info("GENERATING TREATED-DMA HTML REPORT")
    logger.info("=" * 70)

    # =========================================================================
    # 1. VALIDATION
    # =========================================================================

    required_summary_cols = [
        "DMA",
        "Months",
        "Mean_Comp_Traffic",
        "Median_Comp_Traffic",
        "Total_Exits",
        "Total_Exits_LY",
        "Mean_Stores",
        "Aggregate_Comp_Traffic",
    ]

    missing_summary_cols = [
        col
        for col in required_summary_cols
        if col not in treated_summary.columns
    ]

    if missing_summary_cols:
        raise ValueError(
            "treated_summary is missing required columns: "
            f"{missing_summary_cols}"
        )

    required_treated_cols = [
        "DMA",
        "Traffic_Month",
        "Comp_Traffic",
        "Exits",
        "Exits_LY",
    ]

    missing_treated_cols = [
        col
        for col in required_treated_cols
        if col not in treated_dma_month.columns
    ]

    if missing_treated_cols:
        raise ValueError(
            "treated_dma_month is missing required columns: "
            f"{missing_treated_cols}"
        )

    # =========================================================================
    # 2. TREATED-DMA METRICS
    # =========================================================================

    treated_dmas = (
        treated_summary["DMA"]
        .dropna()
        .unique()
    )

    n_treated_dmas = len(treated_dmas)

    treated_months = (
        treated_dma_month["Traffic_Month"]
        .dropna()
        .nunique()
    )

    treated_comp = (
        treated_dma_month["Comp_Traffic"]
        .dropna()
    )

    # -------------------------------------------------------------------------
    # Aggregate treated-DMA exits
    #
    # This is intentionally calculated from the underlying exits rather than
    # averaging DMA-level comp percentages.
    # -------------------------------------------------------------------------

    treated_total_exits = (
        treated_dma_month["Exits"]
        .sum()
    )

    treated_total_exits_ly = (
        treated_dma_month["Exits_LY"]
        .sum()
    )

    if treated_total_exits_ly > 0:

        treated_aggregate_comp = (
            (
                treated_total_exits
                - treated_total_exits_ly
            )
            /
            treated_total_exits_ly
            *
            100
        )

    else:

        treated_aggregate_comp = np.nan

    # =========================================================================
    # 3. TREATED-DMA RANKINGS
    # =========================================================================

    treated_summary_sorted = (
        treated_summary
        .sort_values(
            "Aggregate_Comp_Traffic",
            ascending=False,
        )
        .copy()
    )

    top_treated_dma = (
        treated_summary_sorted
        .head(10)
    )

    bottom_treated_dma = (
        treated_summary_sorted
        .sort_values(
            "Aggregate_Comp_Traffic",
            ascending=True,
        )
        .head(10)
    )

    # =========================================================================
    # 4. MONTHLY TREATED-DMA AGGREGATION
    # =========================================================================

    treated_monthly = (
        treated_dma_month
        .groupby("Traffic_Month")
        .agg(
            Exits=("Exits", "sum"),
            Exits_LY=("Exits_LY", "sum"),
            Mean_Comp_Traffic=("Comp_Traffic", "mean"),
            DMA_Count=("DMA", "nunique"),
        )
        .reset_index()
    )

    treated_monthly["Aggregate_Comp_Traffic"] = np.nan

    valid_months = (
        treated_monthly["Exits_LY"] > 0
    )

    treated_monthly.loc[
        valid_months,
        "Aggregate_Comp_Traffic",
    ] = (
        (
            treated_monthly.loc[
                valid_months,
                "Exits",
            ]
            -
            treated_monthly.loc[
                valid_months,
                "Exits_LY",
            ]
        )
        /
        treated_monthly.loc[
            valid_months,
            "Exits_LY",
        ]
        *
        100
    )

    # =========================================================================
    # 5. NATIONAL CONTEXT
    # =========================================================================

    national_comp = (
        national["Comp_Traffic"]
        .dropna()
    )

    national_mean_comp = (
        national_comp.mean()
        if len(national_comp)
        else np.nan
    )

    # =========================================================================
    # 6. TREATED VS NATIONAL DIFFERENCE
    # =========================================================================

    if (
        pd.notna(treated_aggregate_comp)
        and pd.notna(national_mean_comp)
    ):

        treated_vs_national = (
            treated_aggregate_comp
            - national_mean_comp
        )

    else:

        treated_vs_national = np.nan

    # =========================================================================
    # 7. HELPER FUNCTIONS
    # =========================================================================

    def fmt_pct(value):
        if pd.isna(value):
            return "N/A"
        return f"{value:.2f}%"

    def fmt_num(value):
        if pd.isna(value):
            return "N/A"
        return f"{value:,.0f}"

    def fmt_date(value):
        if pd.isna(value):
            return "N/A"
        return pd.to_datetime(value).strftime("%Y-%m-%d")

    def html_table(
        dataframe,
        columns=None,
        rename=None,
        pct_columns=None,
        number_columns=None,
    ):

        df = dataframe.copy()

        if columns is not None:
            df = df[
                [
                    col
                    for col in columns
                    if col in df.columns
                ]
            ]

        if rename:
            df = df.rename(
                columns=rename
            )

        pct_columns = pct_columns or []
        number_columns = number_columns or []

        for col in pct_columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: (
                        f"{x:.2f}%"
                        if pd.notna(x)
                        else "N/A"
                    )
                )

        for col in number_columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: (
                        f"{x:,.0f}"
                        if pd.notna(x)
                        else "N/A"
                    )
                )

        return df.to_html(
            index=False,
            escape=False,
            classes="data-table",
        )

    # =========================================================================
    # 8. PREPARE HTML TABLES
    # =========================================================================

    treated_summary_table = html_table(
        treated_summary_sorted,
        columns=[
            "DMA",
            "Months",
            "Mean_Comp_Traffic",
            "Median_Comp_Traffic",
            "Total_Exits",
            "Total_Exits_LY",
            "Mean_Stores",
            "Aggregate_Comp_Traffic",
        ],
        rename={
            "DMA": "Treated DMA",
            "Months": "Months",
            "Mean_Comp_Traffic": "Mean Comp Traffic",
            "Median_Comp_Traffic": "Median Comp Traffic",
            "Total_Exits": "Total Exits",
            "Total_Exits_LY": "Total Exits LY",
            "Mean_Stores": "Mean Stores",
            "Aggregate_Comp_Traffic": "Aggregate Comp Traffic",
        },
        pct_columns=[
            "Mean_Comp_Traffic",
            "Median_Comp_Traffic",
            "Aggregate_Comp_Traffic",
        ],
        number_columns=[
            "Total_Exits",
            "Total_Exits_LY",
            "Mean_Stores",
        ],
    )

    top_treated_table = html_table(
        top_treated_dma,
        columns=[
            "DMA",
            "Aggregate_Comp_Traffic",
            "Mean_Comp_Traffic",
            "Months",
            "Mean_Stores",
        ],
        rename={
            "DMA": "Treated DMA",
            "Aggregate_Comp_Traffic": "Aggregate Comp Traffic",
            "Mean_Comp_Traffic": "Mean Comp Traffic",
            "Months": "Months",
            "Mean_Stores": "Mean Stores",
        },
        pct_columns=[
            "Aggregate_Comp_Traffic",
            "Mean_Comp_Traffic",
        ],
        number_columns=[
            "Months",
            "Mean_Stores",
        ],
    )

    bottom_treated_table = html_table(
        bottom_treated_dma,
        columns=[
            "DMA",
            "Aggregate_Comp_Traffic",
            "Mean_Comp_Traffic",
            "Months",
            "Mean_Stores",
        ],
        rename={
            "DMA": "Treated DMA",
            "Aggregate_Comp_Traffic": "Aggregate Comp Traffic",
            "Mean_Comp_Traffic": "Mean Comp Traffic",
            "Months": "Months",
            "Mean_Stores": "Mean Stores",
        },
        pct_columns=[
            "Aggregate_Comp_Traffic",
            "Mean_Comp_Traffic",
        ],
        number_columns=[
            "Months",
            "Mean_Stores",
        ],
    )

    # =========================================================================
    # 9. MONTHLY TREATED-DMA TABLE
    # =========================================================================

    treated_monthly_table = html_table(
        treated_monthly,
        columns=[
            "Traffic_Month",
            "DMA_Count",
            "Exits",
            "Exits_LY",
            "Mean_Comp_Traffic",
            "Aggregate_Comp_Traffic",
        ],
        rename={
            "Traffic_Month": "Month",
            "DMA_Count": "Treated DMAs",
            "Exits": "Exits",
            "Exits_LY": "Exits LY",
            "Mean_Comp_Traffic": "Mean DMA Comp Traffic",
            "Aggregate_Comp_Traffic": "Aggregate Treated-DMA Comp Traffic",
        },
        pct_columns=[
            "Mean_Comp_Traffic",
            "Aggregate_Comp_Traffic",
        ],
        number_columns=[
            "DMA_Count",
            "Exits",
            "Exits_LY",
        ],
    )

    # =========================================================================
    # 10. ALL-DMA DIAGNOSTIC TABLE
    # =========================================================================

    all_dma_table = html_table(
        dma_rankings.head(20),
        columns=[
            "DMA",
            "Aggregate_Comp_Traffic",
        ],
        rename={
            "DMA": "DMA",
            "Aggregate_Comp_Traffic": "Aggregate Comp Traffic",
        },
        pct_columns=[
            "Aggregate_Comp_Traffic",
        ],
    )

    # =========================================================================
    # 11. HTML
    # =========================================================================

    html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="utf-8">

<title>
Comp Traffic Analysis - Treated DMAs
</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    background: #f7f7f7;
    color: #333;
}}

.container {{
    max-width: 1500px;
    margin: auto;
}}

h1 {{
    color: #222;
    margin-bottom: 5px;
}}

h2 {{
    color: #333;
    margin-top: 0;
}}

h3 {{
    color: #444;
}}

.subtitle {{
    color: #666;
    margin-bottom: 30px;
}}

.cards {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 30px;
}}

.card {{
    background: white;
    padding: 20px;
    border-radius: 8px;
    min-width: 210px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.15);
}}

.card h3 {{
    margin-top: 0;
    font-size: 15px;
    color: #666;
}}

.card .value {{
    font-size: 28px;
    font-weight: bold;
    margin-top: 10px;
}}

.section {{
    margin-top: 45px;
    background: white;
    padding: 25px;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}}

.section p {{
    line-height: 1.6;
}}

.highlight {{
    background: #eef5ff;
    border-left: 5px solid #4a78a8;
    padding: 18px;
    margin: 20px 0;
}}

.warning {{
    background: #fff7e6;
    border-left: 5px solid #d99a00;
    padding: 18px;
    margin: 20px 0;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    background: white;
    margin-top: 15px;
}}

th, td {{
    padding: 9px;
    border: 1px solid #ddd;
}}

th {{
    background: #eee;
}}

td {{
    text-align: right;
}}

td:first-child,
th:first-child {{
    text-align: left;
}}

img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 20px auto;
}}

.caption {{
    color: #666;
    font-size: 13px;
    text-align: center;
}}

.small {{
    font-size: 13px;
    color: #666;
}}

</style>

</head>

<body>

<div class="container">

<h1>Comp Traffic Analysis — Treated DMAs</h1>

<p class="subtitle">
Internal analytical report focused on the DMA population selected for
treatment analysis.
</p>


<!-- ================================================================== -->
<!-- EXECUTIVE TREATMENT SUMMARY -->
<!-- ================================================================== -->

<div class="section">

<h2>1. Treatment Population & Primary Result</h2>

<div class="cards">

<div class="card">
<h3>Treated DMAs</h3>
<div class="value">{n_treated_dmas:,}</div>
</div>

<div class="card">
<h3>Treated DMA-Month Rows</h3>
<div class="value">{len(treated_dma_month):,}</div>
</div>

<div class="card">
<h3>Months</h3>
<div class="value">{treated_months:,}</div>
</div>

<div class="card">
<h3>Aggregate Treated-DMA Comp Traffic</h3>
<div class="value">{fmt_pct(treated_aggregate_comp)}</div>
</div>

<div class="card">
<h3>National Mean Comp Traffic</h3>
<div class="value">{fmt_pct(national_mean_comp)}</div>
</div>

<div class="card">
<h3>Treated vs National</h3>
<div class="value">{fmt_pct(treated_vs_national)}</div>
</div>

</div>

<div class="highlight">

<h3>Primary Analytical Result</h3>

<p>
Across the treated DMA population, aggregate traffic was
<b>{fmt_pct(treated_aggregate_comp)}</b> versus the corresponding
last-year traffic baseline.
</p>

<p>
The treated-DMA result is
<b>{fmt_pct(treated_vs_national)}</b>
relative to the national mean comp-traffic result of
<b>{fmt_pct(national_mean_comp)}</b>.
</p>

<p class="small">
The aggregate treated-DMA metric is calculated from total Exits and
total Exits LY across the treated population. It is therefore not an
unweighted average of DMA-level percentage changes.
</p>

</div>

</div>


<!-- ================================================================== -->
<!-- TREATMENT DEFINITION -->
<!-- ================================================================== -->

<div class="section">

<h2>2. Treated DMA Population</h2>

<p>
The primary analytical population for this report consists of
<b>{n_treated_dmas:,} treated DMAs</b>.
All subsequent treatment-focused results are calculated from this
population.
</p>

<p>
The treated-DMA panel contains
<b>{len(treated_dma_month):,}</b>
DMA-month observations covering
<b>{treated_months:,}</b>
months.
</p>

</div>


<!-- ================================================================== -->
<!-- TREATED DMA HEATMAP -->
<!-- ================================================================== -->

<div class="section">

<h2>3. Treated-DMA Comp Traffic Heatmap</h2>

<p>
The heatmap below provides the primary visual diagnostic of comp-traffic
performance across the treated DMA population over time.
</p>

<img
src="../../figures/comp_traffic/dma/dma_comp_heatmap.png"
alt="Treated DMA comp traffic heatmap"
>

<p class="caption">
DMA-level monthly comp traffic across the treated population.
</p>

</div>


<!-- ================================================================== -->
<!-- TREATED DMA SUMMARY -->
<!-- ================================================================== -->

<div class="section">

<h2>4. Treated-DMA Performance Summary</h2>

<p>
The table below ranks the treated DMAs by aggregate comp traffic.
Aggregate comp traffic is calculated using summed Exits and Exits LY
for each DMA.
</p>

{treated_summary_table}

</div>


<!-- ================================================================== -->
<!-- TOP TREATED DMAS -->
<!-- ================================================================== -->

<div class="section">

<h2>5. Strongest Treated DMAs</h2>

<p>
The following treated DMAs have the strongest aggregate comp-traffic
performance.
</p>

{top_treated_table}

</div>


<!-- ================================================================== -->
<!-- BOTTOM TREATED DMAS -->
<!-- ================================================================== -->

<div class="section">

<h2>6. Weakest Treated DMAs</h2>

<p>
The following treated DMAs have the weakest aggregate comp-traffic
performance.
</p>

{bottom_treated_table}

</div>


<!-- ================================================================== -->
<!-- TREATED DMA MONTHLY TREND -->
<!-- ================================================================== -->

<div class="section">

<h2>7. Treated-DMA Monthly Performance</h2>

<p>
The table below shows the evolution of aggregate traffic across the
treated DMA population by month.
</p>

{treated_monthly_table}

</div>


<!-- ================================================================== -->
<!-- NATIONAL CONTEXT -->
<!-- ================================================================== -->

<div class="section">

<h2>8. National Context</h2>

<p>
National traffic is shown below as context. It is not the primary
treatment result.
</p>

<img
src="../../figures/comp_traffic/traffic/national_traffic.png"
alt="National traffic"
>

<img
src="../../figures/comp_traffic/comp/national_comp_traffic.png"
alt="National comp traffic"
>

</div>


<!-- ================================================================== -->
<!-- STORE-LEVEL DISTRIBUTION -->
<!-- ================================================================== -->

<div class="section">

<h2>9. Store-Level Comp Traffic Distribution</h2>

<p>
This distribution provides context for the underlying store-level
variation from which the DMA-level results are constructed.
</p>

<img
src="../../figures/comp_traffic/comp/store_month_comp_distribution.png"
alt="Store-month comp traffic distribution"
>

</div>


<!-- ================================================================== -->
<!-- ALL DMA DIAGNOSTIC -->
<!-- ================================================================== -->

<div class="section">

<h2>10. All-DMA Diagnostic Ranking</h2>

<p>
The following table is provided as a secondary diagnostic only.
The treatment analysis is based on the treated-DMA population above.
</p>

{all_dma_table}

</div>


<!-- ================================================================== -->
<!-- METHODOLOGY -->
<!-- ================================================================== -->

<div class="section">

<h2>11. Methodology</h2>

<p>
Weekly store-level comp traffic is calculated as:
</p>

<p>
<b>
((Exits − Exits LY) / Exits LY) × 100
</b>
</p>

<p>
Weekly observations are aggregated to the store-month level.
Monthly store comp traffic is calculated from summed Exits and
summed Exits LY rather than averaging weekly percentage changes.
</p>

<p>
Store-month observations are mapped to DMA using the store master.
DMA-month comp traffic is then calculated from aggregate DMA Exits and
Exits LY.
</p>

<p>
For the treatment analysis, only the DMAs identified as treated are
included in the primary treated-DMA panel and aggregate calculations.
</p>

<p>
The aggregate treated-DMA result is calculated as:
</p>

<p>
<b>
((Total Treated-DMA Exits − Total Treated-DMA Exits LY)
/
Total Treated-DMA Exits LY) × 100
</b>
</p>

</div>


<!-- ================================================================== -->
<!-- DATA COVERAGE -->
<!-- ================================================================== -->

<div class="section">

<h2>12. Data Coverage</h2>

<p>
Weekly traffic coverage:
<b>
{fmt_date(traffic["WeekEndDate"].min())}
</b>
through
<b>
{fmt_date(traffic["WeekEndDate"].max())}
</b>.
</p>

<p>
National monthly traffic coverage:
<b>
{fmt_date(national["Traffic_Month"].min())}
</b>
through
<b>
{fmt_date(national["Traffic_Month"].max())}
</b>.
</p>

<p>
Treated-DMA monthly coverage:
<b>
{fmt_date(treated_dma_month["Traffic_Month"].min())}
</b>
through
<b>
{fmt_date(treated_dma_month["Traffic_Month"].max())}
</b>.
</p>

</div>


<!-- ================================================================== -->
<!-- DATASET SUMMARY -->
<!-- ================================================================== -->

<div class="section">

<h2>13. Analytical Dataset Summary</h2>

<table>

<tr>
<th>Dataset</th>
<th>Rows</th>
<th>DMAs / Stores</th>
</tr>

<tr>
<td>Weekly Traffic</td>
<td>{len(traffic):,}</td>
<td>{traffic["Store"].nunique():,} stores</td>
</tr>

<tr>
<td>Store-Month</td>
<td>{len(store_month):,}</td>
<td>{store_month["Store"].nunique():,} stores</td>
</tr>

<tr>
<td>DMA-Month</td>
<td>{len(dma_month):,}</td>
<td>{dma_month["DMA"].nunique():,} DMAs</td>
</tr>

<tr>
<td>Treated DMA-Month</td>
<td>{len(treated_dma_month):,}</td>
<td>{n_treated_dmas:,} treated DMAs</td>
</tr>

</table>

</div>


</div>

</body>

</html>
"""

    # =========================================================================
    # 12. WRITE REPORT
    # =========================================================================

    report_path = (
        COMP_REPORT_DIR
        / "comp_traffic_analysis.html"
    )

    report_path.write_text(
        html,
        encoding="utf-8",
    )

    logger.info(
        "Treated-DMA report written to %s",
        report_path,
    )

    return report_path
# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# MAIN
# =============================================================================

def main():

    # =========================================================================
    # 0. INITIALIZATION
    # =========================================================================

    create_output_directories()

    logger.info("")
    logger.info("=" * 80)
    logger.info("STARTING COMP TRAFFIC ANALYSIS")
    logger.info("=" * 80)

    try:

        # =====================================================================
        # 1. LOAD RAW TRAFFIC
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 1: LOADING STORE-LEVEL TRAFFIC")
        logger.info("-" * 80)

        traffic = load_store_traffic()

        logger.info(
            "Traffic loaded successfully: %s rows",
            f"{len(traffic):,}",
        )

        logger.info(
            "Traffic columns: %s",
            traffic.columns.tolist(),
        )

        # =====================================================================
        # 2. LOAD MARKETING SPEND
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 2: LOADING MARKETING SPEND")
        logger.info("-" * 80)

        spend = load_spend()

        logger.info(
            "Spend loaded successfully: %s rows",
            f"{len(spend):,}",
        )

        logger.info(
            "Spend columns: %s",
            spend.columns.tolist(),
        )

        # =====================================================================
        # 3. PREPARE WEEKLY TRAFFIC
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 3: PREPARING WEEKLY TRAFFIC")
        logger.info("-" * 80)

        traffic = prepare_traffic(
            traffic
        )

        logger.info(
            "Weekly traffic prepared: %s rows",
            f"{len(traffic):,}",
        )

        # =====================================================================
        # 4. WEEKLY COVERAGE DIAGNOSTICS
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 4: WEEKLY COVERAGE DIAGNOSTICS")
        logger.info("-" * 80)

        log_weekly_coverage(
            traffic
        )

        log_store_year_coverage(
            traffic
        )

        logger.info(
            "Weekly coverage diagnostics complete."
        )

        # =====================================================================
        # 5. AGGREGATE WEEKLY -> STORE-MONTH
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 5: AGGREGATING WEEKLY TRAFFIC TO STORE-MONTH")
        logger.info("-" * 80)

        store_month = aggregate_store_month(
            traffic
        )

        logger.info("")
        logger.info("=" * 70)
        logger.info("STORE-MONTH OUTPUT SCHEMA")
        logger.info("=" * 70)

        logger.info(
            "Shape: %s rows x %s columns",
            f"{store_month.shape[0]:,}",
            f"{store_month.shape[1]:,}",
        )

        logger.info(
            "Columns:"
        )

        for i, col in enumerate(store_month.columns):
            logger.info(
                "  %s: %r",
                i,
                col,
            )

        logger.info(
            "Index: %s",
            store_month.index,
        )

        logger.info(
            "Head:\n%s",
            store_month.head().to_string(),
        )

        logger.info(
            "Store-month dataset created: %s rows",
            f"{len(store_month):,}",
        )

        if not store_month.empty:

            logger.info(
                "Store-month date range: %s -> %s",
                store_month["Traffic_Month"].min(),
                store_month["Traffic_Month"].max(),
            )

            logger.info(
                "Unique stores represented: %s",
                f"{store_month['Store'].nunique():,}",
            )

        # =====================================================================
        # 6. STORE-MONTH DIAGNOSTICS
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 6: STORE-MONTH DIAGNOSTICS")
        logger.info("-" * 80)

        log_monthly_store_diagnostics(
            store_month
        )

        logger.info(
            "Store-month diagnostics complete."
        )

        # =====================================================================
        # 7. LOAD STORE MASTER
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 7: LOADING STORE MASTER")
        logger.info("-" * 80)

        stores = load_stores()

        logger.info(
            "Store master loaded: %s rows",
            f"{len(stores):,}",
        )

        logger.info(
            "Unique stores: %s",
            f"{stores['Store'].nunique():,}",
        )

        logger.info(
            "Unique DMAs: %s",
            f"{stores['DMA'].nunique():,}",
        )

        # =====================================================================
        # 8. MAP STORE -> DMA
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 8: MAPPING STORE TRAFFIC TO DMA")
        logger.info("-" * 80)

        mapped = map_store_to_dma(
            store_month,
            stores,
        )

        logger.info(
            "Store-month DMA mapping complete: %s rows",
            f"{len(mapped):,}",
        )

        if "DMA" in mapped.columns:

            logger.info(
                "Mapped DMAs: %s",
                f"{mapped['DMA'].nunique():,}",
            )

            logger.info(
                "Rows with DMA: %s",
                f"{mapped['DMA'].notna().sum():,}",
            )

            logger.info(
                "Rows missing DMA: %s",
                f"{mapped['DMA'].isna().sum():,}",
            )

        # =====================================================================
        # 9. AGGREGATE TO ALL-DMA MONTH
        #
        # This remains useful as a diagnostic/reference dataset, but is NOT
        # the primary analysis population.
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 9: AGGREGATING TRAFFIC TO ALL-DMA MONTH")
        logger.info("-" * 80)

        dma_month = aggregate_dma_month(
            mapped
        )

        logger.info(
            "All-DMA-month dataset created: %s rows",
            f"{len(dma_month):,}",
        )

        logger.info(
            "Unique DMAs represented: %s",
            f"{dma_month['DMA'].nunique():,}",
        )

        # =====================================================================
        # 10. NATIONAL MONTHLY TRAFFIC
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 10: CREATING NATIONAL MONTHLY TRAFFIC SERIES")
        logger.info("-" * 80)

        national = aggregate_national_month(
            mapped
        )

        logger.info(
            "National monthly dataset created: %s months",
            f"{len(national):,}",
        )

        if not national.empty:

            logger.info(
                "National date range: %s -> %s",
                national["Traffic_Month"].min(),
                national["Traffic_Month"].max(),
            )

        # =====================================================================
        # 11. IDENTIFY TREATED DMAs
        #
        # PRIMARY PROJECT FILTER
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 11: IDENTIFYING TREATED DMAs")
        logger.info("=" * 80)

        treated_dmas = identify_treated_dmas(
            spend
        )

        logger.info(
            "Treated DMAs identified: %s",
            f"{len(treated_dmas):,}",
        )

        if treated_dmas.empty:

            raise ValueError(
                "No treated DMAs were identified from the marketing spend data."
            )

        for dma in treated_dmas["DMA"]:

            logger.info(
                "  Treated DMA: %s",
                dma,
            )



        spend_dmas = (
            treated_dmas["DMA"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        traffic_dmas = (
            dma_month["DMA"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        spend_dma_set = set(spend_dmas)
        traffic_dma_set = set(traffic_dmas)

        logger.info("")
        logger.info("=" * 80)
        logger.info("DMA KEY MATCHING DIAGNOSTIC")
        logger.info("=" * 80)

        logger.info(
            "Treated spend DMAs: %s",
            f"{len(spend_dma_set):,}",
        )

        logger.info(
            "Traffic DMAs: %s",
            f"{len(traffic_dma_set):,}",
        )

        logger.info(
            "Exact DMA matches: %s",
            f"{len(spend_dma_set & traffic_dma_set):,}",
        )

        logger.info(
            "Spend DMAs missing from traffic: %s",
            f"{len(spend_dma_set - traffic_dma_set):,}",
        )

        logger.info(
            "Traffic DMAs with no spend: %s",
            f"{len(traffic_dma_set - spend_dma_set):,}",
        )

        logger.info(
            "Spend DMA names:\n%s",
            "\n".join(sorted(spend_dma_set)),
        )

        logger.info(
            "Traffic DMA names:\n%s",
            "\n".join(sorted(traffic_dma_set)),
        )

        pd.DataFrame(
            {"DMA": sorted(spend_dma_set)}
        ).to_csv(
            COMP_TABLE_DIR / "diagnostics" / "spend_dma_keys.csv",
            index=False,
        )

        pd.DataFrame(
            {"DMA": sorted(traffic_dma_set)}
        ).to_csv(
            COMP_TABLE_DIR / "diagnostics" / "traffic_dma_keys.csv",
            index=False,
        )


        # =====================================================================
        # 12. FILTER TRAFFIC TO TREATED DMAs
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 12: FILTERING TRAFFIC TO TREATED DMAs")
        logger.info("=" * 80)

        treated_dma_month = filter_treated_dma_month(
            dma_month,
            treated_dmas,
        )

        logger.info(
            "Treated DMA-month rows: %s",
            f"{len(treated_dma_month):,}",
        )

        logger.info(
            "Treated DMAs represented in traffic: %s",
            f"{treated_dma_month['DMA'].nunique():,}",
        )

        missing_treated_dmas = (
                set(
                    treated_dmas["DMA"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                -
                set(
                    treated_dma_month["DMA"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                )
            )

        if missing_treated_dmas:

            logger.warning(
                "Treated DMAs with no traffic observations: %s",
                f"{len(missing_treated_dmas):,}",
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
                "All treated DMAs have traffic observations."
            )

        # =====================================================================
        # 13. BUILD TREATED-DMA PANEL
        #
        # This is where traffic and spend are brought together for the
        # project-specific analysis.
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 13: BUILDING TREATED-DMA PANEL")
        logger.info("=" * 80)

        treated_dma_panel = build_treated_dma_panel(
            treated_dma_month,
            spend,
        )

        logger.info(
            "Treated DMA panel created: %s rows",
            f"{len(treated_dma_panel):,}",
        )

        if treated_dma_panel.empty:

            raise ValueError(
                "Treated DMA panel is empty after combining traffic and spend."
            )

        logger.info(
            "Treated DMAs in panel: %s",
            f"{treated_dma_panel['DMA'].nunique():,}",
        )

        if "Month" in treated_dma_panel.columns:

            logger.info(
                "Panel date range: %s -> %s",
                treated_dma_panel["Month"].min(),
                treated_dma_panel["Month"].max(),
            )

        # =====================================================================
        # 14. AGGREGATE TREATED-DMA MONTH
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 14: AGGREGATING TREATED-DMA MONTH")
        logger.info("=" * 80)

        treated_dma_aggregated = aggregate_treated_dma_month(
            treated_dma_panel
        )

        logger.info(
            "Treated DMA-month observations aggregated: %s",
            f"{len(treated_dma_month):,}",
        )

        logger.info(
            "Treated DMAs contributing: %s",
            f"{treated_dma_month['DMA'].nunique():,}",
        )

        logger.info(
            "Treated-market monthly observations: %s",
            f"{len(treated_dma_aggregated):,}",
        )

        logger.info(
            "Aggregated date range: %s -> %s",
            treated_dma_aggregated["Traffic_Month"].min(),
            treated_dma_aggregated["Traffic_Month"].max(),
        )

       
        # =====================================================================
        # 15. TREATED-DMA SUMMARY
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 15: CREATING TREATED-DMA SUMMARY")
        logger.info("=" * 80)

        treated_summary = create_treated_dma_summary(
            treated_dma_month
        )

        logger.info(
            "Treated-DMA summary created: %s rows",
            f"{len(treated_summary):,}",
        )

        # =====================================================================
        # 16. ALL-DMA RANKINGS
        #
        # Secondary diagnostic only.
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 16: CREATING ALL-DMA DIAGNOSTIC RANKINGS")
        logger.info("-" * 80)

        dma_rankings = create_dma_rankings(
            dma_month
        )

        logger.info(
            "All-DMA rankings generated: %s rows",
            f"{len(dma_rankings):,}",
        )

        logger.info(
            "All-DMA rankings are SECONDARY diagnostics only."
        )

        # =====================================================================
        # 17. DATA QUALITY DIAGNOSTICS
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 17: DATA QUALITY DIAGNOSTICS")
        logger.info("-" * 80)

        run_data_quality_diagnostics(
            traffic=traffic,
            store_month=store_month,
            mapped=mapped,
            dma_month=dma_month,
            national=national,
        )

        logger.info(
            "Data quality diagnostics complete."
        )

        # =====================================================================
        # 18. NATIONAL / STORE FIGURES
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 18: GENERATING NATIONAL / STORE FIGURES")
        logger.info("-" * 80)

        plot_national_traffic(
            national
        )

        plot_national_comp(
            national
        )

        plot_store_comp_distribution(
            store_month
        )

        logger.info(
            "National/store figures generated."
        )

        # =====================================================================
        # 19. TREATED-DMA FIGURES
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 19: GENERATING TREATED-DMA FIGURES")
        logger.info("-" * 80)

        plot_dma_comp_heatmap(
            treated_dma_month
        )

        logger.info(
            "Treated-DMA figures generated."
        )

        # =====================================================================
        # 20. EXPORT DATASETS
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 20: EXPORTING DATASETS")
        logger.info("-" * 80)

        export_datasets(
            traffic=traffic,
            store_month=store_month,
            mapped=mapped,
            dma_month=dma_month,
            national=national,
        )

        logger.info(
            "Core datasets exported."
        )

        logger.info(
            "Exporting project-specific treated-DMA datasets."
        )

        treated_dma_panel.to_csv(
            COMP_TABLE_DIR
            / "treated_dma_panel.csv",
            index=False,
        )

        treated_dma_month.to_csv(
            COMP_TABLE_DIR
            / "treated_dma_month.csv",
            index=False,
        )

        treated_dma_aggregated.to_csv(
            COMP_TABLE_DIR
            / "treated_market_month.csv",
            index=False,
        )

        treated_summary.to_csv(
            COMP_TABLE_DIR
            / "treated_dma_summary.csv",
            index=False,
        )

        logger.info(
            "Treated-DMA datasets exported."
        )

        # =====================================================================
        # 21. HTML REPORT
        # =====================================================================

        logger.info("")
        logger.info("-" * 80)
        logger.info("STEP 21: GENERATING HTML REPORT")
        logger.info("-" * 80)

        report_path = generate_html_report(
            traffic=traffic,
            store_month=store_month,
            mapped=mapped,

            # Keep all-DMA data available for diagnostics.
            dma_month=dma_month,

            national=national,

            # Primary project population.
            treated_dma_month=treated_dma_month,
            treated_summary=treated_summary,

            # Secondary diagnostic.
            dma_rankings=dma_rankings,
        )

        logger.info(
            "HTML report generated: %s",
            report_path,
        )

        # =====================================================================
        # 22. FINAL SUMMARY
        # =====================================================================

        logger.info("")
        logger.info("=" * 80)
        logger.info("COMP TRAFFIC ANALYSIS COMPLETE")
        logger.info("=" * 80)

        logger.info(
            "Weekly store rows: %s",
            f"{len(traffic):,}",
        )

        logger.info(
            "Store-month rows: %s",
            f"{len(store_month):,}",
        )

        logger.info(
            "All-DMA-month rows: %s",
            f"{len(dma_month):,}",
        )

        logger.info(
            "Treated-DMA panel rows: %s",
            f"{len(treated_dma_panel):,}",
        )

        logger.info(
            "Treated-DMA-month rows: %s",
            f"{len(treated_dma_month):,}",
        )

        logger.info(
            "Treated-market-month rows: %s",
            f"{len(treated_dma_aggregated):,}",
        )

        logger.info(
            "Treated DMAs: %s",
            f"{len(treated_dmas):,}",
        )

        logger.info(
            "National months: %s",
            f"{len(national):,}",
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
            "COMP TRAFFIC ANALYSIS FAILED."
        )

        raise


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()