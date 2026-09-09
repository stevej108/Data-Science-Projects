"""
02_clean_data.py

Purpose
-------
Clean and standardize the raw datasets.

Outputs
-------
data/interim/
    spend_clean.parquet
    traffic_dma.parquet
    stores_clean.parquet
"""

import logging
import re

import pandas as pd

from config import (
    INTERIM_DATA_DIR,
    SPEND_FILE,
    STORE_FILE,
    TRAFFIC_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

#------------------------------------------------
# Helper Functions
#------------------------------------------------
def parse_excel_dates(series):
    """
    Parse dates that may contain either:
        - mm/dd/yyyy strings
        - Excel serial dates
    """

    numeric = pd.to_numeric(series, errors="coerce")

    excel_dates = pd.to_datetime(
        numeric,
        unit="D",
        origin="1899-12-30"
    )

    string_dates = pd.to_datetime(
        series,
        errors="coerce"
    )

    return string_dates.fillna(excel_dates)

def clean_numeric(series: pd.Series) -> pd.Series:
    """
    Convert formatted numeric strings to numeric dtype.

    Handles:
    - commas
    - dollar signs
    - percentages
    - whitespace
    - dash placeholders
    - blank values
    """

    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({
            "": None,
            "nan": None,
            "-": None,
            "–": None,
            "—": None,
        })
        .pipe(pd.to_numeric, errors="coerce")
    )

# =====================================================
# Spend
# =====================================================

def clean_spend():
    """
    Clean and reshape the raw marketing spend dataset.

    Returns
    -------
    pd.DataFrame
        DMA-month marketing spend dataset with one row per
        DMA-month combination.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("MARKETING SPEND CLEANING")
    logger.info("=" * 70)

    # =====================================================
    # 1. LOAD RAW DATA
    # =====================================================

    df = pd.read_csv(SPEND_FILE)

    logger.info(
        f"Raw spend rows:    {len(df):,}"
    )

    logger.info(
        f"Raw spend columns: {len(df.columns):,}"
    )

    # =====================================================
    # 2. IDENTIFY MONTH COLUMNS
    # =====================================================

    month_pattern = r"^\d{2}-[A-Za-z]{3}$"

    month_cols = [
        c
        for c in df.columns
        if re.match(month_pattern, str(c))
    ]

    if not month_cols:
        raise ValueError(
            "No monthly spend columns were identified."
        )

    logger.info(
        f"Monthly spend columns identified: "
        f"{len(month_cols)}"
    )

    logger.info(
        f"Month range: "
        f"{month_cols[0]} -> {month_cols[-1]}"
    )

    # =====================================================
    # 3. CLEAN DMA FIELD BEFORE MELTING
    # =====================================================

    if "DMA" not in df.columns:
        raise ValueError(
            "Spend dataset does not contain a 'DMA' column."
        )

    # Preserve the original value for diagnostics.
    df["DMA_Original"] = df["DMA"]

    # Standardize obvious whitespace issues.
    df["DMA"] = (
        df["DMA"]
        .astype("string")
        .str.strip()
    )

    # Convert empty strings to missing values.
    df["DMA"] = (
        df["DMA"]
        .replace("", pd.NA)
    )

    # =====================================================
    # 4. IDENTIFY INVALID / NON-DMA ROWS
    # =====================================================

    invalid_dma_patterns = [
        "TOTAL",
        "FORECAST",
        "EXCLUDE",
        "NOTES",
        "NATIONAL",
        "INCLUDES",
    ]

    invalid_pattern = "|".join(
        re.escape(pattern)
        for pattern in invalid_dma_patterns
    )

    missing_dma_rows = df[
        df["DMA"].isna()
    ].copy()

    invalid_dma_rows = df[
        df["DMA"]
        .str.contains(
            invalid_pattern,
            case=False,
            na=False
        )
    ].copy()

    logger.info("")
    logger.info("DMA QUALITY")

    logger.info(
        f"Rows with missing DMA: "
        f"{len(missing_dma_rows):,}"
    )

    logger.info(
        f"Rows with invalid/non-DMA labels: "
        f"{len(invalid_dma_rows):,}"
    )

    if not missing_dma_rows.empty:

        logger.warning("")
        logger.warning(
            "Rows with missing DMA will be removed:"
        )

        logger.warning(
            missing_dma_rows[
                ["DMA_Original"]
            ]
            .to_string(index=False)
        )

    if not invalid_dma_rows.empty:

        logger.warning("")
        logger.warning(
            "Rows with invalid/non-DMA labels "
            "will be removed:"
        )

        logger.warning(
            invalid_dma_rows[
                ["DMA_Original"]
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

    # Remove rows that cannot represent a DMA.
    df = df[
        df["DMA"].notna()
        & ~df["DMA"].str.contains(
            invalid_pattern,
            case=False,
            na=False
        )
    ].copy()

    logger.info(
        f"Valid DMA rows remaining: "
        f"{len(df):,}"
    )

    # =====================================================
    # 5. MELT TO DMA-MONTH FORMAT
    # =====================================================

    spend = df.melt(
        id_vars=[
            "DMA",
            "DMA_Original"
        ],
        value_vars=month_cols,
        var_name="Month",
        value_name="Spend"
    )

    # =====================================================
    # 6. PARSE MONTH
    # =====================================================

    spend["Month"] = pd.to_datetime(
        spend["Month"],
        format="%y-%b",
        errors="coerce"
    )

    invalid_dates = spend["Month"].isna().sum()

    if invalid_dates > 0:
        raise ValueError(
            f"Found {invalid_dates:,} spend rows "
            "with invalid Month values."
        )

    # =====================================================
    # 7. CLEAN SPEND VALUES
    # =====================================================

    spend["Spend"] = (
        spend["Spend"]
        .fillna("0")
        .astype(str)
        .str.strip()
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("K", "", regex=False)
    )

    spend["Spend"] = pd.to_numeric(
        spend["Spend"],
        errors="coerce"
    )

    invalid_spend = spend["Spend"].isna().sum()

    if invalid_spend > 0:

        logger.warning(
            f"Found {invalid_spend:,} spend values "
            "that could not be converted to numeric. "
            "These will be set to zero."
        )

        spend.loc[
            spend["Spend"].isna(),
            "Spend"
        ] = 0

    # Raw file values are expressed in thousands.
    spend["Spend"] *= 1000

    # =====================================================
    # 8. NEGATIVE SPEND CHECK
    # =====================================================

    negative_spend = spend[
        spend["Spend"] < 0
    ]

    if not negative_spend.empty:

        logger.warning(
            f"Found {len(negative_spend):,} "
            "negative spend observations."
        )

        logger.warning(
            negative_spend[
                ["DMA", "Month", "Spend"]
            ]
            .to_string(index=False)
        )

    # =====================================================
    # 9. DUPLICATE DMA-MONTH CHECK
    # =====================================================

    duplicates = spend[
        spend.duplicated(
            ["DMA", "Month"],
            keep=False
        )
    ].sort_values(
        ["DMA", "Month"]
    )

    if not duplicates.empty:

        logger.error(
            f"Found {len(duplicates):,} rows involved "
            "in duplicate DMA-month combinations."
        )

        logger.error(
            duplicates[
                ["DMA", "Month", "Spend"]
            ]
            .to_string(index=False)
        )

        raise ValueError(
            "Duplicate DMA-Month combinations found "
            "in cleaned spend data."
        )

    # =====================================================
    # 10. SORT
    # =====================================================

    spend = (
        spend
        .sort_values(
            ["DMA", "Month"]
        )
        .reset_index(drop=True)
    )

    # =====================================================
    # 11. FINAL QA
    # =====================================================

    logger.info("")
    logger.info("FINAL SPEND DATASET")
    logger.info("-" * 70)

    logger.info(
        f"Rows:                {len(spend):,}"
    )

    logger.info(
        f"DMAs:                {spend['DMA'].nunique():,}"
    )

    logger.info(
        f"Months:              {spend['Month'].nunique():,}"
    )

    logger.info(
        f"Date range:          "
        f"{spend['Month'].min().date()} "
        f"-> "
        f"{spend['Month'].max().date()}"
    )

    logger.info(
        f"Missing DMA:         "
        f"{spend['DMA'].isna().sum():,}"
    )

    logger.info(
        f"Missing Spend:       "
        f"{spend['Spend'].isna().sum():,}"
    )

    logger.info(
        f"Total Spend:         "
        f"${spend['Spend'].sum():,.0f}"
    )

    logger.info("=" * 70)

    return spend

# =====================================================
# Store Master
# =====================================================

def clean_stores():

    logger.info("Cleaning store master...")

    df = pd.read_csv(STORE_FILE)

    keep_cols = [
        "Store",
        "DMA",
        "MSA",
        "Open/Closed Status",
        "State",
        " Selling Square Feet ",
        "Trade Area Population",
        "DMA Population"
    ]

    stores = df[keep_cols].copy()

    stores.columns = (
        stores.columns
        .str.strip()
        .str.replace(" ", "_")
    )

    stores = stores.rename(columns={
        "Open/Closed_Status": "Status",
        "Selling_Square_Feet": "Selling_SqFt",
        "Trade_Area_Population": "TradeAreaPopulation",
        "DMA_Population": "DMAPopulation"
    })

    numeric_cols = [
    "Selling_SqFt",
    "TradeAreaPopulation",
    "DMAPopulation",
    ]

    for col in numeric_cols:
        stores[col] = clean_numeric(stores[col])

    logger.info("\nStore Data Types")

    for col in numeric_cols:
        logger.info(
            f"{col:<25} {stores[col].dtype}"
        )

    return stores


# =====================================================
# Traffic
# =====================================================

def clean_traffic(stores):

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAFFIC CLEANING DIAGNOSTICS")
    logger.info("=" * 70)

    # =====================================================
    # 1. LOAD RAW TRAFFIC
    # =====================================================

    traffic = pd.read_csv(TRAFFIC_FILE)

    

    logger.info("")
    logger.info("RAW TRAFFIC DATA")
    logger.info("-" * 70)
    logger.info(f"Raw rows              : {len(traffic):,}")
    logger.info(f"Raw columns           : {len(traffic.columns)}")

    logger.info("")
    logger.info("Raw columns:")
    logger.info(traffic.columns.tolist())

    # =====================================================
    # 2. RAW DATE DIAGNOSTICS
    # =====================================================

    logger.info("")
    logger.info("RAW DATE FIELD")
    logger.info("-" * 70)

    logger.info(
        f"WeekEndDate nulls     : "
        f"{traffic['WeekEndDate'].isna().sum():,}"
    )

    logger.info("")
    logger.info("Raw WeekEndDate sample:")
    logger.info(
        traffic["WeekEndDate"]
        .head(10)
        .to_string()
    )

    # Preserve the raw date column for diagnostics
    traffic["_Raw_WeekEndDate"] = traffic["WeekEndDate"]

    # =====================================================
    # 3. PARSE DATES
    # =====================================================

    traffic["Week End Date"] = parse_excel_dates(
        traffic["WeekEndDate"]
    )

    failed_dates = traffic["Week End Date"].isna().sum()

    logger.info("")
    logger.info("PARSED DATE DIAGNOSTICS")
    logger.info("-" * 70)
    logger.info(f"Failed date parses     : {failed_dates:,}")

    if failed_dates > 0:
        logger.warning("Rows with failed date parsing:")
        logger.warning(
            traffic.loc[
                traffic["Week End Date"].isna(),
                ["Store", "WeekEndDate"]
            ]
            .head(20)
            .to_string(index=False)
        )

    if traffic["Week End Date"].notna().any():

        logger.info(
            f"Parsed date range      : "
            f"{traffic['Week End Date'].min()} "
            f"to "
            f"{traffic['Week End Date'].max()}"
        )

    # =====================================================
    # 4. CREATE MONTH
    # =====================================================

    traffic["Month"] = (
        traffic["Week End Date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    jan = traffic[
        traffic["Month"] == pd.Timestamp("2026-01-01")
    ]

    logger.info("")
    logger.info("=" * 70)
    logger.info("JANUARY 2026 AFTER MONTH CREATION")
    logger.info("=" * 70)
    logger.info(
        f"January rows: {len(jan):,}"
    )
    logger.info(
        f"January stores: {jan['Store'].nunique():,}"
    )

    logger.info("")
    logger.info("RAW TRAFFIC MONTH COVERAGE")
    logger.info("-" * 70)

    raw_month_coverage = (
        traffic
        .groupby("Month")
        .size()
        .rename("Raw_Rows")
    )

    logger.info(
        raw_month_coverage.to_string()
    )

    # Explicitly show months
    raw_months = (
        traffic["Month"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    logger.info("")
    logger.info(
        f"Raw traffic months ({len(raw_months)}):"
    )
    logger.info(
        ", ".join(
            month.strftime("%Y-%m")
            for month in raw_months
        )
    )

    # =====================================================
    # 5. CLEAN NUMERIC FIELDS
    # =====================================================

    traffic["Exits"] = clean_numeric(
        traffic[" Exits "]
    )

    traffic["# Sale Txns"] = clean_numeric(
        traffic["# Sale Txns"]
    )

    logger.info("")
    logger.info("NUMERIC FIELD DIAGNOSTICS")
    logger.info("-" * 70)

    logger.info(
        f"Missing Exits         : "
        f"{traffic['Exits'].isna().sum():,}"
    )

    logger.info(
        f"Missing Transactions  : "
        f"{traffic['# Sale Txns'].isna().sum():,}"
    )

    # =====================================================
    # 6. STORE-MONTH AGGREGATION
    # =====================================================

    monthly = (
        traffic
        .groupby(
            ["Store", "Month"],
            as_index=False
        )
        .agg(
            Traffic=("Exits", "sum"),
            Transactions=("# Sale Txns", "sum")
        )
    )

    jan_monthly = monthly[
        monthly["Month"] == pd.Timestamp("2026-01-01")
    ]

    logger.info("")
    logger.info("JANUARY 2026 AFTER STORE-MONTH AGGREGATION")
    logger.info("-" * 70)
    logger.info(
        f"January store-month rows: {len(jan_monthly):,}"
    )
    logger.info(
        f"January stores: {jan_monthly['Store'].nunique():,}"
    )

    logger.info("")
    logger.info("STORE-MONTH AGGREGATION")
    logger.info("-" * 70)

    logger.info(
        f"Store-month rows      : {len(monthly):,}"
    )

    logger.info(
        f"Unique stores         : "
        f"{monthly['Store'].nunique():,}"
    )

    logger.info(
        f"Unique months         : "
        f"{monthly['Month'].nunique():,}"
    )

    logger.info(
        f"Date range            : "
        f"{monthly['Month'].min()} "
        f"to "
        f"{monthly['Month'].max()}"
    )

    logger.info("")
    logger.info("Store-month rows by month:")
    logger.info(
        monthly
        .groupby("Month")
        .size()
        .rename("Store_Month_Rows")
        .to_string()
    )

    # =====================================================
    # 7. MERGE STORE → DMA
    # =====================================================

    monthly = monthly.merge(
        stores[["Store", "DMA"]],
        on="Store",
        how="left",
        indicator=True
    )

    jan_monthly = monthly[
        monthly["Month"] == pd.Timestamp("2026-01-01")
    ]

    logger.info("")
    logger.info("JANUARY 2026 AFTER STORE → DMA MERGE")
    logger.info("-" * 70)

    logger.info(
        f"January rows: {len(jan_monthly):,}"
    )

    logger.info(
        f"January stores: {jan_monthly['Store'].nunique():,}"
    )

    logger.info(
        f"January DMAs: "
        f"{jan_monthly['DMA'].nunique():,}"
    )

    logger.info(
        f"January missing DMA rows: "
        f"{jan_monthly['DMA'].isna().sum():,}"
    )

    logger.info("")
    logger.info("January DMA coverage:")
    logger.info(
        jan_monthly["DMA"]
        .value_counts(dropna=False)
        .to_string()
    )

    logger.info("")
    logger.info("STORE → DMA MERGE")
    logger.info("-" * 70)

    merge_status = monthly["_merge"].value_counts()

    logger.info(
        merge_status.to_string()
    )

    missing_dma = monthly["DMA"].isna().sum()

    logger.info(
        f"Traffic rows with missing DMA: {missing_dma:,}"
    )

    if missing_dma > 0:

        logger.warning("")
        logger.warning(
            "Stores with missing DMA assignments:"
        )

        missing_dma_stores = (
            monthly.loc[
                monthly["DMA"].isna(),
                "Store"
            ]
            .drop_duplicates()
            .sort_values()
        )

        logger.warning(
            missing_dma_stores.to_string(index=False)
        )

        logger.warning("")
        logger.warning(
            "Missing DMA rows by month:"
        )

        logger.warning(
            monthly.loc[
                monthly["DMA"].isna()
            ]
            .groupby("Month")
            .size()
            .rename("Missing_DMA_Rows")
            .to_string()
        )

    monthly = monthly.drop(
        columns="_merge"
    )

    # =====================================================
    # 8. DMA-MONTH AGGREGATION
    # =====================================================

    dma_month = (
        monthly
        .groupby(
            ["DMA", "Month"],
            as_index=False
        )
        .agg(
            Traffic=("Traffic", "sum"),
            Transactions=("Transactions", "sum")
        )
    )

    jan_dma = dma_month[
        dma_month["Month"] == pd.Timestamp("2026-01-01")
    ]

    logger.info("")
    logger.info("JANUARY 2026 AFTER DMA AGGREGATION")
    logger.info("-" * 70)

    logger.info(
        f"January DMA-month rows: {len(jan_dma):,}"
    )

    logger.info(
        f"January DMAs: {jan_dma['DMA'].nunique():,}"
    )

    logger.info(
        "\n%s",
        jan_dma.sort_values("DMA").to_string(index=False)
    )

    logger.info("")
    logger.info("FINAL DMA-MONTH TRAFFIC")
    logger.info("-" * 70)

    logger.info(
        f"DMA-month rows        : {len(dma_month):,}"
    )

    logger.info(
        f"Unique DMAs           : "
        f"{dma_month['DMA'].nunique():,}"
    )

    logger.info(
        f"Unique months         : "
        f"{dma_month['Month'].nunique():,}"
    )

    logger.info(
        f"Date range            : "
        f"{dma_month['Month'].min()} "
        f"to "
        f"{dma_month['Month'].max()}"
    )

    logger.info("")
    logger.info("DMA-month rows by month:")

    dma_month_by_month = (
        dma_month
        .groupby("Month")
        .agg(
            DMA_Obs=("DMA", "count"),
            Traffic=("Traffic", "sum"),
            Transactions=("Transactions", "sum")
        )
    )

    logger.info(
        dma_month_by_month.to_string()
    )

    logger.info("")
    logger.info("DMA coverage:")

    dma_coverage = (
        dma_month
        .groupby("DMA")
        .agg(
            Months=("Month", "nunique"),
            Traffic_Obs=("Traffic", "count")
        )
        .sort_values(
            ["Months", "DMA"]
        )
    )

    logger.info(
        dma_coverage.to_string()
    )

    # =====================================================
    # 9. FINAL MONTH LIST
    # =====================================================

    final_months = (
        dma_month["Month"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    logger.info("")
    logger.info(
        f"Final traffic months ({len(final_months)}):"
    )

    logger.info(
        ", ".join(
            month.strftime("%Y-%m")
            for month in final_months
        )
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("END TRAFFIC CLEANING DIAGNOSTICS")
    logger.info("=" * 70)

    return dma_month



# =====================================================
# Data Quality Report
# =====================================================

def generate_data_quality_report(spend, traffic, stores):

    logger.info("")
    logger.info("=" * 60)
    logger.info("DATA QUALITY REPORT")
    logger.info("=" * 60)

    # -----------------------------
    # Spend
    # -----------------------------

    logger.info("")
    logger.info("Marketing Spend")
    logger.info("-" * 60)
    logger.info(f"DMAs:                 {spend['DMA'].nunique():>8}")
    logger.info(f"Months:               {spend['Month'].nunique():>8}")
    logger.info(f"Records:              {len(spend):>8,}")
    logger.info(f"Missing Spend:        {spend['Spend'].isna().sum():>8}")

    # -----------------------------
    # Traffic
    # -----------------------------

    logger.info("")
    logger.info("Traffic")
    logger.info("-" * 60)
    logger.info(f"DMAs:                 {traffic['DMA'].nunique():>8}")
    logger.info(f"Months:               {traffic['Month'].nunique():>8}")
    logger.info(f"DMA-Month Records:    {len(traffic):>8,}")
    logger.info(f"Total Traffic:        {traffic['Traffic'].sum():>12,.0f}")
    logger.info(f"Total Transactions:   {traffic['Transactions'].sum():>12,.0f}")
    logger.info(traffic.head())

    logger.info(traffic.shape)

    logger.info(traffic.dtypes)

    duplicate_dma_month = traffic.duplicated(
        subset=["DMA", "Month"]
    ).sum()

    logger.info(f"Duplicate DMA-Months: {duplicate_dma_month:>8}")

    logger.info("")
    logger.info("Traffic months:")
    logger.info(
        ", ".join(
            traffic["Month"]
            .drop_duplicates()
            .sort_values()
            .dt.strftime("%Y-%m")
            .tolist()
        )
    )

    logger.info("")
    logger.info("Traffic DMA coverage:")
    logger.info(
        traffic
        .groupby("DMA")["Month"]
        .nunique()
        .sort_values()
        .to_string()
    )
    # -----------------------------
    # Store Master
    # -----------------------------

    logger.info("")
    logger.info("Store Master")
    logger.info("-" * 60)

    open_stores = (
        stores["Status"]
        .str.upper()
        .eq("OPEN")
        .sum()
    )

    closed_stores = (
        stores["Status"]
        .str.upper()
        .eq("CLOSED")
        .sum()
    )

    logger.info(f"Total Stores:         {len(stores):>8}")
    logger.info(f"Open Stores:          {open_stores:>8}")
    logger.info(f"Closed Stores:        {closed_stores:>8}")
    logger.info(f"Distinct DMAs:        {stores['DMA'].nunique():>8}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("End Data Quality Report")
    logger.info("=" * 60)
# =====================================================
# Main
# =====================================================

def main():

    logger.info("Starting data cleaning...")

    spend = clean_spend()

    stores = clean_stores()

    traffic = clean_traffic(stores)

    logger.info("Saving cleaned datasets...")

    spend.to_parquet(
        INTERIM_DATA_DIR / "spend_clean.parquet",
        index=False
    )

    stores.to_parquet(
        INTERIM_DATA_DIR / "stores_clean.parquet",
        index=False
    )

    traffic.to_parquet(
        INTERIM_DATA_DIR / "traffic_dma.parquet",
        index=False
    )

    generate_data_quality_report(
        spend,
        traffic,
        stores
    )

    logger.info("Data cleaning complete.")


if __name__ == "__main__":
    main()