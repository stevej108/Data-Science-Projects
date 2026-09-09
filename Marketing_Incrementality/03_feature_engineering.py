"""
03_feature_engineering.py

Purpose
-------
Create the master analytical panel used by all downstream models.

Inputs
------
data/interim/
    spend_clean.parquet
    traffic_dma.parquet
    stores_clean.parquet

Outputs
-------
data/processed/
    panel_data.parquet
"""

import logging
import numpy as np
import pandas as pd

from config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    TABLE_DIR,
    MAX_LAG,
    ROLLING_WINDOWS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# Load Data
# ==========================================================

def load_data():

    logger.info("Loading cleaned datasets...")

    spend = pd.read_parquet(INTERIM_DATA_DIR / "spend_clean.parquet")
    traffic = pd.read_parquet(INTERIM_DATA_DIR / "traffic_dma.parquet")
    stores = pd.read_parquet(INTERIM_DATA_DIR / "stores_clean.parquet")

    logger.info("=" * 70)
    logger.info("SPEND DMA QA")
    logger.info("=" * 70)

    missing_dma_spend = spend[
        spend["DMA"].isna()
    ]

    logger.info(
        f"Missing spend DMA rows: "
        f"{len(missing_dma_spend)}"
    )

    if not missing_dma_spend.empty:

        logger.info(
            "\n%s",
            missing_dma_spend.to_string(index=False)
        )

    return spend, traffic, stores


# ==========================================================
# Aggregate Store Features
# ==========================================================

def aggregate_store_features(stores):

    logger.info("Aggregating store metrics to DMA...")

    stores = stores.copy()

    stores = stores[stores["Status"].str.upper() == "OPEN"]

    dma_features = (
        stores
        .groupby("DMA", as_index=False)
        .agg(
            Store_Count=("Store", "nunique"),
            Total_Selling_SqFt=("Selling_SqFt", "sum"),
            Avg_Selling_SqFt=("Selling_SqFt", "mean"),
            DMA_Population=("DMAPopulation", "first"),
            TradeAreaPopulation=("TradeAreaPopulation", "sum")
        )
    )

    return dma_features



#---------------------------------------------
def compare_dma_sets(spend, traffic, stores):

    logger.info("=" * 70)
    logger.info("DMA COMPARISON")

    spend_dmas = set(
        spend["DMA"]
        .dropna()
        .unique()
    )

    traffic_dmas = set(
        traffic["DMA"]
        .dropna()
        .unique()
    )

    store_dmas = set(
        stores["DMA"]
        .dropna()
        .unique()
    )

    logger.info(
        f"Spend DMAs: {len(spend_dmas)}"
    )

    logger.info(
        f"Traffic DMAs: {len(traffic_dmas)}"
    )

    logger.info(
        f"Store DMAs: {len(store_dmas)}"
    )


    logger.info("\nSpend missing from Traffic")

    for dma in sorted(spend_dmas - traffic_dmas):

        logger.info(f"  {dma}")


    logger.info("\nSpend missing from Stores")

    for dma in sorted(spend_dmas - store_dmas):

        logger.info(f"  {dma}")

    logger.info("=" * 70)
#----------------------------------------------------
    
# ==========================================================
# Merge Panel
# ==========================================================



def merge_panel(spend, traffic, dma_features):

    logger.info("TRAFFIC ENTERING MERGE_PANEL")

    print(
        traffic[
            traffic["DMA"].str.contains(
                "Columbus|Rochester",
                case=False,
                na=False
            )
        ][
            ["DMA", "DMA_Original"]
        ]
        .drop_duplicates()
    )

    logger.info("Merging datasets...")

    logger.info("=" * 60)
    logger.info("SPEND DUPLICATES")

    spend_dupes = spend[
        spend.duplicated(
            ["DMA", "Month"],
            keep=False
        )
    ].sort_values(["DMA", "Month"])

    logger.info(f"Duplicate rows: {len(spend_dupes)}")

    print(spend_dupes)

    logger.info("=" * 60)
    logger.info("TRAFFIC DUPLICATES")

    traffic_dupes = traffic[
        traffic.duplicated(
            ["DMA", "Month"],
            keep=False
        )
    ].sort_values(["DMA", "Month"])

    logger.info(f"Duplicate rows: {len(traffic_dupes)}")

    print(traffic_dupes)

    logger.info("=" * 60)
    logger.info("DMA FEATURE DUPLICATES")

    feature_dupes = dma_features[
        dma_features.duplicated(
            "DMA",
            keep=False
        )
    ]

    logger.info(f"Duplicate rows: {len(feature_dupes)}")

    print(feature_dupes)
    
    # ------------------------------------------------------
    # HARD DMA VALIDATION
    # ------------------------------------------------------

    missing_spend_dma = spend["DMA"].isna().sum()

    if missing_spend_dma > 0:

        bad_spend = (
            spend.loc[
                spend["DMA"].isna(),
                ["DMA", "Month", "Spend"]
            ]
            .sort_values("Month")
        )

        logger.error(
            f"Found {missing_spend_dma} spend rows with missing DMA."
        )

        logger.error(
            f"\n{bad_spend.to_string(index=False)}"
        )

        raise ValueError(
            f"Spend dataset contains "
            f"{missing_spend_dma} rows with missing DMA."
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info("PRE-MERGE MONTH COVERAGE")
    logger.info("=" * 70)

    logger.info(
        f"Spend months: "
        f"{spend['Month'].min()} → {spend['Month'].max()}"
    )

    logger.info(
        f"Traffic months: "
        f"{traffic['Month'].min()} → {traffic['Month'].max()}"
    )

    logger.info("")
    logger.info("Spend rows by month:")
    logger.info(
        spend.groupby("Month")
        .size()
        .to_string()
    )

    logger.info("")
    logger.info("Traffic rows by month:")
    logger.info(
        traffic.groupby("Month")
        .size()
        .to_string()
    )

    logger.info("")
    logger.info("January 2026 pre-merge:")
    logger.info(
        f"Spend rows: "
        f"{(spend['Month'] == pd.Timestamp('2026-01-01')).sum():,}"
    )

    logger.info(
        f"Traffic rows: "
        f"{(traffic['Month'] == pd.Timestamp('2026-01-01')).sum():,}"
    )

    df = spend.merge(
        traffic,
        on=["DMA", "Month"],
        how="left",
        indicator=True
    )
    logger.info(df["_merge"].value_counts())
    if df["DMA"].isna().any():
        raise ValueError(
            "Spend → Traffic merge produced rows with missing DMA."
        )

    if df["Month"].isna().any():
        raise ValueError(
            "Spend → Traffic merge produced rows with missing Month."
        )

    df = df.merge(
        dma_features,
        on="DMA",
        how="left"
    )

    missing_store_features = df["Store_Count"].isna().sum()

    if missing_store_features > 0:

        bad_dmas = (
            df.loc[
                df["Store_Count"].isna(),
                ["DMA"]
            ]
            .drop_duplicates()
        )

        logger.error(
            f"Found {missing_store_features} rows with "
            "missing store features."
        )

        logger.error(
            f"\n{bad_dmas.to_string(index=False)}"
        )

        raise ValueError(
            "Panel contains DMA values without matching "
            "store features."
        )

    merge_counts = (
        df["_merge"]
        .value_counts()
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("SPEND → TRAFFIC MERGE QA")
    logger.info("=" * 70)

    logger.info(
        f"Spend DMA-month rows: "
        f"{len(spend):,}"
    )

    logger.info(
        f"Traffic DMA-month rows: "
        f"{len(traffic):,}"
    )

    logger.info(
        f"Matched rows: "
        f"{merge_counts.get('both', 0):,}"
    )

    logger.info(
        f"Spend rows without traffic: "
        f"{merge_counts.get('left_only', 0):,}"
    )

    logger.info(
        f"Traffic rows not represented in spend: "
        f"{merge_counts.get('right_only', 0):,}"
    )

    return df


# ==========================================================
# Calendar Features
# ==========================================================

def create_calendar_features(df):

    logger.info("Creating calendar features...")

    df["Year"] = df["Month"].dt.year

    df["Month_Number"] = df["Month"].dt.month

    df["Month_Name"] = df["Month"].dt.month_name()

    df["Quarter"] = df["Month"].dt.quarter

    df["Holiday_Season"] = (
        df["Month_Number"]
        .isin([11, 12])
        .astype(int)
    )

    return df


# ==========================================================
# Marketing Features
# ==========================================================

def create_marketing_features(df):

    logger.info("Creating marketing features...")

    df = df.sort_values(["DMA", "Month"])

    group = df.groupby("DMA")

    for lag in [1, 2, 3, 6]:

        df[f"Spend_Lag_{lag}"] = group["Spend"].shift(lag)

    for window in ROLLING_WINDOWS:

        df[f"Rolling_Spend_{window}"] = (
            group["Spend"]
            .transform(
                lambda x:
                x.rolling(window, min_periods=1).mean()
            )
        )

    df["Cumulative_Spend"] = (
        group["Spend"]
        .cumsum()
    )

    df["Spend_pct_change"] = (
        group["Spend"]
        .pct_change()
    )

    negative_spend = df[df["Spend"] < 0]

    if not negative_spend.empty:
        logger.warning(
            f"Found {len(negative_spend)} negative spend records."
        )
        logger.warning(negative_spend[["DMA", "Month", "Spend"]])

    df["Log_Spend"] = np.log1p(df["Spend"].clip(lower=0))

    mean_spend = group["Spend"].transform("mean")

    df["Spend_Index"] = df["Spend"] / mean_spend

    threshold = group["Spend"].transform(
        lambda x: x.quantile(.75)
    )

    df["High_Spend"] = (
        df["Spend"] > threshold
    ).astype(int)

    return df


# ==========================================================
# Traffic Features
# ==========================================================

def create_traffic_features(df):

    logger.info("Creating traffic features...")

    group = df.groupby("DMA")

    df["Traffic_Lag_1"] = group["Traffic"].shift(1)

    df["Traffic_Lag_12"] = group["Traffic"].shift(12)

    df["Rolling_Traffic_3"] = (
        group["Traffic"]
        .transform(
            lambda x:
            x.rolling(3, min_periods=1).mean()
        )
    )

    df["Traffic_pct_change"] = (
        group["Traffic"]
        .pct_change()
    )

    return df


# ==========================================================
# Store Features
# ==========================================================

def create_store_features(df):

    logger.info("Creating normalized store metrics...")

    df["Traffic_per_Store"] = (
        df["Traffic"] /
        df["Store_Count"]
    )

    df["Transactions_per_Store"] = (
        df["Transactions"] /
        df["Store_Count"]
    )

    df["Traffic_per_1000SqFt"] = (
        df["Traffic"] /
        (df["Total_Selling_SqFt"] / 1000)
    )

    df["Spend_per_Store"] = (
        df["Spend"] /
        df["Store_Count"]
    )

    df["Spend_per_1000SqFt"] = (
        df["Spend"] /
        (df["Total_Selling_SqFt"] / 1000)
    )

    return df


# ==========================================================
# Validation
# ==========================================================

def validate_panel(df):

    logger.info("Running validation checks...")

    duplicates = df.duplicated(
        subset=["DMA", "Month"]
    ).sum()

    logger.info(f"Duplicate DMA-Month rows : {duplicates}")

    logger.info(
        f"Missing Spend : {df['Spend'].isna().sum()}"
    )

    logger.info(
        f"Missing Traffic : {df['Traffic'].isna().sum()}"
    )

    logger.info(
        f"Missing Store Count : {df['Store_Count'].isna().sum()}"
    )

    if duplicates > 0:
        raise ValueError(
            "Duplicate DMA-Month combinations found."
        )

    logger.info("Validation complete.")


def traffic_coverage_qa(traffic, panel=None):
    """
    Diagnose traffic data coverage before and after panel construction.

    Parameters
    ----------
    traffic : pd.DataFrame
        Clean traffic DMA-month dataset before merging.

    panel : pd.DataFrame, optional
        Final analytical panel after merging.

    Returns
    -------
    dict
        Traffic coverage diagnostics.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAFFIC COVERAGE QA")
    logger.info("=" * 70)

    traffic = traffic.copy()

    # ------------------------------------------------------
    # Standardize dates
    # ------------------------------------------------------

    traffic["Month"] = pd.to_datetime(
        traffic["Month"]
    )

    # ------------------------------------------------------
    # Basic coverage
    # ------------------------------------------------------

    traffic_dmas = (
        traffic["DMA"]
        .dropna()
        .unique()
    )

    traffic_months = (
        traffic["Month"]
        .dropna()
        .sort_values()
        .unique()
    )

    logger.info(
        f"Traffic rows:       {len(traffic):,}"
    )

    logger.info(
        f"Traffic DMAs:       {len(traffic_dmas):,}"
    )

    logger.info(
        f"Traffic months:     {len(traffic_months):,}"
    )

    logger.info(
        f"Traffic date range: "
        f"{traffic['Month'].min().date()} "
        f"to "
        f"{traffic['Month'].max().date()}"
    )

    # ------------------------------------------------------
    # Duplicate DMA-month combinations
    # ------------------------------------------------------

    duplicates = (
        traffic
        .duplicated(
            subset=["DMA", "Month"]
        )
        .sum()
    )

    logger.info(
        f"Duplicate DMA-month rows: {duplicates:,}"
    )

    # ------------------------------------------------------
    # Expected DMA-month grid
    # ------------------------------------------------------

    expected_pairs = (
        len(traffic_dmas)
        * len(traffic_months)
    )

    actual_pairs = (
        traffic[
            ["DMA", "Month"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    structural_gaps = (
        expected_pairs
        - actual_pairs
    )

    logger.info(
        f"Expected DMA-month combinations: "
        f"{expected_pairs:,}"
    )

    logger.info(
        f"Actual DMA-month combinations:   "
        f"{actual_pairs:,}"
    )

    logger.info(
        f"Structural missing combinations: "
        f"{structural_gaps:,}"
    )

    # ------------------------------------------------------
    # Identify missing DMA-month combinations
    # ------------------------------------------------------

    expected_index = pd.MultiIndex.from_product(
        [
            sorted(traffic_dmas),
            sorted(traffic_months)
        ],
        names=["DMA", "Month"]
    )

    actual_index = pd.MultiIndex.from_frame(
        traffic[
            ["DMA", "Month"]
        ]
        .drop_duplicates()
    )

    missing_pairs = (
        expected_index
        .difference(actual_index)
    )

    missing_pairs_df = (
        missing_pairs
        .to_frame(index=False)
    )

    # ------------------------------------------------------
    # Missing structural observations by month
    # ------------------------------------------------------

    logger.info("")
    logger.info("Structural gaps by month")

    if missing_pairs_df.empty:

        logger.info(
            "No structural DMA-month gaps detected."
        )

    else:

        gaps_by_month = (
            missing_pairs_df
            .groupby("Month")
            .size()
            .sort_index()
        )

        for month, count in gaps_by_month.items():

            logger.warning(
                f"  {month.date()}: "
                f"{count} missing DMA observations"
            )

    # ------------------------------------------------------
    # Missing structural observations by DMA
    # ------------------------------------------------------

    logger.info("")
    logger.info("Structural gaps by DMA")

    if missing_pairs_df.empty:

        logger.info(
            "No structural DMA gaps detected."
        )

    else:

        gaps_by_dma = (
            missing_pairs_df
            .groupby("DMA")
            .size()
            .sort_values(
                ascending=False
            )
        )

        for dma, count in gaps_by_dma.items():

            logger.warning(
                f"  {dma}: "
                f"{count} missing months"
            )

    # ------------------------------------------------------
    # Missing Traffic values
    # ------------------------------------------------------

    missing_traffic = (
        traffic["Traffic"]
        .isna()
        .sum()
    )

    missing_traffic_pct = (
        missing_traffic
        / len(traffic)
        * 100
        if len(traffic) > 0
        else 0
    )

    logger.info("")
    logger.info(
        f"Missing Traffic values: "
        f"{missing_traffic:,} "
        f"({missing_traffic_pct:.2f}%)"
    )

    # ------------------------------------------------------
    # Missing Traffic by month
    # ------------------------------------------------------

    missing_by_month = (
        traffic
        .groupby("Month")
        ["Traffic"]
        .apply(lambda x: x.isna().sum())
    )

    missing_by_month = (
        missing_by_month[
            missing_by_month > 0
        ]
        .sort_index()
    )

    logger.info("")
    logger.info("Missing Traffic values by month")

    if missing_by_month.empty:

        logger.info(
            "No missing Traffic values by month."
        )

    else:

        for month, count in missing_by_month.items():

            logger.warning(
                f"  {month.date()}: "
                f"{count} missing values"
            )

    # ------------------------------------------------------
    # Missing Traffic by DMA
    # ------------------------------------------------------

    missing_by_dma = (
        traffic
        .groupby("DMA")
        ["Traffic"]
        .apply(lambda x: x.isna().sum())
    )

    missing_by_dma = (
        missing_by_dma[
            missing_by_dma > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    logger.info("")
    logger.info("Missing Traffic values by DMA")

    if missing_by_dma.empty:

        logger.info(
            "No missing Traffic values by DMA."
        )

    else:

        for dma, count in missing_by_dma.items():

            logger.warning(
                f"  {dma}: "
                f"{count} missing values"
            )

    # ------------------------------------------------------
    # Compare with final panel
    # ------------------------------------------------------

    if panel is not None:

        logger.info("")
        logger.info("Traffic Coverage After Panel Merge")

        panel_missing = (
            panel["Traffic"]
            .isna()
            .sum()
        )

        logger.info(
            f"Panel rows: "
            f"{len(panel):,}"
        )

        logger.info(
            f"Panel rows with missing Traffic: "
            f"{panel_missing:,}"
        )

        # --------------------------------------------------
        # Identify panel rows missing traffic
        # --------------------------------------------------

        panel_missing_rows = (
            panel.loc[
                panel["Traffic"].isna(),
                ["DMA", "Month"]
            ]
        )

        if not panel_missing_rows.empty:

            logger.warning(
                "Panel DMA-month observations "
                "with missing Traffic:"
            )

            for _, row in (
                panel_missing_rows
                .sort_values(
                    ["Month", "DMA"]
                )
                .iterrows()
            ):

                logger.warning(
                    f"  {row['DMA']} | "
                    f"{row['Month'].date()}"
                )

    # ------------------------------------------------------
    # Export structural gaps
    # ------------------------------------------------------

    output_path = (
        TABLE_DIR
        / "traffic_coverage_gaps.csv"
    )

    if not missing_pairs_df.empty:

        missing_pairs_df.to_csv(
            output_path,
            index=False
        )

        logger.info(
            f"Traffic coverage gaps saved to "
            f"{output_path}"
        )

    else:

        logger.info(
            "No traffic coverage gaps to export."
        )

    logger.info("=" * 70)

    return {
        "traffic_dmas": len(traffic_dmas),
        "traffic_months": len(traffic_months),
        "expected_pairs": expected_pairs,
        "actual_pairs": actual_pairs,
        "structural_gaps": structural_gaps,
        "missing_traffic": missing_traffic,
        "missing_pairs": missing_pairs_df
    }

# ==========================================================
# Panel Summary Report
# ==========================================================

def panel_summary(df):
    """
    Produce QA summaries for the analytical panel.

    Outputs
    -------
    panel_summary_statistics.csv
    dma_summary.csv
    month_summary.csv
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL PANEL SUMMARY")
    logger.info("=" * 70)

    # ------------------------------------------------------
    # Dataset Shape
    # ------------------------------------------------------

    logger.info(f"Rows:      {len(df):,}")
    logger.info(f"Columns:   {len(df.columns)}")
    logger.info(f"DMAs:      {df['DMA'].nunique()}")
    logger.info(f"Months:    {df['Month'].nunique()}")

    logger.info(
        f"Date Range: "
        f"{df['Month'].min().date()} "
        f"to "
        f"{df['Month'].max().date()}"
    )

    # ------------------------------------------------------
    # Duplicate Keys
    # ------------------------------------------------------

    duplicates = df.duplicated(
        subset=["DMA", "Month"]
    ).sum()

    logger.info("")
    logger.info(f"Duplicate DMA-Month rows: {duplicates}")

    # ------------------------------------------------------
    # Missing Values
    # ------------------------------------------------------

    logger.info("")
    logger.info("Critical Modeling Variables")

    for col in [
        "Spend",
        "Traffic",
        "Transactions"
    ]:

        if col not in df.columns:
            continue

        missing_count = df[col].isna().sum()

        logger.info(
            f"{col:<20}"
            f"missing={missing_count:>6,} "
            f"({missing_count / len(df) * 100:5.2f}%)"
        )

    # ------------------------------------------------------
    # Numeric Column Types
    # ------------------------------------------------------

    logger.info("")
    logger.info("Numeric Data Types")

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:

        logger.info(
            f"{col:<30}{df[col].dtype}"
        )

    # ------------------------------------------------------
    # Descriptive Statistics
    # ------------------------------------------------------

    logger.info("")
    logger.info("Descriptive Statistics")

    summary_cols = [
        c for c in [
            "Spend",
            "Traffic",
            "Transactions",
            "Store_Count",
            "Total_Selling_SqFt",
            "Traffic_per_Store",
            "Spend_per_Store"
        ]
        if c in df.columns
    ]

    summary = (
        df[summary_cols]
        .describe()
        .round(2)
    )

    logger.info("\n%s", summary)

    summary.to_csv(
        TABLE_DIR / "panel_summary_statistics.csv"
    )

    # ======================================================
    # DMA SUMMARY
    # ======================================================

    logger.info("")
    logger.info("Building DMA summary...")

    dma_summary = (

        df.groupby("DMA", as_index=False)

        .agg(

            Months=("Month", "nunique"),

            Store_Count=("Store_Count", "max"),

            Total_Selling_SqFt=("Total_Selling_SqFt", "max"),

            Total_Spend=("Spend", "sum"),

            Avg_Monthly_Spend=("Spend", "mean"),

            Median_Monthly_Spend=("Spend", "median"),

            Total_Traffic=("Traffic", "sum"),

            Avg_Monthly_Traffic=("Traffic", "mean"),

            Median_Monthly_Traffic=("Traffic", "median"),

            Total_Transactions=("Transactions", "sum"),

            Avg_Monthly_Transactions=("Transactions", "mean")

        )

    )

    dma_summary["Spend_per_Store"] = (

        dma_summary["Total_Spend"]

        / dma_summary["Store_Count"]

    )

    dma_summary["Traffic_per_Store"] = (

        dma_summary["Total_Traffic"]

        / dma_summary["Store_Count"]

    )

    dma_summary["Spend_per_1000SqFt"] = (

        dma_summary["Total_Spend"]

        / (dma_summary["Total_Selling_SqFt"] / 1000)

    )

    dma_summary["Traffic_per_1000SqFt"] = (

        dma_summary["Total_Traffic"]

        / (dma_summary["Total_Selling_SqFt"] / 1000)

    )

    dma_summary = dma_summary.sort_values(
        "Total_Spend",
        ascending=False
    )

    dma_summary.to_csv(

        TABLE_DIR / "dma_summary.csv",

        index=False

    )

    logger.info(
        f"DMA Summary written ({len(dma_summary)} DMAs)"
    )

    # ======================================================
    # MONTH SUMMARY
    # ======================================================

    logger.info("")
    logger.info("Building Monthly summary...")

    month_summary = (

        df.groupby("Month", as_index=False)

        .agg(

            Active_DMAs=("DMA", "nunique"),

            Total_Spend=("Spend", "sum"),

            Avg_DMA_Spend=("Spend", "mean"),

            Total_Traffic=("Traffic", "sum"),

            Avg_DMA_Traffic=("Traffic", "mean"),

            Total_Transactions=("Transactions", "sum"),

            Avg_DMA_Transactions=("Transactions", "mean")

        )

    )

    month_summary["Spend_pct_change"] = (

        month_summary["Total_Spend"]

        .pct_change()

    )

    month_summary["Traffic_pct_change"] = (

        month_summary["Total_Traffic"]

        .pct_change()

    )

    month_summary.to_csv(

        TABLE_DIR / "month_summary.csv",

        index=False

    )

    logger.info(
        f"Monthly Summary written ({len(month_summary)} months)"
    )

    # ======================================================
    # Totals
    # ======================================================

    logger.info("")
    logger.info("Dataset Totals")

    logger.info(
        f"Total Spend:        ${df['Spend'].sum():,.0f}"
    )

    logger.info(
        f"Total Traffic:      {df['Traffic'].sum():,.0f}"
    )

    logger.info(
        f"Total Transactions: {df['Transactions'].sum():,.0f}"
    )

    logger.info("")
    logger.info("Reports Saved")

    logger.info(
        f"  panel_summary_statistics.csv"
    )

    logger.info(
        f"  dma_summary.csv"
    )

    logger.info(
        f"  month_summary.csv"
    )

    logger.info("=" * 70)

# ==========================================================
# Main
# ==========================================================

def main():

    spend, traffic, stores = load_data()


    dma_features = aggregate_store_features(stores)

    logger.info(f"Spend shape: {spend.shape}")
    logger.info(f"Traffic shape: {traffic.shape}")

    logger.info(f"Spend months: {spend['Month'].min()} -> {spend['Month'].max()}")
    logger.info(f"Traffic months: {traffic['Month'].min()} -> {traffic['Month'].max()}")

    compare_dma_sets(
    spend,
    traffic,
    stores
   )
    traffic_qa = traffic_coverage_qa(
        traffic
    )
    
    panel = merge_panel(
        spend,
        traffic,
        dma_features
    )

    traffic_qa_panel = traffic_coverage_qa(
        traffic,
        panel=panel
    )

    panel = create_calendar_features(panel)

    panel = create_marketing_features(panel)

    panel = create_traffic_features(panel)

    panel = create_store_features(panel)

    validate_panel(panel)

    panel_summary(panel)

    panel.to_parquet(
        PROCESSED_DATA_DIR / "panel_data.parquet",
        index=False
    )


    logger.info("")
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Rows          : {len(panel):,}")
    logger.info(f"DMAs          : {panel['DMA'].nunique()}")
    logger.info(f"Months        : {panel['Month'].nunique()}")
    logger.info(f"Columns       : {len(panel.columns)}")
    logger.info(panel.head())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()