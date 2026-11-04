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
    merchandise = pd.read_parquet(INTERIM_DATA_DIR / "merchandise_clean.parquet")

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

    return spend, traffic, stores, merchandise


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

#=========================================================
# Map Merchandise to DMA
#=========================================================

def map_merchandise_to_dma(
    merch,
    stores
):
    """
    Map store-level merchandise records to standardized DMA.

    Store master is treated as the authoritative
    Store -> DMA mapping.
    """

    logger.info(
        "Mapping merchandise stores to DMA..."
    )

    store_dma = (
        stores[
            ["Store", "DMA"]
        ]
        .dropna(subset=["Store"])
        .drop_duplicates(subset=["Store"])
        .copy()
    )

    merch = merch.merge(
        store_dma,
        on="Store",
        how="left",
        validate="many_to_one",
        indicator=True
    )

    matched = (
        merch["_merge"] == "both"
    ).sum()

    unmatched = (
        merch["_merge"] == "left_only"
    ).sum()

    logger.info(
        f"Matched merchandise rows   : {matched:,}"
    )

    logger.info(
        f"Unmatched merchandise rows : {unmatched:,}"
    )

    logger.info(
        f"Match rate                  : "
        f"{matched / len(merch):.2%}"
    )

    merch = merch.drop(
        columns="_merge"
    )

    return merch


#=========================================================
# Merchandise DMA Mapping QA
#=========================================================

def merchandise_dma_qa(merch):
    """
    Report merchandise records without a DMA assignment.
    """

    logger.info(
        "Running merchandise DMA mapping QA..."
    )

    unmatched = (
        merch[
            merch["DMA"].isna()
        ]
        .groupby(
            ["Store", "Store_Location"],
            dropna=False,
            as_index=False
        )
        .agg(
            Rows=("Merch_Group", "size"),
            Groups=("Merch_Group", "nunique"),
            Sales_Dollars=("Sales_Dollars", "sum"),
            Sales_Units=("Sales_Units", "sum")
        )
        .sort_values(
            "Sales_Dollars",
            ascending=False
        )
    )

    output_path = (
        TABLE_DIR /
        "merchandise"
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    unmatched.to_csv(
        output_path /
        "unmatched_merchandise_stores.csv",
        index=False
    )

    logger.info(
        f"Unmatched merchandise stores: "
        f"{unmatched['Store'].nunique():,}"
    )

    return unmatched



# ==========================================================
# Compare DMA Sets
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


#=========================================================
# Aggregate Merchandise to DMA Quarter
#=========================================================

def aggregate_merchandise_dma_quarter(merch):
    """
    Aggregate merchandise data to DMA x Quarter.

    Produces total comp merchandise sales and units
    across stores and merchandise groups.
    """

    logger.info(
        "Aggregating merchandise to DMA-quarter..."
    )

    valid = (
        merch
        .dropna(
            subset=[
                "DMA",
                "Quarter_Period"
            ]
        )
        .copy()
    )

    dma_merch = (
        valid
        .groupby(
            [
                "DMA",
                "Quarter_Period"
            ],
            as_index=False
        )
        .agg(
            Merch_Sales_Dollars=(
                "Sales_Dollars",
                "sum"
            ),
            Merch_Sales_Units=(
                "Sales_Units",
                "sum"
            ),
            Merch_Stores=(
                "Store",
                "nunique"
            ),
            Merch_Groups=(
                "Merch_Group",
                "nunique"
            )
        )
        .sort_values(
            [
                "DMA",
                "Quarter_Period"
            ]
        )
        .reset_index(drop=True)
    )

    logger.info(
        f"DMA-quarter merchandise shape: "
        f"{dma_merch.shape}"
    )

    return dma_merch


#=========================================================
# Merchandise Features
#=========================================================

def create_merchandise_features(
    dma_merch
):
    """
    Create DMA-level quarterly merchandise features.

    Includes:
        - Average unit retail
        - Per-store merchandise metrics
        - Prior-quarter merchandise performance
        - Year-over-year merchandise performance
    """

    logger.info(
        "Creating merchandise features..."
    )

    df = (
        dma_merch
        .sort_values(
            [
                "DMA",
                "Quarter_Period"
            ]
        )
        .copy()
    )

    #-----------------------------------------------------
    # Average Unit Retail
    #-----------------------------------------------------

    df["Merch_AUR"] = np.where(
        df["Merch_Sales_Units"] != 0,
        (
            df["Merch_Sales_Dollars"]
            / df["Merch_Sales_Units"]
        ),
        np.nan
    )

    #-----------------------------------------------------
    # Per Store Measures
    #-----------------------------------------------------

    df["Merch_Sales_per_Store"] = np.where(
        df["Merch_Stores"] > 0,
        (
            df["Merch_Sales_Dollars"]
            / df["Merch_Stores"]
        ),
        np.nan
    )

    df["Merch_Units_per_Store"] = np.where(
        df["Merch_Stores"] > 0,
        (
            df["Merch_Sales_Units"]
            / df["Merch_Stores"]
        ),
        np.nan
    )

    #-----------------------------------------------------
    # Prior Quarter
    #-----------------------------------------------------

    df["Merch_Sales_Lag_1Q"] = (
        df
        .groupby("DMA")[
            "Merch_Sales_Dollars"
        ]
        .shift(1)
    )

    df["Merch_Units_Lag_1Q"] = (
        df
        .groupby("DMA")[
            "Merch_Sales_Units"
        ]
        .shift(1)
    )

    df["Merch_AUR_Lag_1Q"] = (
        df
        .groupby("DMA")[
            "Merch_AUR"
        ]
        .shift(1)
    )

    # -----------------------------------------------------
    # Store Coverage / Composition
    # -----------------------------------------------------

    df["Merch_Store_Count_Change"] = (
        df.groupby("DMA")["Merch_Stores"]
        .diff()
    )

    df["Merch_Store_Count_YoY"] = (
        df.groupby("DMA")["Merch_Stores"]
        .diff(4)
    )

    df["Merch_Store_Count_YoY_Pct"] = (
        df.groupby("DMA")["Merch_Stores"]
        .pct_change(4)
    )

    # -----------------------------------------------------
    # Lagged Per-Store Merchandise Performance
    # -----------------------------------------------------

    df["Merch_Sales_per_Store_Lag_1Q"] = (
        df.groupby("DMA")[
            "Merch_Sales_per_Store"
        ]
        .shift(1)
    )

    df["Merch_Units_per_Store_Lag_1Q"] = (
        df.groupby("DMA")[
            "Merch_Units_per_Store"
        ]
        .shift(1)
    )

    #-----------------------------------------------------
    # Year-over-Year
    #-----------------------------------------------------

    df["Merch_Sales_YoY"] = (
        df
        .groupby("DMA")[
            "Merch_Sales_Dollars"
        ]
        .pct_change(4)
    )

    df["Merch_Units_YoY"] = (
        df
        .groupby("DMA")[
            "Merch_Sales_Units"
        ]
        .pct_change(4)
    )

    df["Merch_AUR_YoY"] = (
        df
        .groupby("DMA")[
            "Merch_AUR"
        ]
        .pct_change(4)
    )

    # -----------------------------------------------------
    # Per-Store YoY Performance
    # -----------------------------------------------------

    df["Merch_Sales_per_Store_YoY"] = (
        df.groupby("DMA")[
            "Merch_Sales_per_Store"
        ]
        .pct_change(4)
    )

    df["Merch_Units_per_Store_YoY"] = (
        df.groupby("DMA")[
            "Merch_Units_per_Store"
        ]
        .pct_change(4)
    )

    logger.info(
        f"Created merchandise features: "
        f"{df.shape}"
    )

    return df


# ==========================================================
# Aggregate Merchandise to DMA x Group x Quarter
# ==========================================================

def aggregate_merchandise_dma_group_quarter(
    merchandise
):
    """
    Aggregate merchandise performance to:

        DMA x Merchandise Group x Quarter

    Preserves merchandise-group detail for downstream
    relationship and cross-correlation analysis.
    """

    logger.info(
        "Aggregating merchandise to "
        "DMA x Group x Quarter..."
    )

    valid = (
        merchandise
        .dropna(
            subset=[
                "DMA",
                "Merch_Group",
                "Quarter_Period"
            ]
        )
        .copy()
    )

    group_quarter = (
        valid
        .groupby(
            [
                "DMA",
                "Merch_Group",
                "Quarter_Period"
            ],
            as_index=False
        )
        .agg(
            Group_Sales_Dollars=(
                "Sales_Dollars",
                "sum"
            ),
            Group_Sales_Units=(
                "Sales_Units",
                "sum"
            ),
            Group_Stores=(
                "Store",
                "nunique"
            )
        )
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
    # Group AUR
    # ------------------------------------------------------

    group_quarter["Group_AUR"] = np.where(
        group_quarter["Group_Sales_Units"] != 0,
        (
            group_quarter["Group_Sales_Dollars"]
            / group_quarter["Group_Sales_Units"]
        ),
        np.nan
    )

    # ------------------------------------------------------
    # Per-store performance
    # ------------------------------------------------------

    group_quarter["Group_Sales_per_Store"] = np.where(
        group_quarter["Group_Stores"] > 0,
        (
            group_quarter["Group_Sales_Dollars"]
            / group_quarter["Group_Stores"]
        ),
        np.nan
    )

    group_quarter["Group_Units_per_Store"] = np.where(
        group_quarter["Group_Stores"] > 0,
        (
            group_quarter["Group_Sales_Units"]
            / group_quarter["Group_Stores"]
        ),
        np.nan
    )

    # ------------------------------------------------------
    # Merchandise Mix
    # ------------------------------------------------------

    quarter_totals = (
        group_quarter
        .groupby(
            [
                "DMA",
                "Quarter_Period"
            ]
        )[
            [
                "Group_Sales_Dollars",
                "Group_Sales_Units"
            ]
        ]
        .transform("sum")
    )

    group_quarter["Group_Dollar_Share"] = np.where(
        quarter_totals["Group_Sales_Dollars"] != 0,
        (
            group_quarter["Group_Sales_Dollars"]
            / quarter_totals["Group_Sales_Dollars"]
        ),
        np.nan
    )

    group_quarter["Group_Unit_Share"] = np.where(
        quarter_totals["Group_Sales_Units"] != 0,
        (
            group_quarter["Group_Sales_Units"]
            / quarter_totals["Group_Sales_Units"]
        ),
        np.nan
    )

    group_quarter = group_quarter.sort_values(
        [
            "DMA",
            "Merch_Group",
            "Quarter_Period"
        ]
    )

    group_cols = [
        "Group_Sales_Dollars",
        "Group_Sales_Units",
        "Group_Sales_per_Store",
        "Group_Units_per_Store",
        "Group_Dollar_Share",
        "Group_Unit_Share",
        "Group_AUR"
    ]

    for col in group_cols:

        group_quarter[
            f"{col}_Lag_1Q"
        ] = (
            group_quarter
            .groupby(
                [
                    "DMA",
                    "Merch_Group"
                ]
            )[col]
            .shift(1)
        )
    

    logger.info(
        f"DMA-Group-Quarter shape: "
        f"{group_quarter.shape}"
    )

    logger.info(
        f"Merchandise Groups: "
        f"{group_quarter['Merch_Group'].nunique():,}"
    )

    group_quarter.to_parquet(
        INTERIM_DATA_DIR /
        "merchandise_dma_group_quarter.parquet",
        index=False
    )

    return group_quarter


# ==========================================================
# Merchandise Coverage QA
# ==========================================================

def merchandise_coverage_qa(
    dma_merchandise,
    panel
):
    """
    Evaluate merchandise coverage relative to the
    DMAs used in the monthly modeling panel.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("MERCHANDISE COVERAGE QA")
    logger.info("=" * 70)

    panel_dmas = set(
        panel["DMA"]
        .dropna()
        .unique()
    )

    merch_dmas = set(
        dma_merchandise["DMA"]
        .dropna()
        .unique()
    )

    matched_dmas = (
        panel_dmas & merch_dmas
    )

    missing_dmas = (
        panel_dmas - merch_dmas
    )

    extra_dmas = (
        merch_dmas - panel_dmas
    )

    

    logger.info(
        f"Panel DMAs             : {len(panel_dmas):,}"
    )

    logger.info(
        f"Merchandise DMAs       : {len(merch_dmas):,}"
    )

    logger.info(
        f"Matched Panel DMAs     : {len(matched_dmas):,}"
    )

    logger.info(
        f"Panel DMAs Missing Merch: {len(missing_dmas):,}"
    )

    logger.info(
        f"Merch-Only DMAs        : {len(extra_dmas):,}"
    )

    logger.info(
        f"Panel DMA Coverage     : "
        f"{len(matched_dmas) / len(panel_dmas):.2%}"
    )

    if missing_dmas:

        logger.warning(
            "Panel DMAs without merchandise:"
        )

        for dma in sorted(missing_dmas):
            logger.warning(f"  {dma}")

    # ------------------------------------------------------
    # Quarter coverage by modeling DMA
    # ------------------------------------------------------

    coverage = (
        dma_merchandise[
            dma_merchandise["DMA"].isin(
                panel_dmas
            )
        ]
        .groupby(
            "DMA",
            as_index=False
        )
        .agg(
            Merch_Quarters=(
                "Quarter_Period",
                "nunique"
            ),
            Merch_Start=(
                "Quarter_Period",
                "min"
            ),
            Merch_End=(
                "Quarter_Period",
                "max"
            ),
            Merch_Stores_Min=(
                "Merch_Stores",
                "min"
            ),
            Merch_Stores_Max=(
                "Merch_Stores",
                "max"
            )
        )
    )

    coverage["Merch_Store_Range"] = (
        coverage["Merch_Stores_Max"]
        - coverage["Merch_Stores_Min"]
    )

    coverage["Merch_Store_Range_Pct"] = np.where(
        coverage["Merch_Stores_Min"] > 0,
        (
            coverage["Merch_Stores_Max"]
            / coverage["Merch_Stores_Min"]
            - 1
        ),
        np.nan
    )

    coverage = coverage.sort_values(
        "Merch_Store_Range_Pct",
        ascending=False
    )

    # Include completely missing panel DMAs
    dma_frame = pd.DataFrame({
        "DMA": sorted(panel_dmas)
    })

    coverage = dma_frame.merge(
        coverage,
        on="DMA",
        how="left"
    )

    output_dir = (
        TABLE_DIR /
        "merchandise"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    coverage.to_csv(
        output_dir /
        "merchandise_coverage_by_dma.csv",
        index=False
    )

    logger.info("")
    logger.info(
        "Merchandise coverage table saved."
    )
    logger.info("=" * 70)

    return coverage


# ==========================================================
# Merchandise Quarter Continuity QA
# ==========================================================

def merchandise_continuity_qa(
    dma_merchandise,
    panel
):
    """
    Validate quarterly continuity for merchandise
    histories within modeling DMAs.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("MERCHANDISE QUARTER CONTINUITY QA")
    logger.info("=" * 70)

    panel_dmas = (
        panel["DMA"]
        .dropna()
        .unique()
    )

    merch = (
        dma_merchandise[
            dma_merchandise["DMA"].isin(
                panel_dmas
            )
        ]
        .copy()
    )

    merch["Quarter"] = (
        merch["Quarter_Period"]
        .dt.to_period("Q")
    )

    rows = []

    for dma, group in merch.groupby("DMA"):

        quarters = (
            group["Quarter"]
            .dropna()
            .drop_duplicates()
            .sort_values()
        )

        if len(quarters) == 0:
            continue

        expected = pd.period_range(
            start=quarters.min(),
            end=quarters.max(),
            freq="Q"
        )

        missing = expected.difference(
            quarters
        )

        rows.append({
            "DMA": dma,
            "Observed_Quarters": len(quarters),
            "Expected_Quarters": len(expected),
            "Missing_Quarters": len(missing),
            "Continuous": len(missing) == 0,
            "First_Quarter": str(quarters.min()),
            "Last_Quarter": str(quarters.max()),
            "Missing_Quarter_List": (
                ", ".join(
                    str(q) for q in missing
                )
            )
        })

    continuity = pd.DataFrame(rows)

    continuity.to_csv(
        TABLE_DIR /
        "merchandise" /
        "merchandise_quarter_continuity.csv",
        index=False
    )

    logger.info(
        f"DMAs evaluated       : "
        f"{len(continuity):,}"
    )

    logger.info(
        f"Continuous histories : "
        f"{continuity['Continuous'].sum():,}"
    )

    logger.info(
        f"Histories with gaps  : "
        f"{(~continuity['Continuous']).sum():,}"
    )

    logger.info("=" * 70)

    return continuity




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
    """
    Create traffic-level and comp-traffic time-series features.

    Traffic
    -------
    Represents absolute DMA-month traffic levels.

    Comp_Traffic
    ------------
    Represents YoY traffic performance for comparable stores:

        Comp_Traffic =
            Comp_Exits / Comp_Exits_LY - 1

    Notes
    -----
    Comp_Traffic is constructed upstream in 02_clean_data.py.
    This function only creates time-series transformations
    from the existing metric.
    """

    logger.info("Creating traffic features...")

    # ------------------------------------------------------
    # Sort before creating time-series features
    # ------------------------------------------------------

    df = (
        df
        .sort_values(
            ["DMA", "Month"]
        )
        .reset_index(drop=True)
    )

    group = df.groupby(
        "DMA",
        group_keys=False
    )

    # ======================================================
    # Absolute Traffic Features
    # ======================================================

    df["Traffic_Lag_1"] = (
        group["Traffic"]
        .shift(1)
    )

    df["Traffic_Lag_12"] = (
        group["Traffic"]
        .shift(12)
    )

    df["Rolling_Traffic_3"] = (
        group["Traffic"]
        .transform(
            lambda x:
            x.rolling(
                3,
                min_periods=1
            ).mean()
        )
    )

    df["Traffic_pct_change"] = (
        group["Traffic"]
        .pct_change(
            fill_method=None
        )
    )

    # ======================================================
    # Comp Traffic Features
    # ======================================================

    if "Comp_Traffic" in df.columns:

        logger.info(
            "Creating comp traffic features..."
        )

        # --------------------------------------------------
        # Prior-month comp performance
        # --------------------------------------------------

        df["Comp_Traffic_Lag_1"] = (
            group["Comp_Traffic"]
            .shift(1)
        )

        # --------------------------------------------------
        # 3-month rolling comp trend
        # --------------------------------------------------

        df["Rolling_Comp_Traffic_3"] = (
            group["Comp_Traffic"]
            .transform(
                lambda x:
                x.rolling(
                    3,
                    min_periods=1
                ).mean()
            )
        )

        logger.info(
            f"Comp Traffic observations : "
            f"{df['Comp_Traffic'].notna().sum():,}"
        )

        logger.info(
            f"Comp Traffic Lag 1        : "
            f"{df['Comp_Traffic_Lag_1'].notna().sum():,}"
        )

        logger.info(
            f"Rolling Comp Traffic 3    : "
            f"{df['Rolling_Comp_Traffic_3'].notna().sum():,}"
        )

    else:

        logger.warning(
            "Comp_Traffic not found. "
            "Comp traffic features were not created."
        )

    if "Comp_Store_Coverage" in df.columns:

        logger.info("")
        logger.info("Comp Store Coverage")

        logger.info(
            df["Comp_Store_Coverage"]
            .describe()
            .to_string()
        )

        low_coverage = (
            df["Comp_Store_Coverage"] < 0.80
        ).sum()

        logger.info(
            f"DMA-months below 80% coverage : "
            f"{low_coverage:,}"
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
# Add Merchandise to Monthly Panel
# ==========================================================

def merge_merchandise_to_panel(
    panel,
    dma_merchandise
):
    """
    Attach quarterly merchandise features to the
    monthly DMA modeling panel.

    Quarterly values are repeated across the three
    constituent months. No monthly interpolation
    is performed.
    """

    logger.info(
        "Merging quarterly merchandise features "
        "to monthly panel..."
    )

    df = panel.copy()
    merch = dma_merchandise.copy()

    # ------------------------------------------------------
    # Create common quarterly key
    # ------------------------------------------------------

    df["Merch_Quarter"] = (
        df["Month"]
        .dt.to_period("Q")
    )

    merch["Merch_Quarter"] = (
        merch["Quarter_Period"]
        .dt.to_period("Q")
    )

    # Quarter_Period is redundant after key creation
    merch = merch.drop(
        columns=["Quarter_Period"],
        errors="ignore"
    )

    # ------------------------------------------------------
    # Merge
    # ------------------------------------------------------

    df = df.merge(
        merch,
        on=[
            "DMA",
            "Merch_Quarter"
        ],
        how="left",
        validate="many_to_one"
    )

    logger.info(
        f"Panel rows after merchandise merge: "
        f"{len(df):,}"
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
# Merchandise Panel QA
# ==========================================================

def merchandise_panel_qa(panel):
    """
    Validate merchandise coverage after merging
    quarterly features onto the monthly panel.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("MERCHANDISE PANEL QA")
    logger.info("=" * 70)

    merch_cols = [
        "Merch_Sales_Dollars",
        "Merch_Sales_Units",
        "Merch_Stores",
        "Merch_Groups",
        "Merch_AUR",
        "Merch_Sales_per_Store",
        "Merch_Units_per_Store",

        "Merch_Store_Count_Change",
        "Merch_Store_Count_YoY",
        "Merch_Store_Count_YoY_Pct",

        "Merch_Sales_Lag_1Q",
        "Merch_Units_Lag_1Q",
        "Merch_AUR_Lag_1Q",

        "Merch_Sales_per_Store_Lag_1Q",
        "Merch_Units_per_Store_Lag_1Q",

        "Merch_Sales_YoY",
        "Merch_Units_YoY",
        "Merch_AUR_YoY",

        "Merch_Sales_per_Store_YoY",
        "Merch_Units_per_Store_YoY"
    ]

    for col in merch_cols:

        if col not in panel.columns:
            continue

        missing = (
            panel[col]
            .isna()
            .sum()
        )

        pct = (
            missing /
            len(panel)
        )

        logger.info(
            f"{col:<28} "
            f"missing={missing:>4,} "
            f"({pct:>6.2%})"
        )

    duplicate_rows = (
        panel
        .duplicated(
            subset=[
                "DMA",
                "Month"
            ]
        )
        .sum()
    )

    logger.info("")
    logger.info(
        f"Duplicate DMA-month rows: "
        f"{duplicate_rows:,}"
    )

    logger.info("=" * 70)

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

    # ------------------------------------------------------
    # Load Data
    # ------------------------------------------------------

    spend, traffic, stores, merchandise = load_data()

    logger.info("")
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 60)

    logger.info(f"Spend shape       : {spend.shape}")
    logger.info(f"Traffic shape     : {traffic.shape}")
    logger.info(f"Stores shape      : {stores.shape}")
    logger.info(f"Merchandise shape : {merchandise.shape}")

    logger.info(
        f"Spend months      : "
        f"{spend['Month'].min()} -> {spend['Month'].max()}"
    )

    logger.info(
        f"Traffic months    : "
        f"{traffic['Month'].min()} -> {traffic['Month'].max()}"
    )

    logger.info(
        f"Merch quarters    : "
        f"{merchandise['Quarter_Period'].min()} -> "
        f"{merchandise['Quarter_Period'].max()}"
    )

    # ------------------------------------------------------
    # Store / DMA Features
    # ------------------------------------------------------

    dma_features = aggregate_store_features(
        stores
    )

    # ------------------------------------------------------
    # Merchandise Processing
    # ------------------------------------------------------

    logger.info("")
    logger.info("=" * 60)
    logger.info("MERCHANDISE FEATURE ENGINEERING")
    logger.info("=" * 60)

    # Map store-level merchandise records to standardized DMA
    merchandise = map_merchandise_to_dma(
        merchandise,
        stores
    )

    # Identify unmatched merchandise stores
    merchandise_dma_qa(
        merchandise
    )

    # Aggregate Store x Group x Quarter
    # to DMA x Quarter
    dma_merchandise = aggregate_merchandise_dma_quarter(
        merchandise
    )

    # Create quarterly merchandise features
    dma_merchandise = create_merchandise_features(
        dma_merchandise
    )

    # ------------------------------------------------------
    # Merchandise Group Feature Layer
    # ------------------------------------------------------

    dma_group_merchandise = (
        aggregate_merchandise_dma_group_quarter(
            merchandise
        )
    )

    dma_group_merchandise.to_parquet(
        INTERIM_DATA_DIR /
        "merchandise_dma_group_quarter.parquet",
        index=False
    )

    logger.info(
        "Saved "
        "merchandise_dma_group_quarter.parquet"
    )

    # Save standalone merchandise feature table
    dma_merchandise.to_parquet(
        INTERIM_DATA_DIR /
        "merchandise_dma_quarter.parquet",
        index=False
    )

    logger.info(
        "Saved merchandise_dma_quarter.parquet"
    )

    # ------------------------------------------------------
    # DMA Comparison
    # ------------------------------------------------------

    compare_dma_sets(
        spend,
        traffic,
        stores
    )

    # ------------------------------------------------------
    # Traffic Coverage QA
    # ------------------------------------------------------

    traffic_qa = traffic_coverage_qa(
        traffic
    )

    # ------------------------------------------------------
    # Build Core Monthly Panel
    # ------------------------------------------------------

    panel = merge_panel(
        spend,
        traffic,
        dma_features
    )

    traffic_qa_panel = traffic_coverage_qa(
        traffic,
        panel=panel
    )

    # ------------------------------------------------------
    # Calendar Features
    # ------------------------------------------------------

    panel = create_calendar_features(
        panel
    )

    # ------------------------------------------------------
    # Marketing Features
    # ------------------------------------------------------

    panel = create_marketing_features(
        panel
    )

    # ------------------------------------------------------
    # Traffic Features
    # ------------------------------------------------------

    panel = create_traffic_features(
        panel
    )

    # ------------------------------------------------------
    # Store Features
    # ------------------------------------------------------

    panel = create_store_features(
        panel
    )

    # ======================================================
    # Merchandise Validation
    # ======================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("MERCHANDISE VALIDATION")
    logger.info("=" * 60)

    # ------------------------------------------------------
    # Coverage relative to actual modeling panel
    # ------------------------------------------------------

    merchandise_coverage = merchandise_coverage_qa(
        dma_merchandise,
        panel
    )

    # ------------------------------------------------------
    # Quarter continuity
    # ------------------------------------------------------

    merchandise_continuity = merchandise_continuity_qa(
        dma_merchandise,
        panel
    )

    # ------------------------------------------------------
    # Merchandise validation summary
    # ------------------------------------------------------

    logger.info("")
    logger.info("Merchandise Validation Summary")
    logger.info("-" * 60)

    logger.info(
        f"DMA-Quarter rows       : "
        f"{len(dma_merchandise):,}"
    )

    logger.info(
        f"All Merchandise DMAs   : "
        f"{dma_merchandise['DMA'].nunique():,}"
    )

    logger.info(
        f"Panel DMAs             : "
        f"{panel['DMA'].nunique():,}"
    )

    if not merchandise_coverage.empty:

        covered_dmas = (
            merchandise_coverage[
                "Merch_Quarters"
            ]
            .notna()
            .sum()
        )

        logger.info(
            f"Panel DMAs with Merch  : "
            f"{covered_dmas:,}"
        )

    if not merchandise_continuity.empty:

        continuous_dmas = (
            merchandise_continuity[
                "Continuous"
            ]
            .sum()
        )

        gap_dmas = (
            ~merchandise_continuity[
                "Continuous"
            ]
        ).sum()

        logger.info(
            f"Continuous Histories   : "
            f"{continuous_dmas:,}"
        )

        logger.info(
            f"Histories with Gaps    : "
            f"{gap_dmas:,}"
        )

    # ======================================================
    # Merchandise Panel Merge
    # ======================================================

    panel_merch = merge_merchandise_to_panel(
        panel,
        dma_merchandise
    )

    merchandise_panel_qa(
        panel_merch
    )

    panel_merch.to_parquet(
        PROCESSED_DATA_DIR /
        "panel_data_merch.parquet",
        index=False
    )

    # ------------------------------------------------------
    # Core Panel Validation
    # ------------------------------------------------------

    validate_panel(
        panel
    )

    panel_summary(
        panel
    )

    # ------------------------------------------------------
    # Save Existing Core Panel
    # ------------------------------------------------------

    panel.to_parquet(
        PROCESSED_DATA_DIR /
        "panel_data.parquet",
        index=False
    )

    # ------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------

    logger.info("")
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)

    logger.info(
        f"Rows          : {len(panel):,}"
    )

    logger.info(
        f"DMAs          : "
        f"{panel['DMA'].nunique():,}"
    )

    logger.info(
        f"Months        : "
        f"{panel['Month'].nunique():,}"
    )

    logger.info(
        f"Columns       : "
        f"{len(panel.columns):,}"
    )

    logger.info("")
    logger.info("Merchandise Features")
    logger.info("-" * 60)

    logger.info(
        f"DMA-Quarters  : "
        f"{len(dma_merchandise):,}"
    )

    logger.info(
        f"Merch DMAs    : "
        f"{dma_merchandise['DMA'].nunique():,}"
    )

    logger.info(
        f"Merch Columns : "
        f"{len(dma_merchandise.columns):,}"
    )

    logger.info("")
    logger.info(panel.head())

    logger.info("=" * 60)


if __name__ == "__main__":
    main()