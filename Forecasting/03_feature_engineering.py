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
    cx = pd.read_parquet(INTERIM_DATA_DIR / "cx_clean.parquet")
    weather = pd.read_parquet(INTERIM_DATA_DIR / "weather_monthly.parquet")

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

    return spend, traffic, stores, cx, weather


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


# ==========================================================
# Aggregate CX Features
# ==========================================================

def aggregate_cx_features(cx, stores):

    logger.info("Aggregating CX metrics to DMA...")

    cx = cx.copy()
    stores = stores.copy()

    # -----------------------------------------------
    # Attach DMA to CX records
    # -----------------------------------------------

    store_dma = stores[
        ["Store", "DMA"]
    ].drop_duplicates()

    cx = cx.merge(
        store_dma,
        on="Store",
        how="left"
    )

    logger.info(
        f"CX rows missing DMA: "
        f"{cx['DMA'].isna().sum():,}"
    )

    # -----------------------------------------------
    # Convert quarter to month-level
    # -----------------------------------------------

    quarter_months = {
        "Q1": [1, 2, 3],
        "Q2": [4, 5, 6],
        "Q3": [7, 8, 9],
        "Q4": [10, 11, 12]
    }

    records = []

    for _, row in cx.iterrows():

        months = quarter_months.get(
            row["quarter"],
            []
        )

        for m in months:

            new_row = row.copy()

            new_row["Month"] = pd.Timestamp(
                year=int(row["year"]),
                month=m,
                day=1
            )

            records.append(new_row)

    cx_monthly = pd.DataFrame(records)

    # -----------------------------------------------
    # Grade score mapping
    # -----------------------------------------------

    grade_map = {
        "A": 4,
        "B": 3,
        "C": 2,
        "D": 1
    }

    cx_monthly["Facility_Grade_Score"] = (
        cx_monthly["facility_grade"]
        .map(grade_map)
    )

    cx_monthly["Rating_Grade_Score"] = (
        cx_monthly["rating_grade"]
        .map(grade_map)
    )

    # -----------------------------------------------
    # DMA-Month aggregation
    # -----------------------------------------------

    dma_cx = (
        cx_monthly
        .groupby(
            ["DMA", "Month"],
            as_index=False
        )
        .agg(

            # -----------------------------
            # Sales / Performance
            # -----------------------------
            Avg_Comp_Sales=(
                "Comp_Sales",
                "mean"
            ),

            # -----------------------------
            # CX Grades
            # -----------------------------
            Avg_Facility_Grade=(
                "Facility_Grade_Score",
                "mean"
            ),

            Avg_Rating_Grade=(
                "Rating_Grade_Score",
                "mean"
            ),

            # -----------------------------
            # Competition
            # -----------------------------
            Avg_Competitors=(
                "Direct_Competitors",
                "mean"
            ),

            # -----------------------------
            # Coverage
            # -----------------------------
            CX_Store_Count=(
                "Store",
                "nunique"
            ),

            # -----------------------------
            # Grade Mix
            # -----------------------------
            Pct_A_Facility=(
                "facility_grade",
                lambda x: (x == "A").mean()
            ),

            Pct_B_Facility=(
                "facility_grade",
                lambda x: (x == "B").mean()
            ),

            Pct_C_Facility=(
                "facility_grade",
                lambda x: (x == "C").mean()
            ),

            # -----------------------------
            # Urbanicity Mix
            # -----------------------------
            Pct_Urban=(
                "urbanicity",
                lambda x: (
                    x.str.contains(
                        "urban",
                        case=False,
                        na=False
                    ) &
                    ~x.str.contains(
                        "suburban",
                        case=False,
                        na=False
                    )
                ).mean()
            ),

            Pct_Suburban=(
                "urbanicity",
                lambda x: (
                    x.str.contains(
                        "suburban",
                        case=False,
                        na=False
                    )
                ).mean()
            ),

            # -----------------------------
            # Competitor Presence
            # -----------------------------
            Pct_Marshalls=(
                "marshalls_in_same_center",
                lambda x: (x > 0).mean()
            ),

            Pct_Ross=(
                "ross_stores_in_same_center",
                lambda x: (x > 0).mean()
            ),

            Pct_TJMaxx=(
                "t_j_maxx_in_same_center",
                lambda x: (x > 0).mean()
            )

        )
    )

    logger.info(
        f"DMA-Month CX rows: "
        f"{len(dma_cx):,}"
    )

    logger.info(
        f"Unique DMAs: "
        f"{dma_cx['DMA'].nunique():,}"
    )

    logger.info(
        f"Date Range: "
        f"{dma_cx['Month'].min()} -> "
        f"{dma_cx['Month'].max()}"
    )

    return dma_cx


# ==========================================================
# DMA Weather Mapping
# ==========================================================

DMA_WEATHER_REGION = {

    # -------------------------------------
    # NY
    # -------------------------------------

    "New York, NY": "NY",
    "Philadelphia, PA": "NY",
    "Boston, MA (Manchester, NH)": "NY",
    "Providence, RI-New Bedford, MA": "NY",
    "Hartford & New Haven, CT": "NY",
    "Buffalo, NY": "NY",
    "Rochester, NY": "NY",

    # -------------------------------------
    # FL
    # -------------------------------------

    "Miami-Ft. Lauderdale, FL": "FL",
    "Orlando-Daytona Beach-Melbourne, FL": "FL",
    "Tampa-St. Petersburg (Sarasota), FL": "FL",
    "West Palm Beach-Ft. Pierce, FL": "FL",

    # -------------------------------------
    # CHI
    # -------------------------------------

    "Chicago, IL": "CHI",
    "Milwaukee, WI": "CHI",
    "Cleveland-Akron (Canton), OH": "CHI",
    "Columbus, OH": "CHI",
    "Cincinnati, OH": "CHI",
    "Detroit, MI": "CHI",

    # -------------------------------------
    # WA
    # -------------------------------------

    "Seattle-Tacoma, WA": "WA",
    "Portland, OR": "WA",

    # -------------------------------------
    # LA
    # -------------------------------------

    "Los Angeles, CA": "LA",
    "San Diego, CA": "LA",
    "Las Vegas, NV": "LA",
    "Phoenix, AZ": "LA",

    # -------------------------------------
    # TX
    # -------------------------------------

    "Dallas-Ft. Worth, TX": "TX",
    "Houston, TX": "TX",
    "San Antonio, TX": "TX",
    "Austin, TX": "TX",
    "Harlingen-Weslaco-Brownsville-McAllen, TX": "TX"
}

DMA_WEATHER_REGION.update({

    # East Coast
    "Atlanta, GA": "FL",
    "Baltimore, MD": "NY",
    "Charlotte, NC": "NY",
    "Washington, DC (Hagerstown, MD)": "NY",

    # Florida
    "Ft. Myers-Naples, FL": "FL",

    # Texas / Plains
    "El Paso, TX (Las Cruces, NM)": "TX",
    "Oklahoma City, OK": "TX",

    # Midwest
    "Minneapolis-St. Paul, MN": "CHI",

    # Southwest
    "Phoenix (Prescott), AZ": "LA",

    # California
    "Sacramento-Stockton-Modesto, CA": "LA",

    # Special Case
    "Puerto Rico": "FL"

})


# ==========================================================
# Aggregate Weather Features
# ==========================================================

def aggregate_weather_features(
    weather
):

    logger.info(
        "Creating weather features..."
    )

    weather = weather.copy()

    regions = [
        "NY",
        "FL",
        "CHI",
        "WA",
        "LA",
        "TX"
    ]

    records = []

    for region in regions:

        temp_col = (
            f"Avg_Temp_{region}"
        )

        weather[temp_col] = (

            weather[
                f"TMAX_Degrees_Fahrenheit_{region}"
            ]

            +

            weather[
                f"TMIN_Degrees_Fahrenheit_{region}"
            ]

        ) / 2

        region_df = weather[
            [
                "Month",

                temp_col,

                f"PRCP_Inches_{region}",
                f"SNOW_Inches_{region}",
                f"SNWD_Inches_{region}"

            ]
        ].copy()

        region_df["Weather_Region"] = (
            region
        )

        region_df = region_df.rename(
            columns={

                temp_col:
                    "Avg_Temp",

                f"PRCP_Inches_{region}":
                    "Monthly_Precip",

                f"SNOW_Inches_{region}":
                    "Monthly_Snow",

                f"SNWD_Inches_{region}":
                    "Snow_Depth"

            }
        )

        records.append(
            region_df
        )

    weather_features = pd.concat(
        records,
        ignore_index=True
    )

    logger.info(
        f"Weather rows: "
        f"{len(weather_features):,}"
    )

    return weather_features


# ==========================================================
# Assign Weather Regions
# ==========================================================

def map_dma_weather_regions(
    df
):

    logger.info(
        "Assigning weather regions..."
    )

    df = df.copy()

    df["Weather_Region"] = (
        df["DMA"]
        .map(
            DMA_WEATHER_REGION
        )
    )

    missing = (
        df["Weather_Region"]
        .isna()
        .sum()
    )

    logger.info(
        f"Rows without weather region: "
        f"{missing:,}"
    )

    logger.info(
        "\nWeather Region Coverage"
    )

    logger.info(
        df["Weather_Region"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    missing_weather = (

        df.loc[
            df["Weather_Region"].isna(),
            "DMA"
        ]

        .drop_duplicates()

        .sort_values()

    )

    logger.info(
        "\nDMAs Missing Weather Mapping"
    )

    logger.info(
        missing_weather.to_string(
            index=False
        )
    )

    return df


# ==========================================================
# Weather Features
# ==========================================================

def create_weather_extremes(panel):

    logger.info(
        "Creating weather features..."
    )

    panel = panel.copy()

    # --------------------------------------------------
    # Temperature Extremes
    # --------------------------------------------------

    panel["Hot_Weather"] = (
        panel["Avg_Temp"] >= 85
    ).astype(int)

    panel["Cold_Weather"] = (
        panel["Avg_Temp"] <= 35
    ).astype(int)

    # --------------------------------------------------
    # Precipitation Extremes
    # --------------------------------------------------

    rain_threshold = (
        panel["Monthly_Precip"]
        .quantile(.75)
    )

    panel["Heavy_Rain"] = (
        panel["Monthly_Precip"]
        >=
        rain_threshold
    ).astype(int)

    # --------------------------------------------------
    # Snow Events
    # --------------------------------------------------

    panel["Snow_Event"] = (
        panel["Monthly_Snow"] > 0
    ).astype(int)

    snow_threshold = (
        panel["Monthly_Snow"]
        .quantile(.75)
    )

    panel["Heavy_Snow"] = (
        panel["Monthly_Snow"]
        >=
        snow_threshold
    ).astype(int)

    # --------------------------------------------------
    # Temperature Anomaly
    # --------------------------------------------------

    typical_temp = (

        panel

        .groupby(
            [
                "Weather_Region",
                "Month_Number"
            ],
            as_index=False
        )

        .agg(
            Typical_Temp=(
                "Avg_Temp",
                "mean"
            )
        )

    )

    panel = panel.merge(

        typical_temp,

        on=[
            "Weather_Region",
            "Month_Number"
        ],

        how="left"

    )

    panel["Temp_Anomaly"] = (

        panel["Avg_Temp"]

        -

        panel["Typical_Temp"]

    )

    logger.info(
        f"Heavy Rain Threshold: "
        f"{rain_threshold:.2f}"
    )

    logger.info(
        f"Heavy Snow Threshold: "
        f"{snow_threshold:.2f}"
    )

    logger.info(
        f"Hot Weather Months: "
        f"{panel['Hot_Weather'].sum():,}"
    )

    logger.info(
        f"Cold Weather Months: "
        f"{panel['Cold_Weather'].sum():,}"
    )

    logger.info(
        f"Snow Event Months: "
        f"{panel['Snow_Event'].sum():,}"
    )

    return panel

#---------------------------------------------
def compare_dma_sets(spend, traffic, stores, dma_cx):

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

    cx_dmas = set(
        dma_cx["DMA"]
        .dropna()
        .unique()
    )

    logger.info(
        f"CX DMAs: {len(cx_dmas)}"
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

    logger.info("\nCX missing from Traffic")

    for dma in sorted(cx_dmas - traffic_dmas):
        logger.info(f"  {dma}")    

    logger.info("=" * 70)
#----------------------------------------------------
    
# ==========================================================
# Merge Panel
# ==========================================================



def merge_panel(spend, traffic, dma_features, dma_cx, weather_features):

    logger.info("TRAFFIC ENTERING MERGE_PANEL")
    
    logger.info(
        traffic.columns.tolist()
    )
    
    # print(
    #     traffic[
    #         traffic["DMA"].str.contains(
    #             "Columbus|Rochester",
    #             case=False,
    #             na=False
    #         )
    #     ][
    #         ["DMA", "DMA_Original"]
    #     ]
    #     .drop_duplicates()
    # )

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

    df = df.merge(
        dma_cx,
        on=["DMA", "Month"],
        how="left"
    )

    missing_cx = (
        df["Avg_Comp_Sales"]
        .isna()
        .sum()
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("CX COVERAGE")
    logger.info("=" * 70)

    logger.info(
        f"Rows missing CX: "
        f"{missing_cx:,}"
    )

    logger.info(
        f"Rows with CX: "
        f"{len(df) - missing_cx:,}"
    )

    df = map_dma_weather_regions(
        df
    )

    df = df.merge(

        weather_features,

        on=[
            "Month",
            "Weather_Region"
        ],

        how="left"

    )

    df["Monthly_Snow"] = (
        df["Monthly_Snow"]
        .fillna(0)
    )

    df["Snow_Depth"] = (
        df["Snow_Depth"]
        .fillna(0)
    )


    logger.info("")
    logger.info("=" * 70)
    logger.info("WEATHER COVERAGE")
    logger.info("=" * 70)

    for col in [
        "Avg_Temp",
        "Monthly_Precip",
        "Monthly_Snow"
    ]:

        logger.info(
            f"{col:<25}"
            f"{df[col].isna().sum():,}"
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

    monthly_spend = (
        df.groupby("Month")["Spend"]
        .transform("sum")
    )

    df["Spend_Share"] = (
        df["Spend"] /
        monthly_spend
    )

    df["Spend_3M_Change"] = (
        df["Spend"] -
        group["Spend"].shift(3)
    )

    df["Spend_Adstock"] = (
        df["Spend"]
        + 0.5 * group["Spend"].shift(1)
        + 0.25 * group["Spend"].shift(2)
    ).fillna(df["Spend"])

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

    df["Traffic_YoY"] = (
        df["Traffic"] /
        df["Traffic_Lag_12"]
    ) - 1

    df["Traffic_vs_Rolling3"] = (
        df["Traffic"] /
        df["Rolling_Traffic_3"]
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

    df["Traffic_per_Capita"] = (
        df["Traffic"] /
        df["DMA_Population"]
    )

    df["Stores_per_Million"] = (
        df["Store_Count"] /
        (df["DMA_Population"] / 1_000_000)
    )


    df["TradeArea_per_Store"] = (
        df["TradeAreaPopulation"] /
        df["Store_Count"]
    )

    return df


# ==========================================================
# Transaction Features
# ==========================================================

def create_transaction_features(df):

    logger.info("Creating transaction features...")

    df = df.sort_values(
        ["DMA", "Month"]
    )

    group = df.groupby("DMA")

    # ------------------------------------------------------
    # Lag Features
    # ------------------------------------------------------

    for lag in [1, 3, 6, 12]:

        df[f"Transactions_Lag_{lag}"] = (
            group["Transactions"]
            .shift(lag)
        )

    # ------------------------------------------------------
    # Rolling Means
    # ------------------------------------------------------

    for window in [3, 6, 12]:

        df[f"Rolling_Transactions_{window}"] = (
            group["Transactions"]
            .transform(
                lambda x: x.rolling(
                    window,
                    min_periods=1
                ).mean()
            )
        )

    # ------------------------------------------------------
    # Growth Metrics
    # ------------------------------------------------------

    df["Transactions_pct_change"] = (
        group["Transactions"]
        .pct_change()
    )

    df["Transactions_YoY"] = (
        (
            df["Transactions"] /
            df["Transactions_Lag_12"]
        ) - 1
    )

    # ------------------------------------------------------
    # Conversion Metrics
    # ------------------------------------------------------

    df["Transaction_Rate"] = (
        df["Transactions"] /
        df["Traffic"]
    )

    df["Traffic_per_Transaction"] = (
        df["Traffic"] /
        df["Transactions"]
    )

    # ------------------------------------------------------
    # Relative Performance
    # ------------------------------------------------------

    df["Transactions_vs_Rolling3"] = (
        df["Transactions"] /
        df["Rolling_Transactions_3"]
    )

    # ------------------------------------------------------
    # Safety Checks
    # ------------------------------------------------------

    replace_cols = [
        "Transaction_Rate",
        "Traffic_per_Transaction",
        "Transactions_YoY",
        "Transactions_vs_Rolling3"
    ]

    for col in replace_cols:

        df[col] = (
            df[col]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

    return df

# ==========================================================
# CX Features
# ==========================================================

def create_cx_features(df):

    logger.info("Creating CX features...")

    # ------------------------------------------------------
    # Composite Quality Scores
    # ------------------------------------------------------

    df["CX_Composite"] = (
        df[
            [
                "Avg_Facility_Grade",
                "Avg_Rating_Grade"
            ]
        ]
        .mean(axis=1)
    )

    # ------------------------------------------------------
    # Competition Metrics
    # ------------------------------------------------------

    df["Competitor_Density"] = (
        df["Avg_Competitors"] /
        df["Store_Count"]
    )

    df["Competitor_Presence"] = (
        (
            df["Pct_Marshalls"] +
            df["Pct_Ross"] +
            df["Pct_TJMaxx"]
        ) > 0
    ).astype(int)

    # ------------------------------------------------------
    # Quality Indicators
    # ------------------------------------------------------

    df["High_Grade_DMA"] = (
        df["Pct_A_Facility"] >= 0.50
    ).astype(int)

    df["Low_Grade_DMA"] = (
        df["Pct_C_Facility"] >= 0.50
    ).astype(int)

    # ------------------------------------------------------
    # Store Coverage
    # ------------------------------------------------------

    df["CX_Coverage"] = (
        df["CX_Store_Count"] /
        df["Store_Count"]
    )

    # ------------------------------------------------------
    # Comp Sales Metrics
    # ------------------------------------------------------

    df["Comp_Sales_Index"] = (
        df["Avg_Comp_Sales"] /
        df["Avg_Comp_Sales"]
        .mean()
    )

    # ------------------------------------------------------
    # Urbanicity Metrics
    # ------------------------------------------------------

    df["Urban_Suburban_Ratio"] = (
        df["Pct_Urban"] /
        df["Pct_Suburban"]
    )

    # ------------------------------------------------------
    # Cleanup Infinite Values
    # ------------------------------------------------------

    replace_cols = [
        "Competitor_Density",
        "CX_Coverage",
        "Comp_Sales_Index",
        "Urban_Suburban_Ratio"
    ]

    for col in replace_cols:

        df[col] = (
            df[col]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
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

    # --------------------------------------------------
    # Load Data
    # --------------------------------------------------

    spend, traffic, stores, cx, weather = load_data()

    # --------------------------------------------------
    # Aggregate Features
    # --------------------------------------------------

    dma_features = aggregate_store_features(
        stores
    )

    dma_cx = aggregate_cx_features(
        cx,
        stores
    )

    weather_features = (
        aggregate_weather_features(
            weather
        )
    )

    logger.info(
        f"Spend shape: {spend.shape}"
    )

    logger.info(
        f"Traffic shape: {traffic.shape}"
    )

    logger.info(
        f"CX shape: {cx.shape}"
    )

    logger.info(
        f"DMA CX shape: {dma_cx.shape}"
    )

    logger.info(
        f"Spend months: "
        f"{spend['Month'].min()} -> "
        f"{spend['Month'].max()}"
    )

    logger.info(
        f"Traffic months: "
        f"{traffic['Month'].min()} -> "
        f"{traffic['Month'].max()}"
    )

    logger.info(
        f"Weather shape: "
        f"{weather.shape}"
    )

    logger.info(
        f"Weather feature shape: "
        f"{weather_features.shape}"
    )

    logger.info(
        f"Weather months: "
        f"{weather_features['Month'].min()} -> "
        f"{weather_features['Month'].max()}"
    )

    logger.info(
        f"Weather regions: "
        f"{weather_features['Weather_Region'].nunique()}"
    )
    

    # --------------------------------------------------
    # DMA QA
    # --------------------------------------------------

    compare_dma_sets(
        spend,
        traffic,
        stores,
        dma_cx
    )

    traffic_qa = traffic_coverage_qa(
        traffic
    )

    # --------------------------------------------------
    # Build Panel
    # --------------------------------------------------

    panel = merge_panel(
        spend,
        traffic,
        dma_features,
        dma_cx,
        weather_features
    )

    traffic_qa_panel = traffic_coverage_qa(
        traffic,
        panel=panel
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("WEATHER COVERAGE")
    logger.info("=" * 60)

    for col in [
        "Avg_Temp",
        "Monthly_Precip",
        "Monthly_Snow"
    ]:

        if col in panel.columns:

            logger.info(
                f"{col:<30}"
                f"{panel[col].isna().sum():,}"
            )
    
    # --------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------

    panel = create_calendar_features(
        panel
    )

    panel = create_marketing_features(
        panel
    )

    panel = create_traffic_features(
        panel
    )

    panel = create_transaction_features(
        panel
    )

    panel = create_store_features(
        panel
    )

    panel = create_cx_features(
        panel
    )

    panel = create_weather_extremes(
        panel
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    validate_panel(
        panel
    )

    panel_summary(
        panel
    )

    important_features = [
        "Traffic",
        "Transactions",
        "Spend",
        "Store_Count",
        "Avg_Comp_Sales",
        "CX_Composite",
        "Avg_Temp",
        "Monthly_Precip",
        "Monthly_Snow",
        "Hot_Weather",
        "Cold_Weather",
        "Heavy_Rain",
        "Snow_Event",
        "Heavy_Snow",
        "Temp_Anomaly"
    ]

    logger.info("")
    logger.info("=" * 60)
    logger.info("FEATURE NULL CHECK")
    logger.info("=" * 60)

    for col in important_features:

        if col in panel.columns:

            logger.info(
                f"{col:<30}"
                f"{panel[col].isna().sum():,}"
            )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    panel.to_parquet(
        PROCESSED_DATA_DIR / "panel_data.parquet",
        index=False
    )

    # --------------------------------------------------
    # Final Logging
    # --------------------------------------------------

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

    logger.info(
        f"Date Range    : "
        f"{panel['Month'].min()} -> "
        f"{panel['Month'].max()}"
    )

    logger.info("")
    logger.info("Sample Rows:")

    logger.info(
        panel.head().to_string()
    )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
