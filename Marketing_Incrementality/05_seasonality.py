"""
04_seasonality.py

Prepare the analytical panel for time-series modeling.

This script:

1. Loads panel_data.parquet
2. Validates model readiness
3. Creates seasonality features
4. Performs time-series decomposition
5. Generates QA reports
6. Saves panel_model.parquet
"""


import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    TABLE_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

def load_panel():

    logger.info("Loading analytical panel...")

    panel = pd.read_parquet(
        PROCESSED_DATA_DIR / "panel_data.parquet"
    )

    panel["Month"] = pd.to_datetime(panel["Month"])

    logger.info(
        f"Loaded {len(panel):,} observations."
    )

    return panel

# ==========================================================
# Panel Validation
# ==========================================================

def validate_panel(panel):
    """
    Comprehensive validation of the engineered panel prior to
    creating seasonality features.

    This function is intended to answer one question:

        "Is this panel ready for statistical modeling?"
    """

    logger.info("=" * 70)
    logger.info("MODEL PANEL VALIDATION")
    logger.info("=" * 70)

    n_dmas = panel["DMA"].nunique(dropna=True)
    n_months = panel["Month"].nunique()

    expected_rows = n_dmas * n_months
    actual_rows = len(panel)

    logger.info(f"Rows                : {actual_rows:,}")
    logger.info(f"Columns             : {len(panel.columns)}")
    logger.info(f"DMAs                : {n_dmas}")
    logger.info(f"Months              : {n_months}")

    logger.info("")
    logger.info(
        f"Date Range          : "
        f"{panel['Month'].min().date()} "
        f"to "
        f"{panel['Month'].max().date()}"
    )

    logger.info("")
    logger.info("-" * 70)
    logger.info("PANEL COMPLETENESS")
    logger.info("-" * 70)

    logger.info(f"Expected Rows       : {expected_rows:,}")
    logger.info(f"Actual Rows         : {actual_rows:,}")
    logger.info(
        f"Coverage            : {actual_rows / expected_rows:.1%}"
    )

    logger.info("")
    logger.info("-" * 70)
    logger.info("MISSING VALUES")
    logger.info("-" * 70)

    critical_columns = [

        "DMA",
        "Month",
        "Spend",
        "Traffic",
        "Transactions",
        "Store_Count",
        "Total_Selling_SqFt",
        "DMA_Population",
        "TradeAreaPopulation"

    ]

    for col in critical_columns:

        missing = panel[col].isna().sum()

        logger.info(
            f"{col:<25}"
            f"{missing:>6} "
            f"({missing/actual_rows:6.1%})"
        )

    logger.info("")
    logger.info("-" * 70)
    logger.info("MERGE STATUS")
    logger.info("-" * 70)

    if "_merge" in panel.columns:

        logger.info(panel["_merge"].value_counts())

    logger.info("")
    logger.info("-" * 70)
    logger.info("DUPLICATE CHECKS")
    logger.info("-" * 70)

    duplicates = panel.duplicated(
        subset=["DMA", "Month"]
    ).sum()

    logger.info(f"Duplicate DMA-Month : {duplicates}")

    logger.info("")
    logger.info("-" * 70)
    logger.info("NEGATIVE VALUES")
    logger.info("-" * 70)

    logger.info(
        f"Negative Spend      : {(panel['Spend'] < 0).sum()}"
    )

    logger.info(
        f"Negative Traffic    : {(panel['Traffic'] < 0).sum()}"
    )

    logger.info("")
    logger.info("-" * 70)
    logger.info("DMA COVERAGE")
    logger.info("-" * 70)

    dma_coverage = (

        panel
        .groupby("DMA")
        .agg(

            Months=("Month", "nunique"),
            Traffic_Obs=("Traffic", "count"),
            Spend_Obs=("Spend", "count"),
            Store_Count=("Store_Count", "first")

        )
        .sort_values("Months")

    )

    logger.info(dma_coverage)

    logger.info("")
    logger.info("-" * 70)
    logger.info("SPEND SUMMARY")
    logger.info("-" * 70)

    logger.info(panel["Spend"].describe())

    logger.info("")
    logger.info("-" * 70)
    logger.info("TRAFFIC SUMMARY")
    logger.info("-" * 70)

    logger.info(panel["Traffic"].describe())

    logger.info("")
    logger.info("-" * 70)
    logger.info("MODEL READINESS")
    logger.info("-" * 70)

    complete = panel[

        panel[
            [
                "Spend",
                "Traffic",
                "Store_Count"
            ]
        ].notna().all(axis=1)

    ]

    logger.info(
        f"Complete observations : {len(complete):,}"
    )

    logger.info(
        f"Ready for modeling    : {len(complete)/actual_rows:.1%}"
    )

    logger.info("=" * 70)


# ==========================================================
# Export Validation Summary
# ==========================================================

def export_validation_summary(panel):
    """
    Export modeling validation metrics and DMA coverage reports.
    """

    output_dir = TABLE_DIR / "seasonality"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    n_dmas = panel["DMA"].nunique(dropna=True)
    n_months = panel["Month"].nunique()

    expected_rows = n_dmas * n_months
    actual_rows = len(panel)

    summary = pd.DataFrame({

        "Metric": [

            "Rows",
            "Columns",
            "DMAs",
            "Months",
            "Expected Rows",
            "Actual Rows",
            "Panel Coverage",
            "Date Start",
            "Date End",
            "Missing DMA",
            "Missing Traffic",
            "Missing Spend",
            "Missing Transactions",
            "Missing Store Count",
            "Negative Spend",
            "Negative Traffic",
            "Duplicate DMA-Month"

        ],

        "Value": [

            actual_rows,
            len(panel.columns),
            n_dmas,
            n_months,
            expected_rows,
            actual_rows,
            round(actual_rows / expected_rows, 4),
            panel["Month"].min(),
            panel["Month"].max(),
            panel["DMA"].isna().sum(),
            panel["Traffic"].isna().sum(),
            panel["Spend"].isna().sum(),
            panel["Transactions"].isna().sum(),
            panel["Store_Count"].isna().sum(),
            (panel["Spend"] < 0).sum(),
            (panel["Traffic"] < 0).sum(),
            panel.duplicated(
                ["DMA", "Month"]
            ).sum()

        ]

    })

    summary.to_csv(
        TABLE_DIR /"seasonality" / "model_validation_summary.csv",
        index=False
    )

    dma_coverage = (

        panel
        .groupby("DMA")
        .agg(

            Months=("Month", "nunique"),
            Traffic_Observations=("Traffic", "count"),
            Spend_Observations=("Spend", "count"),
            Store_Count=("Store_Count", "first")

        )
        .sort_values("Months")

    )

    dma_coverage.to_csv(
        TABLE_DIR /"seasonality" / "dma_model_coverage.csv"
    )

    logger.info(
        "Saved model_validation_summary.csv"
    )

    logger.info(
        "Saved dma_model_coverage.csv"
    )    

# ==========================================================
# Time Series Features
# ==========================================================

def create_time_features(panel):
    """
    Create generic time-series features.

    These become the foundation for later
    seasonality and modeling work.
    """

    logger.info("=" * 70)
    logger.info("CREATING TIME FEATURES")
    logger.info("=" * 70)

    df = panel.copy()

    df = df.sort_values(
        ["DMA", "Month"]
    )

    # ------------------------------------------------------
    # Sequential month index
    # ------------------------------------------------------

    start_month = df["Month"].min()

    df["Time_Index"] = (
        (
            df["Month"].dt.year
            - start_month.year
        ) * 12
        +
        (
            df["Month"].dt.month
            - start_month.month
        )
    )

    # ------------------------------------------------------
    # Month within year
    # ------------------------------------------------------

    df["Month_of_Year"] = (
        df["Month"].dt.month
    )

    # ------------------------------------------------------
    # Quarter
    # ------------------------------------------------------

    df["Quarter"] = (
        df["Month"].dt.quarter
    )

    # ------------------------------------------------------
    # Year
    # ------------------------------------------------------

    df["Year"] = (
        df["Month"].dt.year
    )

    # ------------------------------------------------------
    # Beginning / End of Year flags
    # ------------------------------------------------------

    df["Start_of_Year"] = (
        df["Month_of_Year"] <= 2
    ).astype(int)

    df["End_of_Year"] = (
        df["Month_of_Year"] >= 11
    ).astype(int)

    # ------------------------------------------------------
    # Holiday retail period
    # ------------------------------------------------------

    df["Holiday_Period"] = (
        df["Month_of_Year"]
        .isin([11, 12])
    ).astype(int)

    logger.info("Added:")

    logger.info("  Time_Index")
    logger.info("  Month_of_Year")
    logger.info("  Quarter")
    logger.info("  Year")
    logger.info("  Start_of_Year")
    logger.info("  End_of_Year")
    logger.info("  Holiday_Period")

    logger.info("")

    return df

def summarize_time_features(df):

    logger.info("=" * 70)
    logger.info("TIME FEATURE SUMMARY")
    logger.info("=" * 70)

    cols = [
        "Time_Index",
        "Month_of_Year",
        "Quarter",
        "Year",
        "Start_of_Year",
        "End_of_Year",
        "Holiday_Period",
    ]

    logger.info(df[cols].describe(include="all"))

# ==========================================================
# Cyclical Calendar Features
# ==========================================================

def create_cyclical_features(panel):
    """
    Create cyclical encodings for seasonal variables.
    """

    logger.info("=" * 70)
    logger.info("CREATING CYCLICAL FEATURES")
    logger.info("=" * 70)

    df = panel.copy()

    month = df["Month_of_Year"]

    df["Month_Sin"] = np.sin(
        2 * np.pi * month / 12
    )

    df["Month_Cos"] = np.cos(
        2 * np.pi * month / 12
    )

    quarter = df["Quarter"]

    df["Quarter_Sin"] = np.sin(
        2 * np.pi * quarter / 4
    )

    df["Quarter_Cos"] = np.cos(
        2 * np.pi * quarter / 4
    )

    logger.info("Created cyclical calendar features.")

    return df

# ==========================================================
# Fourier Seasonality
# ==========================================================

def create_fourier_features(
    panel,
    period=12,
    order=3
):
    """
    Create Fourier seasonal terms.
    """

    logger.info("=" * 70)
    logger.info("CREATING FOURIER FEATURES")
    logger.info("=" * 70)

    df = panel.copy()

    t = df["Time_Index"]

    for k in range(1, order + 1):

        df[f"Fourier_Sin_{k}"] = np.sin(
            2 * np.pi * k * t / period
        )

        df[f"Fourier_Cos_{k}"] = np.cos(
            2 * np.pi * k * t / period
        )

    logger.info(
        f"Created {order} Fourier harmonics."
    )

    return df

# ==========================================================
# Retail Seasonality Flags
# ==========================================================

def create_retail_seasonality(panel):
    """
    Retail-specific seasonal indicators.
    """

    logger.info("=" * 70)
    logger.info("CREATING RETAIL SEASONAL FEATURES")
    logger.info("=" * 70)

    df = panel.copy()

    m = df["Month_of_Year"]

    df["Back_to_School"] = (
        m.isin([7, 8])
    ).astype(int)

    df["Holiday_Shopping"] = (
        m.isin([11, 12])
    ).astype(int)

    df["Post_Holiday"] = (
        m == 1
    ).astype(int)

    df["Spring"] = (
        m.isin([3, 4, 5])
    ).astype(int)

    df["Summer"] = (
        m.isin([6, 7, 8])
    ).astype(int)

    df["Fall"] = (
        m.isin([9, 10, 11])
    ).astype(int)

    df["Winter"] = (
        m.isin([12, 1, 2])
    ).astype(int)

    logger.info("Retail seasonal indicators created.")

    return df

# ==========================================================
# Seasonality Summary
# ==========================================================

def summarize_seasonality(panel):
    """
    Summarize seasonal feature distributions.
    """

    logger.info("=" * 70)
    logger.info("SEASONAL FEATURE SUMMARY")
    logger.info("=" * 70)

    seasonal_cols = [
        "Month_Sin",
        "Month_Cos",
        "Quarter_Sin",
        "Quarter_Cos",
        "Back_to_School",
        "Holiday_Shopping",
        "Post_Holiday",
        "Spring",
        "Summer",
        "Fall",
        "Winter"
    ]

    logger.info(panel[seasonal_cols].describe())

# ============================================================
# Feature Registry
# ============================================================

def build_feature_registry(df):
    """
    Create documentation for every feature that enters the model.
    """

    logger.info("Building feature registry...")

    registry = []

    feature_groups = {

        "Identifier": [
            "DMA",
            "Month"
        ],

        "Target": [
            "Traffic",
            "Transactions"
        ],

        "Marketing": [
            "Spend",
            "Log_Spend",
            "Spend_Index",
            "High_Spend"
        ],

        "Marketing Lags": [
            "Spend_Lag_1",
            "Spend_Lag_2",
            "Spend_Lag_3",
            "Spend_Lag_6"
        ],

        "Marketing Rolling": [
            "Rolling_Spend_3",
            "Rolling_Spend_6",
            "Rolling_Spend_12",
            "Cumulative_Spend"
        ],

        "Traffic History": [
            "Traffic_Lag_1",
            "Traffic_Lag_12",
            "Rolling_Traffic_3"
        ],

        "Store Metrics": [
            "Store_Count",
            "Total_Selling_SqFt",
            "Avg_Selling_SqFt",
            "Spend_per_Store",
            "Spend_per_1000SqFt",
            "Traffic_per_Store",
            "Traffic_per_1000SqFt"
        ],

        "Calendar": [
            "Year",
            "Quarter",
            "Month_Number",
            "Holiday_Season"
        ],

        "Seasonality": [
            "Month_Sin",
            "Month_Cos",
            "Quarter_Sin",
            "Quarter_Cos",
            "Linear_Trend",
            "Trend_Squared"
        ],

        "Fourier": [
            "Fourier_Sin_1",
            "Fourier_Cos_1",
            "Fourier_Sin_2",
            "Fourier_Cos_2"
        ]
    }

    for group, cols in feature_groups.items():

        for col in cols:

            if col not in df.columns:
                continue

            registry.append({

                "Feature": col,

                "Group": group,

                "Type": str(df[col].dtype),

                "Missing": int(df[col].isna().sum()),

                "Missing_Pct":
                    round(df[col].isna().mean()*100,2)

            })

    registry = pd.DataFrame(registry)

    registry.to_csv(
        TABLE_DIR /
        "seasonality" /
        "feature_registry.csv",
        index=False
    )

    logger.info(
        "Saved feature_registry.csv"
    )

# ============================================================
# Curated Modeling Dataset
# ============================================================

MODEL_COLUMNS = [

    # identifiers
    "DMA",
    "Month",

    # target
    "Traffic",

    # treatment
    "Spend",

    # marketing
    "Log_Spend",
    "Spend_Index",
    "High_Spend",

    "Spend_Lag_1",
    "Spend_Lag_2",
    "Spend_Lag_3",
    "Spend_Lag_6",

    "Rolling_Spend_3",
    "Rolling_Spend_6",
    "Rolling_Spend_12",

    # traffic history
    "Traffic_Lag_1",
    "Traffic_Lag_12",
    "Rolling_Traffic_3",

    # stores
    "Store_Count",
    "Total_Selling_SqFt",
    "Avg_Selling_SqFt",

    "Spend_per_Store",
    "Spend_per_1000SqFt",

    # calendar
    "Year",
    "Quarter",
    "Month_Number",
    "Holiday_Season",

    # seasonality
    "Month_Sin",
    "Month_Cos",

    "Quarter_Sin",
    "Quarter_Cos",

    "Linear_Trend",
    "Trend_Squared",

    # Fourier
    "Fourier_Sin_1",
    "Fourier_Cos_1",
    "Fourier_Sin_2",
    "Fourier_Cos_2"
]


# ============================================================
# Build Model Panel
# ============================================================

def build_model_panel(panel):

    logger.info("Building final model panel...")

    cols = [
        c
        for c in MODEL_COLUMNS
        if c in panel.columns
    ]

    model = panel[cols].copy()

    # --------------------------------------------------------
    # Hard structural validation
    # --------------------------------------------------------

    logger.info("Validating final model panel structure...")

    # 1. DMA cannot be missing
    missing_dma = model["DMA"].isna().sum()

    if missing_dma > 0:
        bad_rows = model.loc[
            model["DMA"].isna(),
            ["DMA", "Month"]
        ]

        logger.error(
            f"Found {missing_dma} rows with missing DMA."
        )
        logger.error(
            f"\n{bad_rows.to_string(index=False)}"
        )

        raise ValueError(
            f"Model panel contains {missing_dma} rows with missing DMA."
        )

    # 2. Month cannot be missing
    missing_month = model["Month"].isna().sum()

    if missing_month > 0:
        raise ValueError(
            f"Model panel contains {missing_month} rows with missing Month."
        )

    # 3. DMA-Month combinations must be unique
    duplicate_rows = model.duplicated(
        subset=["DMA", "Month"]
    ).sum()

    if duplicate_rows > 0:
        duplicates = model.loc[
            model.duplicated(
                subset=["DMA", "Month"],
                keep=False
            ),
            ["DMA", "Month"]
        ].sort_values(["DMA", "Month"])

        logger.error(
            f"Found {duplicate_rows} duplicate DMA-Month rows."
        )
        logger.error(
            f"\n{duplicates.to_string(index=False)}"
        )

        raise ValueError(
            f"Model panel contains {duplicate_rows} duplicate DMA-Month rows."
        )

    # 4. Validate expected panel dimensions
    n_dmas = model["DMA"].nunique()
    n_months = model["Month"].nunique()
    expected_rows = n_dmas * n_months
    actual_rows = len(model)

    if actual_rows != expected_rows:

        logger.error(
            "Panel completeness validation failed."
        )
        logger.error(
            f"Expected rows : {expected_rows:,}"
        )
        logger.error(
            f"Actual rows   : {actual_rows:,}"
        )

        raise ValueError(
            f"Invalid panel structure: expected "
            f"{expected_rows:,} DMA-Month rows but found "
            f"{actual_rows:,}."
        )

    # --------------------------------------------------------
    # Final sort
    # --------------------------------------------------------

    model = model.sort_values(
        ["DMA", "Month"]
    ).reset_index(drop=True)

    logger.info(
        f"Rows    : {len(model):,}"
    )

    logger.info(
        f"Columns : {len(model.columns)}"
    )

    logger.info(
        f"DMAs    : {n_dmas:,}"
    )

    logger.info(
        f"Months  : {n_months:,}"
    )

    logger.info(
        "Final model panel validation passed."
    )

    return model

# ============================================================
# Save Model Dataset
# ============================================================

def save_model_panel(model):

    logger.info(
        "Saving model panel..."
    )

    OUTPUT = (
        PROCESSED_DATA_DIR /
        "model_panel.parquet"
    )

    model.to_parquet(
        OUTPUT,
        index=False
    )

    logger.info(
        f"Saved {OUTPUT}"
    )

#=============================================================
# Model Panel Summary
#=============================================================

def model_panel_summary(model):
    """
    Final QA report for the modeling dataset.
    """

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL MODEL PANEL SUMMARY")
    logger.info("=" * 80)

    # ---------------------------------------------------------
    # Dataset Overview
    # ---------------------------------------------------------

    logger.info("")
    logger.info("DATASET OVERVIEW")
    logger.info("-" * 80)

    logger.info(f"Rows                 : {len(model):,}")
    logger.info(f"Columns              : {len(model.columns)}")
    logger.info(f"Unique DMAs          : {model['DMA'].nunique():,}")
    logger.info(f"Months               : {model['Month'].nunique():,}")

    logger.info(
        f"Date Range           : "
        f"{model['Month'].min().date()} "
        f"to "
        f"{model['Month'].max().date()}"
    )

    logger.info(
        f"Memory Usage         : "
        f"{model.memory_usage(deep=True).sum()/1024**2:.2f} MB"
    )

    # ---------------------------------------------------------
    # Duplicate Check
    # ---------------------------------------------------------

    logger.info("")
    logger.info("DATA INTEGRITY")
    logger.info("-" * 80)

    duplicates = model.duplicated(
        subset=["DMA", "Month"]
    ).sum()

    logger.info(
        f"Duplicate DMA-Month  : {duplicates}"
    )

    # ---------------------------------------------------------
    # Missing Values
    # ---------------------------------------------------------

    logger.info("")
    logger.info("MISSING VALUES")
    logger.info("-" * 80)

    missing = (
        model
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing = missing[missing > 0]

    if len(missing) == 0:

        logger.info("No missing values found.")

    else:

        for col, value in missing.items():

            pct = value / len(model)

            logger.info(
                f"{col:<30}"
                f"{value:>6} "
                f"({pct:.1%})"
            )

    # ---------------------------------------------------------
    # Constant Features
    # ---------------------------------------------------------

    logger.info("")
    logger.info("CONSTANT FEATURES")
    logger.info("-" * 80)

    constant = []

    for col in model.columns:

        if model[col].nunique(dropna=False) <= 1:
            constant.append(col)

    if constant:

        for c in constant:
            logger.info(c)

    else:

        logger.info("None")

    # ---------------------------------------------------------
    # Numeric Summary
    # ---------------------------------------------------------

    logger.info("")
    logger.info("NUMERIC FEATURE SUMMARY")
    logger.info("-" * 80)

    numeric = model.select_dtypes(include=np.number)

    summary = pd.DataFrame({

        "Missing":
            numeric.isna().sum(),

        "Mean":
            numeric.mean(),

        "Std":
            numeric.std(),

        "Min":
            numeric.min(),

        "Max":
            numeric.max()

    })

    logger.info("\n%s", summary.round(2))

    # ---------------------------------------------------------
    # Correlation Diagnostics
    # ---------------------------------------------------------

    logger.info("")
    logger.info("HIGH CORRELATIONS (>0.95)")
    logger.info("-" * 80)

    corr = numeric.corr().abs()

    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)
    )

    found = False

    for col in upper.columns:

        high = upper[col][upper[col] > 0.95]

        for idx, value in high.items():

            logger.info(
                f"{idx:<30}"
                f"{col:<30}"
                f"{value:.3f}"
            )

            found = True

    if not found:

        logger.info("None")

    # ---------------------------------------------------------
    # Target Summary
    # ---------------------------------------------------------

    logger.info("")
    logger.info("TARGET VARIABLE")
    logger.info("-" * 80)

    if "Traffic" in model.columns:

        logger.info(
            f"Mean Traffic         : "
            f"{model['Traffic'].mean():,.0f}"
        )

        logger.info(
            f"Median Traffic       : "
            f"{model['Traffic'].median():,.0f}"
        )

        logger.info(
            f"Missing Traffic      : "
            f"{model['Traffic'].isna().sum()}"
        )

    # ---------------------------------------------------------
    # Treatment Summary
    # ---------------------------------------------------------

    logger.info("")
    logger.info("SPEND SUMMARY")
    logger.info("-" * 80)

    logger.info(
        f"Total Spend          : "
        f"${model['Spend'].sum():,.0f}"
    )

    logger.info(
        f"Average Spend        : "
        f"${model['Spend'].mean():,.0f}"
    )

    logger.info(
        f"Median Spend         : "
        f"${model['Spend'].median():,.0f}"
    )

    logger.info(
        f"Zero Spend Months    : "
        f"{(model['Spend']==0).sum()}"
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("MODEL PANEL READY")
    logger.info("=" * 80)

#=========================
# Main
# ========================
def main():

    logger.info("=" * 70)
    logger.info("TIME SERIES PREPARATION")
    logger.info("=" * 70)

    panel = load_panel()

    validate_panel(panel)

    export_validation_summary(panel)

    panel = create_time_features(panel)

    summarize_time_features(panel)

    panel = create_cyclical_features(panel)

    panel = create_fourier_features(panel)

    panel = create_retail_seasonality(panel)

    summarize_seasonality(panel)

    build_feature_registry(panel)

    model_panel = build_model_panel(panel)

    save_model_panel(model_panel)

    model_panel_summary(model_panel)

    logger.info("=" * 70)
    logger.info("Validation complete.")
    logger.info("=" * 70)


#     traffic_coverage = (
#         model_panel
#         .assign(Traffic_Missing=model_panel["Traffic"].isna())
#         .groupby("DMA")
#         .agg(
#             Traffic_Obs=("Traffic", "count"),
#             Traffic_Missing=("Traffic_Missing", "sum"),
#             Total_Obs=("Traffic", "size"),
#         )
#     )

#     traffic_coverage["Traffic_Coverage"] = (
#         traffic_coverage["Traffic_Obs"]
#         / traffic_coverage["Total_Obs"]
#     )

#     traffic_coverage = traffic_coverage.sort_values("Traffic_Coverage")

#     missing_by_month = (
#     model_panel
#     .assign(Traffic_Missing=model_panel["Traffic"].isna())
#     .groupby("Month")
#     .agg(
#         Traffic_Obs=("Traffic", "count"),
#         Traffic_Missing=("Traffic_Missing", "sum"),
#         Total_Obs=("Traffic", "size"),
#     )
# )

#     missing_by_month["Traffic_Coverage"] = (
#         missing_by_month["Traffic_Obs"]
#         / missing_by_month["Total_Obs"]
#     )
#     logger.info("Missing Traffic by Month:")
#     logger.info("\n%s", missing_by_month.to_string())

#     logger.info("Missing Traffic by DMA:")
#     logger.info("\n%s", traffic_coverage.to_string())

if __name__ == "__main__":
    main()   