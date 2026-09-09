"""
04_model_readiness_validation.py

Purpose:
    Final preparation of modeling dataset.

Tasks:
    - Forward-fill slow-moving CX attributes
    - Remove audit / merge columns
    - Remove zero-variance fields
    - Validate model readiness
    - Save model_panel.parquet

Inputs:
    panel_data.parquet

Outputs:
    model_panel.parquet
"""

import logging

import pandas as pd

from config import (
    PROCESSED_DATA_DIR,
    LOG_LEVEL,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

def load_panel():

    logger.info("Loading panel data...")

    file_path = (
        PROCESSED_DATA_DIR /
        "panel_data.parquet"
    )

    df = pd.read_parquet(file_path)

    logger.info(
        f"Loaded panel: "
        f"{len(df):,} rows x "
        f"{len(df.columns)} columns"
    )

    return df


# ---------------------------------------------------------------------
# Forward Fill CX Features
# ---------------------------------------------------------------------

def forward_fill_cx(df):

    logger.info(
        "Forward filling CX features..."
    )

    cx_cols = [

        "Avg_Comp_Sales",
        "Avg_Facility_Grade",
        "Avg_Rating_Grade",
        "Avg_Competitors",
        "CX_Store_Count",

        "Pct_A_Facility",
        "Pct_B_Facility",
        "Pct_C_Facility",

        "Pct_Urban",
        "Pct_Suburban",

        "Pct_Marshalls",
        "Pct_Ross",
        "Pct_TJMaxx",

        "CX_Composite",
        "Comp_Sales_Index",
        "Competitor_Density",
        "Urban_Suburban_Ratio",
        "CX_Coverage"

    ]

    before_missing = (
        df[cx_cols]
        .isna()
        .sum()
        .sum()
    )

    df = df.sort_values(
        ["DMA", "Month"]
    )

    df[cx_cols] = (
        df
        .groupby("DMA")[cx_cols]
        .ffill()
    )

    after_missing = (
        df[cx_cols]
        .isna()
        .sum()
        .sum()
    )

    logger.info(
        f"CX nulls before: {before_missing:,}"
    )

    logger.info(
        f"CX nulls after : {after_missing:,}"
    )

    return df


# ---------------------------------------------------------------------
# Drop Modeling-Unfriendly Columns
# ---------------------------------------------------------------------

def drop_unused_columns(df):

    logger.info(
        "Dropping unused columns..."
    )

    cols_to_drop = [

        "DMA_Original_x",
        "DMA_Original_y",
        "_merge",

        "Month_Name",

        "Low_Grade_DMA"

    ]

    existing_cols = [
        col
        for col in cols_to_drop
        if col in df.columns
    ]

    df = df.drop(
        columns=existing_cols
    )

    logger.info(
        f"Dropped {len(existing_cols)} columns"
    )

    logger.info(
        f"Columns removed: "
        f"{existing_cols}"
    )

    return df


# ---------------------------------------------------------------------
# Leakage Audit
# ---------------------------------------------------------------------

def leakage_audit(df):

    logger.info("")
    logger.info("=" * 70)
    logger.info("POTENTIAL LEAKAGE AUDIT")
    logger.info("=" * 70)

    leakage_candidates = [

        # Current-period transactional metrics
        "Transactions",
        "Transaction_Rate",
        "Transactions_per_Store",

        # Current-period traffic-derived metrics
        "Traffic_per_Store",
        "Traffic_per_1000SqFt",
        "Traffic_per_Capita",

        # Ratios involving traffic
        "Traffic_per_Transaction",

        # Target-adjacent variables
        "Traffic"

    ]

    found_cols = [
        col
        for col in leakage_candidates
        if col in df.columns
    ]

    logger.info(
        f"Potential leakage fields found: "
        f"{len(found_cols)}"
    )

    for col in found_cols:

        logger.warning(
            f"Review required: {col}"
        )

    logger.info("")
    logger.info(
        "IMPORTANT:"
    )

    logger.info(
        "Columns listed above are not "
        "necessarily leakage."
    )

    logger.info(
        "Confirm whether each variable "
        "would be known at forecast time."
    )

    logger.info("=" * 70)


# ---------------------------------------------------------------------
# Readiness QA
# ---------------------------------------------------------------------

def readiness_report(df):

    logger.info("")
    logger.info("=" * 70)
    logger.info("MODEL READINESS REPORT")
    logger.info("=" * 70)

    logger.info(
        f"Rows: {len(df):,}"
    )

    logger.info(
        f"Columns: {len(df.columns):,}"
    )

    logger.info(
        f"DMAs: "
        f"{df['DMA'].nunique():,}"
    )

    logger.info(
        f"Months: "
        f"{df['Month'].nunique():,}"
    )

    logger.info(
        f"Date Range: "
        f"{df['Month'].min()} -> "
        f"{df['Month'].max()}"
    )

    logger.info("")
    logger.info("Top Missing Fields")

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    logger.info(
        missing.head(20)
        .to_string()
    )

    duplicates = df.duplicated(
        ["DMA", "Month"]
    ).sum()

    logger.info("")
    logger.info(
        f"Duplicate DMA-Month Rows: "
        f"{duplicates:,}"
    )

    if duplicates > 0:

        raise ValueError(
            "Duplicate DMA-Month keys detected."
        )

    logger.info("")
    logger.info(
        "Readiness validation complete."
    )

    logger.info("=" * 70)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

def save_model_panel(df):

    output_file = (
        PROCESSED_DATA_DIR /
        "model_panel.parquet"
    )

    df.to_parquet(
        output_file,
        index=False
    )

    logger.info(
        f"Saved {output_file.name}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    panel = load_panel()

    panel = forward_fill_cx(
        panel
    )

    panel = drop_unused_columns(
        panel
    )

    leakage_audit(
        panel
    )

    readiness_report(
        panel
    )

    save_model_panel(
        panel
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "MODEL PANEL READY"
    )
    logger.info("=" * 70)


if __name__ == "__main__":

    main()