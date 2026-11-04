"""
06a_merchandise_cross_correlation.py

Merchandise Cross-Correlation Analysis
--------------------------------------

Purpose
-------
Evaluate temporal relationships between merchandise
performance and store traffic.

Identify whether specific merchandise groups are
associated with higher or lower store traffic.

Primary Grain
-------------
DMA x Quarter

Outputs
-------
Figures
Tables
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import pearsonr

from config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    OUTPUT_DIR,
    STYLE
)


# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(__name__)


# ==========================================================
# Output Folders
# ==========================================================

REPORT_DIR = OUTPUT_DIR / "reports"


def create_output_folders():

    folders = [

        FIGURE_DIR /
        "merchandise_cross_correlation",

        FIGURE_DIR /
        "merchandise_cross_correlation" /
        "overall",

        FIGURE_DIR /
        "merchandise_cross_correlation" /
        "lag_analysis",

        FIGURE_DIR /
        "merchandise_cross_correlation" /
        "groups",

        FIGURE_DIR /
        "merchandise_cross_correlation" /
        "dma",

        FIGURE_DIR /
        "merchandise_cross_correlation" /
        "diagnostics",

        TABLE_DIR /
        "merchandise_cross_correlation",

        TABLE_DIR /
        "merchandise_cross_correlation" /
        "groups",

        REPORT_DIR
    ]

    for folder in folders:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )



# ==========================================================
# Load Data
# ==========================================================

def load_data():

    logger.info(
        "Loading merchandise analysis data..."
    )

    panel = pd.read_parquet(
        PROCESSED_DATA_DIR /
        "panel_data.parquet"
    )

    merchandise = pd.read_parquet(
        INTERIM_DATA_DIR /
        "merchandise_dma_quarter.parquet"
    )

    merchandise_group = pd.read_parquet(
        INTERIM_DATA_DIR /
        "merchandise_dma_group_quarter.parquet"
    )

    panel["Month"] = pd.to_datetime(
        panel["Month"]
    )

    merchandise["Quarter_Period"] = pd.to_datetime(
        merchandise["Quarter_Period"]
    )

    merchandise_group[
        "Quarter_Period"
    ] = pd.to_datetime(
        merchandise_group[
            "Quarter_Period"
        ]
    )

    logger.info(
        f"Panel rows             : "
        f"{len(panel):,}"
    )

    logger.info(
        f"Merchandise rows       : "
        f"{len(merchandise):,}"
    )

    logger.info(
        f"Merch group rows       : "
        f"{len(merchandise_group):,}"
    )

    return (
        panel,
        merchandise,
        merchandise_group
    )

# ==========================================================
# Quarterly Traffic Panel
# ==========================================================

def create_quarterly_traffic_panel(panel):

    logger.info(
        "Creating DMA-quarter traffic panel..."
    )

    df = panel.copy()

    df["Quarter_Period"] = (
        df["Month"]
        .dt.to_period("Q")
        .dt.to_timestamp()
    )

    quarterly = (
        df
        .groupby(
            [
                "DMA",
                "Quarter_Period"
            ],
            as_index=False
        )
        .agg(
            Traffic=(
                "Traffic",
                "sum"
            ),
            Transactions=(
                "Transactions",
                "sum"
            ),
            Months_Observed=(
                "Month",
                "nunique"
            ),
            Store_Count=(
                "Store_Count",
                "mean"
            )
        )
    )

    quarterly["Traffic_per_Store"] = np.where(
        quarterly["Store_Count"] > 0,
        quarterly["Traffic"] / quarterly["Store_Count"],
        np.nan
    )

    # ------------------------------------------------------
    # Require complete traffic quarters
    # ------------------------------------------------------

    partial = quarterly[
        quarterly["Months_Observed"] < 3
    ]

    if not partial.empty:

        logger.warning(
            f"Excluding {len(partial):,} "
            "DMA-quarter rows with incomplete "
            "monthly traffic coverage."
        )

    quarterly = quarterly[
        quarterly["Months_Observed"] == 3
    ].copy()

    logger.info(
        f"Quarterly traffic rows: "
        f"{len(quarterly):,}"
    )

    return quarterly


def create_merchandise_analysis_panel(
    quarterly_traffic,
    merchandise
):

    logger.info(
        "Creating merchandise analysis panel..."
    )

    panel = quarterly_traffic.merge(
        merchandise,
        on=[
            "DMA",
            "Quarter_Period"
        ],
        how="inner",
        validate="one_to_one"
    )

    logger.info(
        f"Analysis rows : {len(panel):,}"
    )

    logger.info(
        f"DMAs          : "
        f"{panel['DMA'].nunique():,}"
    )

    logger.info(
        f"Quarters      : "
        f"{panel['Quarter_Period'].nunique():,}"
    )

    logger.info(
        f"Quarter range : "
        f"{panel['Quarter_Period'].min()} -> "
        f"{panel['Quarter_Period'].max()}"
    )

    return panel


# ==========================================================
# Merchandise Group Analysis Panel
# ==========================================================

def create_group_analysis_panel(
    quarterly_traffic,
    merchandise_group
):
    """
    Merge DMA-quarter traffic with merchandise-group
    performance.

    Resulting grain:
        DMA x Merchandise Group x Quarter
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("CREATING MERCHANDISE GROUP ANALYSIS PANEL")
    logger.info("=" * 70)

    group_panel = quarterly_traffic.merge(
        merchandise_group,
        on=["DMA", "Quarter_Period"],
        how="inner",
        validate="one_to_many"
    )

    logger.info(
        f"Group analysis rows : {len(group_panel):,}"
    )

    logger.info(
        f"DMAs                : "
        f"{group_panel['DMA'].nunique():,}"
    )

    logger.info(
        f"Quarters            : "
        f"{group_panel['Quarter_Period'].nunique():,}"
    )

    logger.info(
        f"Merchandise groups  : "
        f"{group_panel['Merch_Group'].nunique():,}"
    )

    return group_panel


# ==========================================================
# Overall Merchandise Correlations
# ==========================================================

def overall_merchandise_correlations(
    analysis_panel
):
    """
    Measure pooled relationships between merchandise
    performance and quarterly traffic.

    These are descriptive correlations and do not
    represent causal effects.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("OVERALL MERCHANDISE CORRELATIONS")
    logger.info("=" * 70)

    variables = [
        "Merch_Sales_Dollars",
        "Merch_Sales_Units",
        "Merch_AUR",
        "Merch_Sales_per_Store",
        "Merch_Units_per_Store",
        "Merch_Sales_Lag_1Q",
        "Merch_Units_Lag_1Q",
        "Merch_AUR_Lag_1Q",
        "Merch_Sales_per_Store_Lag_1Q",
        "Merch_Units_per_Store_Lag_1Q"
    ]

    rows = []

    for variable in variables:

        if variable not in analysis_panel.columns:
            logger.warning(
                f"{variable} not found. Skipping."
            )
            continue

        subset = (
            analysis_panel[
                [
                    variable,
                    "Traffic"
                ]
            ]
            .dropna()
        )

        if len(subset) < 3:
            continue

        corr, p_value = pearsonr(
            subset[variable],
            subset["Traffic"]
        )

        rows.append({
            "Variable": variable,
            "Correlation": corr,
            "P_Value": p_value,
            "N": len(subset)
        })

        logger.info(
            f"{variable:<35} "
            f"r={corr:>7.3f} | "
            f"p={p_value:>8.4f} | "
            f"N={len(subset):>3}"
        )

    results = (
        pd.DataFrame(rows)
        .sort_values(
            "Correlation",
            ascending=False
        )
    )

    results.to_csv(
        TABLE_DIR /
        "merchandise_cross_correlation" /
        "overall_merchandise_correlations.csv",
        index=False
    )

    return results


# ==========================================================
# Within-DMA Merchandise Correlations
# ==========================================================

def within_dma_correlations(
    analysis_panel
):

    logger.info("")
    logger.info("=" * 70)
    logger.info("WITHIN-DMA MERCHANDISE CORRELATIONS")
    logger.info("=" * 70)

    df = analysis_panel.copy()

    variables = [
        "Merch_Sales_Dollars",
        "Merch_Sales_Units",
        "Merch_AUR",
        "Merch_Sales_per_Store",
        "Merch_Units_per_Store",
        "Merch_Sales_Lag_1Q",
        "Merch_Units_Lag_1Q",
        "Merch_AUR_Lag_1Q",
        "Merch_Sales_per_Store_Lag_1Q",
        "Merch_Units_per_Store_Lag_1Q"
    ]

    # ------------------------------------------------------
    # Demean traffic within DMA
    # ------------------------------------------------------

    df["Traffic_Within_DMA"] = (
        df["Traffic"]
        - df.groupby("DMA")["Traffic"]
        .transform("mean")
    )

    rows = []

    for variable in variables:

        if variable not in df.columns:
            continue

        within_col = f"{variable}_Within_DMA"

        df[within_col] = (
            df[variable]
            - df.groupby("DMA")[variable]
            .transform("mean")
        )

        subset = (
            df[
                [
                    within_col,
                    "Traffic_Within_DMA"
                ]
            ]
            .dropna()
        )

        if len(subset) < 3:
            continue

        corr, p_value = pearsonr(
            subset[within_col],
            subset["Traffic_Within_DMA"]
        )

        rows.append({
            "Variable": variable,
            "Within_DMA_Correlation": corr,
            "P_Value": p_value,
            "N": len(subset)
        })

        logger.info(
            f"{variable:<35} "
            f"r={corr:>7.3f} | "
            f"p={p_value:>8.4f} | "
            f"N={len(subset):>3}"
        )

    results = pd.DataFrame(rows)

    results.to_csv(
        TABLE_DIR /
        "merchandise_cross_correlation" /
        "within_dma_merchandise_correlations.csv",
        index=False
    )

    return results


# ==========================================================
# DMA + Quarter Adjusted Correlations
# ==========================================================

def dma_quarter_adjusted_correlations(
    analysis_panel
):
    """
    Measure merchandise-traffic relationships after
    removing both:

        1. Persistent DMA-level differences
        2. Common quarter-level differences

    This is equivalent to residualizing each variable
    against DMA and Quarter fixed effects before
    computing the correlation.

    These are descriptive screening relationships,
    not causal estimates.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("DMA + QUARTER ADJUSTED CORRELATIONS")
    logger.info("=" * 70)

    df = analysis_panel.copy()

    variables = [
        "Merch_Sales_Dollars",
        "Merch_Sales_Units",
        "Merch_AUR",
        "Merch_Sales_per_Store",
        "Merch_Units_per_Store",
        "Merch_Sales_Lag_1Q",
        "Merch_Units_Lag_1Q",
        "Merch_AUR_Lag_1Q",
        "Merch_Sales_per_Store_Lag_1Q",
        "Merch_Units_per_Store_Lag_1Q"
    ]

    # ------------------------------------------------------
    # Two-Way Demeaning Helper
    # ------------------------------------------------------

    def two_way_demean(data, column):
        """
        Remove DMA and Quarter fixed effects using
        two-way demeaning.

        residual =
            value
            - DMA mean
            - Quarter mean
            + overall mean
        """

        return (
            data[column]
            - data.groupby("DMA")[column]
            .transform("mean")
            - data.groupby("Quarter_Period")[column]
            .transform("mean")
            + data[column].mean()
        )

    # ------------------------------------------------------
    # Adjust Traffic
    # ------------------------------------------------------

    df["Traffic_Adjusted"] = (
        two_way_demean(
            df,
            "Traffic"
        )
    )

    rows = []

    # ------------------------------------------------------
    # Adjust Merchandise Variables
    # ------------------------------------------------------

    for variable in variables:

        if variable not in df.columns:

            logger.warning(
                f"{variable} not found. Skipping."
            )

            continue

        adjusted_col = (
            f"{variable}_Adjusted"
        )

        df[adjusted_col] = (
            two_way_demean(
                df,
                variable
            )
        )

        subset = (
            df[
                [
                    adjusted_col,
                    "Traffic_Adjusted"
                ]
            ]
            .dropna()
        )

        if len(subset) < 3:
            continue

        corr, p_value = pearsonr(
            subset[adjusted_col],
            subset["Traffic_Adjusted"]
        )

        rows.append({
            "Variable": variable,
            "Adjusted_Correlation": corr,
            "P_Value": p_value,
            "N": len(subset)
        })

        logger.info(
            f"{variable:<35} "
            f"r={corr:>7.3f} | "
            f"p={p_value:>8.4f} | "
            f"N={len(subset):>3}"
        )

    results = pd.DataFrame(rows)

    results.to_csv(
        TABLE_DIR /
        "merchandise_cross_correlation" /
        "dma_quarter_adjusted_correlations.csv",
        index=False
    )

    return results


# ==========================================================
# Merchandise Group Correlations
# ==========================================================

def merchandise_group_correlations(
    group_panel
):
    """
    Evaluate merchandise-group relationships with traffic.

    For each merchandise group, calculate:

        1. Raw sales correlation
        2. Sales-per-store correlation
        3. Dollar-share correlation
        4. DMA-adjusted correlation
        5. DMA + Quarter adjusted correlation

    Correlations are descriptive and should not be
    interpreted as causal effects.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("MERCHANDISE GROUP CORRELATIONS")
    logger.info("=" * 70)

    rows = []

    groups = sorted(
        group_panel["Merch_Group"]
        .dropna()
        .unique()
    )

    for group in groups:

        df = (
            group_panel[
                group_panel["Merch_Group"] == group
            ]
            .copy()
        )

        # --------------------------------------------------
        # Helper
        # --------------------------------------------------

        def calculate_correlation(x, y):

            subset = (
                pd.DataFrame({
                    "x": x,
                    "y": y
                })
                .dropna()
            )

            if len(subset) < 3:
                return np.nan, np.nan, len(subset)

            corr, p_value = pearsonr(
                subset["x"],
                subset["y"]
            )

            return corr, p_value, len(subset)

        # --------------------------------------------------
        # Raw Sales
        # --------------------------------------------------

        raw_corr, raw_p, raw_n = (
            calculate_correlation(
                df["Group_Sales_Dollars"],
                df["Traffic"]
            )
        )

        # --------------------------------------------------
        # Sales Per Store
        # --------------------------------------------------

        per_store_corr, per_store_p, _ = (
            calculate_correlation(
                df["Group_Sales_per_Store"],
                df["Traffic"]
            )
        )

        # --------------------------------------------------
        # Merchandise Mix / Dollar Share
        # --------------------------------------------------

        share_corr, share_p, _ = (
            calculate_correlation(
                df["Group_Dollar_Share"],
                df["Traffic"]
            )
        )

        # --------------------------------------------------
        # Within-DMA Adjustment
        # --------------------------------------------------

        df["Traffic_Within_DMA"] = (
            df["Traffic"]
            - df.groupby("DMA")["Traffic"]
            .transform("mean")
        )

        df["Group_Sales_Within_DMA"] = (
            df["Group_Sales_Dollars"]
            - df.groupby("DMA")[
                "Group_Sales_Dollars"
            ]
            .transform("mean")
        )

        within_corr, within_p, _ = (
            calculate_correlation(
                df["Group_Sales_Within_DMA"],
                df["Traffic_Within_DMA"]
            )
        )

        # --------------------------------------------------
        # DMA + Quarter Adjustment
        # --------------------------------------------------

        df["Traffic_Adjusted"] = (
            df["Traffic"]
            - df.groupby("DMA")["Traffic"]
            .transform("mean")
            - df.groupby("Quarter_Period")["Traffic"]
            .transform("mean")
            + df["Traffic"].mean()
        )

        df["Group_Sales_Adjusted"] = (
            df["Group_Sales_Dollars"]
            - df.groupby("DMA")[
                "Group_Sales_Dollars"
            ]
            .transform("mean")
            - df.groupby("Quarter_Period")[
                "Group_Sales_Dollars"
            ]
            .transform("mean")
            + df["Group_Sales_Dollars"].mean()
        )

        adjusted_corr, adjusted_p, _ = (
            calculate_correlation(
                df["Group_Sales_Adjusted"],
                df["Traffic_Adjusted"]
            )
        )

        df["Group_Share_Adjusted"] = (
            df["Group_Dollar_Share"]
            - df.groupby("DMA")[
                "Group_Dollar_Share"
            ].transform("mean")
            - df.groupby("Quarter_Period")[
                "Group_Dollar_Share"
            ].transform("mean")
            + df["Group_Dollar_Share"].mean()
        )

        adjusted_share_corr, adjusted_share_p, _ = (
            calculate_correlation(
                df["Group_Share_Adjusted"],
                df["Traffic_Adjusted"]
            )
        )


         # --------------------------------------------------
        # Adjust Traffic Per Store
        # --------------------------------------------------

        df["Traffic_per_Store_Adjusted"] = (
            df["Traffic_per_Store"]
            - df.groupby("DMA")[
                "Traffic_per_Store"
            ].transform("mean")
            - df.groupby("Quarter_Period")[
                "Traffic_per_Store"
            ].transform("mean")
            + df["Traffic_per_Store"].mean()
        )

        # --------------------------------------------------
        # DMA + Quarter Adjusted Sales Per Store
        # --------------------------------------------------

        df["Group_Sales_per_Store_Adjusted"] = (
            df["Group_Sales_per_Store"]
            - df.groupby("DMA")[
                "Group_Sales_per_Store"
            ].transform("mean")
            - df.groupby("Quarter_Period")[
                "Group_Sales_per_Store"
            ].transform("mean")
            + df["Group_Sales_per_Store"].mean()
        )

        adjusted_per_store_corr, adjusted_per_store_p, _ = (
            calculate_correlation(
                df["Group_Sales_per_Store_Adjusted"],
                df["Traffic_Adjusted"]
            )
        )

        adjusted_per_store_both_corr, adjusted_per_store_both_p, _ = (
            calculate_correlation(
                df["Group_Sales_per_Store_Adjusted"],
                df["Traffic_per_Store_Adjusted"]
            )
        )

       

        # --------------------------------------------------
        # Store Results
        # --------------------------------------------------

        rows.append({
            "Merch_Group": group,
            "Raw_Correlation": raw_corr,
            "Raw_P_Value": raw_p,
            "Per_Store_Correlation": per_store_corr,
            "Per_Store_P_Value": per_store_p,
            "Share_Correlation": share_corr,
            "Share_P_Value": share_p,
            "Within_DMA_Correlation": within_corr,
            "Within_DMA_P_Value": within_p,
            "Adjusted_Correlation": adjusted_corr,
            "Adjusted_P_Value": adjusted_p,
            "Adjusted_Share_Correlation": adjusted_share_corr,
            "Adjusted_Share_P_Value": adjusted_share_p,
            "N": raw_n,
            "DMAs": df["DMA"].nunique(),
            "Quarters": df["Quarter_Period"].nunique(),
            "Adjusted_Per_Store_Correlation": adjusted_per_store_corr,
            "Adjusted_Per_Store_P_Value": adjusted_per_store_p,
            "Adjusted_Per_Store_Both_Correlation": adjusted_per_store_both_corr,
            "Adjusted_Per_Store_Both_P_Value": adjusted_per_store_both_p,
        })

    results = pd.DataFrame(rows)

    results = results.sort_values(
        "Adjusted_Correlation",
        ascending=False
    ).reset_index(drop=True)

    results.to_csv(
        TABLE_DIR /
        "merchandise_cross_correlation" /
        "groups" /
        "merchandise_group_correlations.csv",
        index=False
    )

    return results


# ==========================================================
# Lagged Merchandise Group Correlations
# ==========================================================

def lagged_merchandise_group_correlations(
    group_panel
):
    """
    Evaluate whether prior-quarter merchandise-group
    performance is associated with current-quarter traffic.

    Primary relationship:

        Group Sales per Store (t-1)
                    vs.
        Traffic per Store (t)

    Both variables are adjusted for:
        - persistent DMA differences
        - common quarter effects

    These relationships are descriptive and should not
    be interpreted as causal effects.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("LAGGED MERCHANDISE GROUP CORRELATIONS")
    logger.info("=" * 70)

    rows = []

    groups = sorted(
        group_panel["Merch_Group"]
        .dropna()
        .unique()
    )

    for group in groups:

        df = (
            group_panel[
                group_panel["Merch_Group"] == group
            ]
            .copy()
            .sort_values(
                ["DMA", "Quarter_Period"]
            )
        )

        # --------------------------------------------------
        # Helper
        # --------------------------------------------------

        def calculate_correlation(x, y):

            subset = (
                pd.DataFrame({
                    "x": x,
                    "y": y
                })
                .dropna()
            )

            if len(subset) < 3:
                return np.nan, np.nan, len(subset)

            corr, p_value = pearsonr(
                subset["x"],
                subset["y"]
            )

            return corr, p_value, len(subset)

        # --------------------------------------------------
        # Validate Lag Column
        # --------------------------------------------------

        lag_col = "Group_Sales_per_Store_Lag_1Q"

        if lag_col not in df.columns:

            logger.warning(
                f"{lag_col} not found for {group}. "
                "Skipping."
            )

            continue

        # --------------------------------------------------
        # Raw Lagged Relationship
        #
        # Prior-quarter group sales/store
        # vs current traffic/store
        # --------------------------------------------------

        raw_lag_corr, raw_lag_p, raw_lag_n = (
            calculate_correlation(
                df[lag_col],
                df["Traffic_per_Store"]
            )
        )

        # --------------------------------------------------
        # Within-DMA Adjustment
        # --------------------------------------------------

        df["Lag_Sales_per_Store_Within_DMA"] = (
            df[lag_col]
            - df.groupby("DMA")[lag_col]
            .transform("mean")
        )

        df["Traffic_per_Store_Within_DMA"] = (
            df["Traffic_per_Store"]
            - df.groupby("DMA")[
                "Traffic_per_Store"
            ]
            .transform("mean")
        )

        within_lag_corr, within_lag_p, _ = (
            calculate_correlation(
                df["Lag_Sales_per_Store_Within_DMA"],
                df["Traffic_per_Store_Within_DMA"]
            )
        )

        # --------------------------------------------------
        # DMA + Quarter Adjusted Lagged Merchandise
        # --------------------------------------------------

        df["Lag_Sales_per_Store_Adjusted"] = (
            df[lag_col]
            - df.groupby("DMA")[lag_col]
            .transform("mean")
            - df.groupby("Quarter_Period")[lag_col]
            .transform("mean")
            + df[lag_col].mean()
        )

        # --------------------------------------------------
        # DMA + Quarter Adjusted Current Traffic
        # --------------------------------------------------

        df["Traffic_per_Store_Adjusted"] = (
            df["Traffic_per_Store"]
            - df.groupby("DMA")[
                "Traffic_per_Store"
            ]
            .transform("mean")
            - df.groupby("Quarter_Period")[
                "Traffic_per_Store"
            ]
            .transform("mean")
            + df["Traffic_per_Store"].mean()
        )

        adjusted_lag_corr, adjusted_lag_p, adjusted_n = (
            calculate_correlation(
                df["Lag_Sales_per_Store_Adjusted"],
                df["Traffic_per_Store_Adjusted"]
            )
        )

        # --------------------------------------------------
        # Store Results
        # --------------------------------------------------

        rows.append({
            "Merch_Group": group,

            "Raw_Lag_Correlation":
                raw_lag_corr,

            "Raw_Lag_P_Value":
                raw_lag_p,

            "Within_DMA_Lag_Correlation":
                within_lag_corr,

            "Within_DMA_Lag_P_Value":
                within_lag_p,

            "Adjusted_Lag_Correlation":
                adjusted_lag_corr,

            "Adjusted_Lag_P_Value":
                adjusted_lag_p,

            "N":
                adjusted_n,

            "DMAs":
                df["DMA"].nunique(),

            "Quarters":
                df["Quarter_Period"].nunique()
        })

    results = pd.DataFrame(rows)

    results = results.sort_values(
        "Adjusted_Lag_Correlation",
        ascending=False
    ).reset_index(drop=True)

    results.to_csv(
        TABLE_DIR /
        "merchandise_cross_correlation" /
        "groups" /
        "lagged_merchandise_group_correlations.csv",
        index=False
    )

    return results


# ==========================================================
# Current vs Lagged Group Comparison
# ==========================================================

def compare_current_vs_lagged_groups(
    current_results,
    lagged_results
):
    """
    Compare contemporaneous and prior-quarter adjusted
    merchandise-group relationships with traffic.

    Current:
        Adjusted group sales/store
        vs adjusted traffic/store

    Lagged:
        Adjusted prior-quarter group sales/store
        vs adjusted current-quarter traffic/store
    """

    logger.info(
        "Creating current vs lagged merchandise "
        "group comparison..."
    )

    comparison = (
        current_results[
            [
                "Merch_Group",
                "Adjusted_Per_Store_Both_Correlation"
            ]
        ]
        .merge(
            lagged_results[
                [
                    "Merch_Group",
                    "Adjusted_Lag_Correlation"
                ]
            ],
            on="Merch_Group",
            how="inner"
        )
    )

    comparison = comparison.rename(
        columns={
            "Adjusted_Per_Store_Both_Correlation":
                "Current_Correlation",
            "Adjusted_Lag_Correlation":
                "Lag_1Q_Correlation"
        }
    )

    comparison["Correlation_Change"] = (
        comparison["Lag_1Q_Correlation"]
        - comparison["Current_Correlation"]
    )

    comparison["Absolute_Change"] = (
        comparison["Correlation_Change"].abs()
    )

    comparison = comparison.sort_values(
        "Current_Correlation",
        ascending=False
    ).reset_index(drop=True)

    output_file = (
        TABLE_DIR
        / "merchandise_cross_correlation"
        / "groups"
        / "current_vs_lagged_group_correlations.csv"
    )

    comparison.to_csv(
        output_file,
        index=False
    )

    logger.info(
        f"Saved current vs lagged comparison: "
        f"{output_file}"
    )

    return comparison


# ==========================================================
# Current vs Lagged Visualization
# ==========================================================

def plot_current_vs_lagged_groups(comparison):
    """
    Plot contemporaneous versus prior-quarter adjusted
    merchandise-group correlations with traffic.

    Excludes nonstandard merchandise categories from the
    primary business-facing visualization.
    """

    logger.info(
        "Creating current vs lagged merchandise "
        "group visualization..."
    )

    # --------------------------------------------------
    # Exclude Nonstandard Categories
    # --------------------------------------------------

    excluded_groups = [
        "Central Balance",
        "Unassigned dB Group",
        "Leased"
    ]

    plot_df = comparison[
        ~comparison["Merch_Group"].isin(
            excluded_groups
        )
    ].copy()

    # Sort for visual readability
    plot_df = plot_df.sort_values(
        "Current_Correlation",
        ascending=True
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    y_positions = np.arange(
        len(plot_df)
    )

    # Connector between lagged and current
    for y, (_, row) in zip(
        y_positions,
        plot_df.iterrows()
    ):
        ax.plot(
            [
                row["Lag_1Q_Correlation"],
                row["Current_Correlation"]
            ],
            [y, y],
            linewidth=2,
            alpha=0.6
        )

    # Lagged points
    ax.scatter(
        plot_df["Lag_1Q_Correlation"],
        y_positions,
        s=80,
        label="Prior Quarter"
    )

    # Current points
    ax.scatter(
        plot_df["Current_Correlation"],
        y_positions,
        s=80,
        label="Current Quarter"
    )

    # Zero reference
    ax.axvline(
        0,
        linewidth=1,
        linestyle="--",
        alpha=0.7
    )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        plot_df["Merch_Group"]
    )

    ax.set_xlabel(
        "Adjusted Correlation with Traffic per Store"
    )

    ax.set_ylabel(
        "Merchandise Group"
    )

    ax.set_title(
        "Current vs Prior-Quarter Merchandise "
        "Relationship with Traffic"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        axis="x",
        alpha=0.25
    )

    plt.tight_layout()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_file = (
        FIGURE_DIR
        / "merchandise_cross_correlation"
        / "groups"
        / "current_vs_lagged_group_correlations.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(
        f"Saved current vs lagged figure: "
        f"{output_file}"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    # ------------------------------------------------------
    # Setup
    # ------------------------------------------------------

    create_output_folders()

    logger.info("")
    logger.info("=" * 70)
    logger.info("MERCHANDISE CROSS-CORRELATION ANALYSIS")
    logger.info("=" * 70)

    # ------------------------------------------------------
    # Load Data
    # ------------------------------------------------------

    (
        panel,
        merchandise,
        merchandise_group
    ) = load_data()

    # ------------------------------------------------------
    # Source Data Summary
    # ------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("SOURCE DATA")
    logger.info("=" * 70)

    logger.info(
        f"Monthly panel rows       : "
        f"{len(panel):,}"
    )

    logger.info(
        f"Monthly panel DMAs       : "
        f"{panel['DMA'].nunique():,}"
    )

    logger.info(
        f"Monthly panel months     : "
        f"{panel['Month'].nunique():,}"
    )

    logger.info(
        f"Monthly panel range      : "
        f"{panel['Month'].min().date()} -> "
        f"{panel['Month'].max().date()}"
    )

    logger.info("")

    logger.info(
        f"Merchandise rows         : "
        f"{len(merchandise):,}"
    )

    logger.info(
        f"Merchandise DMAs         : "
        f"{merchandise['DMA'].nunique():,}"
    )

    logger.info(
        f"Merchandise quarters     : "
        f"{merchandise['Quarter_Period'].nunique():,}"
    )

    logger.info(
        f"Merchandise range        : "
        f"{merchandise['Quarter_Period'].min().date()} -> "
        f"{merchandise['Quarter_Period'].max().date()}"
    )

    logger.info("")

    logger.info(
        f"Merch group rows         : "
        f"{len(merchandise_group):,}"
    )

    logger.info(
        f"Merchandise groups       : "
        f"{merchandise_group['Merch_Group'].nunique():,}"
    )

    # ------------------------------------------------------
    # Create Quarterly Traffic Panel
    # ------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("QUARTERLY TRAFFIC PANEL")
    logger.info("=" * 70)

    quarterly_traffic = (
        create_quarterly_traffic_panel(
            panel
        )
    )

    logger.info(
        f"Complete DMA-quarter rows : "
        f"{len(quarterly_traffic):,}"
    )

    logger.info(
        f"DMAs                      : "
        f"{quarterly_traffic['DMA'].nunique():,}"
    )

    logger.info(
        f"Quarters                  : "
        f"{quarterly_traffic['Quarter_Period'].nunique():,}"
    )

    logger.info(
        f"Quarter range             : "
        f"{quarterly_traffic['Quarter_Period'].min().date()} -> "
        f"{quarterly_traffic['Quarter_Period'].max().date()}"
    )

    # ------------------------------------------------------
    # Merge Merchandise + Traffic
    # ------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("MERCHANDISE ANALYSIS PANEL")
    logger.info("=" * 70)

    analysis_panel = (
        create_merchandise_analysis_panel(
            quarterly_traffic,
            merchandise
        )
    )

    # ------------------------------------------------------
    # Basic Validation
    # ------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS PANEL VALIDATION")
    logger.info("=" * 70)

    duplicate_rows = (
        analysis_panel
        .duplicated(
            subset=[
                "DMA",
                "Quarter_Period"
            ]
        )
        .sum()
    )

    logger.info(
        f"Duplicate DMA-quarter rows : "
        f"{duplicate_rows:,}"
    )

    critical_columns = [
        "Traffic",
        "Transactions",
        "Store_Count",
        "Merch_Sales_Dollars",
        "Merch_Sales_Units",
        "Merch_Stores",
        "Merch_AUR",
        "Merch_Sales_per_Store",
        "Merch_Units_per_Store"
    ]

    logger.info("")
    logger.info("Critical Variable Missingness")
    logger.info("-" * 70)

    for col in critical_columns:

        if col not in analysis_panel.columns:

            logger.warning(
                f"{col:<30} NOT FOUND"
            )

            continue

        missing = (
            analysis_panel[col]
            .isna()
            .sum()
        )

        pct = (
            missing /
            len(analysis_panel)
        )

        logger.info(
            f"{col:<30} "
            f"missing={missing:>5,} "
            f"({pct:>6.2%})"
        )

    # ------------------------------------------------------
    # Coverage by Quarter
    # ------------------------------------------------------

    logger.info("")
    logger.info("Coverage by Quarter")
    logger.info("-" * 70)

    quarter_coverage = (
        analysis_panel
        .groupby(
            "Quarter_Period",
            as_index=False
        )
        .agg(
            DMAs=(
                "DMA",
                "nunique"
            ),
            Observations=(
                "DMA",
                "size"
            ),
            Traffic=(
                "Traffic",
                "sum"
            ),
            Merch_Sales_Dollars=(
                "Merch_Sales_Dollars",
                "sum"
            ),
            Merch_Sales_Units=(
                "Merch_Sales_Units",
                "sum"
            )
        )
        .sort_values(
            "Quarter_Period"
        )
    )

    for _, row in quarter_coverage.iterrows():

        quarter = row["Quarter_Period"]

        logger.info(
            f"{quarter.year}Q{quarter.quarter} | "
            f"DMAs={int(row['DMAs']):>2} | "
            f"Rows={int(row['Observations']):>3}"
        )


    # ------------------------------------------------------
    # Save QA Tables
    # ------------------------------------------------------

    output_dir = (
        TABLE_DIR /
        "merchandise_cross_correlation"
    )

    analysis_panel.to_csv(
        output_dir /
        "merchandise_analysis_panel.csv",
        index=False
    )

    quarter_coverage.to_csv(
        output_dir /
        "quarter_coverage.csv",
        index=False
    )

    logger.info("")
    logger.info(
        "Saved merchandise_analysis_panel.csv"
    )

    logger.info(
        "Saved quarter_coverage.csv"
    )

    # ------------------------------------------------------
    # Merchandise Group Coverage
    # ------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("MERCHANDISE GROUP COVERAGE")
    logger.info("=" * 70)

    panel_dmas = set(
        analysis_panel["DMA"]
        .dropna()
        .unique()
    )

    group_subset = (
        merchandise_group[
            merchandise_group[
                "DMA"
            ].isin(panel_dmas)
        ]
        .copy()
    )

    group_summary = (
        group_subset
        .groupby(
            "Merch_Group",
            as_index=False
        )
        .agg(
            DMAs=(
                "DMA",
                "nunique"
            ),
            Quarters=(
                "Quarter_Period",
                "nunique"
            ),
            Observations=(
                "DMA",
                "size"
            ),
            Sales_Dollars=(
                "Group_Sales_Dollars",
                "sum"
            ),
            Sales_Units=(
                "Group_Sales_Units",
                "sum"
            )
        )
        .sort_values(
            "Sales_Dollars",
            ascending=False
        )
    )

    group_summary.to_csv(
        output_dir /
        "groups" /
        "merchandise_group_coverage.csv",
        index=False
    )

    logger.info(
        f"Merchandise groups : "
        f"{len(group_summary):,}"
    )

    logger.info(
        f"Group observations : "
        f"{len(group_subset):,}"
    )

    logger.info("")
    logger.info(
        group_summary[
            [
                "Merch_Group",
                "DMAs",
                "Quarters",
                "Observations"
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Overall Merchandise Relationships
    # ------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("MERCHANDISE RELATIONSHIP ANALYSIS")
    logger.info("=" * 70)

    overall_results = (
        overall_merchandise_correlations(
            analysis_panel
        )
    )

    # ------------------------------------------------------
    # Within-DMA Merchandise Relationships
    # ------------------------------------------------------

    within_dma_results = (
        within_dma_correlations(
            analysis_panel
        )
    )

    # ------------------------------------------------------
    # DMA + Quarter Adjusted Relationships
    # ------------------------------------------------------

    adjusted_results = (
        dma_quarter_adjusted_correlations(
            analysis_panel
        )
    )


    # ------------------------------------------------------
    # Merchandise Group Analysis Panel
    # ------------------------------------------------------

    group_analysis_panel = (
        create_group_analysis_panel(
            quarterly_traffic,
            merchandise_group
        )
    )


    # ------------------------------------------------------
    # Merchandise Group Correlations
    # ------------------------------------------------------

    group_results = (
        merchandise_group_correlations(
            group_analysis_panel
        )
    )


    # ------------------------------------------------------
    # Lagged Merchandise Group Correlations
    # ------------------------------------------------------

    lagged_group_results = (
        lagged_merchandise_group_correlations(
            group_analysis_panel
        )
    )

    # ------------------------------------------------------
    # Current vs Lagged Comparison
    # ------------------------------------------------------

    current_vs_lagged = (
        compare_current_vs_lagged_groups(
            group_results,
            lagged_group_results
        )
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "CURRENT VS PRIOR-QUARTER GROUP RELATIONSHIPS"
    )
    logger.info("=" * 70)

    logger.info(
        "\n"
        + current_vs_lagged.round(3)
        .to_string(index=False)
    )

    logger.info("=" * 70)


    # ------------------------------------------------------
    # Current vs Lagged Visualization
    # ------------------------------------------------------

    plot_current_vs_lagged_groups(
        current_vs_lagged
    )

    # ------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("FIRST PASS COMPLETE")
    logger.info("=" * 70)

    logger.info(
        f"Analysis rows       : "
        f"{len(analysis_panel):,}"
    )

    logger.info(
        f"Analysis DMAs       : "
        f"{analysis_panel['DMA'].nunique():,}"
    )

    logger.info(
        f"Analysis quarters   : "
        f"{analysis_panel['Quarter_Period'].nunique():,}"
    )

    logger.info(
        f"Merchandise groups  : "
        f"{len(group_summary):,}"
    )

    logger.info(
        f"Adjusted results    : "
        f"{len(adjusted_results):,}"
    )


    logger.info("")
    logger.info("=" * 70)
    logger.info("MERCHANDISE GROUP SUMMARY")
    logger.info("=" * 70)

    display_columns = [
        "Merch_Group",
        "Raw_Correlation",
        "Adjusted_Correlation",
        "Adjusted_Per_Store_Correlation",
        "Adjusted_Share_Correlation",
        "Adjusted_Per_Store_Both_Correlation",
        "N"
    ]

    logger.info(
        "\n" +
        group_results[
            display_columns
        ].round(3).to_string(index=False)
    )


    logger.info("")
    logger.info("=" * 70)
    logger.info("LAGGED MERCHANDISE GROUP SUMMARY")
    logger.info("=" * 70)

    lag_display_columns = [
        "Merch_Group",
        "Raw_Lag_Correlation",
        "Within_DMA_Lag_Correlation",
        "Adjusted_Lag_Correlation",
        "N"
    ]

    logger.info(
        "\n" +
        lagged_group_results[
            lag_display_columns
        ].round(3).to_string(index=False)
    )

       
    logger.info("=" * 70)


if __name__ == "__main__":
    main()



