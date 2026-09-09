"""
===============================================================
Exploratory Data Analysis
Marketing Incrementality Project
===============================================================

Purpose
-------
Perform exploratory data analysis on the final analytical panel.

Outputs
-------
Figures:
    outputs/figures/eda/

Tables:
    outputs/tables/eda

Reports:
    outputs/reports/

Author:
    JM Steve

Last Updated:
    2026-07-30
===============================================================
"""

# =============================================================
# Standard Library
# =============================================================

import logging
from pathlib import Path

# =============================================================
# Third Party
# =============================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================
# Project Imports
# =============================================================

from config import (
    PROCESSED_DATA_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    OUTPUT_DIR,
    FIG_SIZE,
    DPI,
    STYLE
)

# =============================================================
# Configuration
# =============================================================

REPORT_DIR = OUTPUT_DIR / "reports"

plt.style.use(STYLE)

# =============================================================
# Logging
# =============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# =============================================================
# Helper Functions
# =============================================================

def create_output_folders():
    """
    Create all required output folders.
    """

    folders = [

        FIGURE_DIR / "eda",

        FIGURE_DIR / "eda" / "spend",

        FIGURE_DIR / "eda" / "traffic",

        FIGURE_DIR / "eda" / "stores",

        FIGURE_DIR / "eda" / "relationships",

        FIGURE_DIR / "eda" / "seasonality",

        FIGURE_DIR / "eda" / "diagnostics",

        TABLE_DIR / "eda",

        REPORT_DIR,

    ]

    for folder in folders:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

logger.info(FIGURE_DIR)
logger.info(TABLE_DIR)
logger.info(REPORT_DIR)        


def load_panel():
    """
    Load the final analytical panel.
    """

    logger.info("Loading panel dataset...")

    panel = pd.read_parquet(
        PROCESSED_DATA_DIR / "panel_data.parquet"
    )

    panel["Month"] = pd.to_datetime(panel["Month"])

    logger.info(
        f"Loaded {len(panel):,} observations."
    )

    return panel



def save_plot(fig, folder, filename):
    """
    Save figure.
    """

    path = (
        FIGURE_DIR
        / "eda"
        / folder
        / f"{filename}.png"
    )
    
    logger.info(f"Saving {path}")

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)

    logger.info(f"Saved {filename}")

#============================================================
# Spend Analysis
#============================================================
def plot_monthly_spend(df):

    logger.info("Monthly Spend")

    monthly = (

        df.groupby("Month")["Spend"]

        .sum()

    )

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        monthly.index,
        monthly.values,
        linewidth=2
    )

    ax.set_title("Monthly Marketing Spend")

    ax.set_ylabel("Spend ($)")
    ax.set_xlabel("Month")

    ax.grid(alpha=.3)

    save_plot(
        fig,
        "spend",
        "monthly_spend"
    )

def plot_spend_distribution(df):

    logger.info("Spend Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        df["Spend"],
        bins=30
    )

    ax.set_title("Marketing Spend Distribution")
    ax.set_xlabel("Spend ($)")
    ax.set_ylabel("Frequency")

    save_plot(
        fig,
        "spend",
        "spend_distribution"
    )  

def plot_spend_by_dma(df):

    logger.info("Spend by DMA")

    dma = (

        df.groupby("DMA")["Spend"]

        .sum()

        .sort_values()

    )

    fig, ax = plt.subplots(figsize=(10,10))

    ax.barh(
        dma.index,
        dma.values
    )

    ax.set_title("Total Spend by DMA")
    ax.set_ylabel("DMA")
    ax.set_xlabel("Spend ($)")

    save_plot(
        fig,
        "spend",
        "spend_by_dma"
    )

def export_spend_heatmap_data(df):

    logger.info("Spend Heatmap")

    heatmap = (

        df.pivot_table(

            index="DMA",

            columns="Month",

            values="Spend",

            aggfunc="sum"

        )

    )

    heatmap.to_csv(

        TABLE_DIR /

        "spend_heatmap.csv"

    )  

def export_spend_statistics(df):

    logger.info("Spend Statistics")

    stats = (

        df["Spend"]

        .describe()

        .round(2)

    )

    stats.to_csv(

        TABLE_DIR /

        "spend_statistics.csv"

    )

def spend_analysis(df):

    logger.info("")

    logger.info("=" * 60)

    logger.info("SPEND ANALYSIS")

    logger.info("=" * 60)

    plot_monthly_spend(df)

    plot_spend_distribution(df)

    plot_spend_by_dma(df)

    export_spend_heatmap_data(df)

    export_spend_statistics(df)    


#============================================================
# Traffic  Analysis
#============================================================   

# =============================================================
# Monthly Traffic
# =============================================================

def plot_monthly_traffic(df):
    """
    Plot total monthly traffic across all DMAs.
    """

    logger.info("Monthly Traffic")

    monthly = (
        df.groupby("Month")["Traffic"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        monthly.index,
        monthly.values,
        linewidth=2
    )

    ax.set_title("Monthly Store Traffic")

    ax.set_ylabel("Traffic")
    ax.set_xlabel("Month")

    ax.grid(alpha=0.3)

    save_plot(
        fig,
        "traffic",
        "monthly_traffic"
    )  

# =============================================================
# Traffic Distribution
# =============================================================

def plot_traffic_distribution(df):
    """
    Distribution of monthly traffic observations.
    """

    logger.info("Traffic Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        df["Traffic"].dropna(),
        bins=30
    )

    ax.set_title("Traffic Distribution")

    ax.set_xlabel("Traffic")
    ax.set_ylabel("Frequency")

    save_plot(
        fig,
        "traffic",
        "traffic_distribution"
    )

# =============================================================
# Traffic by DMA
# =============================================================

def plot_traffic_by_dma(df):
    """
    Total traffic by DMA.
    """

    logger.info("Traffic by DMA")

    dma = (

        df.groupby("DMA")["Traffic"]

        .sum()

        .sort_values()

    )

    fig, ax = plt.subplots(figsize=(10,10))

    ax.barh(
        dma.index,
        dma.values
    )

    ax.set_title("Total Traffic by DMA")

    ax.set_xlabel("Traffic")

    save_plot(
        fig,
        "traffic",
        "traffic_by_dma"
    )

# =============================================================
# Traffic Per Store
# =============================================================

def plot_traffic_per_store(df):
    """
    Distribution of traffic normalized by store count.
    """

    logger.info("Traffic per Store")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        df["Traffic_per_Store"].dropna(),
        bins=30
    )

    ax.set_title("Traffic per Store")
    ax.set_xlabel("Traffic")
    ax.set_ylabel("Frequency")

    save_plot(
        fig,
        "traffic",
        "traffic_per_store_distribution"
    )

# =============================================================
# Traffic per 1000 SqFt
# =============================================================

def plot_traffic_per_sqft(df):

    logger.info("Traffic per 1000 SqFt")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        df["Traffic_per_1000SqFt"].dropna(),
        bins=30
    )

    ax.set_title("Traffic per 1000 SqFt")

    ax.set_xlabel("Traffic")
    ax.set_ylabel("Frequency")

    save_plot(
        fig,
        "traffic",
        "traffic_per_1000sqft_distribution"
    )

# =============================================================
# Monthly Traffic Heatmap
# =============================================================

def export_traffic_heatmap_data(df):

    logger.info("Traffic Heatmap")

    heatmap = (

        df.pivot_table(

            index="DMA",

            columns="Month",

            values="Traffic",

            aggfunc="sum"

        )

    )

    heatmap.to_csv(

        TABLE_DIR /

        "traffic_heatmap.csv"

    )

# =============================================================
# Traffic Summary
# =============================================================

def export_traffic_statistics(df):

    logger.info("Traffic Statistics")

    stats = (

        df[

            [

                "Traffic",

                "Traffic_per_Store",

                "Traffic_per_1000SqFt"

            ]

        ]

        .describe()

        .round(2)

    )

    stats.to_csv(

        TABLE_DIR /

        "traffic_statistics.csv"

    )

# =============================================================
# Traffic Analysis
# =============================================================

def traffic_analysis(df):

    logger.info("")

    logger.info("=" * 60)

    logger.info("TRAFFIC ANALYSIS")

    logger.info("=" * 60)

    plot_monthly_traffic(df)

    plot_traffic_distribution(df)

    plot_traffic_by_dma(df)

    plot_traffic_per_store(df)

    plot_traffic_per_sqft(df)

    export_traffic_heatmap_data(df)

    export_traffic_statistics(df)    


#---------------------------------
# Store Analysis
# ---------------------------------

# =============================================================
# Store Count by DMA
# =============================================================

def plot_store_count(df):
    """
    Number of stores within each DMA.
    """

    logger.info("Store Count by DMA")

    stores = (
        df.groupby("DMA")["Store_Count"]
        .max()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10,10))

    ax.barh(
        stores.index,
        stores.values
    )

    ax.set_title("Store Count by DMA")

    ax.set_xlabel("Stores")

    save_plot(
        fig,
        "stores",
        "store_count_by_dma"
    )

# =============================================================
# Total Selling Square Feet
# =============================================================

def plot_total_sqft(df):

    logger.info("Selling Square Feet")

    sqft = (
        df.groupby("DMA")["Total_Selling_SqFt"]
        .max()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10,10))

    ax.barh(
        sqft.index,
        sqft.values
    )

    ax.set_title("Total Selling Square Feet")

    ax.set_xlabel("Square Feet")

    save_plot(
        fig,
        "stores",
        "total_selling_sqft"
    )

# =============================================================
# Average Store Size
# =============================================================

def plot_avg_store_size(df):

    logger.info("Average Store Size")

    avg = (
        df.groupby("DMA")["Avg_Selling_SqFt"]
        .max()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10,10))

    ax.barh(
        avg.index,
        avg.values
    )

    ax.set_title("Average Store Size")

    ax.set_xlabel("Average SqFt")

    save_plot(
        fig,
        "stores",
        "average_store_size"
    )

# =============================================================
# DMA Population
# =============================================================

def plot_dma_population(df):

    logger.info("DMA Population")

    pop = (
        df.groupby("DMA")["DMA_Population"]
        .max()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10,10))

    ax.barh(
        pop.index,
        pop.values
    )

    ax.set_title("DMA Population")

    ax.set_xlabel("Population")

    save_plot(
        fig,
        "stores",
        "dma_population"
    )

# =============================================================
# Trade Area Population
# =============================================================

def plot_trade_area_population(df):

    logger.info("Trade Area Population")

    trade = (
        df.groupby("DMA")["TradeAreaPopulation"]
        .max()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10,10))

    ax.barh(
        trade.index,
        trade.values
    )

    ax.set_title("Trade Area Population")

    ax.set_xlabel("Population")

    save_plot(
        fig,
        "stores",
        "trade_area_population"
    )

# =============================================================
# Store Statistics
# =============================================================

def export_store_statistics(df):

    logger.info("Store Statistics")

    stats = (
        df[
            [
                "Store_Count",
                "Total_Selling_SqFt",
                "Avg_Selling_SqFt",
                "DMA_Population",
                "TradeAreaPopulation"
            ]
        ]
        .describe()
        .round(2)
    )

    stats.to_csv(
        TABLE_DIR /
        "store_statistics.csv"
    )

# =============================================================
# Store Analysis
# =============================================================

def store_analysis(df):

    logger.info("")

    logger.info("=" * 60)
    logger.info("STORE ANALYSIS")
    logger.info("=" * 60)

    plot_store_count(df)

    plot_total_sqft(df)

    plot_avg_store_size(df)

    plot_dma_population(df)

    plot_trade_area_population(df)

    export_store_statistics(df)   



#-------------------------------
# Relationship Analysis
#-------------------------------

# =============================================================
# Spend vs Traffic
# =============================================================

def plot_spend_vs_traffic(df):
    """
    Scatter plot of Spend vs Traffic.
    """

    logger.info("Spend vs Traffic")

    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(
        df["Spend"],
        df["Traffic"],
        alpha=.65
    )

    ax.set_title("Marketing Spend vs Store Traffic")

    ax.set_xlabel("Spend")

    ax.set_ylabel("Traffic")

    ax.grid(alpha=.3)

    save_plot(
        fig,
        "relationships",
        "spend_vs_traffic"
    )

# =============================================================
# Spend vs Transactions
# =============================================================

def plot_spend_vs_transactions(df):

    logger.info("Spend vs Transactions")

    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(
        df["Spend"],
        df["Transactions"],
        alpha=.65
    )

    ax.set_xlabel("Spend")

    ax.set_ylabel("Transactions")

    ax.set_title("Spend vs Transactions")

    save_plot(
        fig,
        "relationships",
        "spend_vs_transactions"
    )

# =============================================================
# Spend vs Traffic per Store
# =============================================================

def plot_spend_vs_traffic_per_store(df):

    logger.info("Spend vs Traffic per Store")

    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(
        df["Spend"],
        df["Traffic_per_Store"],
        alpha=.6
    )

    ax.set_title("Spend vs Traffic per Store")

    ax.set_xlabel("Spend")

    ax.set_ylabel("Traffic per Store")

    save_plot(
        fig,
        "relationships",
        "spend_vs_traffic_per_store"
    )

# =============================================================
# Spend vs Traffic per 1000 SqFt
# =============================================================

def plot_spend_vs_sqft(df):

    logger.info("Spend vs Traffic per 1000 SqFt")

    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(
        df["Spend"],
        df["Traffic_per_1000SqFt"],
        alpha=.6
    )

    ax.set_title("Spend vs Traffic per 1000 SqFt")
    ax.set_xlabel("Spend ($)")
    ax.set_ylabel("Traffic per 1000 SqFt")

    save_plot(
        fig,
        "relationships",
        "spend_vs_1000sqft"
    )

# =============================================================
# Correlation Matrix
# =============================================================

def correlation_matrix(df):

    logger.info("Correlation Matrix")

    numeric = df.select_dtypes(include=np.number)

    corr = numeric.corr()

    corr.to_csv(
        TABLE_DIR /
        "correlation_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(12,10))

    im = ax.imshow(corr)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(
        corr.columns,
        rotation=90,
        fontsize=7
    )

    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(
        corr.columns,
        fontsize=7
    )

    fig.colorbar(im)

    save_plot(
        fig,
        "relationships",
        "correlation_matrix"
    )

# =============================================================
# Correlation Ranking
# =============================================================

def export_correlation_table(df):

    logger.info("Correlation Table")

    numeric = df.select_dtypes(include=np.number)

    corr = (
        numeric
        .corr()["Traffic"]
        .sort_values(
            ascending=False
        )
        .round(3)
    )

    corr.to_csv(
        TABLE_DIR /
        "traffic_correlations.csv"
    )

# =============================================================
# Spend Lag Correlations
# =============================================================

def export_lag_correlations(df):

    logger.info("Lag Correlations")

    cols = [

        "Spend",

        "Spend_Lag_1",

        "Spend_Lag_2",

        "Spend_Lag_3",

        "Spend_Lag_6"

    ]

    corr = (
        df[
            cols +
            ["Traffic"]
        ]
        .corr()["Traffic"]
        .round(3)
    )

    corr.to_csv(
        TABLE_DIR /
        "lag_correlations.csv"
    )

# =============================================================
# High vs Low Spend
# =============================================================

def export_high_spend_summary(df):

    logger.info("High Spend Comparison")

    summary = (
        df.groupby("High_Spend")
        .agg(

            AvgTraffic=("Traffic","mean"),

            AvgTransactions=("Transactions","mean"),

            AvgSpend=("Spend","mean")

        )
        .round(2)
    )

    summary.to_csv(

        TABLE_DIR /

        "high_spend_summary.csv"

    )

# =============================================================
# Relationship Statistics
# =============================================================

def export_relationship_statistics(df):

    stats = (

        df[

            [

                "Spend",

                "Traffic",

                "Transactions"

            ]

        ]

        .describe()

        .round(2)

    )

    stats.to_csv(

        TABLE_DIR /

        "relationship_statistics.csv"

    )

# =============================================================
# Relationship Analysis
# =============================================================

def relationship_analysis(df):

    logger.info("")

    logger.info("=" * 60)

    logger.info("RELATIONSHIP ANALYSIS")

    logger.info("=" * 60)

    plot_spend_vs_traffic(df)

    plot_spend_vs_transactions(df)

    plot_spend_vs_traffic_per_store(df)

    plot_spend_vs_sqft(df)

    correlation_matrix(df)

    export_correlation_table(df)

    export_lag_correlations(df)

    export_high_spend_summary(df)

    export_relationship_statistics(df)                    

#------------------------------
# Seasonality Analysis
# -----------------------------

# =============================================================
# Average Traffic by Calendar Month
# =============================================================

def plot_avg_monthly_traffic(df):
    """
    Average traffic for each calendar month.
    """

    logger.info("Average Monthly Traffic")

    monthly = (
        df.groupby("Month_Number")["Traffic"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        monthly.index,
        monthly.values,
        marker="o",
        linewidth=2
    )

    ax.set_xticks(range(1,13))

    ax.set_xlabel("Month")

    ax.set_ylabel("Average Traffic")

    ax.set_title("Average Monthly Traffic")

    ax.grid(alpha=.3)

    save_plot(
        fig,
        "seasonality",
        "average_monthly_traffic"
    )

# =============================================================
# Average Monthly Spend
# =============================================================

def plot_avg_monthly_spend(df):

    logger.info("Average Monthly Spend")

    monthly = (
        df.groupby("Month_Number")["Spend"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(
        monthly.index,
        monthly.values,
        marker="o",
        linewidth=2
    )

    ax.set_xticks(range(1,13))

    ax.set_title("Average Monthly Marketing Spend")

    ax.grid(alpha=.3)

    save_plot(
        fig,
        "seasonality",
        "average_monthly_spend"
    )

# =============================================================
# Year-over-Year Traffic
# =============================================================

def plot_yoy_traffic(df):

    logger.info("Year-over-Year Traffic")

    pivot = (
        df.groupby(
            ["Year","Month_Number"]
        )["Traffic"]
        .sum()
        .unstack(0)
    )

    logger.info("YOY TRAFFIC PIVOT")
    logger.info("\n%s", pd.pivot)

    fig, ax = plt.subplots(figsize=(10,5))

    for year in pivot.columns:

        ax.plot(
            pivot.index,
            pivot[year],
            marker="o",
            label=str(year)
        )

    ax.legend()

    ax.set_xticks(range(1,13))

    ax.set_title("Traffic by Month (Year-over-Year)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Traffic")

    save_plot(
        fig,
        "seasonality",
        "traffic_yoy"
    )

# =============================================================
# Seasonal Index
# =============================================================

def export_seasonal_index(df):

    logger.info("Seasonal Index")

    monthly = (
        df.groupby("Month_Number")["Traffic"]
        .mean()
    )

    seasonal = monthly / monthly.mean()

    seasonal.to_csv(
        TABLE_DIR /
        "traffic_seasonal_index.csv"
    )

from statsmodels.tsa.seasonal import STL
# =============================================================
# STL Decomposition
# =============================================================

def plot_stl(df):

    logger.info("STL Decomposition")

    ts = (
        df.groupby("Month")["Traffic"]
        .sum()
        .sort_index()
    )

    stl = STL(
        ts,
        period=12
    )

    result = stl.fit()

    fig = result.plot()

    fig.set_size_inches(12,8)

    save_plot(
        fig,
        "seasonality",
        "stl_decomposition"
    )

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def plot_traffic_acf(df):

    logger.info("Traffic ACF")

    ts = (
        df.groupby("Month")["Traffic"]
        .sum()
        .dropna()
    )

    logger.info(
        f"Traffic ACF observations: {len(ts)}"
    )

    if len(ts) < 5:
        logger.warning(
            "Not enough observations for ACF plot."
        )
        return


    max_lag = min(
        18,
        len(ts) - 2
    )

    fig, ax = plt.subplots(figsize=(10,5))

    plot_acf(
        ts,
        ax=ax,
        lags=max_lag
    )

    ax.set_title(
        "Traffic Autocorrelation"
    )

    save_plot(
        fig,
        "seasonality",
        "traffic_acf"
    )

def plot_traffic_pacf(df):

    logger.info("Traffic PACF")
    
    ts = (
        df.groupby("Month")["Traffic"]
        .sum()
        .dropna()
    )

    logger.info(
        f"Traffic PACF observations: {len(ts)}"
    )

    if len(ts) < 5:
        logger.warning(
            "Not enough observations for PACF plot."
        )
        return


    max_lag = min(
        18,
        (len(ts) // 2) - 1
    )

    fig, ax = plt.subplots(figsize=(10,5))

    plot_pacf(
        ts,
        ax=ax,
        lags=max_lag
    )

    ax.set_title(
        "Traffic Partial Autocorrelation"
    )

    save_plot(
        fig,
        "seasonality",
        "traffic_pacf"
    )

def seasonality_analysis(df):

    logger.info("")
    logger.info("=" * 60)
    logger.info("SEASONALITY ANALYSIS")
    logger.info("=" * 60)

    plot_avg_monthly_traffic(df)

    plot_avg_monthly_spend(df)

    plot_yoy_traffic(df)

    export_seasonal_index(df)

    plot_stl(df)

    plot_traffic_acf(df)

    plot_traffic_pacf(df)

#------------------------------
# Diagnostic Analysis
#------------------------------

# =============================================================
# Missing Values
# =============================================================

def export_missing_values(df):
    """
    Export missing value summary.
    """

    logger.info("Missing Value Report")

    missing = pd.DataFrame({

        "Missing_Count": df.isna().sum(),

        "Percent":

        (df.isna().mean()*100).round(2)

    })

    missing = (

        missing

        .sort_values(

            "Percent",

            ascending=False

        )

    )

    missing.to_csv(

        TABLE_DIR /

        "missing_values.csv"

    )

# =============================================================
# Duplicate Observations
# =============================================================

def export_duplicates(df):

    logger.info("Duplicate Check")

    dup = df.duplicated()

    report = pd.DataFrame({

        "Duplicate Rows":[dup.sum()],

        "Percent":[

            round(

                dup.mean()*100,

                3

            )

        ]

    })

    report.to_csv(

        TABLE_DIR /

        "duplicates.csv",

        index=False

    )

# =============================================================
# Numeric Summary
# =============================================================

def export_numeric_summary(df):

    logger.info("Numeric Summary")

    summary = (

        df

        .select_dtypes(

            include=np.number

        )

        .describe()

        .round(2)

    )

    summary.to_csv(

        TABLE_DIR /

        "numeric_summary.csv"

    )

# =============================================================
# Boxplots
# =============================================================

def plot_boxplots(df):

    logger.info("Boxplots")

    cols = [

        "Spend",

        "Traffic",

        "Transactions",

        "Traffic_per_Store",

        "Spend_per_Store"

    ]

    for col in cols:

        fig, ax = plt.subplots(figsize=(6,5))

        ax.boxplot(

            df[col].dropna()

        )

        ax.set_title(col)

        save_plot(

            fig,

            "diagnostics",

            f"{col}_boxplot"

        )

# =============================================================
# Outlier Report
# =============================================================

def export_outliers(df):

    logger.info("Outlier Report")

    numeric = df.select_dtypes(

        include=np.number

    )

    results = []

    for col in numeric.columns:

        q1 = numeric[col].quantile(.25)

        q3 = numeric[col].quantile(.75)

        iqr = q3-q1

        lower = q1-1.5*iqr

        upper = q3+1.5*iqr

        count = (

            (numeric[col]<lower)

            |

            (numeric[col]>upper)

        ).sum()

        results.append({

            "Variable":col,

            "Outliers":count

        })

    pd.DataFrame(results).to_csv(

        TABLE_DIR /

        "outlier_report.csv",

        index=False

    )

# =============================================================
# Shape Statistics
# =============================================================

def export_distribution_shape(df):

    logger.info("Distribution Shape")

    numeric = df.select_dtypes(

        include=np.number

    )

    stats = pd.DataFrame({

        "Skewness":

        numeric.skew(),

        "Kurtosis":

        numeric.kurt()

    })

    stats.to_csv(

        TABLE_DIR /

        "distribution_shape.csv"

    )

from statsmodels.stats.outliers_influence import variance_inflation_factor
# =============================================================
# Variance Inflation Factor
# =============================================================

def export_vif(df):

    logger.info("Variance Inflation Factors")

    cols = [

        "Spend",

        "Store_Count",

        "Total_Selling_SqFt",

        "DMA_Population",

        "TradeAreaPopulation"

    ]

    x = (

        df[cols]

        .dropna()

        .copy()

    )

    vif = pd.DataFrame()

    vif["Variable"] = cols

    vif["VIF"] = [

        variance_inflation_factor(

            x.values,

            i

        )

        for i in range(len(cols))

    ]

    vif.to_csv(

        TABLE_DIR /

        "vif.csv",

        index=False

    )

# =============================================================
# Panel Balance
# =============================================================

def export_panel_balance(df):

    logger.info("Panel Balance")

    balance = (

        df.groupby("DMA")

        .size()

        .reset_index(

            name="Months"

        )

    )

    balance.to_csv(

        TABLE_DIR /

        "panel_balance.csv",

        index=False

    )

# =============================================================
# Time Coverage
# =============================================================

def export_time_coverage(df):

    logger.info("Time Coverage")

    coverage = (

        df.groupby("Month")

        .size()

        .reset_index(

            name="DMAs"

        )

    )

    coverage.to_csv(

        TABLE_DIR /

        "time_coverage.csv",

        index=False

    )

# =============================================================
# DMA Coverage
# =============================================================

def export_dma_summary(df):

    logger.info("DMA Coverage")

    summary = (

        df.groupby("DMA")

        .agg(

            Months=("Month","count"),

            AvgSpend=("Spend","mean"),

            AvgTraffic=("Traffic","mean")

        )

        .round(2)

    )

    summary.to_csv(

        TABLE_DIR /

        "dma_summary.csv"

    )

# =============================================================
# Diagnostic Analysis
# =============================================================

def diagnostic_analysis(df):

    logger.info("")

    logger.info("=" * 60)

    logger.info("MODEL DIAGNOSTICS")

    logger.info("=" * 60)

    export_missing_values(df)

    export_duplicates(df)

    export_numeric_summary(df)

    plot_boxplots(df)

    export_outliers(df)

    export_distribution_shape(df)

    export_vif(df)

    export_panel_balance(df)

    export_time_coverage(df)

    export_dma_summary(df)                                                          
                                
#---------------------------
# Dataset Overview
#---------------------------

# =============================================================
# Dataset Overview
# =============================================================

def dataset_overview(df):
    """
    Generate a high-level overview of the modeling dataset.
    """

    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET OVERVIEW")
    logger.info("=" * 60)

    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Columns: {len(df.columns)}")

    logger.info(
        f"Date Range: "
        f"{df['Month'].min().date()} "
        f"to "
        f"{df['Month'].max().date()}"
    )

    logger.info(f"DMAs: {df['DMA'].nunique():,}")

    logger.info(
        f"Observations per DMA (avg): "
        f"{len(df)/df['DMA'].nunique():.1f}"
    )

    logger.info("")

    logger.info(df.dtypes)

    logger.info("")

    logger.info(df.describe(include="all"))

    # ---------------------------------------------------------
    # Dataset Metadata
    # ---------------------------------------------------------

    overview = pd.DataFrame({

        "Metric":[
            "Rows",
            "Columns",
            "DMAs",
            "Start Date",
            "End Date",
            "Missing Values"
        ],

        "Value":[
            len(df),
            len(df.columns),
            df["DMA"].nunique(),
            df["Month"].min(),
            df["Month"].max(),
            int(df.isna().sum().sum())
        ]

    })

    overview.to_csv(
        TABLE_DIR / "dataset_overview.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Column Summary
    # ---------------------------------------------------------

    column_summary = pd.DataFrame({

        "Column": df.columns,

        "Type": df.dtypes.astype(str),

        "Missing":

            df.isna().sum().values,

        "Unique":

            df.nunique().values

    })

    column_summary.to_csv(

        TABLE_DIR /

        "column_summary.csv",

        index=False

    )

    logger.info("Dataset overview complete.")

#-----------------------------------
# Generate HTML Report
#-----------------------------------

#--------------------------------
# Helper Functions
#--------------------------------
# ============================================================
# HTML Figure Gallery
# ============================================================

def html_gallery(folder):

    folder = Path(folder)

    images = sorted(folder.glob("*.png"))

    if not images:
        return "<p><i>No figures generated.</i></p>"

    html = ""

    for img in images:

        relative = img.relative_to(OUTPUT_DIR)

        title = img.stem.replace("_", " ").title()

        html += f"""
        <div class="figure">
            <h3>{title}</h3>
            <img src="../{relative.as_posix()}" width="900">
        </div>
        """

    return html

# ============================================================
# HTML Table List
# ============================================================

def html_tables(max_rows=25):
    """
    Convert generated CSV tables into HTML sections.

    Parameters
    ----------
    max_rows : int
        Maximum rows displayed per table.
    """

    html = ""

    tables = sorted(TABLE_DIR.glob("*.csv"))

    if not tables:
        return "<p>No tables generated.</p>"


    for table in tables:

        logger.info(
            f"Adding table to HTML report: {table.name}"
        )

        try:

            df = pd.read_csv(table)

            html += f"""

<details>

<summary>
<b>{table.name}</b>
</summary>

<br>

"""

            html += df.head(max_rows).to_html(
                index=False,
                classes="dataframe"
            )


            if len(df) > max_rows:

                html += f"""

<p>
Showing first {max_rows} rows of {len(df):,}.
</p>

"""


            html += """

</details>

<br>

"""


        except Exception as e:

            logger.warning(
                f"Could not render {table.name}: {e}"
            )


    return html
# =============================================================
# HTML Report
# =============================================================

def generate_html_report(df):

    logger.info("Generating HTML report...")

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_spend = df["Spend"].sum()

    total_traffic = df["Traffic"].sum()

    corr = df["Spend"].corr(df["Traffic"])

    html = f"""

<html>

<head>

<title>Marketing Incrementality EDA</title>

<style>

body{{
    font-family:Arial;
    margin:40px;
    background:#F7F7F7;
}}

h1{{
    color:#003865;
}}

h2{{
    border-bottom:2px solid #003865;
    padding-bottom:5px;
}}

table{{
    border-collapse:collapse;
}}

td,th{{
    border:1px solid #ccc;
    padding:8px;
}}

.figure{{
    margin-bottom:50px;
}}

img{{
    border:1px solid gray;
}}

.dataframe {{
    width: 100%;
    margin-top: 10px;
    margin-bottom: 20px;
    font-size: 12px;
}}

.dataframe th {{
    background:#003865;
    color:white;
}}

details {{
    background:white;
    padding:15px;
    border:1px solid #ccc;
    margin-bottom:15px;
}}

summary {{
    cursor:pointer;
    font-size:16px;
}}

</style>

</head>

<body>

<h1>Marketing Incrementality EDA</h1>

<h2>Dataset Summary</h2>

<table>

<tr><th>Metric</th><th>Value</th></tr>

<tr><td>Rows</td><td>{len(df):,}</td></tr>

<tr><td>Columns</td><td>{len(df.columns)}</td></tr>

<tr><td>DMAs</td><td>{df["DMA"].nunique()}</td></tr>

<tr><td>Date Range</td><td>{df["Month"].min().date()} -
{df["Month"].max().date()}</td></tr>

<tr><td>Total Spend</td><td>${total_spend:,.0f}</td></tr>

<tr><td>Total Traffic</td><td>{total_traffic:,.0f}</td></tr>

<tr><td>Spend vs Traffic Correlation</td><td>{corr:.3f}</td></tr>

</table>

<h2>Spend Analysis</h2>

{html_gallery(FIGURE_DIR / "eda" / "spend")}

<h2>Traffic Analysis</h2>

{html_gallery(FIGURE_DIR / "eda" / "traffic")}

<h2>Store Analysis</h2>

{html_gallery(FIGURE_DIR / "eda" / "stores")}

<h2>Relationship Analysis</h2>

{html_gallery(FIGURE_DIR / "eda" / "relationships")}

<h2>Seasonality Analysis</h2>

{html_gallery(FIGURE_DIR / "eda" / "seasonality")}

<h2>Diagnostic Analysis</h2>

{html_gallery(FIGURE_DIR / "eda" / "diagnostics")}

<h2>Generated Tables</h2>

{html_tables()}

</body>

</html>

"""

    report = REPORT_DIR / "EDA_Report.html"

    report.write_text(html, encoding="utf-8")

    logger.info(f"Saved report to {report}")


def main():

    logger.info("=" * 70)
    logger.info("MARKETING INCREMENTALITY EDA")
    logger.info("=" * 70)

    create_output_folders()

    panel = load_panel()

    dataset_overview(panel)

    spend_analysis(panel)

    traffic_analysis(panel)

    store_analysis(panel)

    relationship_analysis(panel)

    seasonality_analysis(panel)

    diagnostic_analysis(panel)

    generate_html_report(panel)


    logger.info(panel.info())
    logger.info("=" * 70)
    logger.info("EDA COMPLETE")
    logger.info("=" * 70)    

if __name__ == "__main__":
    main()                         
