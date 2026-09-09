"""
06_cross_correlation.py

Cross-Correlation Analysis
--------------------------

Purpose
-------
Evaluate temporal relationships between marketing spend
and store traffic prior to causal modeling.

Outputs
-------
Figures
Tables
HTML Report
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import pearsonr

from config import (
    PROCESSED_DATA_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    OUTPUT_DIR,
    STYLE
)

#=========================================================
# Output Folders
#=========================================================

REPORT_DIR = OUTPUT_DIR / "reports"


def create_output_folders():

    folders = [

        FIGURE_DIR / "cross_correlation",

        FIGURE_DIR / "cross_correlation" / "overall",

        FIGURE_DIR / "cross_correlation" / "lag_analysis",

        FIGURE_DIR / "cross_correlation" / "dma",

        FIGURE_DIR / "cross_correlation" / "rolling",

        FIGURE_DIR / "cross_correlation" / "diagnostics",

        FIGURE_DIR / "cross_correlation" / "adjusted",

        TABLE_DIR / "cross_correlation",

        REPORT_DIR

    ]

    for folder in folders:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger(__name__)

#=========================================================
# Load Data
#=========================================================

def load_model_panel():

    logger.info("Loading model panel...")

    panel_df = pd.read_parquet(

        PROCESSED_DATA_DIR /
        "model_panel.parquet"

    )

    panel_df["Month"] = pd.to_datetime(
        panel_df["Month"]
    )

    panel= panel_df.copy()
    panel = panel[panel["DMA"].notna()]
    panel = panel[panel["Traffic"].notna()]

    logger.info(
        f"Loaded {len(panel):,} rows"
    )

    logger.info(
        f"{panel['DMA'].nunique()} DMAs"
    )

    return panel

#=========================================================
# Validation
#=========================================================

def validate_panel(panel):

    logger.info("=" * 70)
    logger.info("MODEL PANEL")
    logger.info("=" * 70)

    logger.info(
        f"Rows     : {len(panel):,}"
    )

    logger.info(
        f"DMAs     : {panel['DMA'].nunique()}"
    )

    logger.info(
        f"Months   : {panel['Month'].nunique()}"
    )

    logger.info(
        f"Date Min : {panel['Month'].min().date()}"
    )

    logger.info(
        f"Date Max : {panel['Month'].max().date()}"
    )

    logger.info("")

#=========================================================
# Figure Helper
#=========================================================

def save_plot(

    fig,

    folder,

    filename

):

    path = (

        FIGURE_DIR
        / "cross_correlation"
        / folder
        / f"{filename}.png"

    )

    fig.tight_layout()

    fig.savefig(

        path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close(fig)

    logger.info(
        f"Saved {filename}"
    )

#=========================================================
# Overall Pearson Correlation
#=========================================================

def overall_correlation(
    df,
    spend_col="Spend",
    traffic_col="Traffic",
    analysis_name="raw"
):
    """
    Compute overall Spend vs Traffic correlation.
    """

    logger.info(
        f"Computing {analysis_name} overall correlation..."
    )


    corr = (
        df[
            [
                spend_col,
                traffic_col
            ]
        ]
        .corr()
        .iloc[0,1]
    )


    logger.info(
        f"{analysis_name} correlation: {corr:.3f}"
    )


    summary = pd.DataFrame({

        "Analysis":[
            analysis_name
        ],

        "Spend Variable":[
            spend_col
        ],

        "Traffic Variable":[
            traffic_col
        ],

        "Correlation":[
            corr
        ]

    })


    summary.to_csv(

        TABLE_DIR /
        "cross_correlation" /
        f"{analysis_name}_overall_correlation.csv",

        index=False

    )


    return corr

#=========================================================
# Lagged Correlation Table
#=========================================================

def lagged_correlation_table(
    df,
    spend_col="Spend",
    traffic_col="Traffic",
    max_lag=12,
    analysis_name="raw"
):
    """
    Correlation between Spend and future Traffic.
    """

    logger.info(
        f"Computing {analysis_name} lagged correlations..."
    )


    rows = []


    for lag in range(max_lag + 1):

        shifted = (
            df[traffic_col]
            .shift(-lag)
        )


        corr = (
            df[spend_col]
            .corr(shifted)
        )


        rows.append({

            "Analysis": analysis_name,
            "Lag": lag,
            "Correlation": corr

        })


    results = pd.DataFrame(rows)


    results.to_csv(

        TABLE_DIR /
        "cross_correlation" /
        f"{analysis_name}_lagged_correlations.csv",

        index=False

    )


    return results

#=========================================================
# Lag Correlation Plot
#=========================================================

def lagged_correlation_plot(
    results,
    analysis_name="raw"
):
    """
    Plot lag correlations.
    """

    logger.info(
        f"Plotting {analysis_name} lag correlations..."
    )


    fig, ax = plt.subplots(
        figsize=(10,5)
    )


    ax.bar(
        results["Lag"],
        results["Correlation"]
    )


    ax.axhline(
        0,
        color="black",
        linewidth=1
    )


    ax.set_xlabel(
        "Lag (Months)"
    )

    ax.set_ylabel(
        "Correlation"
    )


    ax.set_title(
        f"{analysis_name.title()} Spend vs Traffic Lag Correlation"
    )


    save_plot(
        fig,
        "lag_analysis",
        f"{analysis_name}_lagged_correlation"
    )

#=========================================================
# Rolling Correlation
#=========================================================

def rolling_correlation(
    df,
    window=6
):
    """
    Rolling Spend vs Traffic correlation.
    """

    logger.info("Computing rolling correlation...")

    monthly = (

        df
        .groupby("Month")
        .agg({

            "Spend": "sum",
            "Traffic": "sum"

        })

    )

    rolling = (
        monthly["Spend"]
        .rolling(window)
        .corr(monthly["Traffic"])
    )

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        rolling.index,
        rolling.values
    )

    ax.axhline(
        0,
        linestyle="--"
    )

    ax.set_title(
        "Rolling Spend-Traffic Correlation"
    )

    save_plot(
        fig,
        "rolling",
        "rolling_correlation"
    )

    rolling.to_csv(
        TABLE_DIR /
        "cross_correlation" /
        "rolling_correlation.csv"
    )

    logger.info(
        "Saved rolling correlation"
    )

    return rolling

#=========================================================
# DMA Correlations
#=========================================================

def dma_correlation_summary(df):
    """
    Correlation by DMA.
    """

    logger.info("Computing DMA correlations...")

    rows = []

    for dma, group in df.groupby("DMA"):

        corr = group["Spend"].corr(
            group["Traffic"]
        )

        rows.append({

            "DMA": dma,
            "Correlation": corr

        })

    results = (
        pd.DataFrame(rows)
        .sort_values(
            "Correlation",
            ascending=False
        )
    )

    results.to_csv(

        TABLE_DIR /
        "cross_correlation" /
        "dma_correlations.csv",

        index=False

    )

    return results


#=========================================================
# Export Summary
#=========================================================

def export_correlation_summary(
    overall_corr,
    lag_results,
    dma_results
):
    """
    Export high-level correlation summary.
    """

    best = lag_results.loc[
        lag_results["Correlation"].idxmax()
    ]

    summary = pd.DataFrame({

        "Metric":[

            "Overall Correlation",
            "Best Lag",
            "Best Lag Correlation",
            "Best DMA",
            "Best DMA Correlation"

        ],

        "Value":[

            overall_corr,
            int(best["Lag"]),
            best["Correlation"],
            dma_results.iloc[0]["DMA"],
            dma_results.iloc[0]["Correlation"]

        ]

    })

    summary.to_csv(

        TABLE_DIR /
        "cross_correlation" /
        "correlation_summary.csv",

        index=False

    )

    logger.info(
        "Saved correlation_summary.csv"
    )

#=========================================================
# National Monthly Time Series
#=========================================================

def create_monthly_series(df):
    """
    Aggregate panel to one monthly national time series.
    """

    logger.info(
        "Creating monthly national time series..."
    )

    agg_dict = {

        "Spend": "sum",
        "Traffic": "sum"

    }


    # Optional metrics

    optional_metrics = [

        "Transactions"

    ]


    for metric in optional_metrics:

        if metric in df.columns:

            agg_dict[metric] = "sum"

        else:

            logger.warning(
                f"{metric} not found. Skipping aggregation."
            )


    monthly = (
        df
        .groupby(
            "Month",
            as_index=False
        )
        .agg(agg_dict)
        .sort_values("Month")
    )


    monthly.to_csv(

        TABLE_DIR /
        "cross_correlation" /
        "monthly_timeseries.csv",

        index=False

    )


    logger.info(
        f"Created {len(monthly)} monthly observations"
    )


    logger.info(
        f"Columns: {monthly.columns.tolist()}"
    )


    return monthly


def validate_monthly_series(monthly):

    logger.info("="*70)
    logger.info("NATIONAL MONTHLY SERIES")
    logger.info("="*70)

    logger.info(
        monthly.isna().sum()
    )

    if monthly["Traffic"].isna().any():

        raise ValueError(
            "Traffic contains missing values. "
            "CCF cannot proceed."
        )
#=========================================================
# Cross Correlation Function
#=========================================================

#=========================================================
# Detrending
#=========================================================

def detrend_series(
    monthly
):
    """
    Remove linear trend from Spend and Traffic.

    Creates residual series representing
    deviations from expected trend.
    """

    logger.info(
        "Creating detrended series..."
    )

    df = monthly.copy()

    x = np.arange(len(df))


    for col in [
        "Spend",
        "Traffic"
    ]:

        coefficients = np.polyfit(
            x,
            df[col],
            deg=1
        )

        trend = np.polyval(
            coefficients,
            x
        )

        df[f"{col}_Trend"] = trend

        df[f"{col}_Detrended"] = (
            df[col]
            -
            trend
        )


    df.to_csv(

        TABLE_DIR /
        "cross_correlation" /
        "adjusted" /
        "detrended_timeseries.csv",

        index=False

    )

    return df



from scipy.signal import correlate

def compute_ccf(
    monthly,
    spend_col="Spend",
    traffic_col="Traffic",
    max_lag=12,
    filename="cross_correlation_function.csv"
):
    """
    Compute normalized cross-correlation between
    Spend and Traffic.
    """

    logger.info("Computing Cross-Correlation Function...")

    spend = monthly[spend_col].values.astype(float)
    traffic = monthly[traffic_col].values.astype(float)

    spend = (spend - spend.mean()) / spend.std()
    traffic = (traffic - traffic.mean()) / traffic.std()

    ccf = correlate(
        traffic,
        spend,
        mode="full"
    )

    lags = np.arange(
        -len(spend)+1,
        len(spend)
    )

    ccf = ccf / len(spend)

    results = pd.DataFrame({

        "Lag": lags,
        "Correlation": ccf

    })

    results = results[
        results["Lag"].between(
            -max_lag,
            max_lag
        )
    ]

    results.to_csv(

        TABLE_DIR /
        "cross_correlation" /
        filename,

        index=False

    )

    return results

#=========================================================
# Cross Correlation Plot
#=========================================================

def plot_ccf(
    results,
    analysis_name="raw"
):
    """
    Plot Cross Correlation Function.
    """

    logger.info(f"Plotting {analysis_name} CCF...")

    fig, ax = plt.subplots(figsize=(11,6))

    ax.vlines(
        results["Lag"],
        0,
        results["Correlation"],
        linewidth=2
    )

    ax.scatter(
        results["Lag"],
        results["Correlation"],
        s=35
    )

    ax.axhline(
        0,
        color="black"
    )

    ax.axvline(
        0,
        color="gray",
        linestyle="--"
    )

    ax.set_xlabel("Lag (Months)")
    ax.set_ylabel("Correlation")

    ax.set_title(
        f"Cross Correlation Function\n{analysis_name.capitalize()} Spend vs Traffic"
    )

    save_plot(
        fig,
        "overall",
        f"cross_correlation_function_{analysis_name}"
    )

#=========================================================
# Peak Cross Correlation
#=========================================================

def summarize_ccf(
    results,
    analysis_name="raw"
):
    """
    Summarize strongest cross correlation.
    """

    peak = results.loc[
        results["Correlation"]
        .abs()
        .idxmax()
    ]


    logger.info("=" * 60)
    logger.info(
        f"{analysis_name.upper()} CCF SUMMARY"
    )
    logger.info("=" * 60)


    logger.info(
        f"Peak Lag: {peak['Lag']}"
    )

    logger.info(
        f"Correlation: {peak['Correlation']:.3f}"
    )


    summary = pd.DataFrame({

        "Analysis":[
            analysis_name
        ],

        "Peak Lag":[
            peak["Lag"]
        ],

        "Peak Correlation":[
            peak["Correlation"]
        ]

    })


    summary.to_csv(

        TABLE_DIR /
        "cross_correlation" /
        f"{analysis_name}_ccf_summary.csv",

        index=False

    )


    return summary


def compare_ccf(
    raw,
    adjusted
):

    raw_peak = raw.loc[
        raw["Correlation"]
        .abs()
        .idxmax()
    ]


    adj_peak = adjusted.loc[
        adjusted["Correlation"]
        .abs()
        .idxmax()
    ]


    comparison = pd.DataFrame({

        "Series":[
            "Raw",
            "Detrended"
        ],

        "Peak Lag":[
            raw_peak["Lag"],
            adj_peak["Lag"]
        ],

        "Peak Correlation":[
            raw_peak["Correlation"],
            adj_peak["Correlation"]
        ]

    })


    comparison.to_csv(

        TABLE_DIR /
        "cross_correlation" /
        "ccf_comparison.csv",

        index=False

    )

    return comparison

def plot_ccf_comparison(
    raw,
    adjusted
):

    fig, ax = plt.subplots(
        figsize=(12,6)
    )


    ax.plot(
        raw["Lag"],
        raw["Correlation"],
        marker="o",
        label="Raw"
    )


    ax.plot(
        adjusted["Lag"],
        adjusted["Correlation"],
        marker="o",
        label="Detrended"
    )


    ax.axhline(
        0,
        color="black"
    )


    ax.axvline(
        0,
        linestyle="--"
    )


    ax.set_xlabel(
        "Lag (Months)"
    )

    ax.set_ylabel(
        "Correlation"
    )


    ax.set_title(
        "Raw vs Detrended Cross Correlation"
    )


    ax.legend()


    save_plot(
        fig,
        "adjusted",
        "raw_vs_detrended_ccf"
    )

#=========================================================
# HTML Header
#=========================================================

def html_header():
    """
    Return HTML header and stylesheet.
    """

    return f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<title>Marketing Cross-Correlation Report</title>

<style>

body {{

    font-family: Arial, Helvetica, sans-serif;
    margin: 40px;
    background: #f7f7f7;
    color: #222;

}}

h1 {{

    color: #003B5C;
    border-bottom: 3px solid #003B5C;
    padding-bottom: 10px;

}}

h2 {{

    color: #003B5C;
    margin-top: 40px;

}}

h3 {{

    color: #444;

}}

p {{

    line-height: 1.6;

}}

table {{

    border-collapse: collapse;
    width: 100%;
    margin-top: 15px;
    margin-bottom: 25px;

}}

th {{

    background: #003B5C;
    color: white;
    padding: 10px;

}}

td {{

    border: 1px solid #ddd;
    padding: 8px;

}}

tr:nth-child(even) {{

    background: #f2f2f2;

}}

.section {{

    background: white;
    padding: 25px;
    margin-bottom: 30px;
    border-radius: 8px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);

}}

.card-container {{

    display: flex;
    justify-content: space-between;
    gap: 20px;
    margin: 25px 0;

}}

.card {{

    flex: 1;
    background: white;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.10);

}}

.card-title {{

    color: #666;
    font-size: 14px;

}}

.card-value {{

    font-size: 28px;
    font-weight: bold;
    color: #003B5C;
    margin-top: 10px;

}}

.figure {{

    text-align: center;
    margin-top: 20px;
    margin-bottom: 20px;

}}

.figure img {{

    max-width: 100%;
    border: 1px solid #ddd;
    border-radius: 6px;

}}

.footer {{

    margin-top: 60px;
    font-size: 12px;
    color: gray;
    text-align: center;

}}

</style>

</head>

<body>

<h1>
Marketing Spend vs Traffic<br>
Cross-Correlation Analysis Report
</h1>

<p>

This report summarizes the temporal relationship between
marketing spend and store traffic using Pearson correlation,
lagged correlation analysis, rolling correlations, DMA-level
correlations, and cross-correlation functions (CCF).

</p>

"""

#=========================================================
# HTML Summary Cards
#=========================================================

def html_summary_cards(
    monthly,
    overall_corr,
    raw_ccf,
    adjusted_ccf,
    dma_results
):
    """
    Build executive summary KPI cards for the HTML report.
    """

    #-----------------------------------------------------
    # Summary Metrics
    #-----------------------------------------------------

    n_months = len(monthly)

    date_min = monthly["Month"].min().strftime("%b %Y")

    date_max = monthly["Month"].max().strftime("%b %Y")

    raw_peak = raw_ccf.loc[
        raw_ccf["Correlation"].abs().idxmax()
    ]

    adjusted_peak = adjusted_ccf.loc[
        adjusted_ccf["Correlation"].abs().idxmax()
    ]

    top_dma = dma_results.iloc[0]

    #-----------------------------------------------------
    # HTML
    #-----------------------------------------------------

    html = f"""

<div class="section">

<h2>Executive Summary</h2>

<p>

This dashboard summarizes the primary findings from the
national cross-correlation analysis between marketing spend
and store traffic.

</p>


<div class="card-container">

    <div class="card">

        <div class="card-title">
            Monthly Observations
        </div>

        <div class="card-value">
            {n_months}
        </div>

        <small>
            {date_min} – {date_max}
        </small>

    </div>


    <div class="card">

        <div class="card-title">
            Overall Correlation
        </div>

        <div class="card-value">
            {overall_corr:.3f}
        </div>

        <small>
            Pearson Correlation
        </small>

    </div>


    <div class="card">

        <div class="card-title">
            Raw Peak CCF
        </div>

        <div class="card-value">
            Lag {int(raw_peak["Lag"])}
        </div>

        <small>
            Corr = {raw_peak["Correlation"]:.3f}
        </small>

    </div>


    <div class="card">

        <div class="card-title">
            Detrended Peak CCF
        </div>

        <div class="card-value">
            Lag {int(adjusted_peak["Lag"])}
        </div>

        <small>
            Corr = {adjusted_peak["Correlation"]:.3f}
        </small>

    </div>


    <div class="card">

        <div class="card-title">
            Strongest DMA
        </div>

        <div class="card-value">
            {top_dma["DMA"]}
        </div>

        <small>
            Corr = {top_dma["Correlation"]:.3f}
        </small>

    </div>

</div>

</div>

"""

    return html

#=========================================================
# HTML Analysis Sections
#=========================================================

def html_analysis_sections(
    raw_overall_corr,
    raw_lagged,
    detrended_lagged,
    dma_results
):
    """
    Build the analysis sections for the HTML report.
    """

    html = f"""

<div class="section">

<h2>Overall Correlation</h2>

<p>

The overall Pearson correlation summarizes the linear
relationship between national marketing spend and store
traffic across the study period.

</p>

<table>

<tr>

<th>Metric</th>

<th>Value</th>

</tr>

<tr>

<td>Overall Correlation</td>

<td>{raw_overall_corr:.3f}</td>

</tr>

</table>

</div>



<div class="section">

<h2>Lagged Correlation Analysis</h2>

<p>

Lagged correlations evaluate whether marketing spend
is more strongly associated with traffic occurring
several months into the future.

</p>

<h3>Raw Series</h3>

<div class="figure">

<img src="../figures/cross_correlation/lag_analysis/raw_lagged_correlation.png"
width="900">

</div>

{raw_lagged.to_html(
    index=False,
    float_format="{:.3f}".format,
    classes="table"
)}

<br>

<h3>Detrended Series</h3>

<div class="figure">

<img src="../figures/cross_correlation/lag_analysis/detrended_lagged_correlation.png"
width="900">

</div>

{detrended_lagged.to_html(
    index=False,
    float_format="{:.3f}".format,
    classes="table"
)}

</div>



<div class="section">

<h2>Rolling Correlation</h2>

<p>

Rolling six-month correlations illustrate how the
relationship between marketing spend and traffic
changed throughout the study period.

</p>

<div class="figure">

<img src="../figures/cross_correlation/rolling/rolling_correlation.png"
width="900">

</div>

</div>



<div class="section">

<h2>DMA Correlation Rankings</h2>

<p>

DMA-level correlations identify geographic markets
where historical marketing spend has shown the strongest
association with store traffic.

</p>

{dma_results.head(10).to_html(
    index=False,
    float_format="{:.3f}".format,
    classes="table"
)}

</div>



<div class="section">

<h2>Cross-Correlation Function (Raw)</h2>

<div class="figure">

<img src="../figures/cross_correlation/overall/cross_correlation_function_raw.png"
width="900">

</div>

</div>



<div class="section">

<h2>Cross-Correlation Function (Detrended)</h2>

<div class="figure">

<img src="../figures/cross_correlation/overall/cross_correlation_function_detrended.png"
width="900">

</div>

</div>



<div class="section">

<h2>Raw vs. Detrended Comparison</h2>

<p>

Comparing the raw and detrended cross-correlation
functions helps distinguish relationships driven by
shared trends from those that persist after removing
long-term movement and seasonality.

</p>

<div class="figure">

<img src="../figures/cross_correlation/adjusted/raw_vs_detrended_ccf.png"
width="900">

</div>

</div>

"""

    return html

#=========================================================
# HTML Key Findings
#=========================================================

def html_key_findings(
    monthly,
    overall_corr,
    raw_ccf,
    adjusted_ccf,
    dma_results,
    rolling_corr
):
    """
    Build Key Findings section for the HTML report.
    """

    #-----------------------------------------------------
    # Summary Statistics
    #-----------------------------------------------------

    raw_peak = raw_ccf.loc[
        raw_ccf["Correlation"].abs().idxmax()
    ]

    adjusted_peak = adjusted_ccf.loc[
        adjusted_ccf["Correlation"].abs().idxmax()
    ]

    top_dma = dma_results.iloc[0]

    #-----------------------------------------------------
    # Correlation Strength
    #-----------------------------------------------------

    abs_corr = abs(overall_corr)

    if abs_corr >= 0.70:
        corr_strength = "strong"

    elif abs_corr >= 0.40:
        corr_strength = "moderate"

    elif abs_corr >= 0.20:
        corr_strength = "weak"

    else:
        corr_strength = "very weak"

    #-----------------------------------------------------
    # Lag Interpretation
    #-----------------------------------------------------

    if raw_peak["Lag"] == 0:

        lag_comment = (
            "The strongest relationship in the raw data "
            "occurs contemporaneously (Lag 0), suggesting "
            "that changes in marketing spend and traffic "
            "tend to occur during the same month."
        )

    else:

        lag_comment = (
            f"The strongest relationship in the raw data "
            f"occurs at Lag {int(raw_peak['Lag'])}, "
            "indicating that temporal structure exists "
            "between marketing spend and traffic."
        )

    #-----------------------------------------------------
    # Detrending Interpretation
    #-----------------------------------------------------

    if raw_peak["Lag"] != adjusted_peak["Lag"]:

        detrend_comment = (
            f"After detrending, the peak correlation shifts "
            f"from Lag {int(raw_peak['Lag'])} to "
            f"Lag {int(adjusted_peak['Lag'])}. "
            "This suggests that long-term trend was "
            "influencing the raw cross-correlation "
            "structure."
        )

    else:

        detrend_comment = (
            "Detrending did not materially change the "
            "location of the peak correlation, indicating "
            "that the observed lag structure is relatively "
            "stable."
        )

    #-----------------------------------------------------
    # DMA Interpretation
    #-----------------------------------------------------

    dma_comment = (
        f"The strongest DMA-level relationship was observed "
        f"in <b>{top_dma['DMA']}</b> "
        f"(Correlation = {top_dma['Correlation']:.3f}), "
        "suggesting this market exhibited the strongest "
        "historical association between spend and traffic."
    )


    #-----------------------------------------------------
    # Rolling Correlation Summary
    #-----------------------------------------------------

    rolling_clean = rolling_corr.dropna()

    rolling_min = rolling_clean.min()

    rolling_max = rolling_clean.max()

    rolling_mean = rolling_clean.mean()

    rolling_range = rolling_max - rolling_min


    if rolling_range > 0.60:

        rolling_comment = (
            f"Rolling six-month correlations varied "
            f"substantially throughout the study period "
            f"(Range: {rolling_min:.3f} to {rolling_max:.3f}; "
            f"Mean: {rolling_mean:.3f}). "
            "This indicates that the relationship between "
            "marketing spend and traffic was not stable "
            "through time."
        )

    elif rolling_range > 0.30:

        rolling_comment = (
            f"Rolling correlations showed moderate variation "
            f"(Range: {rolling_min:.3f} to {rolling_max:.3f}; "
            f"Mean: {rolling_mean:.3f}), suggesting some "
            "changes in the strength of the spend–traffic "
            "relationship over time."
        )

    else:

        rolling_comment = (
            f"Rolling correlations remained relatively stable "
            f"(Range: {rolling_min:.3f} to {rolling_max:.3f}; "
            f"Mean: {rolling_mean:.3f}), indicating a "
            "consistent relationship between spend and "
            "traffic throughout the study period."
        )

    #-----------------------------------------------------
    # HTML
    #-----------------------------------------------------

    html = f"""

<div class="section">

<h2>Key Findings</h2>

<ul>

<li>

Overall national spend and traffic exhibited a
<b>{corr_strength}</b> positive relationship
(Pearson Correlation = <b>{overall_corr:.3f}</b>).

</li>

<li>

{lag_comment}

</li>

<li>

The maximum raw cross-correlation occurred at
<b>Lag {int(raw_peak["Lag"])}</b>
(Correlation = <b>{raw_peak["Correlation"]:.3f}</b>).

</li>

<li>

The maximum detrended cross-correlation occurred at
<b>Lag {int(adjusted_peak["Lag"])}</b>
(Correlation = <b>{adjusted_peak["Correlation"]:.3f}</b>).

</li>

<li>

{detrend_comment}

</li>

<li>

{dma_comment}

</li>

<li>

{rolling_comment}

</li>

<li>

Overall, the cross-correlation analysis provides
preliminary evidence that marketing spend and store
traffic exhibit measurable temporal association,
although additional causal modeling is required to
estimate true incremental impact.

</li>

</ul>

</div>

"""

    return html

#=========================================================
# Generate HTML Report
#=========================================================

def generate_html_report(
    monthly,
    raw_overall_corr,
    raw_lagged,
    raw_ccf,
    detrended_lagged,
    detrended_ccf,
    dma_results,
    rolling_results
):
    """
    Generate the Cross-Correlation HTML report.
    """

    logger.info("Generating HTML report...")

    html = ""

    #-----------------------------------------------------
    # Header
    #-----------------------------------------------------

    html += html_header()

    #-----------------------------------------------------
    # Executive Summary
    #-----------------------------------------------------

    html += html_summary_cards(
        monthly,
        raw_overall_corr,
        raw_ccf,
        detrended_ccf,
        dma_results
    )

    #-----------------------------------------------------
    # Analysis Sections
    #-----------------------------------------------------

    html += html_analysis_sections(
        raw_overall_corr,
        raw_lagged,
        detrended_lagged,
        dma_results
    )

    #-----------------------------------------------------
    # Key Findings
    #-----------------------------------------------------

    html += html_key_findings(
        monthly,
        raw_overall_corr,
        raw_ccf,
        detrended_ccf,
        dma_results,
        rolling_results
    )

    #-----------------------------------------------------
    # Footer
    #-----------------------------------------------------

    html += """

<div class="footer">

<hr>

<p>

Generated automatically by
<b>06_cross_correlation.py</b>

</p>

<p>

Marketing Incrementality Project

</p>

</div>

</body>

</html>

"""

    #-----------------------------------------------------
    # Save
    #-----------------------------------------------------

    report_path = (
        REPORT_DIR /
        "cross_correlation_report.html"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)

    logger.info(
        f"Saved HTML report to {report_path}"
    )

    return report_path

#=========================
# Main
# ========================

def main():

    logger.info("=" * 70)
    logger.info("Begin Cross-Correlation Analysis")
    logger.info("=" * 70)


    #=========================================================
    # Load + Validate Panel
    #=========================================================

    panel = load_model_panel()

    validate_panel(panel)


    #=========================================================
    # Create National Monthly Series
    #=========================================================

    monthly = create_monthly_series(
        panel
    )

    validate_monthly_series(
        monthly
    )


    #=========================================================
    # DMA Analysis
    #=========================================================

    logger.info(
        "Running DMA correlation analysis..."
    )

    dma_results = dma_correlation_summary(
        panel
    )


    #=========================================================
    # Raw Analysis
    #=========================================================

    logger.info(
        "Running raw correlation analysis..."
    )


    # Overall correlation

    raw_overall_corr = overall_correlation(
        monthly,
        spend_col="Spend",
        traffic_col="Traffic",
        analysis_name="raw"
    )


    # Lag correlations

    raw_lagged = lagged_correlation_table(
        monthly,
        spend_col="Spend",
        traffic_col="Traffic",
        max_lag=12,
        analysis_name="raw"
    )


    lagged_correlation_plot(
        raw_lagged,
        analysis_name="raw"
    )


    # Rolling correlation

    rolling_results =rolling_correlation(
        monthly,
        window=6
    )


    # Raw CCF

    raw_ccf = compute_ccf(
        monthly,
        spend_col="Spend",
        traffic_col="Traffic",
        filename="raw_ccf.csv"
    )


    plot_ccf(
        raw_ccf,
        analysis_name="raw"
    )


    raw_ccf_summary = summarize_ccf(
        raw_ccf,
        analysis_name="raw"
    )


    #=========================================================
    # Detrended Analysis
    #=========================================================

    logger.info(
        "Running detrended correlation analysis..."
    )


    detrended = detrend_series(
        monthly
    )


    # Overall detrended correlation

    detrended_overall_corr = overall_correlation(
        detrended,
        spend_col="Spend_Detrended",
        traffic_col="Traffic_Detrended",
        analysis_name="detrended"
    )


    # Detrended lag correlations

    detrended_lagged = lagged_correlation_table(
        detrended,
        spend_col="Spend_Detrended",
        traffic_col="Traffic_Detrended",
        max_lag=12,
        analysis_name="detrended"
    )


    lagged_correlation_plot(
        detrended_lagged,
        analysis_name="detrended"
    )


    # Detrended CCF

    detrended_ccf = compute_ccf(
        detrended,
        spend_col="Spend_Detrended",
        traffic_col="Traffic_Detrended",
        filename="detrended_ccf.csv"
    )


    plot_ccf(
        detrended_ccf,
        analysis_name="detrended"
    )


    detrended_ccf_summary = summarize_ccf(
        detrended_ccf,
        analysis_name="detrended"
    )


    #=========================================================
    # Raw vs Detrended Comparison
    #=========================================================

    logger.info(
        "Comparing raw vs detrended results..."
    )


    compare_ccf(
        raw_ccf,
        detrended_ccf
    )


    plot_ccf_comparison(
        raw_ccf,
        detrended_ccf
    )


    #=========================================================
    # Final Summary Export
    #=========================================================

    export_correlation_summary(
        raw_overall_corr,
        raw_lagged,
        dma_results
    )

    #=========================================================
    # HTML Report
    #=========================================================

    generate_html_report(
        monthly,
        raw_overall_corr,
        raw_lagged,
        raw_ccf,
        detrended_lagged,
        detrended_ccf,
        dma_results,
        rolling_results
    )


    logger.info("=" * 70)
    logger.info("Cross-Correlation Analysis complete.")
    logger.info("=" * 70)



if __name__ == "__main__":
    main()