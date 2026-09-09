"""
07_future_forecast_reporting.py

Purpose:
    Generate future forecast.

Inputs:
    forecast_metrics.json
    future_forecast.parquet
    future_national_forecast.parquet

Outputs:
    future_forecast_report.html
"""

import json
import logging

import numpy as np
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import shap


from pathlib import Path

from config import (
    PROCESSED_DATA_DIR,
    LOG_LEVEL,
    REPORT_DIR,
    FIGURE_DIR
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
def load_artifacts():

    forecast_dir = (
        PROCESSED_DATA_DIR /
        "forecasting"
    )

    future = pd.read_parquet(
        forecast_dir /
        "future_forecast.parquet"
    )

    future_national = pd.read_parquet(
        forecast_dir /
        "future_national_forecast.parquet"
    )

    future_yoy = pd.read_parquet(
        forecast_dir /
        "future_yoy_forecast.parquet"
    )

    with open(
        forecast_dir /
        "forecast_metrics.json",
        "r"
    ) as f:

        metrics = json.load(f)

    return (
        future,
        future_national,
        future_yoy,
        metrics
    )


def create_national_forecast_plot(
    future_national
):

    logger.info(
        "Creating national forecast plot..."
    )

    output_path = (
        FIGURE_DIR /
        "future_national_forecast.png"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        future_national["Month"],
        future_national["Forecast"],
        marker="o",
        linewidth=3,
        color="#1f77b4"
    )

    # Format Y-axis in millions

    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda x, pos:
            f"{x/1_000_000:.1f}M"
        )
    )

    # Add point labels

    for _, row in future_national.iterrows():

        ax.annotate(
            f"{row['Forecast']/1_000_000:.1f}M",

            (
                row["Month"],
                row["Forecast"]
            ),

            textcoords="offset points",

            xytext=(0, 10),

            ha="center",

            fontsize=9,

            fontweight="bold"
        )

    ax.set_title(
        "Six-Month National Traffic Forecast",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Forecasted Traffic"
    )

    ax.grid(
        alpha=.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def create_top_dma_forecast_plot(
    future_yoy
):

    logger.info(
        "Creating top DMA growth plot..."
    )

    output_path = (
        FIGURE_DIR /
        "top_dma_forecasts.png"
    )

    dma_growth = (

        future_yoy

        .groupby(
            "DMA",
            as_index=False
        )

        .agg(
            YoY_Growth_Pct=(
                "YoY_Growth_Pct",
                "mean"
            )
        )

        .sort_values(
            "YoY_Growth_Pct",
            ascending=False
        )

        .head(15)

    )

    plt.figure(
        figsize=(12, 6)
    )

    bars = plt.barh(

        dma_growth["DMA"],

        dma_growth[
            "YoY_Growth_Pct"
        ]

    )

    plt.axvline(
        0,
        color="black",
        linewidth=1
    )

    plt.title(
        "Top DMA Forecasted YoY Growth"
    )

    plt.xlabel(
        "Projected YoY Traffic Growth (%)"
    )

    for bar in bars:

        width = (
            bar.get_width()
        )

        plt.text(

            width,

            bar.get_y()
            +
            bar.get_height()/2,

            f"{width:.1f}%",

            va="center",

            fontsize=8

        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def create_forecast_distribution(
    future
):

    output_path = (
        FIGURE_DIR /
        "forecast_distribution.png"
    )

    plt.figure(
        figsize=(8,5)
    )

    plt.hist(
        future["Forecast"],
        bins=20,
        edgecolor="black"
    )

    plt.title(
        "DMA Forecast Distribution"
    )

    plt.xlabel(
        "Forecast Traffic"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


def create_major_dma_trends(
    future_yoy
):

    logger.info(
        "Creating major DMA YoY trends..."
    )

    output_path = (
        FIGURE_DIR /
        "major_dma_trends.png"
    )

    top_dmas = (

        future_yoy

        .groupby(
            "DMA"
        )["Forecast"]

        .sum()

        .nlargest(5)

        .index

    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for dma in top_dmas:

        subset = (

            future_yoy[
                future_yoy["DMA"]
                == dma
            ]

            .sort_values(
                "Month"
            )

        )

        ax.plot(

            subset["Month"],

            subset[
                "YoY_Growth_Pct"
            ],

            marker="o",

            linewidth=2,

            label=dma

        )

        for _, row in subset.iterrows():

            ax.annotate(

                f"{row['YoY_Growth_Pct']:.1f}%",

                (
                    row["Month"],
                    row[
                        "YoY_Growth_Pct"
                    ]
                ),

                textcoords=
                    "offset points",

                xytext=(0, 8),

                ha="center",

                fontsize=7

            )

    ax.axhline(
        0,
        color="black",
        linewidth=1
    )

    ax.set_title(
        "Projected YoY Traffic Trend for Largest DMAs",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel(
        "YoY Traffic Growth (%)"
    )

    ax.grid(
        alpha=.3
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close()

    return output_path



# ---------------------------------------------------------------------
# Generate Figures
# ---------------------------------------------------------------------

def generate_figures(
    future,
    future_national,
    future_yoy
):

    figures = {

        "national_forecast":
            create_national_forecast_plot(
                future_national
            ),

        "top_dma_forecasts":
            create_top_dma_forecast_plot(
                future_yoy
            ),

        "distribution":
            create_forecast_distribution(
                future
            ),

        "dma_trends":
            create_major_dma_trends(
                future_yoy
            )

    }

    return figures

# ---------------------------------------------------------------------
# Executive Summary
# ---------------------------------------------------------------------

def build_summary(
    future_national,
    metrics
):

    peak_month = (
        future_national
        .loc[
            future_national["Forecast"]
            .idxmax()
        ]
    )

    peak_date = (
        pd.to_datetime(
            peak_month["Month"]
        )
        .strftime("%B %Y")
    )

    peak_volume = (
        peak_month["Forecast"]
    )

    dma_wape = (
        metrics["dma_metrics"]["WAPE"]
    )

    headline = (
        f"National traffic is projected to peak "
        f"in {peak_date} at approximately "
        f"{peak_volume:,.0f} visits. "
        f"The forecasting model achieved a "
        f"validation WAPE of {dma_wape:.1%}, "
        f"supporting confidence in the "
        f"six-month outlook."
    )

    return headline

# ---------------------------------------------------------------------
# KPI Card
# ---------------------------------------------------------------------

def metric_card(
    label,
    value
):

    return f"""
    <div class="card">
        <div class="label">
            {label}
        </div>

        <div class="value">
            {value}
        </div>
    </div>
    """


# ---------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------

def generate_html_report(
    future,
    future_national,
    future_yoy,
    metrics,
    figures
):

    logger.info(
        "Generating future forecast report..."
    )

    # ----------------------------------------------------
    # Figure Paths
    # ----------------------------------------------------

    report_path = (
        REPORT_DIR /
        "future_forecast_report.html"
    )

    figure_paths = {}

    for key, path in figures.items():

        figure_path = Path(
            path
        ).resolve()

        relative_path = os.path.relpath(
            figure_path,
            start=report_path.parent.resolve()
        )

        figure_paths[key] = (
            Path(relative_path)
            .as_posix()
        )

    # ----------------------------------------------------
    # Forecast Metrics
    # ----------------------------------------------------

    dma_wape = (
        metrics["dma_metrics"]["WAPE"]
    )

    dma_r2 = (
        metrics["dma_metrics"]["R2"]
    )

    forecast_horizon = (
        future_national["Month"]
        .nunique()
    )

    dma_count = (
        future["DMA"]
        .nunique()
    )

    total_forecast = (
        future["Forecast"]
        .sum()
    )

    peak_month = (
        future_national
        .loc[
            future_national["Forecast"]
            .idxmax()
        ]
    )

    peak_month_name = (
        pd.to_datetime(
            peak_month["Month"]
        )
        .strftime("%B %Y")
    )

    peak_volume = (
        peak_month["Forecast"]
    )

    headline = build_summary(
        future_national,
        metrics
    )

    # ----------------------------------------------------
    # Forecast Tables
    # ----------------------------------------------------

    top_dma_forecasts = (

        future_yoy

        .groupby(
            "DMA",
            as_index=False
        )

        .agg(

            Forecast=(
                "Forecast",
                "sum"
            ),

            Avg_YoY_Growth_Pct=(
                "YoY_Growth_Pct",
                "mean"
            )

        )

        .sort_values(
            "Forecast",
            ascending=False
        )

        .head(15)

    )

    top_dma_forecasts[
        "Avg_YoY_Growth_Pct"
    ] = (
        top_dma_forecasts[
            "Avg_YoY_Growth_Pct"
        ]
        .round(1)
    )

    top_dma_html = (
        top_dma_forecasts
        .round({
            "Forecast": 0
        })
        .to_html(
            index=False,
            classes="report-table"
        )
    )

    growth = (

        future_yoy

        .groupby(
            "DMA",
            as_index=False
        )

        .agg(

            Forecast=(
                "Forecast",
                "sum"
            ),

            Avg_YoY_Growth_Pct=(
                "YoY_Growth_Pct",
                "mean"
            )

        )

    )

    top_growth_html = (

        growth

        .sort_values(
            "Avg_YoY_Growth_Pct",
            ascending=False
        )

        .head(10)

        .round({
            "Forecast": 0,
            "Avg_YoY_Growth_Pct": 1
        })

        .to_html(
            index=False,
            classes="report-table"
        )

    )

    top_decline_html = (

        growth

        .sort_values(
            "Avg_YoY_Growth_Pct"
        )

        .head(10)

        .round({
            "Forecast": 0,
            "Avg_YoY_Growth_Pct": 1
        })

        .to_html(
            index=False,
            classes="report-table"
        )

    )

    # ----------------------------------------------------
    # HTML
    # ----------------------------------------------------

    report_html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="utf-8">

<title>
Future Traffic Forecast Report
</title>

<style>

body {{
    font-family: Arial, Helvetica, sans-serif;
    background: #f4f6f8;
    color: #1f2933;
    margin: 0;
}}

.container {{
    max-width: 1400px;
    margin: auto;
    padding: 30px;
}}

header {{
    background: #102a43;
    color: white;
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 20px;
}}

section {{
    background: white;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.08);
}}

.cards {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(220px,1fr));
    gap: 12px;
}}

.card {{
    background: #eef2f7;
    border-radius: 10px;
    padding: 16px;
}}

.label {{
    font-size: 12px;
    color: #627d98;
}}

.value {{
    font-size: 24px;
    font-weight: 700;
    color: #102a43;
}}

.report-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}

.report-table th,
.report-table td {{
    border: 1px solid #ddd;
    padding: 8px;
}}

.report-table th {{
    background: #eef2f7;
}}

.figure {{
    text-align: center;
}}

.figure img {{
    max-width: 100%;
    border: 1px solid #ddd;
    border-radius: 6px;
}}

.takeaway {{
    background: #eef6ff;
    border-left: 4px solid #3578c6;
    padding: 16px;
    border-radius: 6px;
}}

</style>

</head>

<body>

<div class="container">

<header>

<h1>
Future Traffic Forecast Outlook
</h1>

<p>
Six-Month DMA-Level Traffic Forecast
</p>

</header>

<section>

<h2>
Executive Summary
</h2>

<div class="takeaway">

{headline}

</div>

</section>

<section>

<h2>
Forecast KPI Summary
</h2>

<div class="cards">

{metric_card(
    "Validation WAPE",
    f"{dma_wape:.1%}"
)}

{metric_card(
    "Validation R²",
    f"{dma_r2:.3f}"
)}

{metric_card(
    "Forecast Horizon",
    f"{forecast_horizon} Months"
)}

{metric_card(
    "DMAs Forecasted",
    f"{dma_count}"
)}

{metric_card(
    "Peak Month",
    peak_month_name
)}

{metric_card(
    "Peak Traffic",
    f"{peak_volume/1_000_000:.1f}M"
)}

{metric_card(
    "Total Forecast Visits",
    f"{total_forecast/1_000_000:.1f}M"
)}

</div>

</section>

<section>

<h2>
National Forecast Outlook
</h2>

<div class="figure">

<img
src="{figure_paths['national_forecast']}"
alt="National Forecast"
>

</div>

</section>

<section>

<h2>
Largest Forecasted DMA Markets
</h2>

<div class="figure">

<img
src="{figure_paths['top_dma_forecasts']}"
alt="Top DMA's"
>

</div>

<br>

{top_dma_html}

</section>

<section>

<h2>
Forecast Distribution
</h2>

<div class="figure">

<img
src="{figure_paths['distribution']}"
alt="Forecast Distribution"
>

</div>

</section>

<section>

<h2>
Major DMA Forecast Trends
</h2>

<div class="figure">

<img
src="{figure_paths['dma_trends']}"
alt="DMA Forecast Trends"
>

</div>

</section>

<section>

<h2>
Most Stable Markets
</h2>

{top_growth_html}

</section>

<section>

<h2>
Largest Projected Declines
</h2>

{top_decline_html}

</section>

<section>

<h2>
Forecast Methodology
</h2>

<p>

Forecasts were generated using a tuned
XGBoost regression model trained across
34 DMA markets.

The final model achieved:

</p>

<ul>

<li>DMA WAPE: {dma_wape:.1%}</li>

<li>DMA R²: {dma_r2:.3f}</li>

<li>6-Month Forecast Horizon</li>

<li>Traffic History Features</li>

<li>Transaction Features</li>

<li>Store Network Variables</li>

<li>Competitive Density Metrics</li>

<li>Calendar Seasonality Indicators</li>

</ul>

</section>

<footer>

Generated automatically by
07_future_forecast_reporting.py

</footer>

</div>

</body>

</html>
"""

    output_file = (
        REPORT_DIR /
        "future_forecast_report.html"
    )

    output_file.write_text(
        report_html,
        encoding="utf-8"
    )

    logger.info(
        f"Report saved: {output_file}"
    )

    return output_file



# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():

    future, future_national, future_yoy, metrics = (
        load_artifacts()
    )

    figures = generate_figures(
        future,
        future_national,
        future_yoy
    )

    generate_html_report(
        future,
        future_national,
        future_yoy,
        metrics,
        figures
    )


if __name__ == "__main__":

    main()







