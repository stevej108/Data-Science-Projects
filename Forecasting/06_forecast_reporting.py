"""
06_forecast_reporting.py

Purpose:
    Generate forecasting validation report.

Inputs:
    forecast_metrics.json
    xgb_predictions.parquet
    dma_performance.parquet
    feature_importance.parquet
    national_forecast.parquet

Outputs:
    forecast_report.html
"""

import json
import logging

import numpy as np
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt
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

    predictions = pd.read_parquet(
        forecast_dir /
        "xgb_predictions.parquet"
    )

    dma_performance = pd.read_parquet(
        forecast_dir /
        "dma_performance.parquet"
    )

    feature_importance = pd.read_parquet(
        forecast_dir /
        "feature_importance.parquet"
    )

    national = pd.read_parquet(
        forecast_dir /
        "national_forecast.parquet"
    )

    with open(
        forecast_dir /
        "forecast_metrics.json",
        "r"
    ) as f:

        metrics = json.load(f)

    return (
        predictions,
        dma_performance,
        feature_importance,
        national,
        metrics
    )


# ---------------------------------------------------------------------
# Figure Helper
# ---------------------------------------------------------------------

def figure_to_base64():

    buffer = BytesIO()

    plt.savefig(
        buffer,
        format="png",
        bbox_inches="tight"
    )

    plt.close()

    buffer.seek(0)

    return base64.b64encode(
        buffer.read()
    ).decode("utf-8")


# ---------------------------------------------------------------------
# DMA Validation Plot
# ---------------------------------------------------------------------

def create_dma_validation_plot(
    predictions
):

    logger.info(
        "Creating DMA validation plot..."
    )

    output_path = (
        FIGURE_DIR /
        "dma_validation.png"
    )

    dma_summary = (
        predictions
        .groupby(
            "DMA",
            as_index=False
        )
        .agg(
            Actual=("Actual", "sum"),
            Forecast=("Forecast", "sum")
        )
    )

    dma_summary = (
        dma_summary
        .sort_values(
            "Actual",
            ascending=False
        )
        .head(15)
    )

    x = np.arange(
        len(dma_summary)
    )

    width = 0.40

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        x - width / 2,
        dma_summary["Actual"],
        width,
        label="Actual"
    )

    plt.bar(
        x + width / 2,
        dma_summary["Forecast"],
        width,
        label="Forecast"
    )

    plt.xticks(
        x,
        dma_summary["DMA"],
        rotation=45,
        ha="right"
    )

    plt.title(
        "Top DMA Validation: Actual vs Forecast"
    )

    plt.ylabel(
        "Traffic"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close()

    return output_path

# ---------------------------------------------------------------------
# Feature Importance Plot
# ---------------------------------------------------------------------

def create_feature_importance_plot(
    importance
):

    logger.info(
        "Creating feature importance plot..."
    )

    logger.info(
        f"Shape: {importance.shape}"
    )

    logger.info(
        importance.columns.tolist()
    )
    

    

    output_path = (
            FIGURE_DIR /
            "feature_importance.png"
            )

    top_features = (
        importance
        .head(15)
        .sort_values(
            "Importance"
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.title(
        "Top 15 Feature Importance"
    )

    plt.xlabel(
        "Importance"
    )

    plt.savefig(
            output_path,
            bbox_inches="tight"
        )
    
    plt.close()
    
    return output_path


# ---------------------------------------------------------------------
# DMA Error Distribution
# ---------------------------------------------------------------------

def create_dma_error_distribution(
    dma_performance
):

    logger.info(
        "Creating DMA error distribution..."
    )

    output_path = (
        FIGURE_DIR /
        "dma_error_distribution.png"
        )

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        dma_performance["APE"],
        bins=15,
        edgecolor="black"
    )

    plt.title(
        "DMA Forecast Error Distribution"
    )

    plt.xlabel(
        "APE"
    )

    plt.ylabel(
        "DMA Count"
    )

    plt.savefig(
            output_path,
            bbox_inches="tight"
        )
    
    plt.close()
    
    return output_path


# ---------------------------------------------------------------------
# Worst DMA Forecast Plot
# ---------------------------------------------------------------------

def create_worst_dma_plot(
    dma_performance
):

    logger.info(
        "Creating worst DMA plot..."
    )

    output_path = (
        FIGURE_DIR /
        "worst_dma.png"
        )

    worst = (
        dma_performance
        .head(10)
        .sort_values(
            "APE"
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        worst.index,
        worst["APE"]
    )

    plt.title(
        "Highest Forecast Error DMAs"
    )

    plt.xlabel(
        "Average Percentage Error"
    )

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
    predictions, 
    feature_importance,
    dma_performance
    
):

    logger.info("=" * 70)
    logger.info(
        "GENERATING FORECAST FIGURES"
    )
    logger.info("=" * 70)

    figures = {

        "dma_validation":
            create_dma_validation_plot(
            predictions
        ),

        "feature_importance":
            create_feature_importance_plot(
                feature_importance
            ),

        "dma_error_distribution":
            create_dma_error_distribution(
                dma_performance
            ),

        "worst_dma":
            create_worst_dma_plot(
                dma_performance
            )

    }

    logger.info(
        "Forecast figures generated: %s",
        len(figures)
    )

    return figures


# ---------------------------------------------------------------------
# Executive Summary
# ---------------------------------------------------------------------

def build_summary(metrics):

    dma_wape = (
        metrics["dma_metrics"]["WAPE"]
    )

    if dma_wape < .15:

        headline = (
            "Forecast accuracy was strong "
            "across DMA markets."
        )

    elif dma_wape < .25:

        headline = (
            "Forecast accuracy was acceptable "
            "with moderate DMA variation."
        )

    else:

        headline = (
            "Forecast performance suggests "
            "additional model refinement."
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
    predictions,
    dma_performance,
    feature_importance,
    national,
    metrics,
    figures
):

    logger.info(
        "Generating forecast report..."
    )

    dma_metrics = metrics["dma_metrics"]

    national_metrics = metrics["national_metrics"]

    headline = build_summary(
        metrics
    )

    # ----------------------------------------------------
    # Tables
    # ----------------------------------------------------

    best_dma = (
        dma_performance
        .sort_values("APE")
        .head(10)
        .round(3)
    )

    worst_dma = (
        dma_performance
        .sort_values(
            "APE",
            ascending=False
        )
        .head(10)
        .round(3)
    )

    top_features = (
        feature_importance
        .head(20)
        .round(4)
    )

    best_dma_html = (
        best_dma
        .to_html(
            classes="report-table"
        )
    )

    worst_dma_html = (
        worst_dma
        .to_html(
            classes="report-table"
        )
    )

    feature_html = (
        top_features
        .to_html(
            index=False,
            classes="report-table"
        )
    )


    report_path = (
        REPORT_DIR /
        "forecast_report.html"
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
    # Summary Stats
    # ----------------------------------------------------

    validation_months = (
        predictions["Month"]
        .nunique()
    )

    dma_count = (
        predictions["DMA"]
        .nunique()
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
Forecast Validation Report
</title>

<style>

body {{
    font-family:
        Arial,
        Helvetica,
        sans-serif;

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

    box-shadow:
        0 1px 3px
        rgba(0,0,0,.08);
}}

.cards {{
    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(220px,1fr)
        );

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
    font-size: 26px;

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

    margin-top: 20px;

    text-align:center;
}}

.figure img {{

    max-width:100%;

    border:1px solid #ddd;

    border-radius:6px;
}}

.takeaway {{

    background:#eef6ff;

    border-left:4px solid #3578c6;

    padding:16px;

    border-radius:6px;
}}

</style>

</head>

<body>

<div class="container">

<header>

<h1>
XGBoost Traffic Forecast Validation
</h1>

<p>
DMA-level retail traffic forecasting evaluation
</p>

</header>

<section>

<h2>
Executive Summary
</h2>

<div class="takeaway">

<strong>
Headline:
</strong>

{headline}

<br><br>

The XGBoost forecasting model achieved a
DMA-level WAPE of
<strong>{dma_metrics['WAPE']:.1%}</strong>
and a DMA-level R² of
<strong>{dma_metrics['R2']:.3f}</strong>.

National forecast accuracy produced a
WAPE of
<strong>{national_metrics['WAPE']:.1%}</strong>.

</div>

</section>

<section>

<h2>
Validation KPI Summary
</h2>

<div class="cards">

{metric_card(
    "DMA WAPE",
    f"{dma_metrics['WAPE']:.1%}"
)}

{metric_card(
    "DMA MAPE",
    f"{dma_metrics['MAPE']:.1%}"
)}

{metric_card(
    "DMA R²",
    f"{dma_metrics['R2']:.3f}"
)}

{metric_card(
    "National WAPE",
    f"{national_metrics['WAPE']:.1%}"
)}

{metric_card(
    "DMAs",
    f"{dma_count}"
)}

{metric_card(
    "Validation Months",
    f"{validation_months}"
)}

</div>

</section>

<section>

<h2>
Top DMA Validation Results
</h2>

<p>

This comparison shows the highest-volume DMAs
in the validation period and compares actual
traffic against the XGBoost forecast.

Markets where actual and forecast traffic are
closely aligned indicate strong predictive
performance.

</p>

<div class="figure">

<img
    src="{figure_paths["dma_validation"]}"
    alt="National Forecast"
>

</div>

</section>

<section>

<h2>
Feature Importance
</h2>

<div class="figure">

<img
    src="{figure_paths['feature_importance']}"
    alt="Feature Importance"
>

</div>

</section>

<section>

<h2>
DMA Error Distribution
</h2>

<div class="figure">

<img
    src="{figure_paths['dma_error_distribution']}"
    alt="DMA Error Distribution"
>

</div>

</section>

<section>

<h2>
Worst Forecasted DMAs
</h2>

<div class="figure">

<img
    src="{figure_paths['worst_dma']}"
    alt="Worst DMA Forecast"
>

</div>

</section>

<section>

<h2>
Top DMA Performance
</h2>

{best_dma_html}

</section>

<section>

<h2>
Worst DMA Performance
</h2>

{worst_dma_html}

</section>

<section>

<h2>
Top Model Features
</h2>

{feature_html}

</section>

<section>

<h2>
Model Interpretation
</h2>

<p>

Recent traffic patterns were the strongest
drivers of forecast performance.

Operational variables including store
network size and competitive density
provided meaningful incremental signal.

Marketing spend variables contributed to
forecast accuracy, although historical
traffic trends remained more influential
than spend alone.

</p>

</section>

<section>

<h2>
Methodology
</h2>

<p>

Traffic forecasts were generated using an
XGBoost regression model trained at the
DMA-month level.

The model incorporates:

</p>

<ul>

<li>Marketing spend history</li>

<li>Traffic lag features</li>

<li>Transaction lag features</li>

<li>Store network characteristics</li>

<li>Customer experience metrics</li>

<li>Calendar seasonality</li>

</ul>

<p>

Validation results were produced using an
out-of-time holdout period rather than
random sampling, providing a more realistic
assessment of forecasting performance.

</p>

</section>

<footer>

Generated automatically by
06_forecast_reporting.py

</footer>

</div>

</body>

</html>
"""

    output_file = (
        REPORT_DIR /
        "forecast_report.html"
    )

    output_file.write_text(
        report_html,
        encoding="utf-8"
    )

    logger.info(
        f"Report saved: {output_file}"
    )

    return output_file


def main():

    (
        predictions,
        dma_performance,
        feature_importance,
        national,
        metrics
    ) = load_artifacts()

    figures = generate_figures(
        predictions,
        feature_importance,
        dma_performance,
        
    )

    generate_html_report(
        predictions,
        dma_performance,
        feature_importance,
        national,
        metrics,
        figures
    )


if __name__ == "__main__":

    main()







