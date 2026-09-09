"""
07_panel_regression.py

Panel Regression Analysis
-------------------------

Purpose
-------
After controlling for DMA-level differences and time effects, is additional marketing spend associated with incremental traffic?.

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

from linearmodels.panel import PanelOLS
from statsmodels.api import add_constant, OLS

from config import (
    PROCESSED_DATA_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    OUTPUT_DIR,
    
)

#=========================================================
# Output Folders
#=========================================================

REPORT_DIR = OUTPUT_DIR / "reports"


def create_output_folders():

    folders = [

        FIGURE_DIR / "panel_regression",

        FIGURE_DIR / "panel_regression" / "coefficients",

        FIGURE_DIR / "panel_regression" / "comparisons",

        FIGURE_DIR / "panel_regression" / "diagnostics",

        FIGURE_DIR / "panel_regression" / "residuals",

        TABLE_DIR / "panel_regression",

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
        / "panel_regression"
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
# Prepare Panel Data
#=========================================================

def prepare_panel_data(panel):
    """
    Prepare panel dataframe for regression modeling.
    """

    logger.info(
        "Preparing panel regression dataset..."
    )

    required_columns = [

        "DMA",
        "Month",
        "Spend",
        "Traffic"

    ]


    missing = [

        col for col in required_columns
        if col not in panel.columns

    ]


    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )


    panel = (

        panel
        .sort_values(
            [
                "DMA",
                "Month"
            ]
        )
        .copy()

    )

    panel = panel.dropna(
        subset=[
            "DMA",
            "Month",
            "Traffic",
            "Spend"
        ]
    )


    # Log transformations
    # Helps reduce skew and interpret elasticity

    panel["Log_Spend"] = np.log1p(
        panel["Spend"]
    )

    panel["Log_Traffic"] = np.log1p(
        panel["Traffic"]
    )


    # Panel index required for linearmodels

    panel = panel.set_index(
        [
            "DMA",
            "Month"
        ]
    )


    logger.info(
        "Panel prepared"
    )


    logger.info(
        f"Shape: {panel.shape}"
    )


    return panel

#=========================================================
# Pooled OLS Regression
#=========================================================

def run_pooled_ols(panel):
    """
    Baseline pooled OLS regression.

    Traffic = Spend + Error
    """

    logger.info(
        "Running pooled OLS model..."
    )


    model_data = panel.reset_index()


    X = model_data[
        [
            "Spend"
        ]
    ]

    y = model_data[
        "Traffic"
    ]


    X = add_constant(X)


    model = OLS(
        y,
        X
    ).fit()


    logger.info(
        model.summary()
    )


    results = pd.DataFrame({

        "Variable":
            model.params.index,

        "Coefficient":
            model.params.values,

        "P_Value":
            model.pvalues.values

    })


    results.to_csv(

        TABLE_DIR /
        "panel_regression" /
        "pooled_ols_coefficients.csv",

        index=False

    )


    return model, results

#=========================================================
# DMA Fixed Effects Model
#=========================================================

def run_dma_fixed_effects(panel):
    """
    Fixed effects regression controlling for DMA differences.
    """

    logger.info(
        "Running DMA fixed effects model..."
    )


    model = PanelOLS(

        dependent=
        panel["Traffic"],

        exog=
        panel[
            [
                "Spend"
            ]
        ],

        entity_effects=True

    )


    results = model.fit(

        cov_type="clustered",

        cluster_entity=True

    )


    logger.info(
        results.summary
    )


    coefficients = pd.DataFrame({

        "Variable":
            results.params.index,

        "Coefficient":
            results.params.values,

        "P_Value":
            results.pvalues.values

    })


    coefficients.to_csv(

        TABLE_DIR /
        "panel_regression" /
        "dma_fixed_effects_coefficients.csv",

        index=False

    )


    return results, coefficients

#=========================================================
# DMA + Time Fixed Effects
#=========================================================

def run_dma_time_fixed_effects(panel):
    """
    Fixed effects regression controlling for DMA
    and month effects.
    """

    logger.info(
        "Running DMA + Time fixed effects model..."
    )


    model = PanelOLS(

        dependent=
        panel["Traffic"],

        exog=
        panel[
            [
                "Spend"
            ]
        ],

        entity_effects=True,

        time_effects=True

    )


    results = model.fit(

        cov_type="clustered",

        cluster_entity=True

    )


    logger.info(
        results.summary
    )


    coefficients = pd.DataFrame({

        "Variable":
            results.params.index,

        "Coefficient":
            results.params.values,

        "P_Value":
            results.pvalues.values

    })


    coefficients.to_csv(

        TABLE_DIR /
        "panel_regression" /
        "dma_time_fixed_effects_coefficients.csv",

        index=False

    )


    return results, coefficients

#=========================================================
# Model Comparison
#=========================================================

def compare_models(
    pooled_model,
    dma_fe_model,
    dma_time_fe_model
):
    """
    Compare regression model results.

    Models:
        1. Pooled OLS
        2. DMA Fixed Effects
        3. DMA + Time Fixed Effects
    """

    logger.info(
        "Comparing regression models..."
    )


    models = {

        "Pooled OLS":
            pooled_model,

        "DMA Fixed Effects":
            dma_fe_model,

        "DMA + Time Fixed Effects":
            dma_time_fe_model

    }


    rows = []


    for name, model in models.items():

        rows.append({

            "Model":
                name,

            "Spend Coefficient":
                model.params.get(
                    "Spend",
                    np.nan
                ),

            "Spend P-Value":
                model.pvalues.get(
                    "Spend",
                    np.nan
                ),

            "R-Squared":

                getattr(
                    model,
                    "rsquared_within",
                    getattr(
                        model,
                        "rsquared",
                        np.nan
                    )
                ),

            "Observations":
                getattr(
                    model,
                    "nobs",
                    np.nan
                )

        })


    comparison = pd.DataFrame(rows)


    comparison.to_csv(

        TABLE_DIR /
        "panel_regression" /
        "model_comparison.csv",

        index=False

    )


    logger.info(
        "Saved model_comparison.csv"
    )


    logger.info(
        "\n%s",
        comparison
    )


    return comparison


#=========================================================
# Plot Model Coefficients
#=========================================================

def plot_model_coefficients(comparison):
    """
    Plot spend coefficients across regression models.
    """

    logger.info(
        "Plotting model coefficients..."
    )


    fig, ax = plt.subplots(
        figsize=(10,6)
    )


    ax.bar(
        comparison["Model"],
        comparison["Spend Coefficient"]
    )


    ax.axhline(
        0,
        color="black",
        linewidth=1
    )


    ax.set_ylabel(
        "Spend Coefficient"
    )


    ax.set_xlabel(
        "Model"
    )


    ax.set_title(
        "Marketing Spend Coefficient Across Models"
    )


    plt.xticks(
        rotation=30,
        ha="right"
    )


    save_plot(
        fig,
        "coefficients",
        "spend_coefficients"
    )

#=========================================================
# Actual vs Predicted Traffic
#=========================================================

def plot_actual_vs_predicted(
    model,
    panel,
    model_name="model"
):
    """
    Plot actual versus predicted traffic.

    Uses PanelOLS predictions and aligns
    them back to the panel observations.
    """

    logger.info(
        f"Plotting actual vs predicted traffic: {model_name}"
    )


    predictions = model.predict(
        fitted=True,
        effects=True
    )

    logger.info(
    f"Prediction rows: {len(predictions)}"
    )

    logger.info(predictions.head())
    logger.info(predictions.describe())

    logger.info(
        f"Panel rows: {len(panel)}"
    )

    logger.info(
        f"Prediction index: {predictions.index}"
    )

    logger.info(
        f"Panel index: {panel.index}"
    )


    # PanelOLS returns predictions indexed by DMA / Month

    pred = (
        predictions["fitted_values"]
        +
        predictions["estimated_effects"]
    )


    results = pd.DataFrame({

        "Predicted":
            pred

    })


    # Attach actual traffic using matching index

    results["Actual"] = (
        panel["Traffic"]
        .reindex(results.index)
    )


    results = results.dropna()


    fig, ax = plt.subplots(
        figsize=(12,5)
    )


    ax.plot(
        np.arange(len(results)),
        results["Actual"].to_numpy(dtype=float),
        label="Actual"
    )


    ax.plot(
        np.arange(len(results)),
        results["Predicted"].to_numpy(dtype=float),
        label="Predicted"
    )


    ax.set_title(
        f"Actual vs Predicted Traffic\n{model_name}"
    )


    ax.set_ylabel(
        "Traffic"
    )


    ax.set_xlabel(
        "DMA-Month Observations"
    )


    ax.legend()


    save_plot(
        fig,
        "comparisons",
        f"actual_vs_predicted_{model_name.replace(' ','_').lower()}"
    )

    results.to_csv(

        TABLE_DIR /
        "panel_regression" /
        "predicted_vs_actual.csv"

    )


    return results

#=========================================================
# Residual Diagnostics
#=========================================================

def plot_residuals(
    prediction_results
):
    """
    Plot regression residuals.
    """

    logger.info(
        "Plotting residual diagnostics..."
    )


    residuals = (

        prediction_results["Actual"]
        -
        prediction_results["Predicted"]

    )


    fig, ax = plt.subplots(
        figsize=(12,5)
    )


    ax.scatter(
        range(len(residuals)),
        residuals
    )


    ax.axhline(
        0,
        linestyle="--"
    )


    ax.set_title(
        "Regression Residuals"
    )


    ax.set_ylabel(
        "Residual"
    )


    ax.set_xlabel(
        "Observation"
    )


    save_plot(
        fig,
        "residuals",
        "residual_plot"
    )

    residuals.to_csv(

        TABLE_DIR /
        "panel_regression" /
        "residuals.csv",

        index=False

    )


    return residuals

#=========================================================
# HTML Summary Cards
#=========================================================

def html_regression_summary_cards(
    comparison
):
    """
    Build summary cards for regression report.
    """

    best_model = comparison.loc[
        comparison["R-Squared"].idxmax()
    ]

    spend_model = comparison.loc[
        comparison["Spend Coefficient"].idxmax()
    ]


    html = f"""

<div class="cards">


<div class="card">

<h3>Best Model</h3>

<p>
{best_model["Model"]}
</p>

</div>



<div class="card">

<h3>Within R²</h3>

<p>
{best_model["R-Squared"]:.3f}
</p>

</div>



<div class="card">

<h3>Spend Coefficient</h3>

<p>
{spend_model["Spend Coefficient"]:.4f}
</p>

</div>



<div class="card">

<h3>Spend P-Value</h3>

<p>
{spend_model["Spend P-Value"]:.4f}
</p>

</div>


</div>

"""

    return html

#=========================================================
# HTML Regression Analysis Sections
#=========================================================

def html_regression_analysis_sections(
    comparison
):
    """
    Build regression analysis sections.
    """

    comparison_html = comparison.to_html(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )


    html = f"""

<div class="section">

<h2>Model Comparison</h2>

<p>

Three regression specifications were evaluated:

<ul>

<li>Pooled OLS</li>

<li>DMA Fixed Effects</li>

<li>DMA + Month Fixed Effects</li>

</ul>

The progression evaluates whether the relationship
between marketing spend and traffic remains after
controlling for market differences and seasonal effects.

</p>


{comparison_html}


</div>



<div class="section">

<h2>Marketing Spend Coefficients</h2>


<div class="figure">

<img src="../figures/panel_regression/coefficients/spend_coefficients.png"
width="900">

</div>


</div>




<div class="section">

<h2>Actual vs Predicted Traffic</h2>


<div class="figure">

<img src="../figures/panel_regression/comparisons/actual_vs_predicted_dma_+_time_fixed_effects.png"
width="900">

</div>


</div>




<div class="section">

<h2>Residual Diagnostics</h2>


<div class="figure">

<img src="../figures/panel_regression/residuals/residual_plot.png"
width="900">

</div>


</div>

"""

    return html

#=========================================================
# HTML Key Findings
#=========================================================

def html_regression_key_findings(
    comparison
):
    """
    Generate automated regression findings.
    """


    best_model = comparison.loc[
        comparison["R-Squared"].idxmax()
    ]


    spend_effect = comparison.loc[
        comparison["Model"]
        ==
        "DMA + Time Fixed Effects"
    ]


    coefficient = (
        spend_effect["Spend Coefficient"]
        .values[0]
        if len(spend_effect)
        else None
    )


    pvalue = (
        spend_effect["Spend P-Value"]
        .values[0]
        if len(spend_effect)
        else None
    )


    significance = (
        "statistically significant"
        if pvalue < 0.05
        else
        "not statistically significant"
    )


    html = f"""

<div class="section">

<h2>Key Findings</h2>

<ul>


<li>

The strongest explanatory model was:

<b>
{best_model["Model"]}
</b>

with a within R² of

<b>
{best_model["R-Squared"]:.3f}
</b>.

</li>



<li>

After controlling for DMA and time effects,
the estimated marketing spend coefficient was:

<b>
{coefficient:.4f}
</b>.

</li>



<li>

The spend relationship was

<b>
{significance}
</b>

with a p-value of

<b>
{pvalue:.4f}
</b>.

</li>



<li>

The regression framework helps separate
incremental marketing effects from differences
caused by market size, geography, and seasonality.

</li>


</ul>


</div>

"""

    return html

#=========================================================
# HTML Header
#=========================================================

def html_header():
    """
    Create HTML report header.
    """

    html = """

<!DOCTYPE html>

<html>

<head>

<title>
Panel Regression Analysis Report
</title>


<style>


body {

    font-family: Arial, sans-serif;
    margin: 40px;

}


h1 {

    color: #333;

}


h2 {

    color: #444;

    border-bottom:
    1px solid #ddd;

    padding-bottom: 5px;

}


.section {

    margin-bottom: 40px;

}


.cards {

    display:flex;

    gap:20px;

    margin-bottom:30px;

}


.card {

    border:1px solid #ddd;

    padding:20px;

    border-radius:8px;

    width:220px;

}


.card h3 {

    margin-top:0;

}


.figure {

    margin-top:20px;

}


table {

    border-collapse:collapse;

    width:100%;

}


th {

    background:#f2f2f2;

}


td, th {

    padding:8px;

    border:1px solid #ddd;

}


.footer {

    margin-top:50px;

    color:#777;

    font-size:12px;

}


</style>


</head>


<body>


<h1>
Marketing Incrementality -
Panel Regression Analysis
</h1>


"""

    return html

#=========================================================
# Generate HTML Report
#=========================================================

def generate_html_report(
    comparison
):
    """
    Generate panel regression HTML report.
    """

    logger.info(
        "Generating regression HTML report..."
    )


    html = ""


    html += html_header()


    html += html_regression_summary_cards(
        comparison
    )


    html += html_regression_analysis_sections(
        comparison
    )


    html += html_regression_key_findings(
        comparison
    )


    html += """

<div class="footer">

<hr>

<p>
Generated automatically by
<b>07_panel_regression.py</b>
</p>

<p>
Marketing Incrementality Project
</p>


</div>


</body>

</html>

"""


    report_path = (
        REPORT_DIR /
        "panel_regression_report.html"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)


    logger.info(
        f"Saved report: {report_path}"
    )


    return report_path


#=========================================================
# Main
#=========================================================
def main():

    logger.info("=" * 70)
    logger.info("Begin Panel Regression Analysis")
    logger.info("=" * 70)


    #=========================================================
    # Load + Validate Panel
    #=========================================================

    create_output_folders()

    panel = load_model_panel()

    validate_panel(panel)

    panel = prepare_panel_data(panel)

    #=========================================================
    # Run Models
    #=========================================================

    pooled_model, pooled_results = run_pooled_ols(panel)

    dma_model, dma_results = run_dma_fixed_effects(panel)

    dma_time_model, dma_time_results = run_dma_time_fixed_effects(panel)

    comparison = compare_models(
        pooled_model,
        dma_model,
        dma_time_model
    )

    #=========================================================
    # Plot Results
    #=========================================================

    plot_model_coefficients(comparison)

    prediction_results = plot_actual_vs_predicted(
        dma_time_model,
        panel,
        model_name="DMA + Time Fixed Effects"
    )

    plot_residuals(prediction_results)

    #=========================================================
    # Generate HTML Report
    #=========================================================
    generate_html_report(comparison)



    logger.info("=" * 70)
    logger.info("Panel Regression Analysis complete.")
    logger.info("=" * 70)



if __name__ == "__main__":
    main()
