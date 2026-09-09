"""
08_difference_in_differences.py

Difference-in-Differences Analysis
----------------------------------

Purpose
-------
Evaluate if traffic changes more in markets exposed to increased marketing compared with markets that were less exposed.

Outputs
-------
Regression results
Treatment effect estimates
Parallel trends diagnostics
Figures
Tables
HTML report
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

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

        FIGURE_DIR / "difference_in_differences",

        FIGURE_DIR / "difference_in_differences" / "parallel_trends",

        FIGURE_DIR / "difference_in_differences" / "effects",

        FIGURE_DIR / "difference_in_differences" / "diagnostics",

        TABLE_DIR / "difference_in_differences",

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
        / "difference_in_differences"
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
# Create Treatment Definition
#=========================================================

def create_treatment_definition(
    panel,
    intervention_date,
    spend_col="Spend",
    treatment_quantile=0.75
):
    """
    Create treatment groups based on the increase
    in marketing spend after the intervention.

    Treatment is assigned to DMAs exhibiting the
    largest increases in spend relative to the
    pre-intervention period.
    """

    logger.info(
        "Creating treatment definition..."
    )

    panel = panel.copy()

    panel["Post"] = (
        panel["Month"] >= intervention_date
    ).astype(int)

    

    #-----------------------------------------------------
    # Average spend before intervention
    #-----------------------------------------------------

    pre = (

        panel.loc[
            panel["Post"] == 0
        ]

        .groupby("DMA")[spend_col]

        .mean()

        .rename("Pre_Spend")

    )

    #-----------------------------------------------------
    # Average spend after intervention
    #-----------------------------------------------------

    post = (

        panel.loc[
            panel["Post"] == 1
        ]

        .groupby("DMA")[spend_col]

        .mean()

        .rename("Post_Spend")

    )

    spend_change = pd.concat(
        [pre, post],
        axis=1
    )

    spend_change = spend_change.dropna()

    spend_change["Spend_Increase"] = (

        spend_change["Post_Spend"]
        -
        spend_change["Pre_Spend"]

    )

    threshold = spend_change[
        "Spend_Increase"
    ].quantile(
        treatment_quantile
    )

    spend_change["Treatment"] = (

        spend_change["Spend_Increase"] >= threshold

    ).astype(int)

    panel = panel.merge(

        spend_change,

        left_on="DMA",

        right_index=True,

        how="left"

    )

    panel["Treatment_Post"] = (
            panel["Treatment"] *
            panel["Post"]
    )

    panel["Spend_Increase_Post"] = (
        panel["Spend_Increase"]
        *
        panel["Post"]
    )

    logger.info(
        f"Treatment threshold: "
        f"{threshold:,.0f}"
    )

    logger.info(
        f"Treated DMAs: "
        f"{spend_change['Treatment'].sum()}"
    )

    logger.info(
        f"Control DMAs: "
        f"{len(spend_change)-spend_change['Treatment'].sum()}"
    )

    spend_change.to_csv(

        TABLE_DIR /
        "difference_in_differences" /
        "treatment_definition.csv"

    )

    return panel


def prepare_panel_index(panel):
    """
    Prepare panel data for PanelOLS.
    """

    panel = panel.copy()

    panel["Month"] = pd.to_datetime(
        panel["Month"]
    )

    return (
        panel
        .set_index(["DMA", "Month"])
        .sort_index()
    )

#=========================================================
# Validate Treatment Definition
#=========================================================

def validate_treatment(panel):
    """
    Validate treatment assignment.
    """

    logger.info("=" * 70)
    logger.info("TREATMENT SUMMARY")
    logger.info("=" * 70)

    summary = (

        panel

        .groupby("Treatment")

        .agg(

            DMAs=("DMA", "nunique"),

            AvgSpend=("Spend", "mean"),

            AvgTraffic=("Traffic", "mean")

        )

    )

    logger.info("\n%s", summary)

    summary.to_csv(

        TABLE_DIR /
        "difference_in_differences" /
        "treatment_summary.csv"

    )

    return summary

#=========================================================
# Difference-in-Differences Model
#=========================================================

def run_difference_in_differences(panel):
    """
    Estimate a Difference-in-Differences model using
    DMA and month fixed effects.

    Treatment and Post indicators are absorbed by the
    fixed effects, so only the interaction term is
    explicitly estimated.

    Returns
    -------
    results : PanelOLSResults

    coefficients : DataFrame
    """

    logger.info(
        "Running Difference-in-Differences model..."
    )

    panel = prepare_panel_index(panel)

    logger.info(
        panel[
            [
                "Traffic",
                "Treatment_Post",
                "Treatment",
                "Post"
            ]
        ].isna().sum()
    )

    model = PanelOLS(

        dependent=
        panel["Traffic"],

        exog=
        panel[
            [
                "Treatment_Post"
            ]
        ],

        entity_effects=True,

        time_effects=True

    )

    results = model.fit(

        cov_type="clustered",

        cluster_entity=True

    )

    logger.info(results.summary)

    coefficients = pd.DataFrame({

        "Variable":
            results.params.index,

        "Coefficient":
            results.params.values,

        "Std_Error":
            results.std_errors.values,

        "T_Statistic":
            results.tstats.values,

        "P_Value":
            results.pvalues.values,

        "Lower_95":
            results.conf_int().iloc[:,0].values,

        "Upper_95":
            results.conf_int().iloc[:,1].values

    })

    coefficients.to_csv(

        TABLE_DIR /
        "difference_in_differences" /
        "did_coefficients.csv",

        index=False

    )



    logger.info(
        "Difference-in-Differences model complete."
    )

    return results, coefficients

#=========================================================
# Run Continuous Spend Difference-in-Differences Model
#=========================================================

def run_continuous_spend_did(panel):
    """
    Estimate Difference-in-Differences using
    continuous spend increase instead of binary treatment.

    Measures incremental traffic impact per dollar
    of increased marketing spend.
    """

    logger.info(
        "Running continuous spend DiD model..."
    )

    panel = prepare_panel_index(panel)

    model = PanelOLS(

        dependent=
        panel["Traffic"],

        exog=
        panel[
            [
                "Spend_Increase_Post"
            ]
        ],

        entity_effects=True,

        time_effects=True

    )


    results = model.fit(

        cov_type="clustered",

        cluster_entity=True

    )


    logger.info(results.summary)


    coefficients = pd.DataFrame({

        "Variable":
            results.params.index,

        "Coefficient":
            results.params.values,

        "Std_Error":
            results.std_errors.values,

        "T_Statistic":
            results.tstats.values,

        "P_Value":
            results.pvalues.values,

        "Lower_95":
            results.conf_int().iloc[:,0].values,

        "Upper_95":
            results.conf_int().iloc[:,1].values

    })


    coefficients.to_csv(

        TABLE_DIR /
        "difference_in_differences" /
        "continuous_spend_did_coefficients.csv",

        index=False

    )


    return results, coefficients

#=========================================================
# Parallel Trends Validation
#=========================================================

def validate_parallel_trends(panel):
    """
    Validate the Difference-in-Differences
    parallel trends assumption.

    Compares average monthly traffic between
    treated and untreated DMAs before and after
    the intervention.
    """

    logger.info(
        "Validating parallel trends assumption..."
    )

    trend_summary = (

        panel

        .groupby(
            [
                "Month",
                "Treatment"
            ]
        )

        .agg(

            Avg_Traffic=(
                "Traffic",
                "mean"
            )

        )

        .reset_index()

    )

    trend_summary["Group"] = np.where(

        trend_summary["Treatment"] == 1,

        "Treatment",

        "Control"

    )

    trend_summary.to_csv(

        TABLE_DIR /
        "difference_in_differences" /
        "parallel_trends.csv",

        index=False

    )

    logger.info(
        "Parallel trends table created."
    )

    return trend_summary

#=========================================================
# Plot Parallel Trends
#=========================================================

def plot_parallel_trends(
    trend_summary,
    panel
    ):
    """
    Plot average monthly traffic for Treatment and Control
    groups to visually assess the parallel trends assumption.
    """

    logger.info(
        "Plotting parallel trends..."
    )

    #-----------------------------------------------------
    # Create monthly summary
    #-----------------------------------------------------

    pre_panel = panel.loc[
        panel["Post"] == 0
    ]


    trend_summary = (

        pre_panel

        .groupby(
            ["Month", "Treatment"]
        )["Traffic"]

        .mean()

        .reset_index()

    )

    trend_summary["Group"] = np.where(

        trend_summary["Treatment"] == 1,

        "Treatment",

        "Control"

    )

    #-----------------------------------------------------
    # Treatment start date
    #-----------------------------------------------------

    treatment_start = (

        panel.loc[
            panel["Post"] == 1,
            "Month"
        ]
        .min()

    )

    #-----------------------------------------------------
    # Plot
    #-----------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12,6)
    )

    colors = {

        "Treatment": "tab:blue",
        "Control": "tab:orange"

    }

    for group in ["Treatment", "Control"]:

        subset = trend_summary[
            trend_summary["Group"] == group
        ]

        ax.plot(

            subset["Month"],

            subset["Traffic"],

            marker="o",

            linewidth=2,

            markersize=6,

            color=colors[group],

            label=group

        )

    #-----------------------------------------------------
    # Treatment start
    #-----------------------------------------------------

    ax.axvline(

        treatment_start,

        color="black",

        linestyle="--",

        linewidth=2,

        label="Treatment Begins"

    )

    ax.text(

        treatment_start,

        ax.get_ylim()[1] * 0.98,

        "Treatment",

        rotation=90,

        va="top",

        ha="right",

        fontsize=9

    )

    #-----------------------------------------------------
    # Formatting
    #-----------------------------------------------------

    ax.set_title(
        "Parallel Trends Validation"
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "Average Monthly Traffic"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    fig.autofmt_xdate()

    save_plot(

        fig,

        "diagnostics",

        "parallel_trends"

    )

    return trend_summary

#=========================================================
# Parallel Trend Statistics
#=========================================================

def calculate_parallel_trend_stat(panel):
    """
    Quantify the parallel trends assumption by comparing
    the pre-treatment trend between Treatment and Control
    groups.

    Returns
    -------
    pd.DataFrame
        Summary of pre-treatment slopes.
    """

    logger.info(
        "Calculating parallel trend statistics..."
    )

    #-----------------------------------------------------
    # Pre-treatment observations only
    #-----------------------------------------------------

    pre = panel.loc[
        panel["Post"] == 0
    ].copy()

    pre["Month_Index"] = (

        pre["Month"]

        - pre["Month"].min()

    ).dt.days

    #-----------------------------------------------------
    # Average traffic by month
    #-----------------------------------------------------

    summary = (

        pre

        .groupby(
            ["Month_Index", "Treatment"]
        )["Traffic"]

        .mean()

        .reset_index()

    )

    results = []

    #-----------------------------------------------------
    # Fit linear trend
    #-----------------------------------------------------

    for treatment in [0, 1]:

        subset = summary[
            summary["Treatment"] == treatment
        ]

        if len(subset) < 2:

            slope = np.nan
            intercept = np.nan

        else:

            slope, intercept = np.polyfit(

                subset["Month_Index"],

                subset["Traffic"],

                1

            )

        results.append({

            "Group":
                "Treatment" if treatment else "Control",

            "Slope":
                slope,

            "Intercept":
                intercept

        })

    results = pd.DataFrame(results)

    results["Pre_Period_Difference"] = (

        results["Slope"]

        -

        results["Slope"].shift()

    )

    results.to_csv(

        TABLE_DIR /
        "difference_in_differences" /
        "parallel_trend_statistics.csv",

        index=False

    )

    logger.info(
        "\n%s",
        results
    )

    return results

#=========================================================
# Treatment Effect Summary
#=========================================================

def summarize_treatment_effect(panel):
    """
    Summarize average traffic before and after treatment
    for Treatment and Control groups.

    Also calculates the manual
    Difference-in-Differences estimate.

    Returns
    -------
    summary : pd.DataFrame
        Traffic summary by group and period

    effect_summary : pd.DataFrame
        DiD calculation summary
    """

    logger.info(
        "Summarizing treatment effects..."
    )


    summary = (

        panel

        .groupby(
            [
                "Treatment",
                "Post"
            ]
        )

        .agg(

            Average_Traffic=(
                "Traffic",
                "mean"
            ),

            Std_Dev=(
                "Traffic",
                "std"
            ),

            Observations=(
                "Traffic",
                "count"
            ),

            Avg_Spend_Increase=(
                "Spend_Increase",
                "mean"
            )

        )

        .reset_index()

    )


    summary["Group"] = summary["Treatment"].map(
        {
            1:"Treatment",
            0:"Control"
        }
    )


    summary["Period"] = summary["Post"].map(
        {
            1:"Post",
            0:"Pre"
        }
    )


    summary = summary[
        [
            "Group",
            "Period",
            "Average_Traffic",
            "Std_Dev",
            "Observations",
            "Avg_Spend_Increase"
        ]
    ]


    #=====================================================
    # Manual Difference-in-Differences Calculation
    #=====================================================

    treatment_pre = summary.loc[
        (summary["Group"]=="Treatment") &
        (summary["Period"]=="Pre"),
        "Average_Traffic"
    ].iloc[0]


    treatment_post = summary.loc[
        (summary["Group"]=="Treatment") &
        (summary["Period"]=="Post"),
        "Average_Traffic"
    ].iloc[0]


    control_pre = summary.loc[
        (summary["Group"]=="Control") &
        (summary["Period"]=="Pre"),
        "Average_Traffic"
    ].iloc[0]


    control_post = summary.loc[
        (summary["Group"]=="Control") &
        (summary["Period"]=="Post"),
        "Average_Traffic"
    ].iloc[0]


    treatment_change = (
        treatment_post -
        treatment_pre
    )


    control_change = (
        control_post -
        control_pre
    )


    did_effect = (
        treatment_change -
        control_change
    )


    effect_summary = pd.DataFrame({

        "Metric":[
            "Treatment Change",
            "Control Change",
            "Difference-in-Differences"
        ],

        "Value":[
            treatment_change,
            control_change,
            did_effect
        ]

    })


    #=====================================================
    # Export
    #=====================================================

    summary.to_csv(
        TABLE_DIR /
        "difference_in_differences" /
        "treatment_summary.csv",
        index=False
    )


    effect_summary.to_csv(
        TABLE_DIR /
        "difference_in_differences" /
        "did_effect_summary.csv",
        index=False
    )


    logger.info(
        "\n%s",
        effect_summary
    )


    return summary, effect_summary

#=========================================================
# Treatment Effect Plot
#=========================================================

def plot_treatment_effect(panel):
    """
    Plot average monthly traffic for Treatment and Control
    groups across the analysis period.
    """

    logger.info(
        "Plotting treatment effect..."
    )

    monthly = (

        panel

        .groupby(
            ["Month","Treatment"]
        )["Traffic"]

        .mean()

        .reset_index()

    )

    monthly["Group"] = np.where(

        monthly["Treatment"] == 1,

        "Treatment",

        "Control"

    )

    treatment_start = (

        panel.loc[
            panel["Post"] == 1,
            "Month"
        ].min()

    )

    fig, ax = plt.subplots(
        figsize=(12,6)
    )

    colors = {

        "Treatment":"tab:blue",

        "Control":"tab:orange"

    }

    for group in ["Treatment","Control"]:

        subset = monthly[
            monthly["Group"] == group
        ]

        ax.plot(

            subset["Month"],

            subset["Traffic"],

            marker="o",

            linewidth=2,

            markersize=6,

            color=colors[group],

            label=group

        )

    ax.axvline(

        treatment_start,

        color="black",

        linestyle="--",

        linewidth=2,

        label="Treatment"

    )

    ax.set_title(
        "Treatment vs Control Traffic"
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "Average Traffic"
    )

    ax.grid(alpha=0.30)

    ax.legend()

    fig.autofmt_xdate()

    save_plot(

        fig,

        "effects",

        "treatment_vs_control"

    )

    return monthly

#=========================================================
# HTML Header
#=========================================================

def html_header():
    """
    HTML report header and styling.
    """

    return """

<html>

<head>

<title>
Difference-in-Differences Report
</title>

<style>

body{
    font-family:Arial;
    margin:40px;
    background:#fafafa;
}

h1{
    color:#1f4e79;
}

h2{
    border-bottom:2px solid #dddddd;
    padding-bottom:6px;
}

table{
    border-collapse:collapse;
    width:100%;
    margin-top:15px;
    margin-bottom:30px;
}

th{
    background:#1f4e79;
    color:white;
    padding:8px;
}

td{
    border:1px solid #dddddd;
    padding:8px;
}

.card{

    display:inline-block;

    width:220px;

    margin:10px;

    padding:15px;

    border:1px solid #cccccc;

    border-radius:8px;

    background:white;

    text-align:center;

}

.figure{

    text-align:center;

    margin-top:20px;

    margin-bottom:35px;

}

.footer{

    color:gray;

    font-size:12px;

}

</style>

</head>

<body>

<h1>

Difference-in-Differences Analysis

</h1>

<p>

Marketing Incrementality Project

</p>

<hr>

"""

#=========================================================
# HTML Summary Cards
#=========================================================

def html_summary_cards(
    panel,
    did_results,
    continuous_results,
    parallel_stats,
    treatment_summary
):
    """
    Summary dashboard cards.
    """

    treatment_dmas = (
        panel.loc[
            panel["Treatment"] == 1,
            "DMA"
        ]
        .nunique()
    )


    control_dmas = (
        panel.loc[
            panel["Treatment"] == 0,
            "DMA"
        ]
        .nunique()
    )


    treatment_month = (
        panel.loc[
            panel["Post"] == 1,
            "Month"
        ]
        .min()
        .strftime("%Y-%m")
    )


    did_coef = (
        did_results.params[
            "Treatment_Post"
        ]
    )

    did_p = (
        did_results.pvalues[
            "Treatment_Post"
        ]
    )


    spend_coef = (
        continuous_results.params[
            "Spend_Increase_Post"
        ]
    )

    spend_p = (
        continuous_results.pvalues[
            "Spend_Increase_Post"
        ]
    )


    html = f"""

<h2>
Summary
</h2>


<div class="card">

<h3>Treated DMAs</h3>

<h1>{treatment_dmas}</h1>

</div>


<div class="card">

<h3>Control DMAs</h3>

<h1>{control_dmas}</h1>

</div>


<div class="card">

<h3>Intervention</h3>

<h1>{treatment_month}</h1>

</div>


<div class="card">

<h3>Binary DiD Effect</h3>

<h1>{did_coef:,.0f}</h1>

<p>
p={did_p:.4f}
</p>

</div>


<div class="card">

<h3>Spend DiD Effect</h3>

<h1>{spend_coef:,.2f}</h1>

<p>
Traffic visits per additional dollar
<br>
p={spend_p:.4f}
</p>

</div>


<hr>

"""

    return html

#=========================================================
# HTML Analysis Sections
#=========================================================

def html_analysis_sections(
    did_results,
    trend_summary
):
    """
    Build HTML analysis sections.
    """

    coefficient = did_results.params["Treatment_Post"]
    pvalue = did_results.pvalues["Treatment_Post"]

    html = f"""

<div class="section">

<h2>Difference-in-Differences Results</h2>

<p>

The Difference-in-Differences model estimates the incremental
change in traffic attributable to the marketing intervention
after controlling for baseline differences between treatment
and control DMAs.

</p>

<table>

<tr>
<th>Metric</th>
<th>Value</th>
</tr>

<tr>
<td>Treatment Effect</td>
<td>{coefficient:,.3f}</td>
</tr>

<tr>
<td>P-Value</td>
<td>{pvalue:.4f}</td>
</tr>

</table>

</div>


<div class="section">

<h2>Parallel Trends Validation</h2>

<p>

A key assumption of Difference-in-Differences is that the
treatment and control groups followed similar trends before
the intervention.

</p>

<div class="figure">

<img src="../figures/difference_in_differences/diagnostics/parallel_trends.png"
width="900">

</div>

</div>


<div class="section">

<h2>Treatment Effect Visualization</h2>

<div class="figure">

<img src="../figures/difference_in_differences/effects/treatment_vs_control.png"
width="900">

</div>

</div>

"""

    return html

#=========================================================
# HTML Key Findings
#=========================================================

def html_key_findings(
    did_results,
    parallel_stats,
    did_effect_summary,
    treatment_summary
):
    """
    Generate narrative findings.
    """

    effect = did_results.params[
        "Treatment_Post"
    ]

    pvalue = did_results.pvalues[
        "Treatment_Post"
    ]


    #---------------------------------------------
    # Parallel trends diagnostic
    #---------------------------------------------

    trend_difference = (
        parallel_stats
        .loc[
            parallel_stats[
                "Pre_Period_Difference"
            ].notna(),
            "Pre_Period_Difference"
        ]
        .iloc[0]
    )


    #---------------------------------------------
    # Manual DiD components
    #---------------------------------------------

    treatment_change = (
        did_effect_summary
        .loc[
            did_effect_summary["Metric"]
            ==
            "Treatment Change",
            "Value"
        ]
        .iloc[0]
    )


    control_change = (
        did_effect_summary
        .loc[
            did_effect_summary["Metric"]
            ==
            "Control Change",
            "Value"
        ]
        .iloc[0]
    )


    did_manual = (
        did_effect_summary
        .loc[
            did_effect_summary["Metric"]
            ==
            "Difference-in-Differences",
            "Value"
        ]
        .iloc[0]
    )


    #---------------------------------------------
    # Baseline traffic difference
    #---------------------------------------------

    treatment_pre = (
        treatment_summary
        .loc[
            (treatment_summary["Group"]=="Treatment")
            &
            (treatment_summary["Period"]=="Pre"),
            "Average_Traffic"
        ]
        .iloc[0]
    )


    control_pre = (
        treatment_summary
        .loc[
            (treatment_summary["Group"]=="Control")
            &
            (treatment_summary["Period"]=="Pre"),
            "Average_Traffic"
        ]
        .iloc[0]
    )


    html = f"""

<div class="section">

<h2>Key Findings</h2>


<ul>


<li>

The Difference-in-Differences model estimated a post-treatment
effect of

<b>{effect:,.0f}</b>

average traffic visits for treated DMAs.

</li>


<li>

The estimated treatment effect was statistically significant
with a p-value of

<b>{pvalue:.4f}</b>.

However, statistical significance should be interpreted
alongside the underlying market trends and treatment design.

</li>


<li>

Before the intervention, Treatment DMAs averaged

<b>{treatment_pre:,.0f}</b>

monthly visits compared with

<b>{control_pre:,.0f}</b>

monthly visits for Control DMAs.

This indicates that treated markets were materially different
from control markets before the spend increase occurred.

</li>


<li>

The pre-treatment traffic difference between groups was

<b>{trend_difference:,.0f}</b>

visits, suggesting the Treatment and Control groups did not
begin from identical baseline conditions.

</li>


<li>

Following intervention:

<ul>

<li>

Treatment DMAs changed by

<b>{treatment_change:,.0f}</b>

visits.

</li>

<li>

Control DMAs changed by

<b>{control_change:,.0f}</b>

visits.

</li>

<li>

The resulting Difference-in-Differences estimate was

<b>{did_manual:,.0f}</b>

visits.

</li>

</ul>

</li>


<li>

Because Treatment DMAs experienced higher baseline traffic
and different pre-treatment trends, the estimated DiD effect
may partially reflect underlying market differences rather
than only incremental marketing impact.

</li>


</ul>


<h3>Assumptions & Limitations</h3>


<ul>


<li>

The Difference-in-Differences approach assumes that treated
and control DMAs would have followed similar trends in the
absence of the intervention.

</li>


<li>

The observed pre-treatment trend differences indicate this
assumption may not fully hold, reducing confidence in the
causal interpretation.

</li>


<li>

The analysis estimates the Average Treatment Effect on the
Treated (ATT), meaning the estimated impact applies only to
the treated DMA population.

</li>


<li>

The upcoming Bayesian Structural Time Series (BSTS) model will
provide an alternative counterfactual estimate that does not
require identifying untreated DMA markets as controls.

</li>


</ul>


</div>

"""

    return html

#=========================================================
# Generate HTML Report
#=========================================================

def generate_html_report(
    panel,
    did_results,
    trend_summary,
    parallel_stats,
    treatment_summary,
    continuous_results,
    did_effect_summary
):
    """
    Generate Difference-in-Differences HTML report.
    """

    logger.info(
        "Generating Difference-in-Differences HTML report..."
    )

    html = ""

    html += html_header()

    html += html_summary_cards(
        panel,
        did_results,
        continuous_results,
        parallel_stats,
        treatment_summary
    )

    html += html_analysis_sections(
        did_results,
        trend_summary
    )

    html += html_key_findings(
        did_results,
        parallel_stats,
        did_effect_summary,
        treatment_summary
    )

    html += """

<hr>

<p>

Generated automatically by
<b>08_difference_in_differences.py</b>

</p>

</body>

</html>

"""

    report_path = (
        REPORT_DIR /
        "difference_in_differences_report.html"
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

#=========================================================
# Main
#=========================================================

def main():

    logger.info("=" * 70)
    logger.info("Begin Difference-in-Differences Analysis")
    logger.info("=" * 70)

    intervention_date = pd.Timestamp("2026-01-01")



    #=====================================================
    # Load + Validate
    #=====================================================

    panel = load_model_panel()

    validate_panel(panel)

    #=====================================================
    # Treatment Definition
    #=====================================================

    logger.info(
        "Creating treatment variables..."
    )

    panel = create_treatment_definition(
        panel, intervention_date=intervention_date
    )

    #=====================================================
    # Difference-in-Differences Model
    #=====================================================

    logger.info(
        "Running Difference-in-Differences model..."
    )

    did_results, coefficients = (
        run_difference_in_differences(panel)
    )

    did_results, coefficients = (
        run_difference_in_differences(panel)
    )

    continuous_results, continuous_coefficients = (
        run_continuous_spend_did(panel)
    )

    #=====================================================
    # Parallel Trends Validation
    #=====================================================

    logger.info(
        "Validating parallel trends..."
    )

    trend_summary = validate_parallel_trends(
        panel
    )

    parallel_stats = (
        calculate_parallel_trend_stat(
            panel
        )
    )

    #=====================================================
    # Diagnostics
    #=====================================================

    logger.info(
        "Creating diagnostic visualizations..."
    )

    plot_parallel_trends(
        trend_summary,
        panel
    )

    plot_treatment_effect(
        panel
    )

    #=====================================================
    # Model Summary
    #=====================================================

    logger.info(
        "Exporting treatment summary..."
    )

    treatment_summary, did_effect_summary = (
        summarize_treatment_effect(panel)
    )

    #=====================================================
    # HTML Report
    #=====================================================

    generate_html_report(

        panel=panel,

        did_results=did_results,

        continuous_results=continuous_results,

        trend_summary=trend_summary,

        parallel_stats=parallel_stats,

        treatment_summary=treatment_summary,

        did_effect_summary=did_effect_summary

    )

    logger.info("=" * 70)
    logger.info("Difference-in-Differences Analysis complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    create_output_folders()
    main()

