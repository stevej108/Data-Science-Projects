"""
12_reporting.py

Final Marketing Incrementality Reporting Layer

Purpose
-------
Assemble the outputs from the analytical pipeline into a single,
self-contained HTML report.

The report combines:
    1. Data / EDA context
    2. Cross-correlation evidence
    3. Panel regression / DiD context
    4. BSTS model validation
    5. Counterfactual validation
    6. Aggregate causal impact
    7. DMA-level heterogeneity
    8. Placebo robustness
    9. Marketing spend robustness
   10. Limitations and analytical conclusion

This script does not fit models or alter analytical results.
It reads persisted tables/figures produced by scripts 01-11.
"""

# =========================================================
# Imports
# =========================================================

import base64
import html
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    FIGURE_DIR,
    TABLE_DIR,
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
)


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# Reporting Paths
# =========================================================

REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_FIGURE_DIR = FIGURE_DIR / "reporting"

CAUSAL_TABLE_DIR = TABLE_DIR / "causal_impact"
ROBUSTNESS_TABLE_DIR = TABLE_DIR / "causal_impact_robustness"


# =========================================================
# Generic Helpers
# =========================================================

def ensure_output_dirs() -> None:
    """Create reporting output directories."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_safe(
    path: Path,
    required: bool = False,
) -> Optional[pd.DataFrame]:
    """Read a CSV with consistent logging and optional failure."""

    if not path.exists():
        message = f"Reporting input not found: {path}"
        if required:
            raise FileNotFoundError(message)
        logger.warning(message)
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        if required:
            raise
        logger.warning(
            "Unable to read reporting input %s: %s",
            path,
            exc,
        )
        return None

    logger.info(
        "Loaded %s (%s rows, %s columns)",
        path.name,
        len(df),
        len(df.columns),
    )

    return df


def fmt_int(value) -> str:
    """Format an integer-like value for report prose."""

    if value is None or pd.isna(value):
        return "—"

    return f"{value:,.0f}"


def fmt_pct(value, digits: int = 1) -> str:
    """Format a decimal as a percentage."""

    if value is None or pd.isna(value):
        return "—"

    return f"{value:.{digits}%}"


def fmt_num(value, digits: int = 2) -> str:
    """Format a numeric value."""

    if value is None or pd.isna(value):
        return "—"

    return f"{value:,.{digits}f}"


def safe_path_name(text: str) -> str:
    """Create a filesystem-safe report asset name."""

    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def image_to_data_uri(path: Path) -> str:
    """Embed an image as a base64 data URI."""

    if not path.exists():
        return ""

    mime = "image/png"

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return f"data:{mime};base64,{encoded}"


def html_table(
    df: Optional[pd.DataFrame],
    max_rows: int = 20,
    round_digits: int = 3,
) -> str:
    """Render a compact HTML table."""

    if df is None or df.empty:
        return '<p class="muted">No table available.</p>'

    display = df.head(max_rows).copy()

    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].round(round_digits)

    return display.to_html(
        index=False,
        classes="data-table",
        border=0,
        escape=True,
    )


def metric_card(
    label: str,
    value: str,
    detail: str = "",
) -> str:
    """Return HTML for a summary metric card."""

    return f"""
    <div class="metric-card">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value">{html.escape(value)}</div>
        <div class="metric-detail">{html.escape(detail)}</div>
    </div>
    """


def section(
    title: str,
    body: str,
    anchor: str,
) -> str:
    """Return a report section."""

    return f"""
    <section id="{html.escape(anchor)}">
        <h2>{html.escape(title)}</h2>
        {body}
    </section>
    """


def figure_html(
    path: Path,
    caption: str,
    width: str = "100%",
) -> str:
    """Embed a report figure."""

    uri = image_to_data_uri(path)

    if not uri:
        return ""

    return f"""
    <figure>
        <img src="{uri}" alt="{html.escape(caption)}" style="width:{width};">
        <figcaption>{html.escape(caption)}</figcaption>
    </figure>
    """


# =========================================================
# Data Loading
# =========================================================

def load_reporting_inputs() -> dict:
    """Load all persisted analytical outputs used in reporting."""

    inputs = {}

    # -----------------------------------------------------
    # Core processed data
    # -----------------------------------------------------

    panel_path = PROCESSED_DATA_DIR / "model_panel.parquet"

    if panel_path.exists():
        inputs["panel"] = pd.read_parquet(panel_path)
        inputs["panel"]["Month"] = pd.to_datetime(
            inputs["panel"]["Month"]
        )
    else:
        logger.warning("Model panel not found: %s", panel_path)

    # -----------------------------------------------------
    # EDA
    # -----------------------------------------------------

    eda_dir = TABLE_DIR / "eda"

    inputs["dataset_overview"] = read_csv_safe(
        eda_dir / "dataset_overview.csv"
    )
    inputs["panel_summary"] = read_csv_safe(
        eda_dir / "panel_summary_statistics.csv"
    )
    inputs["dma_summary"] = read_csv_safe(
        eda_dir / "dma_summary.csv"
    )
    inputs["relationship_statistics"] = read_csv_safe(
        eda_dir / "relationship_statistics.csv"
    )
    inputs["traffic_statistics"] = read_csv_safe(
        eda_dir / "traffic_statistics.csv"
    )
    inputs["spend_statistics"] = read_csv_safe(
        eda_dir / "spend_statistics.csv"
    )

    # -----------------------------------------------------
    # Cross-correlation
    # -----------------------------------------------------

    ccf_dir = TABLE_DIR / "cross_correlation"

    inputs["ccf_comparison"] = read_csv_safe(
        ccf_dir / "ccf_comparison.csv"
    )
    inputs["ccf_correlation_summary"] = read_csv_safe(
        ccf_dir / "correlation_summary.csv"
    )
    inputs["raw_ccf_summary"] = read_csv_safe(
        ccf_dir / "raw_ccf_summary.csv"
    )
    inputs["detrended_ccf_summary"] = read_csv_safe(
        ccf_dir / "detrended_ccf_summary.csv"
    )
    inputs["dma_correlations"] = read_csv_safe(
        ccf_dir / "dma_correlations.csv"
    )

    # -----------------------------------------------------
    # Panel regression
    # -----------------------------------------------------

    regression_dir = TABLE_DIR / "panel_regression"

    inputs["regression_comparison"] = read_csv_safe(
        regression_dir / "model_comparison.csv"
    )
    inputs["dma_time_coefficients"] = read_csv_safe(
        regression_dir / "dma_time_fixed_effects_coefficients.csv"
    )
    inputs["predicted_vs_actual"] = read_csv_safe(
        regression_dir / "predicted_vs_actual.csv"
    )

    # -----------------------------------------------------
    # Difference-in-differences
    # -----------------------------------------------------

    did_dir = TABLE_DIR / "difference_in_differences"

    inputs["did_effect_summary"] = read_csv_safe(
        did_dir / "did_effect_summary.csv"
    )
    inputs["did_coefficients"] = read_csv_safe(
        did_dir / "did_coefficients.csv"
    )
    inputs["continuous_spend_did_coefficients"] = read_csv_safe(
        did_dir / "continuous_spend_did_coefficients.csv"
    )
    inputs["parallel_trend_statistics"] = read_csv_safe(
        did_dir / "parallel_trend_statistics.csv"
    )

    # -----------------------------------------------------
    # BSTS / causal impact
    # -----------------------------------------------------

    inputs["bsts_validation"] = read_csv_safe(
        CAUSAL_TABLE_DIR /
        "bsts_model_validation_summary.csv",
        required=True,
    )

    inputs["counterfactual_validation"] = read_csv_safe(
        CAUSAL_TABLE_DIR /
        "counterfactual_validation_summary.csv",
        required=True,
    )

    inputs["dma_impact"] = read_csv_safe(
        CAUSAL_TABLE_DIR /
        "dma_impact_summary.csv",
        required=True,
    )

    inputs["dma_impact_analysis"] = read_csv_safe(
        CAUSAL_TABLE_DIR /
        "dma_impact_analysis.csv"
    )

    inputs["aggregate_impact"] = read_csv_safe(
        CAUSAL_TABLE_DIR /
        "aggregate_impact_summary.csv",
        required=True,
    )

    inputs["dma_contribution"] = read_csv_safe(
        CAUSAL_TABLE_DIR /
        "dma_contribution_analysis.csv"
    )

    inputs["dma_contribution_summary"] = read_csv_safe(
        CAUSAL_TABLE_DIR /
        "dma_contribution_summary.csv"
    )

    # -----------------------------------------------------
    # Robustness
    # -----------------------------------------------------

    heterogeneity_dir = ROBUSTNESS_TABLE_DIR / "heterogeneity"
    placebo_dir = ROBUSTNESS_TABLE_DIR / "placebo"
    spend_dir = ROBUSTNESS_TABLE_DIR / "marketing_spend"

    inputs["heterogeneity"] = read_csv_safe(
        heterogeneity_dir / "dma_heterogeneity_analysis.csv"
    )

    inputs["heterogeneity_summary"] = read_csv_safe(
        heterogeneity_dir / "dma_heterogeneity_summary.csv"
    )

    inputs["placebo_results"] = read_csv_safe(
        placebo_dir / "placebo_results.csv"
    )

    inputs["placebo_summary"] = read_csv_safe(
        placebo_dir / "placebo_summary.csv"
    )

    inputs["placebo_validation"] = read_csv_safe(
        placebo_dir / "placebo_counterfactual_validation.csv"
    )

    inputs["spend_impact"] = read_csv_safe(
        spend_dir / "dma_spend_impact_analysis.csv"
    )

    inputs["spend_relationships"] = read_csv_safe(
        spend_dir / "spend_impact_relationships.csv"
    )

    inputs["spend_direction"] = read_csv_safe(
        spend_dir / "spend_direction_summary.csv"
    )

    inputs["spend_aggregate"] = read_csv_safe(
        spend_dir / "spend_aggregate_summary.csv"
    )

    return inputs


# =========================================================
# Summary Calculations
# =========================================================

def calculate_report_metrics(inputs: dict) -> dict:
    """Derive high-level report metrics without changing analytical outputs."""

    metrics = {}

    panel = inputs.get("panel")
    aggregate = inputs.get("aggregate_impact")
    dma_impact = inputs.get("dma_impact")
    bsts = inputs.get("bsts_validation")
    counterfactual = inputs.get("counterfactual_validation")
    placebo = inputs.get("placebo_summary")
    spend = inputs.get("spend_aggregate")
    spend_relationships = inputs.get("spend_relationships")
    heterogeneity = inputs.get("heterogeneity")

    if panel is not None and not panel.empty:
        metrics["panel_rows"] = len(panel)
        metrics["panel_dmas"] = panel["DMA"].nunique()
        metrics["panel_months"] = panel["Month"].nunique()
        metrics["panel_start"] = panel["Month"].min()
        metrics["panel_end"] = panel["Month"].max()

    if aggregate is not None and not aggregate.empty:
        row = aggregate.iloc[0]
        metrics["aggregate_incremental"] = row.get(
            "Incremental_Traffic_Mean"
        )
        metrics["aggregate_lower"] = row.get(
            "Impact_Lower_95"
        )
        metrics["aggregate_upper"] = row.get(
            "Impact_Upper_95"
        )
        metrics["aggregate_prob_positive"] = row.get(
            "Probability_Positive"
        )
        metrics["aggregate_lift"] = row.get(
            "Relative_Lift_Mean"
        )
        metrics["aggregate_dmas"] = row.get(
            "DMAs_Included"
        )

    if dma_impact is not None and not dma_impact.empty:
        metrics["positive_dma_count"] = int(
            (dma_impact["Incremental_Traffic_Mean"] > 0).sum()
        )
        metrics["negative_dma_count"] = int(
            (dma_impact["Incremental_Traffic_Mean"] < 0).sum()
        )

    if bsts is not None and not bsts.empty:
        metrics["bsts_pass_count"] = int(
            bsts["converged"].sum()
        ) if "converged" in bsts.columns else np.nan
        metrics["bsts_count"] = len(bsts)
        metrics["bsts_divergence_count"] = int(
            (bsts["divergences"] > 0).sum()
        ) if "divergences" in bsts.columns else np.nan

    if counterfactual is not None and not counterfactual.empty:
        if "counterfactual_valid" in counterfactual.columns:
            metrics["counterfactual_pass_count"] = int(
                counterfactual["counterfactual_valid"].sum()
            )
            metrics["counterfactual_count"] = len(counterfactual)
        if "pre_r2" in counterfactual.columns:
            metrics["counterfactual_mean_r2"] = float(
                counterfactual["pre_r2"].mean()
            )

    if placebo is not None and not placebo.empty:
        row = placebo.iloc[0]
        metrics["placebo_tests"] = row.get("Placebo_Tests")
        metrics["placebo_positive_rate"] = row.get(
            "Placebo_Positive_Rate"
        )
        metrics["placebo_zero_crossing_rate"] = row.get(
            "Placebo_Zero_Crossing_Rate"
        )
        metrics["placebo_strong_positive_rate"] = row.get(
            "Strong_Positive_Posterior_Rate"
        )
        metrics["placebo_strong_negative_rate"] = row.get(
            "Strong_Negative_Posterior_Rate"
        )

    if spend is not None and not spend.empty:
        row = spend.iloc[0]
        metrics["incremental_spend"] = row.get(
            "Total_Incremental_Post_Spend"
        )
        metrics["spend_aggregate_impact"] = row.get(
            "Total_Incremental_Traffic_Mean"
        )
        metrics["spend_impact_per_1k"] = row.get(
            "Aggregate_Impact_Per_Incremental_Spend_1K"
        )

    if spend_relationships is not None and not spend_relationships.empty:
        row = spend_relationships.loc[
            spend_relationships["Relationship"].eq(
                "Spend Change vs Incremental Traffic"
            )
        ]
        if not row.empty:
            row = row.iloc[0]
            metrics["spend_pearson"] = row.get("Pearson_R")
            metrics["spend_pearson_p"] = row.get("Pearson_P_Value")
            metrics["spend_spearman"] = row.get("Spearman_Rho")

    if heterogeneity is not None and not heterogeneity.empty:
        if "Impact_Excludes_Zero" in heterogeneity.columns:
            metrics["credible_direction_count"] = int(
                heterogeneity["Impact_Excludes_Zero"].sum()
            )
        metrics["heterogeneity_count"] = len(heterogeneity)

    return metrics


# =========================================================
# Report Figures
# =========================================================

def build_report_figures(inputs: dict) -> dict:
    """Create a small set of final-report visualizations."""

    figure_paths = {}

    # -----------------------------------------------------
    # Aggregate impact
    # -----------------------------------------------------

    dma_impact = inputs.get("dma_impact")

    if dma_impact is not None and not dma_impact.empty:

        df = dma_impact.copy()
        df = df.sort_values(
            "Incremental_Traffic_Mean",
            ascending=True,
        )

        fig, ax = plt.subplots(
            figsize=(11, 6),
        )

        y = np.arange(len(df))
        x = df["Incremental_Traffic_Mean"].to_numpy()
        lower = df["Impact_Lower_95"].to_numpy()
        upper = df["Impact_Upper_95"].to_numpy()

        ax.errorbar(
            x,
            y,
            xerr=[
                x - lower,
                upper - x,
            ],
            fmt="o",
            capsize=4,
        )

        ax.axvline(
            0,
            linewidth=1.2,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(df["DMA"])
        ax.set_xlabel("Estimated incremental traffic")
        ax.set_title(
            "DMA-level BSTS impact with 95% credible intervals"
        )
        ax.grid(
            axis="x",
            alpha=0.25,
        )
        fig.tight_layout()

        path = REPORT_FIGURE_DIR / "dma_impact_intervals.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figure_paths["dma_impact"] = path

    # -----------------------------------------------------
    # Spend vs impact
    # -----------------------------------------------------

    spend = inputs.get("spend_impact")

    if spend is not None and not spend.empty:

        df = spend.copy()

        fig, ax = plt.subplots(
            figsize=(9, 6),
        )

        ax.scatter(
            df["Monthly_Spend_Change"],
            df["Incremental_Traffic_Mean"],
            s=70,
        )

        for _, row in df.iterrows():
            ax.annotate(
                str(row["DMA"]),
                (
                    row["Monthly_Spend_Change"],
                    row["Incremental_Traffic_Mean"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        ax.axhline(0, linewidth=1.0)
        ax.axvline(0, linewidth=1.0)
        ax.set_xlabel("Monthly spend change")
        ax.set_ylabel("Estimated incremental traffic")
        ax.set_title(
            "Marketing spend change vs. estimated traffic impact"
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()

        path = REPORT_FIGURE_DIR / "spend_vs_impact.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figure_paths["spend_impact"] = path

    # -----------------------------------------------------
    # Spend efficiency
    # -----------------------------------------------------

    if spend is not None and not spend.empty:

        df = spend.sort_values(
            "Impact_Per_Incremental_Spend_1K",
            ascending=True,
        )

        fig, ax = plt.subplots(
            figsize=(11, 6),
        )

        ax.barh(
            df["DMA"],
            df["Impact_Per_Incremental_Spend_1K"],
        )

        ax.axvline(0, linewidth=1.0)
        ax.set_xlabel(
            "Estimated incremental traffic per $1K incremental spend"
        )
        ax.set_title(
            "Descriptive spend-efficiency comparison by DMA"
        )
        ax.grid(
            axis="x",
            alpha=0.25,
        )
        fig.tight_layout()

        path = REPORT_FIGURE_DIR / "spend_efficiency.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figure_paths["spend_efficiency"] = path

    # -----------------------------------------------------
    # Placebo distribution
    # -----------------------------------------------------

    placebo = inputs.get("placebo_results")

    if placebo is not None and not placebo.empty:

        values = pd.to_numeric(
            placebo["Incremental_Traffic_Mean"],
            errors="coerce",
        ).dropna()

        if not values.empty:

            fig, ax = plt.subplots(
                figsize=(9, 5.5),
            )

            ax.hist(
                values,
                bins=min(20, max(5, len(values) // 5)),
                alpha=0.8,
            )

            ax.axvline(0, linewidth=1.0)

            aggregate = inputs.get("aggregate_impact")
            if aggregate is not None and not aggregate.empty:
                actual = aggregate.iloc[0].get(
                    "Incremental_Traffic_Mean"
                )
                if actual is not None and pd.notna(actual):
                    ax.axvline(
                        actual,
                        linewidth=1.5,
                        linestyle="--",
                    )
                    ax.text(
                        actual,
                        ax.get_ylim()[1] * 0.92,
                        "Actual aggregate impact",
                        rotation=90,
                        va="top",
                        ha="right",
                    )

            ax.set_xlabel("Placebo estimated incremental traffic")
            ax.set_ylabel("Count")
            ax.set_title(
                "DMA-level placebo impact distribution"
            )
            ax.grid(
                axis="y",
                alpha=0.25,
            )
            fig.tight_layout()

            path = REPORT_FIGURE_DIR / "placebo_distribution.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            figure_paths["placebo"] = path

    return figure_paths


# =========================================================
# Narrative Builders
# =========================================================

def build_executive_summary(
    metrics: dict,
) -> str:
    """Build the executive summary narrative from calculated metrics."""

    aggregate = metrics.get("aggregate_incremental")
    aggregate_lower = metrics.get("aggregate_lower")
    aggregate_upper = metrics.get("aggregate_upper")
    prob_positive = metrics.get("aggregate_prob_positive")
    aggregate_dmas = metrics.get("aggregate_dmas")

    positive_count = metrics.get("positive_dma_count")
    negative_count = metrics.get("negative_dma_count")

    spend_r = metrics.get("spend_pearson")
    spend_p = metrics.get("spend_pearson_p")

    placebo_cross = metrics.get("placebo_zero_crossing_rate")

    text = []

    if aggregate is not None:
        text.append(
            f"Across the {fmt_int(aggregate_dmas)} production-valid DMAs, "
            f"the BSTS analysis estimates {fmt_int(aggregate)} incremental "
            f"traffic visits relative to the modeled counterfactual. The "
            f"95% credible interval ranges from {fmt_int(aggregate_lower)} "
            f"to {fmt_int(aggregate_upper)}, and the posterior probability "
            f"of a positive aggregate effect is {fmt_pct(prob_positive)}."
        )

    if positive_count is not None and negative_count is not None:
        text.append(
            f"The aggregate result masks meaningful DMA heterogeneity: "
            f"{positive_count} DMAs have positive posterior mean impacts and "
            f"{negative_count} have negative posterior mean impacts."
        )

    if spend_r is not None:
        qualification = "suggestive" if abs(spend_r) >= 0.5 else "weak"
        text.append(
            f"Marketing-spend robustness shows a {qualification} positive "
            f"linear association between absolute spend increases and "
            f"estimated incremental traffic (Pearson r={spend_r:.3f}, "
            f"p={spend_p:.3f}). The corresponding rank-based relationship "
            f"is materially weaker, so the evidence does not establish a "
            f"stable monotonic spend-response relationship."
        )

    if placebo_cross is not None:
        text.append(
            f"DMA-level placebo tests provide an additional robustness check: "
            f"{fmt_pct(placebo_cross)} of placebo impact intervals cross zero. "
            f"A common-universe aggregate placebo benchmark was not feasible "
            f"because the available placebo models did not produce a stable "
            f"common DMA universe across placebo dates."
        )

    return "\n".join(
        f"<p>{html.escape(paragraph)}</p>"
        for paragraph in text
    )


def build_model_credibility_section(inputs: dict) -> str:
    """Create model-validation narrative."""

    bsts = inputs.get("bsts_validation")
    cf = inputs.get("counterfactual_validation")

    if bsts is None or cf is None:
        return '<p class="muted">Validation tables unavailable.</p>'

    bsts_pass = int(bsts["converged"].sum()) if "converged" in bsts else 0
    bsts_total = len(bsts)

    cf_pass = (
        int(cf["counterfactual_valid"].sum())
        if "counterfactual_valid" in cf
        else 0
    )
    cf_total = len(cf)

    mean_r2 = (
        float(cf["pre_r2"].mean())
        if "pre_r2" in cf
        else np.nan
    )

    mean_coverage = (
        float(cf["pre_coverage_95"].mean())
        if "pre_coverage_95" in cf
        else np.nan
    )

    body = f"""
    <p>
        The production BSTS models passed convergence validation for
        <strong>{bsts_pass} of {bsts_total}</strong> modeled DMAs. The
        counterfactual validation passed for <strong>{cf_pass} of
        {cf_total}</strong> DMA models.
    </p>
    <p>
        Mean pre-treatment R² across the validated counterfactuals was
        <strong>{fmt_num(mean_r2, 3)}</strong>, indicating close reproduction
        of the pre-treatment traffic trajectory. Mean 95% interval coverage
        was <strong>{fmt_pct(mean_coverage)}</strong>.
    </p>
    <p>
        These diagnostics support using the fitted models for treatment-effect
        estimation, while the short pre-treatment window remains an important
        limitation on the precision and stability of causal inference.
    </p>
    """

    return body


def build_methodology_section() -> str:
    """Static methodology description."""

    return """
    <p>
        The analysis was designed as a staged incrementality workflow rather
        than a single-model exercise. The pipeline first established the data
        panel and exploratory relationships, then examined temporal association,
        panel regression, and difference-in-differences evidence before moving
        to DMA-specific Bayesian structural time-series models.
    </p>
    <p>
        The BSTS specification uses a standardized target, deterministic trend,
        standardized control-DMA predictors, optional Fourier seasonality, and
        Gaussian observation noise. Models were fitted only on the pre-treatment
        period and then propagated across the full modeling horizon using the
        observed control-DMA trajectories.
    </p>
    <p>
        Treatment impact is defined as observed post-treatment traffic minus the
        posterior counterfactual traffic trajectory. Credible intervals are
        calculated from posterior impact draws directly rather than by combining
        marginal interval bounds.
    </p>
    """


def build_aggregate_section(
    inputs: dict,
    figures: dict,
) -> str:
    """Build aggregate impact section."""

    aggregate = inputs.get("aggregate_impact")

    if aggregate is None or aggregate.empty:
        return '<p class="muted">Aggregate impact summary unavailable.</p>'

    row = aggregate.iloc[0]

    impact = row.get("Incremental_Traffic_Mean")
    lower = row.get("Impact_Lower_95")
    upper = row.get("Impact_Upper_95")
    prob = row.get("Probability_Positive")
    lift = row.get("Relative_Lift_Mean")

    cards = "".join([
        metric_card(
            "Estimated incremental traffic",
            fmt_int(impact),
            "posterior mean",
        ),
        metric_card(
            "95% credible interval",
            f"{fmt_int(lower)} to {fmt_int(upper)}",
            "aggregate treatment effect",
        ),
        metric_card(
            "P(impact > 0)",
            fmt_pct(prob),
            "posterior probability",
        ),
        metric_card(
            "Relative lift",
            fmt_pct(lift),
            "posterior mean",
        ),
    ])

    fig = figures.get("dma_impact")
    fig_html = figure_html(
        fig,
        "DMA-level BSTS impacts and 95% credible intervals",
    ) if fig else ""

    body = f"""
    <div class="metric-grid">{cards}</div>
    <p>
        At the portfolio level, the posterior mean indicates a negative traffic
        effect. However, the credible interval is wide and crosses zero, so the
        aggregate result should not be interpreted as definitive evidence of a
        portfolio-wide traffic decline.
    </p>
    {fig_html}
    {html_table(aggregate, max_rows=5)}
    """

    return body


def build_heterogeneity_section(
    inputs: dict,
) -> str:
    """Build DMA heterogeneity section."""

    heterogeneity = inputs.get("heterogeneity")
    contribution = inputs.get("dma_contribution")

    body_parts = []

    if heterogeneity is not None and not heterogeneity.empty:

        credible = (
            heterogeneity.loc[
                heterogeneity.get(
                    "Impact_Excludes_Zero",
                    False,
                ).astype(bool)
            ]
            if "Impact_Excludes_Zero" in heterogeneity.columns
            else pd.DataFrame()
        )

        if not credible.empty:
            names = credible["DMA"].tolist()
            body_parts.append(
                "<p>DMA-level posterior evidence is heterogeneous. "
                "The following DMAs have 95% impact intervals that exclude zero: "
                f"<strong>{html.escape(', '.join(names))}</strong>.</p>"
            )
        else:
            body_parts.append(
                "<p>No DMA-level 95% impact interval excludes zero in the current "
                "validated production set. Point estimates still differ materially "
                "across DMAs.</p>"
            )

        body_parts.append(
            html_table(
                heterogeneity.sort_values(
                    "Absolute_Impact_Rank"
                ),
                max_rows=20,
            )
        )

    if contribution is not None and not contribution.empty:
        body_parts.append(
            "<h3>DMA contribution context</h3>"
        )
        body_parts.append(
            html_table(
                contribution,
                max_rows=20,
            )
        )

    if not body_parts:
        return '<p class="muted">Heterogeneity outputs unavailable.</p>'

    return "\n".join(body_parts)


def build_placebo_section(
    inputs: dict,
    figures: dict,
) -> str:
    """Build placebo robustness section."""

    summary = inputs.get("placebo_summary")
    results = inputs.get("placebo_results")

    if summary is None or summary.empty:
        return '<p class="muted">Placebo analysis unavailable.</p>'

    row = summary.iloc[0]

    body = f"""
    <p>
        The placebo analysis produced <strong>{fmt_int(row.get('Placebo_Tests'))}</strong>
        valid DMA-by-date tests across <strong>{fmt_int(row.get('DMA_Count'))}</strong>
        DMAs and <strong>{fmt_int(row.get('Placebo_Date_Count'))}</strong> placebo dates.
    </p>
    <p>
        Placebo intervals crossed zero in <strong>{fmt_pct(row.get('Placebo_Zero_Crossing_Rate'))}</strong>
        of tests, with strong positive posterior evidence in
        <strong>{fmt_pct(row.get('Strong_Positive_Posterior_Rate'))}</strong> and strong
        negative evidence in <strong>{fmt_pct(row.get('Strong_Negative_Posterior_Rate'))}</strong>.
    </p>
    <p>
        This supports using placebo tests primarily as a DMA-level robustness check.
        The data did not support a stable common DMA universe across placebo dates,
        so an aggregate placebo benchmark was not treated as a production inference.
    </p>
    """

    fig = figures.get("placebo")
    if fig:
        body += figure_html(
            fig,
            "Distribution of DMA-level placebo impact estimates",
        )

    if results is not None and not results.empty:
        body += html_table(
            results.sort_values(
                "Probability_Positive",
                ascending=False,
            ),
            max_rows=20,
        )

    return body


def build_spend_section(
    inputs: dict,
    figures: dict,
) -> str:
    """Build marketing-spend robustness section."""

    spend_aggregate = inputs.get("spend_aggregate")
    spend_relationships = inputs.get("spend_relationships")
    spend_impact = inputs.get("spend_impact")

    if spend_aggregate is None or spend_aggregate.empty:
        return '<p class="muted">Marketing-spend analysis unavailable.</p>'

    row = spend_aggregate.iloc[0]

    cards = "".join([
        metric_card(
            "Incremental post-period spend",
            f"${fmt_int(row.get('Total_Incremental_Post_Spend'))}",
            "relative to pre-period monthly average",
        ),
        metric_card(
            "Estimated incremental traffic",
            fmt_int(row.get("Total_Incremental_Traffic_Mean")),
            "same validated DMA universe",
        ),
        metric_card(
            "Traffic per $1K incremental spend",
            fmt_num(
                row.get(
                    "Aggregate_Impact_Per_Incremental_Spend_1K"
                ),
                1,
            ),
            "descriptive efficiency metric",
        ),
        metric_card(
            "Positive-impact DMAs",
            fmt_int(row.get("Positive_DMA_Count")),
            f"of {fmt_int(row.get('Eligible_DMA_Count'))} eligible DMAs",
        ),
    ])

    body = f"""
    <div class="metric-grid">{cards}</div>
    <p>
        All eight production-valid DMAs increased spend in the post-treatment
        period. Absolute spend increases were moderately associated with estimated
        incremental traffic, but the percentage-spend-change relationships were
        substantially weaker. This suggests the apparent association is influenced
        by DMA scale and absolute dollar changes rather than a stable normalized
        dose-response relationship.
    </p>
    """

    fig = figures.get("spend_impact")
    if fig:
        body += figure_html(
            fig,
            "Marketing spend change versus estimated incremental traffic",
        )

    fig = figures.get("spend_efficiency")
    if fig:
        body += figure_html(
            fig,
            "Descriptive traffic impact per $1K incremental spend",
        )

    if spend_relationships is not None and not spend_relationships.empty:
        body += "<h3>Spend / impact relationships</h3>"
        body += html_table(
            spend_relationships,
            max_rows=20,
        )

    if spend_impact is not None and not spend_impact.empty:
        body += "<h3>DMA-level spend and impact</h3>"
        body += html_table(
            spend_impact.sort_values(
                "Monthly_Spend_Change",
                ascending=False,
            ),
            max_rows=20,
        )

    return body


def build_earlier_evidence_section(
    inputs: dict,
) -> str:
    """Summarize earlier, non-BSTS evidence without over-weighting it."""

    parts = []

    ccf = inputs.get("ccf_comparison")
    if ccf is not None and not ccf.empty:
        parts.append(
            "<h3>Cross-correlation</h3>"
        )
        parts.append(
            "<p>The cross-correlation analysis was used to characterize temporal "
            "alignment and to compare raw versus detrended relationships. Because "
            "trend can induce misleading correlation, the detrended results are the "
            "more informative diagnostic for timing rather than a causal estimate.</p>"
        )
        parts.append(
            html_table(ccf, max_rows=10)
        )

    regression = inputs.get("regression_comparison")
    if regression is not None and not regression.empty:
        parts.append(
            "<h3>Panel regression</h3>"
        )
        parts.append(
            "<p>The panel regression provides a complementary association model "
            "that controls for DMA and time effects. It is useful as triangulation, "
            "but the BSTS counterfactuals provide the primary treatment-effect estimates.</p>"
        )
        parts.append(
            html_table(regression, max_rows=10)
        )

    did = inputs.get("did_effect_summary")
    if did is not None and not did.empty:
        parts.append(
            "<h3>Difference-in-differences</h3>"
        )
        parts.append(
            "<p>The DiD analysis provides another benchmark for the treatment effect, "
            "subject to the quality of the parallel-trends assumption and treatment-group definition.</p>"
        )
        parts.append(
            html_table(did, max_rows=10)
        )

    if not parts:
        return '<p class="muted">Earlier analytical outputs unavailable.</p>'

    return "\n".join(parts)


def build_limitations_section(
    inputs: dict,
) -> str:
    """Build the limitations section."""

    return """
    <ol>
        <li>
            <strong>Short pre-treatment history.</strong> The primary BSTS models use
            only about eleven pre-treatment monthly observations. This materially
            limits the amount of historical information available for estimating
            control relationships and trend behavior.
        </li>
        <li>
            <strong>Wide posterior uncertainty.</strong> Several DMA- and portfolio-level
            credible intervals are broad and cross zero, so point estimates should not
            be interpreted as precise treatment effects.
        </li>
        <li>
            <strong>DMA heterogeneity.</strong> The portfolio aggregate combines materially
            different DMA responses. A single aggregate effect therefore hides meaningful
            regional variation.
        </li>
        <li>
            <strong>Placebo limitations.</strong> DMA-level placebo tests were feasible, but
            the available data did not provide a stable common DMA universe across all
            placebo dates. No aggregate placebo effect was therefore used as primary evidence.
        </li>
        <li>
            <strong>Spend analysis is associative.</strong> The marketing-spend layer evaluates
            whether estimated effects align with observed spend changes. It does not isolate
            a causal marginal effect of one additional dollar of spend.
        </li>
        <li>
            <strong>Potential unobserved confounding.</strong> The framework controls for observed
            control-DMA trajectories and deterministic time structure, but unmeasured factors
            that change specifically in treated DMAs could still influence the estimated effects.
        </li>
    </ol>
    """


def build_conclusion_section(
    metrics: dict,
) -> str:
    """Build final analytical conclusion."""

    impact = metrics.get("aggregate_incremental")
    prob = metrics.get("aggregate_prob_positive")
    positive = metrics.get("positive_dma_count")
    negative = metrics.get("negative_dma_count")
    spend_r = metrics.get("spend_pearson")

    return f"""
    <p>
        The completed analytical workflow provides evidence that the change in traffic
        after the intervention was <strong>not uniform across DMAs</strong>. The validated
        portfolio estimate is approximately <strong>{fmt_int(impact)}</strong> incremental
        visits in the posterior mean, with a positive-effect probability of
        <strong>{fmt_pct(prob)}</strong>.
    </p>
    <p>
        At the DMA level, <strong>{fmt_int(positive)}</strong> of the production-valid
        DMAs have positive posterior mean effects while <strong>{fmt_int(negative)}</strong>
        have negative effects. This heterogeneity is more informative than a simple
        portfolio-wide positive/negative label and should be central to any business interpretation.
    </p>
    <p>
        The marketing-spend robustness layer shows a moderate positive Pearson association
        between absolute spend increases and estimated traffic impact
        {f'(r={spend_r:.3f})' if spend_r is not None else ''}, but the normalized spend-change
        relationships are weak. The evidence therefore supports a nuanced interpretation:
        <strong>larger marketing investments were concentrated in some of the more positive
        DMA outcomes, but the analysis does not establish a consistent marginal spend-response
        relationship.</strong>
    </p>
    <p>
        Overall, the results are best interpreted as <strong>directional evidence of heterogeneous
        DMA-level treatment response with substantial uncertainty</strong>, rather than as a precise
        estimate of a single portfolio-wide marketing incrementality effect.
    </p>
    """


# =========================================================
# HTML Report
# =========================================================

def build_report_html(
    inputs: dict,
    metrics: dict,
    figures: dict,
) -> str:
    """Assemble the complete HTML report."""

    overview_cards = "".join([
        metric_card(
            "Production-valid DMAs",
            fmt_int(metrics.get("aggregate_dmas")),
            "used in aggregate causal analysis",
        ),
        metric_card(
            "Aggregate incremental traffic",
            fmt_int(metrics.get("aggregate_incremental")),
            "posterior mean",
        ),
        metric_card(
            "P(aggregate impact > 0)",
            fmt_pct(metrics.get("aggregate_prob_positive")),
            "posterior probability",
        ),
        metric_card(
            "Counterfactual mean R²",
            fmt_num(metrics.get("counterfactual_mean_r2"), 3),
            "across validated DMAs",
        ),
    ])

    panel = inputs.get("panel")

    if panel is not None and not panel.empty:
        coverage_detail = (
            f"{fmt_num(metrics.get('panel_dmas'), 0)} DMAs, "
            f"{fmt_num(metrics.get('panel_months'), 0)} months, "
            f"{metrics.get('panel_start').date()} through "
            f"{metrics.get('panel_end').date()}"
        )
    else:
        coverage_detail = "Panel coverage unavailable"

    body = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Marketing Incrementality Analysis — Final Report</title>
        <style>
            body {{
                font-family: Arial, Helvetica, sans-serif;
                margin: 0;
                background: #f5f7fa;
                color: #1f2933;
                line-height: 1.55;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 36px;
            }}
            header {{
                background: #1f2933;
                color: white;
                padding: 36px;
                border-radius: 12px;
                margin-bottom: 24px;
            }}
            h1 {{
                margin: 0 0 8px 0;
                font-size: 34px;
            }}
            h2 {{
                margin-top: 0;
                color: #243b53;
                border-bottom: 2px solid #d9e2ec;
                padding-bottom: 8px;
            }}
            h3 {{
                color: #334e68;
                margin-top: 28px;
            }}
            section {{
                background: white;
                padding: 28px;
                border-radius: 12px;
                margin-bottom: 24px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 14px;
                margin: 18px 0 24px 0;
            }}
            .metric-card {{
                background: #eef2f7;
                padding: 18px;
                border-radius: 10px;
            }}
            .metric-label {{
                font-size: 13px;
                color: #52606d;
                margin-bottom: 6px;
            }}
            .metric-value {{
                font-size: 25px;
                font-weight: 700;
                color: #102a43;
            }}
            .metric-detail {{
                font-size: 12px;
                color: #627d98;
                margin-top: 4px;
            }}
            .data-table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 14px;
                font-size: 12px;
            }}
            .data-table th, .data-table td {{
                border: 1px solid #d9e2ec;
                padding: 7px 8px;
                text-align: left;
            }}
            .data-table th {{
                background: #e9eff5;
                color: #334e68;
            }}
            figure {{
                margin: 24px 0;
                text-align: center;
            }}
            figure img {{
                border: 1px solid #d9e2ec;
                border-radius: 8px;
                background: white;
            }}
            figcaption {{
                font-size: 12px;
                color: #627d98;
                margin-top: 7px;
            }}
            .muted {{
                color: #829ab1;
                font-style: italic;
            }}
            .toc a {{
                color: #1f6feb;
                text-decoration: none;
            }}
            .callout {{
                background: #eef6ff;
                border-left: 4px solid #3578c6;
                padding: 16px;
                margin: 18px 0;
            }}
            footer {{
                color: #627d98;
                font-size: 12px;
                margin-top: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Marketing Incrementality Analysis</h1>
                <div>Final analytical report</div>
                <div style="margin-top:10px;color:#bcccdc;">{html.escape(coverage_detail)}</div>
            </header>

            <section>
                <h2>Executive Summary</h2>
                <div class="metric-grid">{overview_cards}</div>
                {build_executive_summary(metrics)}
                <div class="callout">
                    <strong>Interpretation:</strong> The results support a nuanced
                    DMA-level incrementality story rather than a single precise
                    portfolio-wide effect. Point estimates are informative, but
                    credible intervals and the short historical window materially
                    limit precision.
                </div>
            </section>

            <section class="toc">
                <h2>Report Guide</h2>
                <ol>
                    <li><a href="#methodology">Methodology and analytical design</a></li>
                    <li><a href="#earlier">Earlier analytical evidence</a></li>
                    <li><a href="#credibility">BSTS model and counterfactual validation</a></li>
                    <li><a href="#aggregate">Aggregate causal impact</a></li>
                    <li><a href="#heterogeneity">DMA-level heterogeneity</a></li>
                    <li><a href="#placebo">Placebo robustness</a></li>
                    <li><a href="#spend">Marketing spend robustness</a></li>
                    <li><a href="#limitations">Limitations</a></li>
                    <li><a href="#conclusion">Analytical conclusion</a></li>
                </ol>
            </section>

            {section('Methodology and analytical design', build_methodology_section(), 'methodology')}

            {section('Earlier analytical evidence', build_earlier_evidence_section(inputs), 'earlier')}

            {section('BSTS model and counterfactual validation', build_model_credibility_section(inputs), 'credibility')}

            {section('Aggregate causal impact', build_aggregate_section(inputs, figures), 'aggregate')}

            {section('DMA-level heterogeneity', build_heterogeneity_section(inputs), 'heterogeneity')}

            {section('Placebo robustness', build_placebo_section(inputs, figures), 'placebo')}

            {section('Marketing spend robustness', build_spend_section(inputs, figures), 'spend')}

            {section('Limitations', build_limitations_section(inputs), 'limitations')}

            {section('Analytical conclusion', build_conclusion_section(metrics), 'conclusion')}

            <footer>
                Generated from persisted outputs of the Marketing Incrementality
                analytical pipeline. This reporting layer does not refit models
                or modify the underlying analytical estimates.
            </footer>
        </div>
    </body>
    </html>
    """

    return body


# =========================================================
# Executive HTML Report
# =========================================================
def build_executive_report_html(
    inputs: dict,
    metrics: dict,
    figures: dict,
) -> str:
    """
    Build a concise executive-level HTML summary.

    Uses the same persisted analytical inputs and calculated
    metrics as the detailed report, but presents only the
    highest-level findings, implications, and limitations.
    """

    aggregate = metrics.get(
        "aggregate_incremental"
    )

    lower = metrics.get(
        "aggregate_lower"
    )

    upper = metrics.get(
        "aggregate_upper"
    )

    prob_positive = metrics.get(
        "aggregate_prob_positive"
    )

    aggregate_dmas = metrics.get(
        "aggregate_dmas"
    )

    positive_dmas = metrics.get(
        "positive_dma_count"
    )

    negative_dmas = metrics.get(
        "negative_dma_count"
    )

    cf_pass = metrics.get(
        "counterfactual_pass_count"
    )

    cf_total = metrics.get(
        "counterfactual_count"
    )

    cf_r2 = metrics.get(
        "counterfactual_mean_r2"
    )

    spend_r = metrics.get(
        "spend_pearson"
    )

    spend_p = metrics.get(
        "spend_pearson_p"
    )

    spend_impact = metrics.get(
        "spend_aggregate_impact"
    )

    incremental_spend = metrics.get(
        "incremental_spend"
    )

    placebo_tests = metrics.get(
        "placebo_tests"
    )

    placebo_cross = metrics.get(
        "placebo_zero_crossing_rate"
    )

    # =====================================================
    # Positive / Negative DMA Callouts
    # =====================================================

    dma_impact = inputs.get(
        "dma_impact"
    )

    positive_names = []
    negative_names = []

    if (
        dma_impact is not None
        and
        not dma_impact.empty
    ):

        temp = dma_impact.copy()

        temp[
            "Incremental_Traffic_Mean"
        ] = pd.to_numeric(
            temp[
                "Incremental_Traffic_Mean"
            ],
            errors="coerce",
        )

        positive_names = (
            temp.loc[
                temp[
                    "Incremental_Traffic_Mean"
                ] > 0
            ]
            .sort_values(
                "Incremental_Traffic_Mean",
                ascending=False,
            )
            .head(2)[
                "DMA"
            ]
            .tolist()
        )

        negative_names = (
            temp.loc[
                temp[
                    "Incremental_Traffic_Mean"
                ] < 0
            ]
            .sort_values(
                "Incremental_Traffic_Mean",
                ascending=True,
            )
            .head(2)[
                "DMA"
            ]
            .tolist()
        )

    positive_text = (
        ", ".join(
            positive_names
        )
        if positive_names
        else
        "No positive DMA point estimates identified."
    )

    negative_text = (
        ", ".join(
            negative_names
        )
        if negative_names
        else
        "No negative DMA point estimates identified."
    )

    # =====================================================
    # Executive Interpretation
    # =====================================================

    if (
        aggregate is not None
        and pd.notna(aggregate)
    ):

            aggregate = float(aggregate)

            interval_crosses_zero = (
                lower is not None
                and upper is not None
                and pd.notna(lower)
                and pd.notna(upper)
                and lower <= 0 <= upper
            )

            probability_available = (
                prob_positive is not None
                and pd.notna(prob_positive)
            )

            if probability_available:

                prob_positive_pct = (
                    float(prob_positive) * 100
                    if float(prob_positive) <= 1
                    else float(prob_positive)
                )

                prob_negative_pct = (
                    100 - prob_positive_pct
                )

            else:

                prob_positive_pct = None
                prob_negative_pct = None

            # -------------------------------------------------
            # Positive aggregate effect
            # -------------------------------------------------

            if aggregate > 0:

                if (
                    not interval_crosses_zero
                    and
                    probability_available
                    and
                    prob_positive_pct >= 95
                ):

                    aggregate_interpretation = (
                        f"The model estimates a positive aggregate "
                        f"incremental traffic effect of "
                        f"{fmt_int(aggregate)} visits. The 95% credible "
                        f"interval is entirely above zero and the posterior "
                        f"probability of a positive impact is "
                        f"{prob_positive_pct:.1f}%. This provides strong "
                        f"evidence of a portfolio-level incremental traffic "
                        f"effect."
                    )

                elif interval_crosses_zero:

                    if probability_available:

                        aggregate_interpretation = (
                            f"The model estimates a positive aggregate "
                            f"incremental traffic effect of "
                            f"{fmt_int(aggregate)} visits. However, the 95% "
                            f"credible interval crosses zero and the posterior "
                            f"probability of a positive impact is only "
                            f"{prob_positive_pct:.1f}%. The results therefore "
                            f"provide directional evidence of a positive effect "
                            f"but do not establish a conclusive portfolio-level "
                            f"traffic lift."
                        )

                    else:

                        aggregate_interpretation = (
                            f"The model estimates a positive aggregate "
                            f"incremental traffic effect of "
                            f"{fmt_int(aggregate)} visits, but the 95% "
                            f"credible interval crosses zero. The portfolio-level "
                            f"effect is therefore positive in direction but "
                            f"uncertain."
                        )

                else:

                    aggregate_interpretation = (
                        f"The model estimates a positive aggregate "
                        f"incremental traffic effect of "
                        f"{fmt_int(aggregate)} visits. The credible interval "
                        f"is entirely above zero, indicating a statistically "
                        f"clear positive treatment effect within the modeled "
                        f"uncertainty."
                    )

            # -------------------------------------------------
            # Negative aggregate effect
            # -------------------------------------------------

            elif aggregate < 0:

                if (
                    not interval_crosses_zero
                    and
                    probability_available
                    and
                    prob_negative_pct >= 95
                ):

                    aggregate_interpretation = (
                        f"The model estimates a negative aggregate "
                        f"incremental traffic effect of "
                        f"{fmt_int(aggregate)} visits. The 95% credible "
                        f"interval is entirely below zero and the posterior "
                        f"probability of a negative impact is "
                        f"{prob_negative_pct:.1f}%. This provides strong "
                        f"evidence of a portfolio-level adverse traffic "
                        f"effect."
                    )

                elif interval_crosses_zero:

                    if probability_available:

                        aggregate_interpretation = (
                            f"The model estimates a negative aggregate "
                            f"incremental traffic effect of "
                            f"{fmt_int(aggregate)} visits. However, the 95% "
                            f"credible interval crosses zero and the posterior "
                            f"probability of a negative impact is only "
                            f"{prob_negative_pct:.1f}%. The results therefore "
                            f"provide directional evidence of a negative effect "
                            f"but do not establish a conclusive portfolio-level "
                            f"traffic decline."
                        )

                    else:

                        aggregate_interpretation = (
                            f"The model estimates a negative aggregate "
                            f"incremental traffic effect of "
                            f"{fmt_int(aggregate)} visits, but the 95% "
                            f"credible interval crosses zero. The portfolio-level "
                            f"effect is therefore negative in direction but "
                            f"uncertain."
                        )

                else:

                    aggregate_interpretation = (
                        f"The model estimates a negative aggregate "
                        f"incremental traffic effect of "
                        f"{fmt_int(aggregate)} visits. The credible interval "
                        f"is entirely below zero, indicating a clear negative "
                        f"treatment effect within the modeled uncertainty."
                    )

            # -------------------------------------------------
            # Essentially zero
            # -------------------------------------------------

            else:

                aggregate_interpretation = (
                    "The model estimates essentially no aggregate "
                    "incremental traffic effect. The posterior result "
                    "does not indicate a meaningful portfolio-level "
                    "traffic response to treatment."
                )

    else:

        aggregate_interpretation = (
            "Aggregate impact estimates were unavailable."
        )

    # =====================================================
    # Counterfactual Interpretation
    # =====================================================

    if (
        cf_pass is not None
        and
        cf_total is not None
    ):

        counterfactual_interpretation = (
            f"{fmt_int(cf_pass)} of "
            f"{fmt_int(cf_total)} production DMAs "
            "passed the counterfactual validation criteria"
        )

        if (
            cf_r2 is not None
            and
            pd.notna(cf_r2)
        ):

            counterfactual_interpretation += (
                f", with mean pre-treatment R² of "
                f"{cf_r2:.3f}"
            )

        counterfactual_interpretation += "."

    else:

        counterfactual_interpretation = (
            "Counterfactual validation results were unavailable."
        )

    # =====================================================
    # Spend Interpretation
    # =====================================================

    if (
        spend_r is not None
        and
        pd.notna(spend_r)
    ):

        if (
            spend_p is not None
            and
            pd.notna(spend_p)
        ):

            spend_interpretation = (
                "Absolute marketing-spend increases show a "
                f"moderate positive linear association with "
                f"estimated traffic impact "
                f"(Pearson r={spend_r:.2f}, "
                f"p={spend_p:.3f}). "
                "However, the relationship weakens materially "
                "when spend is expressed as a percentage change, "
                "so the analysis does not establish a consistent "
                "spend-driven dose-response effect."
            )

        else:

            spend_interpretation = (
                "Absolute marketing-spend increases show a "
                f"moderate positive linear association with "
                f"estimated traffic impact "
                f"(Pearson r={spend_r:.2f})."
            )

    else:

        spend_interpretation = (
            "Marketing-spend robustness results were unavailable."
        )

    # =====================================================
    # Placebo Interpretation
    # =====================================================

    if (
        placebo_tests is not None
        and
        pd.notna(placebo_tests)
    ):

        if (
            placebo_cross is not None
            and
            pd.notna(placebo_cross)
        ):

            placebo_interpretation = (
                f"DMA-level placebo testing produced "
                f"{fmt_int(placebo_tests)} valid placebo tests, "
                f"with {fmt_pct(placebo_cross)} of placebo intervals "
                "crossing zero. A stable aggregate placebo benchmark "
                "was not feasible because no common production-valid "
                "DMA universe was available across all placebo dates."
            )

        else:

            placebo_interpretation = (
                f"DMA-level placebo testing produced "
                f"{fmt_int(placebo_tests)} valid placebo tests."
            )

    else:

        placebo_interpretation = (
            "Placebo robustness results were unavailable."
        )


    # =====================================================
    # Bottom-Line Label
    # =====================================================

    if (
        aggregate is not None
        and pd.notna(aggregate)
    ):

        if aggregate > 0:

            if (
                lower is not None
                and upper is not None
                and pd.notna(lower)
                and pd.notna(upper)
                and lower <= 0 <= upper
            ):

                bottom_line_label = (
                    "Positive Direction, High Uncertainty"
                )

            else:

                bottom_line_label = (
                    "Positive Incremental Impact"
                )

        elif aggregate < 0:

            if (
                lower is not None
                and upper is not None
                and pd.notna(lower)
                and pd.notna(upper)
                and lower <= 0 <= upper
            ):

                bottom_line_label = (
                    "Negative Direction, High Uncertainty"
                )

            else:

                bottom_line_label = (
                    "Negative Incremental Impact"
                )

        else:

            bottom_line_label = (
                "No Meaningful Aggregate Impact"
            )

    else:

        bottom_line_label = (
            "Aggregate Impact Unavailable"
        )



    # =====================================================
    # Management Interpretation
    # =====================================================

    if (
        aggregate is not None
        and pd.notna(aggregate)
    ):

        interval_crosses_zero = (
            lower is not None
            and upper is not None
            and pd.notna(lower)
            and pd.notna(upper)
            and lower <= 0 <= upper
        )

        if aggregate > 0:

            if interval_crosses_zero:

                management_interpretation = (
                    "The analysis points toward a positive marketing "
                    "response, but the uncertainty around the aggregate "
                    "effect remains substantial. Management should "
                    "therefore treat the result as directional evidence "
                    "rather than a confirmed portfolio-wide return."
                )

            else:

                management_interpretation = (
                    "The analysis provides evidence of a positive "
                    "incremental traffic response to treatment. "
                    "The next optimization opportunity is to identify "
                    "which markets generated the strongest returns and "
                    "whether additional investment can be concentrated "
                    "in those markets."
                )

        elif aggregate < 0:

            if interval_crosses_zero:

                management_interpretation = (
                    "The analysis points toward a negative marketing "
                    "response, but the uncertainty interval includes "
                    "zero. Management should therefore avoid treating "
                    "the result as evidence that marketing definitively "
                    "reduced traffic."
                )

            else:

                management_interpretation = (
                    "The analysis provides evidence of a negative "
                    "incremental traffic response to treatment. "
                    "This warrants reviewing market-level allocation, "
                    "campaign strategy, and whether incremental spend "
                    "should be redirected or reduced."
                )

        else:

            management_interpretation = (
                "The analysis does not indicate a meaningful aggregate "
                "traffic response to treatment. Future optimization "
                "should focus on identifying market-level differences "
                "rather than assuming a broad portfolio effect."
            )

    else:

        management_interpretation = (
            "Aggregate treatment results were unavailable, so no "
            "portfolio-level management conclusion can be drawn."
        )  


        # =====================================================
    # DMA Heterogeneity Interpretation
    # =====================================================

    if (
        positive_dmas is not None
        and negative_dmas is not None
    ):

        positive_dmas = int(positive_dmas)
        negative_dmas = int(negative_dmas)

        if positive_dmas > 0 and negative_dmas > 0:

            dma_interpretation = (
                f"The portfolio result masks meaningful DMA-level "
                f"heterogeneity. {positive_dmas} DMAs have positive "
                f"posterior mean effects while {negative_dmas} DMAs "
                f"have negative posterior mean effects. The aggregate "
                f"result therefore should not be interpreted as a "
                f"uniform response across markets."
            )

        elif positive_dmas > 0:

            dma_interpretation = (
                f"All validated DMAs with a directional point estimate "
                f"are positive, with {positive_dmas} DMAs showing "
                f"positive posterior mean effects. However, aggregate "
                f"uncertainty should still be considered when assessing "
                f"the portfolio-level result."
            )

        elif negative_dmas > 0:

            dma_interpretation = (
                f"All validated DMAs with a directional point estimate "
                f"are negative, with {negative_dmas} DMAs showing "
                f"negative posterior mean effects. This suggests the "
                f"observed treatment response is not favorable across "
                f"the validated markets."
            )

        else:

            dma_interpretation = (
                "No clear positive or negative DMA-level point estimates "
                "were identified."
            )

    else:

        dma_interpretation = (
            "DMA-level heterogeneity could not be evaluated."
        )      


    # =====================================================
    # Figure
    # =====================================================

    fig = figures.get(
        "dma_impact"
    )

    dma_fig = (
        figure_html(
            fig,
            "DMA-level estimated traffic impacts with 95% credible intervals",
            width="92%",
        )
        if fig
        else ""
    )

    # =====================================================
    # HTML
    # =====================================================

    body = f"""
<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="utf-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >

    <title>
        Marketing Incrementality — Executive Summary
    </title>

    <style>

        body {{
            font-family:
                Arial,
                Helvetica,
                sans-serif;

            margin: 0;

            background: #f4f6f8;

            color: #1f2933;

            line-height: 1.5;
        }}

        .container {{
            max-width: 1050px;

            margin:
                0 auto;

            padding:
                30px;
        }}

        header {{
            background:
                #102a43;

            color:
                white;

            padding:
                30px;

            border-radius:
                12px;

            margin-bottom:
                20px;
        }}

        h1 {{
            margin:
                0 0 8px 0;

            font-size:
                32px;
        }}

        h2 {{
            margin-top:
                0;

            color:
                #243b53;
        }}

        h3 {{
            color:
                #334e68;

            margin-bottom:
                6px;
        }}

        section {{
            background:
                white;

            padding:
                24px;

            border-radius:
                12px;

            margin-bottom:
                18px;

            box-shadow:
                0 1px 3px
                rgba(
                    0,
                    0,
                    0,
                    .08
                );
        }}

        .cards {{
            display:
                grid;

            grid-template-columns:
                repeat(
                    auto-fit,
                    minmax(
                        210px,
                        1fr
                    )
                );

            gap:
                12px;

            margin-top:
                16px;
        }}

        .card {{
            background:
                #eef2f7;

            border-radius:
                10px;

            padding:
                16px;
        }}

        .label {{
            font-size:
                12px;

            color:
                #627d98;
        }}

        .value {{
            font-size:
                24px;

            font-weight:
                700;

            color:
                #102a43;

            margin-top:
                3px;
        }}

        .detail {{
            font-size:
                12px;

            color:
                #627d98;

            margin-top:
                4px;
        }}

        .takeaway {{
            background:
                #eef6ff;

            border-left:
                4px solid
                #3578c6;

            padding:
                15px 18px;

            border-radius:
                6px;
        }}

        .two-col {{
            display:
                grid;

            grid-template-columns:
                1fr 1fr;

            gap:
                18px;
        }}

        .positive {{
            color:
                #1f7a3a;
        }}

        .negative {{
            color:
                #b42318;
        }}

        .muted {{
            color:
                #627d98;
        }}

        .small {{
            font-size:
                12px;
        }}

        footer {{
            color:
                #627d98;

            font-size:
                11px;

            margin-top:
                20px;
        }}

        @media (
            max-width: 750px
        ) {{

            .two-col {{
                grid-template-columns:
                    1fr;
            }}

        }}

    </style>

</head>

<body>

<div class="container">

<header>

    <h1>
        Marketing Incrementality
        — Executive Summary
    </h1>

    <div
        class="muted"
        style="color:#bcccdc;"
    >
        Final validated analytical results
    </div>

</header>

<section>

    <h2>
        Bottom Line: {html.escape(bottom_line_label)}
    </h2>

    <div class="cards">

        {metric_card(
            "Aggregate incremental traffic",
            fmt_int(
                aggregate
            ),
            "posterior mean across production-valid DMAs",
        )}

        {metric_card(
            "95% credible interval",
            (
                f"{fmt_int(lower)} "
                f"to {fmt_int(upper)}"
            ),
            "aggregate treatment effect",
        )}

        {metric_card(
            "Probability of positive impact",
            fmt_pct(
                prob_positive
            ),
            "posterior probability",
        )}

        {metric_card(
            "Validated DMAs",
            fmt_int(
                aggregate_dmas
            ),
            "production analysis universe",
        )}

    </div>

    <div
        class="takeaway"
        style="margin-top:18px;"
    >

        <strong>
            Executive takeaway:
        </strong>

        {html.escape(
            aggregate_interpretation
        )}

    </div>

</section>

<section>

    <h2>
        Model Credibility
    </h2>

    <p>
        {html.escape(
            counterfactual_interpretation
        )}
    </p>

    <p>
        The production BSTS models generally exhibited
        strong sampling diagnostics and strong
        pre-treatment predictive performance.
        This supports use of the estimated counterfactuals
        while preserving the uncertainty reflected in the
        posterior intervals.
    </p>

</section>

<section>

    <h2>
        DMA-Level Response
    </h2>

    <div class="two-col">

        <div>

            <h3 class="positive">
                Largest positive point estimates
            </h3>

            <p>
                {html.escape(
                    positive_text
                )}
            </p>

        </div>

        <div>

            <h3 class="negative">
                Largest negative point estimates
            </h3>

            <p>
                {html.escape(
                    negative_text
                )}
            </p>

        </div>

    </div>

    {dma_fig}

    <p>
        {html.escape(
            dma_interpretation
        )}
    </p>

</section>

<section>

    <h2>
        Marketing Spend
    </h2>

    {metric_card(
        "Incremental post-treatment spend",
        (
            f"${incremental_spend:,.0f}"
            if incremental_spend is not None
            and pd.notna(
                incremental_spend
            )
            else "Unavailable"
        ),
        "observed increase versus pre-treatment monthly baseline",
    )}

    <p>
        {html.escape(
            spend_interpretation
        )}
    </p>

    <p class="small muted">
        Spend analysis is a robustness / association analysis
        and should not be interpreted as causal ROAS.
    </p>

</section>

<section>

    <h2>
        Robustness
    </h2>

    <p>
        {html.escape(
            placebo_interpretation
        )}
    </p>

    <p>
        Taken together, the robustness analyses support
        the credibility of the DMA-level results, but
        they do not eliminate the uncertainty created by
        the short pre-treatment history or the heterogeneous
        DMA response.
    </p>

</section>

<section>

    <h2>
        Management Interpretation
    </h2>

    <p>
        {html.escape(
            management_interpretation
        )}
    </p>

</section>

<section>

    <h2>
        Key Limitation
    </h2>

    <p>
        The analysis uses approximately eleven
        pre-treatment monthly observations.
        This is sufficient to construct a useful
        structured counterfactual in the validated DMAs,
        but it limits precision and makes the results
        less definitive than would be possible with a
        longer historical window.
    </p>

</section>

<footer>

    This executive summary is generated from the same
    persisted analytical outputs as the detailed report.

    See
    <strong>
        Marketing_Incrementality_Final_Report.html
    </strong>
    for the full methodology, diagnostics,
    tables, robustness results, and limitations.

</footer>

</div>

</body>

</html>
"""

    return body


# =========================================================
# Main
# =========================================================

def main():
    """Build the final marketing incrementality report."""

    logger.info("=" * 70)
    logger.info("Building final Marketing Incrementality report")
    logger.info("=" * 70)

    ensure_output_dirs()

    inputs = load_reporting_inputs()

    metrics = calculate_report_metrics(
        inputs
    )

    figures = build_report_figures(
        inputs
    )

    report_html = build_report_html(
        inputs=inputs,
        metrics=metrics,
        figures=figures,
    )

    report_path = (
        REPORT_DIR /
        "Marketing_Incrementality_Final_Report.html"
    )

    report_path.write_text(
        report_html,
        encoding="utf-8",
    )

    logger.info(
        "Saved final report to %s",
        report_path,
    )

    # -----------------------------------------------------
    # Executive summary report
    # -----------------------------------------------------

    executive_html = build_executive_report_html(
        inputs=inputs,
        metrics=metrics,
        figures=figures,
    )

    executive_report_path = (
        REPORT_DIR /
        "Marketing_Incrementality_Executive_Summary.html"
    )

    executive_report_path.write_text(
        executive_html,
        encoding="utf-8",
    )

    if not executive_report_path.exists():
        raise RuntimeError(
            f"Executive summary was not created: "
            f"{executive_report_path}"
        )

    logger.info(
        "Saved executive summary to %s (%s bytes)",
        executive_report_path,
        executive_report_path.stat().st_size,
    )

    # -----------------------------------------------------
    # Save report metric snapshot
    # -----------------------------------------------------

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_path = (
        TABLE_DIR /
        "reporting_metrics.csv"
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    logger.info(
        "Saved reporting metric snapshot to %s",
        metrics_path,
    )

    logger.info("=" * 70)
    logger.info("Final report generation complete")
    logger.info("=" * 70)

    return {
        "report_path": report_path,
        "executive_report_path": executive_report_path,
        "metrics": metrics,
        "figure_paths": figures,
    }


if __name__ == "__main__":
    main()