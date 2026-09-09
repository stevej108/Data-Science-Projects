# ============================================================
# 14_bsts_comp_comparison.py
# ============================================================
"""
Compare BSTS traffic impacts with DMA-level comp traffic.

Purpose
-------
Translate the BSTS relative traffic lift into a theoretical
comp-traffic interpretation for each production-valid DMA.

The BSTS model estimates incremental traffic using total
traffic and a modeled counterfactual.

The comp-traffic dataset measures year-over-year traffic
performance among stores currently identified as comp stores.

Because BSTS Traffic and comp Exits represent different
populations, raw BSTS traffic should NOT be divided directly
by comp-store LY exits.

Instead, the BSTS relative lift is used to translate the
observed comp result into an implied counterfactual comp:

    Implied Counterfactual Comp
        = ((1 + Actual Comp) / (1 + BSTS Relative Lift)) - 1

and:

    Implied Incremental Comp
        = Actual Comp - Implied Counterfactual Comp

This provides a theoretical interpretation of how much of
the observed comp performance could be associated with the
BSTS-estimated intervention lift.

This is NOT a separate causal estimate of comp traffic.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import html
import logging
import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    FIGURE_DIR,
    TABLE_DIR,
    OUTPUT_DIR,
    REPORT_DIR
)


# ============================================================
# CONFIG
# ============================================================

TABLE_DIR = (
    OUTPUT_DIR
    / "tables"
    / "bsts_comp_comparison"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
    / "bsts_comp_comparison"
)

REPORT_DIR = (
    OUTPUT_DIR
    / "reports"
)

BSTS_IMPACT_FILE = (
    OUTPUT_DIR
    / "tables"
    / "causal_impact"
    / "dma_impact_summary.csv"
)

COMP_TRAFFIC_FILE = (
    OUTPUT_DIR
    / "comp_traffic"
    / "traffic_dma_month.parquet"
)

INTERVENTION_DATE = pd.Timestamp(
    "2026-01-01"
)

# Optional: only use these columns if available.
REQUIRED_BSTS_COLUMNS = [
    "DMA",
    "Intervention_Date",
    "Post_Treatment_Months",
    "Observed_Traffic",
    "Counterfactual_Traffic_Mean",
    "Incremental_Traffic_Mean",
    "Probability_Positive",
    "Relative_Lift_Mean",
    "Impact_Lower_95",
    "Impact_Upper_95",
]

REQUIRED_COMP_COLUMNS = [
    "DMA",
    "Traffic_Month",
    "Exits",
    "Exits_LY",
    "Comp_Traffic",
]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)-8s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_output_directories():
    """Create required output directories."""

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_bsts_results():
    """Load production-valid BSTS DMA impact results."""

    logger.info(
        "Loading BSTS DMA impact summary..."
    )

    if not BSTS_IMPACT_FILE.exists():
        raise FileNotFoundError(
            f"BSTS impact file not found: "
            f"{BSTS_IMPACT_FILE}"
        )

    df = pd.read_csv(
        BSTS_IMPACT_FILE
    )

    logger.info(
        "Loaded %,d BSTS DMA results.",
        len(df),
    )

    missing = [
        col
        for col in REQUIRED_BSTS_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "BSTS impact summary is missing "
            f"required columns: {missing}"
        )

    df["Intervention_Date"] = pd.to_datetime(
        df["Intervention_Date"],
        errors="coerce",
    )

    numeric_columns = [
        col
        for col in REQUIRED_BSTS_COLUMNS
        if col not in [
            "DMA",
            "Intervention_Date",
        ]
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


def load_comp_traffic():
    """Load monthly DMA comp traffic."""

    logger.info(
        "Loading DMA monthly comp traffic..."
    )

    if not COMP_TRAFFIC_FILE.exists():
        raise FileNotFoundError(
            f"Comp traffic file not found: "
            f"{COMP_TRAFFIC_FILE}"
        )

    df = pd.read_parquet(
        COMP_TRAFFIC_FILE
    )

    logger.info(
        "Loaded %,d DMA-month observations.",
        len(df),
    )

    missing = [
        col
        for col in REQUIRED_COMP_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Comp traffic file is missing "
            f"required columns: {missing}"
        )

    df["Traffic_Month"] = pd.to_datetime(
        df["Traffic_Month"],
        errors="coerce",
    )

    for col in [
        "Exits",
        "Exits_LY",
        "Comp_Traffic",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


# ============================================================
# DMA NORMALIZATION
# ============================================================

def normalize_dma_name(value):
    """
    Create a common DMA join key.

    BSTS uses names such as:

        New York, NY
        Boston, MA (Manchester, NH)
        Dallas-Ft. Worth, TX

    Comp traffic uses names such as:

        New York
        Boston
        Dallas

    The original DMA names are retained for presentation.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    # --------------------------------------------------------
    # Remove parenthetical market extensions
    # --------------------------------------------------------

    value = re.sub(
        r"\s*\([^)]*\)",
        "",
        value,
    ).strip()

    # --------------------------------------------------------
    # Remove state suffix
    # --------------------------------------------------------

    if "," in value:
        value = value.split(
            ",",
            1,
        )[0].strip()

    # --------------------------------------------------------
    # Standardize punctuation
    # --------------------------------------------------------

    value = value.replace(
        "–",
        "-",
    )

    value = value.replace(
        "—",
        "-",
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    # --------------------------------------------------------
    # Known naming conventions
    # --------------------------------------------------------

    replacements = {
        "Dallas-Ft. Worth": "Dallas",
        "Dallas-Ft Worth": "Dallas",
        "Dallas Fort Worth": "Dallas",

        "Miami-Ft. Lauderdale": "Miami",
        "Miami-Fort Lauderdale": "Miami",

        "Washington DC": "Washington DC",
        "Washington, DC": "Washington DC",

        "Ft. Myers": "Fort Myers",
        "Ft Myers": "Fort Myers",

        "St. Louis": "St. Louis",
        "St Louis": "St. Louis",

        "Tampa-St. Petersburg (Sarasota)": "Tampa",
        "Tampa-St Petersburg": "Tampa",

        "Minneapolis-St. Paul": "Minneapolis",
        "Minneapolis-St Paul": "Minneapolis",

        "Charlotte": "Charlotte",
        "New York": "New York",
        "Los Angeles": "Los Angeles",
        "Chicago": "Chicago",
        "Atlanta": "Atlanta",
        "Houston": "Houston",
        "Philadelphia": "Philadelphia",
        "Boston": "Boston",
        "Puerto Rico": "Puerto Rico",
    }

    return replacements.get(
        value,
        value,
    )


def add_dma_keys(
    bsts,
    comp,
):
    """
    Add normalized DMA keys to both datasets.
    """

    bsts = bsts.copy()
    comp = comp.copy()

    bsts["DMA_Key"] = (
        bsts["DMA"]
        .apply(normalize_dma_name)
    )

    comp["DMA_Key"] = (
        comp["DMA"]
        .apply(normalize_dma_name)
    )

    logger.info(
        "BSTS normalized DMA keys: %d",
        bsts["DMA_Key"].nunique(),
    )

    logger.info(
        "Comp normalized DMA keys: %d",
        comp["DMA_Key"].nunique(),
    )

    return bsts, comp


# ============================================================
# COMP CALCULATIONS
# ============================================================

def calculate_aggregate_comp(
    comp_period,
):
    """
    Calculate aggregate comp traffic from total Exits
    and Exits_LY.

    This intentionally does NOT average monthly
    percentage changes.
    """

    valid = comp_period[
        comp_period["Exits"].notna()
        &
        comp_period["Exits_LY"].notna()
        &
        (comp_period["Exits_LY"] > 0)
    ].copy()

    if valid.empty:
        return np.nan

    actual_exits = valid[
        "Exits"
    ].sum()

    ly_exits = valid[
        "Exits_LY"
    ].sum()

    if ly_exits <= 0:
        return np.nan

    return (
        (
            actual_exits
            /
            ly_exits
        )
        - 1
    )


# ============================================================
# DMA COMPARISON
# ============================================================

def build_dma_comparison(
    bsts,
    comp,
):
    """
    Construct DMA-level BSTS / comp comparison.
    """

    logger.info(
        "Building DMA-level BSTS / comp comparison..."
    )

    results = []

    production_dmas = (
        bsts[
            bsts["DMA_Key"].notna()
        ]
        .copy()
    )

    logger.info(
        "Production-valid BSTS DMAs: %d",
        production_dmas["DMA_Key"].nunique(),
    )

    for _, row in production_dmas.iterrows():

        dma = row["DMA"]
        dma_key = row["DMA_Key"]

        intervention_date = (
            row["Intervention_Date"]
            if pd.notna(
                row["Intervention_Date"]
            )
            else INTERVENTION_DATE
        )

        post_months = row[
            "Post_Treatment_Months"
        ]

        # ----------------------------------------------------
        # Determine BSTS post-treatment end month
        # ----------------------------------------------------

        if (
            pd.notna(post_months)
            and post_months > 0
        ):

            post_months = int(
                post_months
            )

            comp_end_month = (
                intervention_date
                + pd.DateOffset(
                    months=post_months - 1
                )
            )

        else:

            comp_end_month = (
                comp[
                    comp["DMA_Key"]
                    == dma_key
                ]["Traffic_Month"]
                .max()
            )

        # ----------------------------------------------------
        # Select matching comp period
        # ----------------------------------------------------

        comp_period = comp[
            (comp["DMA_Key"] == dma_key)
            &
            (
                comp["Traffic_Month"]
                >= intervention_date
            )
            &
            (
                comp["Traffic_Month"]
                <= comp_end_month
            )
        ].copy()

        if comp_period.empty:

            logger.warning(
                "No comp traffic observations for %s",
                dma,
            )

            continue

        # ----------------------------------------------------
        # Calculate comp metrics
        # ----------------------------------------------------

        valid_comp = comp_period[
            comp_period["Exits"].notna()
            &
            comp_period["Exits_LY"].notna()
            &
            (
                comp_period["Exits_LY"]
                > 0
            )
        ].copy()

        if valid_comp.empty:

            logger.warning(
                "No valid comp observations for %s",
                dma,
            )

            continue

        actual_exits = (
            valid_comp["Exits"].sum()
        )

        ly_exits = (
            valid_comp["Exits_LY"].sum()
        )

        if ly_exits <= 0:

            logger.warning(
                "Invalid LY exits for %s",
                dma,
            )

            continue

        actual_comp = (
            actual_exits
            /
            ly_exits
        ) - 1

        # ----------------------------------------------------
        # BSTS metrics
        # ----------------------------------------------------

        bsts_lift = row[
            "Relative_Lift_Mean"
        ]

        incremental_traffic = row[
            "Incremental_Traffic_Mean"
        ]

        counterfactual_traffic = row[
            "Counterfactual_Traffic_Mean"
        ]

        probability_positive = row[
            "Probability_Positive"
        ]

        impact_lower = row[
            "Impact_Lower_95"
        ]

        impact_upper = row[
            "Impact_Upper_95"
        ]

        # ----------------------------------------------------
        # Theoretical comp translation
        # ----------------------------------------------------

        if (
            pd.notna(actual_comp)
            and pd.notna(bsts_lift)
            and (1 + bsts_lift) != 0
        ):

            implied_counterfactual_comp = (
                (
                    1
                    + actual_comp
                )
                /
                (
                    1
                    + bsts_lift
                )
            ) - 1

            implied_incremental_comp = (
                actual_comp
                -
                implied_counterfactual_comp
            )

        else:

            implied_counterfactual_comp = np.nan
            implied_incremental_comp = np.nan

        # ----------------------------------------------------
        # Directional interpretation
        # ----------------------------------------------------

        if pd.notna(bsts_lift):

            if bsts_lift > 0:
                bsts_direction = "Positive"

            elif bsts_lift < 0:
                bsts_direction = "Negative"

            else:
                bsts_direction = "Neutral"

        else:

            bsts_direction = "Unavailable"

        if pd.notna(actual_comp):

            if actual_comp > 0:
                comp_direction = "Positive"

            elif actual_comp < 0:
                comp_direction = "Negative"

            else:
                comp_direction = "Neutral"

        else:

            comp_direction = "Unavailable"

        if pd.notna(implied_incremental_comp):

            if implied_incremental_comp > 0:
                comp_impact_direction = "Positive"

            elif implied_incremental_comp < 0:
                comp_impact_direction = "Negative"

            else:
                comp_impact_direction = "Neutral"

        else:

            comp_impact_direction = "Unavailable"

        # ----------------------------------------------------
        # Business interpretation
        # ----------------------------------------------------

        if (
            bsts_direction == "Positive"
            and
            comp_direction == "Positive"
        ):

            alignment = "Positive Impact"

        elif (
            bsts_direction == "Positive"
            and
            comp_direction == "Negative"
        ):

            alignment = "Mitigated Decline"

        elif (
            bsts_direction == "Negative"
            and
            comp_direction == "Positive"
        ):

            alignment = "Offsetting Drag"

        elif (
            bsts_direction == "Negative"
            and
            comp_direction == "Negative"
        ):

            alignment = "Negative Impact"

        elif (
            bsts_direction == "Neutral"
            or
            comp_direction == "Neutral"
        ):

            alignment = "Neutral / Mixed"

        else:

            alignment = "Unavailable"


        if alignment == "Positive Impact":

            interpretation = (
                "Positive BSTS-estimated intervention effect "
                "with positive observed comp traffic."
            )

        elif alignment == "Mitigated Decline":

            interpretation = (
                "Observed comp traffic declined, but the BSTS "
                "counterfactual implies the decline may have been "
                "larger without the intervention."
            )

        elif alignment == "Offsetting Drag":

            interpretation = (
                "Observed comp traffic remained positive, but the BSTS "
                "counterfactual implies performance could have been "
                "stronger without the intervention."
            )

        elif alignment == "Negative Impact":

            interpretation = (
                "Negative BSTS-estimated intervention effect accompanied "
                "by negative observed comp traffic."
            )

        else:

            interpretation = (
                "The DMA result is neutral, mixed, or unavailable."
            )    

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "DMA": dma,

                "DMA_Key": dma_key,

                "Intervention_Date":
                    intervention_date,

                "Post_Treatment_Months":
                    post_months,

                "Comp_Start_Month":
                    valid_comp[
                        "Traffic_Month"
                    ].min(),

                "Comp_End_Month":
                    valid_comp[
                        "Traffic_Month"
                    ].max(),

                "Comp_Months_Available":
                    valid_comp[
                        "Traffic_Month"
                    ].nunique(),

                # BSTS
                "BSTS_Observed_Traffic":
                    row[
                        "Observed_Traffic"
                    ],

                "BSTS_Counterfactual_Traffic":
                    counterfactual_traffic,

                "BSTS_Incremental_Traffic":
                    incremental_traffic,

                "BSTS_Relative_Lift":
                    bsts_lift,

                "BSTS_Probability_Positive":
                    probability_positive,

                "BSTS_Impact_Lower_95":
                    impact_lower,

                "BSTS_Impact_Upper_95":
                    impact_upper,

                # Comp
                "Actual_Exits":
                    actual_exits,

                "LY_Exits":
                    ly_exits,

                "Actual_Comp_Traffic":
                    actual_comp,

                # Theoretical translation
                "Implied_Counterfactual_Comp":
                    implied_counterfactual_comp,

                "Implied_Incremental_Comp":
                    implied_incremental_comp,

                # Interpretation
                "BSTS_Direction":
                    bsts_direction,

                "Comp_Direction":
                    comp_direction,

                "Comp_Impact_Direction":
                    comp_impact_direction,

                "DMA_Impact_Classification":
                    alignment,

                "DMA_Interpretation":
                    interpretation,    
            }
        )
        

    comparison = pd.DataFrame(
        results
    )

    logger.info(
        "Constructed %d DMA comparisons.",
        len(comparison),
    )

    return comparison


# ============================================================
# SUMMARY METRICS
# ============================================================

def build_summary(
    comparison,
):
    """
    Build report-level summary metrics.

    Classification framework
    ------------------------
    Positive Impact
        Positive BSTS lift + positive actual comp.

    Mitigated Decline
        Positive BSTS lift + negative actual comp.

    Offsetting Drag
        Negative BSTS lift + positive actual comp.

    Negative Impact
        Negative BSTS lift + negative actual comp.
    """

    if comparison.empty:

        return {
            "dma_count": 0,

            "positive_impact_count": 0,
            "mitigated_decline_count": 0,
            "offsetting_drag_count": 0,
            "negative_impact_count": 0,

            "positive_bsts_count": 0,
            "negative_bsts_count": 0,

            "positive_comp_count": 0,
            "negative_comp_count": 0,

            "mean_bsts_lift": np.nan,
            "mean_actual_comp": np.nan,
            "mean_counterfactual_comp": np.nan,
            "mean_incremental_comp": np.nan,
        }

    classification = (
        comparison[
            "DMA_Impact_Classification"
        ]
        .fillna("Unavailable")
    )

    return {
        # ----------------------------------------------------
        # Population
        # ----------------------------------------------------

        "dma_count":
            len(comparison),

        # ----------------------------------------------------
        # Business classification
        # ----------------------------------------------------

        "positive_impact_count":
            (
                classification
                == "Positive Impact"
            ).sum(),

        "mitigated_decline_count":
            (
                classification
                == "Mitigated Decline"
            ).sum(),

        "offsetting_drag_count":
            (
                classification
                == "Offsetting Drag"
            ).sum(),

        "negative_impact_count":
            (
                classification
                == "Negative Impact"
            ).sum(),

        # ----------------------------------------------------
        # Directional metrics
        # ----------------------------------------------------

        "positive_bsts_count":
            (
                comparison[
                    "BSTS_Relative_Lift"
                ]
                > 0
            ).sum(),

        "negative_bsts_count":
            (
                comparison[
                    "BSTS_Relative_Lift"
                ]
                < 0
            ).sum(),

        "positive_comp_count":
            (
                comparison[
                    "Actual_Comp_Traffic"
                ]
                > 0
            ).sum(),

        "negative_comp_count":
            (
                comparison[
                    "Actual_Comp_Traffic"
                ]
                < 0
            ).sum(),

        # ----------------------------------------------------
        # Portfolio averages
        # ----------------------------------------------------

        "mean_bsts_lift":
            comparison[
                "BSTS_Relative_Lift"
            ].mean(),

        "mean_actual_comp":
            comparison[
                "Actual_Comp_Traffic"
            ].mean(),

        "mean_counterfactual_comp":
            comparison[
                "Implied_Counterfactual_Comp"
            ].mean(),

        "mean_incremental_comp":
            comparison[
                "Implied_Incremental_Comp"
            ].mean(),
    }

# ============================================================
# FIGURES
# ============================================================

def generate_figures(
    comparison,
):
    """Generate DMA comparison figures."""

    figures = {}

    if comparison.empty:
        return figures

    # --------------------------------------------------------
    # BSTS lift vs actual comp
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    x = np.arange(
        len(comparison)
    )

    width = 0.38

    ax.bar(
        x - width / 2,
        comparison[
            "BSTS_Relative_Lift"
        ] * 100,
        width,
        label="BSTS Relative Lift",
    )

    ax.bar(
        x + width / 2,
        comparison[
            "Actual_Comp_Traffic"
        ] * 100,
        width,
        label="Actual Comp Traffic",
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        comparison["DMA"],
        rotation=75,
        ha="right",
    )

    ax.set_ylabel(
        "Percent"
    )

    ax.set_title(
        "BSTS Relative Lift vs Actual Comp Traffic"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        FIGURE_DIR
        / "bsts_vs_actual_comp.png"
    )

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    figures[
        "bsts_vs_actual_comp"
    ] = path

    # --------------------------------------------------------
    # Implied counterfactual vs actual comp
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    x = np.arange(
        len(comparison)
    )

    ax.bar(
        x - width / 2,
        comparison[
            "Actual_Comp_Traffic"
        ] * 100,
        width,
        label="Actual Comp",
    )

    ax.bar(
        x + width / 2,
        comparison[
            "Implied_Counterfactual_Comp"
        ] * 100,
        width,
        label="Implied Counterfactual Comp",
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        comparison["DMA"],
        rotation=75,
        ha="right",
    )

    ax.set_ylabel(
        "Comp Traffic (%)"
    )

    ax.set_title(
        "Actual vs BSTS-Implied Counterfactual Comp Traffic"
    )

    ax.legend()

    fig.tight_layout()

    path = (
        FIGURE_DIR
        / "actual_vs_counterfactual_comp.png"
    )

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    figures[
        "actual_vs_counterfactual_comp"
    ] = path

    # --------------------------------------------------------
    # Implied incremental comp
    # --------------------------------------------------------

    ordered = (
        comparison
        .sort_values(
            "Implied_Incremental_Comp"
        )
    )

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.barh(
        ordered["DMA"],
        ordered[
            "Implied_Incremental_Comp"
        ] * 100,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Implied Incremental Comp Traffic (percentage points)"
    )

    ax.set_title(
        "BSTS-Implied Incremental Comp Traffic"
    )

    fig.tight_layout()

    path = (
        FIGURE_DIR
        / "implied_incremental_comp.png"
    )

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    figures[
        "implied_incremental_comp"
    ] = path

    return figures


# ============================================================
# FORMATTING HELPERS
# ============================================================

def fmt_pct(
    value,
    decimals=2,
):
    """Format decimal as percentage."""

    if (
        value is None
        or pd.isna(value)
    ):
        return "N/A"

    return (
        f"{value * 100:.{decimals}f}%"
    )


def fmt_int(
    value,
):
    """Format integer."""

    if (
        value is None
        or pd.isna(value)
    ):
        return "N/A"

    return f"{value:,.0f}"


def metric_card(
    label,
    value,
    detail="",
):
    """Generate HTML metric card."""

    return f"""
    <div class="card">
        <div class="label">
            {html.escape(str(label))}
        </div>

        <div class="value">
            {html.escape(str(value))}
        </div>

        <div class="detail">
            {html.escape(str(detail))}
        </div>
    </div>
    """


def figure_html(
    path,
    title,
    report_dir,
):
    """Create HTML figure block."""

    if path is None:
        return ""

    relative_path = os.path.relpath(
        path,
        start=report_dir,
    )

    return f"""
    <div class="figure">
        <h3>
            {html.escape(title)}
        </h3>

        <img
            src="{html.escape(relative_path.replace(os.sep, '/'))}"
            alt="{html.escape(title)}"
        >
    </div>
    """
# ============================================================
# HTML REPORT
# ============================================================

def generate_html_report(
    comparison,
    summary,
    figures,
):
    """
    Generate an internal HTML report focused on
    BSTS / comp-traffic reconciliation.
    """

    logger.info(
        "Generating BSTS / comp comparison report..."
    )

    # ========================================================
    # Portfolio interpretation
    # ========================================================

    if comparison.empty:

        headline = (
            "No production-valid DMA comparisons "
            "could be constructed."
        )

        interpretation = (
            "The BSTS and comp-traffic datasets could not "
            "be matched at the DMA level."
        )

        classification_interpretation = (
            "DMA classification results were unavailable."
        )

    else:

        mean_bsts = summary[
            "mean_bsts_lift"
        ]

        mean_actual = summary[
            "mean_actual_comp"
        ]

        mean_cf = summary[
            "mean_counterfactual_comp"
        ]

        mean_incremental = summary[
            "mean_incremental_comp"
        ]

        positive_impact = summary[
            "positive_impact_count"
        ]

        mitigated = summary[
            "mitigated_decline_count"
        ]

        offsetting_drag = summary[
            "offsetting_drag_count"
        ]

        negative_impact = summary[
            "negative_impact_count"
        ]

        # ----------------------------------------------------
        # Headline
        # ----------------------------------------------------

        if (
            pd.notna(mean_incremental)
            and mean_incremental > 0
        ):

            headline = (
                "Across production-valid DMAs, the BSTS "
                "results translate into a positive average "
                "theoretical contribution to comp traffic."
            )

        elif (
            pd.notna(mean_incremental)
            and mean_incremental < 0
        ):

            headline = (
                "Across production-valid DMAs, the BSTS "
                "results translate into a negative average "
                "theoretical contribution to comp traffic."
            )

        else:

            headline = (
                "Across production-valid DMAs, the BSTS "
                "results translate into little or no average "
                "theoretical contribution to comp traffic."
            )

        # ----------------------------------------------------
        # Overall metric interpretation
        # ----------------------------------------------------

        interpretation = (
            "Across the production-valid DMAs, the BSTS model "
            f"estimated an average relative traffic lift of "
            f"{fmt_pct(mean_bsts)}. "
            f"Actual comp performance averaged "
            f"{fmt_pct(mean_actual)}, while the BSTS-implied "
            f"counterfactual comp averaged "
            f"{fmt_pct(mean_cf)}. "
            f"The resulting theoretical incremental comp "
            f"contribution averaged "
            f"{fmt_pct(mean_incremental)}."
        )

        # ----------------------------------------------------
        # Classification interpretation
        # ----------------------------------------------------

        classification_interpretation = (
            f"{fmt_int(positive_impact)} DMAs were classified "
            f"as Positive Impact, "
            f"{fmt_int(mitigated)} as Mitigated Decline, "
            f"{fmt_int(offsetting_drag)} as Offsetting Drag, "
            f"and {fmt_int(negative_impact)} as Negative Impact. "
            "This distribution highlights substantial "
            "DMA-level heterogeneity in the estimated "
            "intervention response."
        )

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    display_columns = [
        "DMA",
        "Comp_Months_Available",

        "BSTS_Incremental_Traffic",
        "BSTS_Relative_Lift",
        "BSTS_Probability_Positive",

        "Actual_Comp_Traffic",
        "Implied_Counterfactual_Comp",
        "Implied_Incremental_Comp",

        "DMA_Impact_Classification",
        "DMA_Interpretation",
    ]

    display = (
        comparison[
            display_columns
        ].copy()
        if not comparison.empty
        else pd.DataFrame(
            columns=display_columns
        )
    )

    if not display.empty:

        # ----------------------------------------------------
        # Format numeric values for report
        # ----------------------------------------------------

        for col in [
            "BSTS_Relative_Lift",
            "BSTS_Probability_Positive",
            "Actual_Comp_Traffic",
            "Implied_Counterfactual_Comp",
            "Implied_Incremental_Comp",
        ]:

            display[col] = display[
                col
            ].map(
                lambda x:
                    fmt_pct(x)
                if pd.notna(x)
                else "N/A"
            )

        display[
            "BSTS_Incremental_Traffic"
        ] = display[
            "BSTS_Incremental_Traffic"
        ].map(
            lambda x:
                fmt_int(x)
            if pd.notna(x)
            else "N/A"
        )

        # ----------------------------------------------------
        # Rename columns for report readability
        # ----------------------------------------------------

        display = display.rename(
            columns={
                "Comp_Months_Available":
                    "Comp Months",

                "BSTS_Incremental_Traffic":
                    "BSTS Incremental Traffic",

                "BSTS_Relative_Lift":
                    "BSTS Lift",

                "BSTS_Probability_Positive":
                    "Probability Positive",

                "Actual_Comp_Traffic":
                    "Actual Comp",

                "Implied_Counterfactual_Comp":
                    "Implied Counterfactual Comp",

                "Implied_Incremental_Comp":
                    "Implied Incremental Comp",

                "DMA_Impact_Classification":
                    "Impact Classification",

                "DMA_Interpretation":
                    "Interpretation",
            }
        )

    table_html = display.to_html(
        index=False,
        escape=False,
        classes="report-table",
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    fig_bsts_comp = figure_html(
        figures.get("bsts_vs_actual_comp"),
        "BSTS Relative Lift vs Actual Comp Traffic",
        REPORT_DIR,
    )

    fig_counterfactual = figure_html(
        figures.get("actual_vs_counterfactual_comp"),
                "Actual vs BSTS-Implied Counterfactual Comp",
                REPORT_DIR,
    )


    fig_incremental = figure_html(
        figures.get("implied_incremental_comp"),
                    "BSTS-Implied Incremental Comp Traffic",
                    REPORT_DIR,
    
    )

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    report_html = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<title>
BSTS / Comp Traffic Comparison
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

h1 {{
    margin: 0 0 8px 0;

    font-size: 32px;
}}

h2 {{
    color: #243b53;

    margin-top: 0;
}}

h3 {{
    color: #334e68;
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
            minmax(
                210px,
                1fr
            )
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
    font-size: 25px;

    font-weight: 700;

    color: #102a43;

    margin-top: 4px;
}}

.detail {{
    font-size: 12px;

    color: #627d98;

    margin-top: 4px;
}}

.takeaway {{
    background: #eef6ff;

    border-left:
        4px solid
        #3578c6;

    padding:
        16px 20px;

    border-radius: 6px;

    margin-top: 20px;
}}

.warning {{
    background: #fff8e6;

    border-left:
        4px solid
        #d99a00;

    padding:
        16px 20px;

    border-radius: 6px;
}}

.report-table {{
    border-collapse:
        collapse;

    width: 100%;

    background: white;

    font-size: 13px;
}}

.report-table th,
.report-table td {{
    border:
        1px solid #ddd;

    padding: 8px;

    text-align: right;
}}

.report-table th {{
    background: #eef2f7;

    color: #243b53;
}}

.report-table td:first-child,
.report-table th:first-child {{
    text-align: left;
}}

.figure {{
    margin-top: 25px;

    text-align: center;
}}

.figure img {{
    max-width: 100%;

    height: auto;

    border:
        1px solid #ddd;

    border-radius: 6px;
}}

.formula {{
    background: #f7f9fb;

    padding: 15px;

    border-radius: 6px;

    font-family:
        "Courier New",
        monospace;

    margin:
        12px 0;
}}

.small {{
    font-size: 12px;

    color: #627d98;
}}

footer {{
    color: #627d98;

    font-size: 11px;

    margin-top: 20px;
}}

@media (
    max-width: 750px
) {{

    .container {{
        padding: 15px;
    }}

}}

# .dma-results-table {{
#     width: 100%;
#     table-layout: fixed;
#     border-collapse: collapse;
#     font-size: 9px;
# }}

# .dma-results-table th,
# .dma-results-table td {{
#     padding: 4px 5px;
#     border: 1px solid #ddd;
#     word-wrap: break-word;
#     overflow-wrap: anywhere;
#     white-space: normal;
# }}

# .dma-results-table th {{
#     font-weight: bold;
# }}

# .dma-results-table th:first-child,
# .dma-results-table td:first-child {{
#     width: 18%;
# }}

# .dma-results-table th:nth-child(2),
# .dma-results-table td:nth-child(2) {{
#     width: 6%;
# }}

# .dma-results-table th:nth-child(3),
# .dma-results-table td:nth-child(3) {{
#     width: 11%;
# }}

# .dma-results-table th:nth-child(4),
# .dma-results-table td:nth-child(4) {{
#     width: 9%;
# }}

# .dma-results-table th:nth-child(5),
# .dma-results-table td:nth-child(5) {{
#     width: 9%;
# }}

# .dma-results-table th:nth-child(6),
# .dma-results-table td:nth-child(6) {{
#     width: 10%;
# }}

# .dma-results-table th:nth-child(7),
# .dma-results-table td:nth-child(7) {{
#     width: 10%;
# }}

# .dma-results-table th:nth-child(8),
# .dma-results-table td:nth-child(8) {{
#     width: 10%;
# }}

# .dma-results-table th:nth-child(9),
# .dma-results-table td:nth-child(9) {{
#     width: 8%;
# }}

# .dma-results-table th:nth-child(10),
# .dma-results-table td:nth-child(10) {{
#     width: 9%;
# }}


</style>

</head>

<body>

<div class="container">

<header>

<h1>
BSTS / Comp Traffic Comparison
</h1>

<div>
The implied counterfactual comp is mathematically derived from the BSTS relative lift and observed comp performance. It is not independently modeled comp performance and should not be interpreted as a separate causal estimate
</div>

</header>

<section>

<h2>
Treatment Population
</h2>

<div class="cards">

{metric_card(
    "Positive Impact",
    fmt_int(
        summary[
            "positive_impact_count"
        ]
    ),
    "positive BSTS effect and positive observed comp",
)}

{metric_card(
    "Mitigated Decline",
    fmt_int(
        summary[
            "mitigated_decline_count"
        ]
    ),
    "positive estimated effect despite negative observed comp",
)}

{metric_card(
    "Offsetting Drag",
    fmt_int(
        summary[
            "offsetting_drag_count"
        ]
    ),
    "negative estimated effect despite positive observed comp",
)}

{metric_card(
    "Negative Impact",
    fmt_int(
        summary[
            "negative_impact_count"
        ]
    ),
    "negative BSTS effect and negative observed comp",
)}

</div>

<div class="takeaway">

<strong>
Headline:
</strong>

{html.escape(headline)}

<p>
{html.escape(interpretation)}
</p>

<section>

<h2>
DMA Response Classification
</h2>

<p>
{html.escape(classification_interpretation)}
</p>


</div>

</section>



<section>

<h2>
How to Interpret the Comparison
</h2>

<p>
The BSTS model estimates the incremental traffic
associated with the intervention by comparing observed
traffic with a modeled counterfactual.
</p>

<p>
The comp-traffic dataset measures year-over-year traffic
performance among stores currently classified as comp stores.
Because the BSTS traffic population and comp-store population
are not identical, the two measures should not be directly
divided against one another.
</p>

<p>
Instead, the BSTS relative lift is used to construct a
theoretical comp counterfactual.
</p>

<div class="formula">

Implied Counterfactual Comp
=
((1 + Actual Comp) /
(1 + BSTS Relative Lift)) - 1

</div>

<div class="formula">

Implied Incremental Comp
=
Actual Comp -
Implied Counterfactual Comp

</div>

<p class="small">

This is a translation of the BSTS result into comp-traffic
terms. It is not a separate causal model of comp traffic and
should not be interpreted as an independent causal estimate.

</p>

</section>


<section>

<h2>
DMA-Level Results
</h2>

{table_html}

</section>


<section>

<h2>
BSTS Lift vs Actual Comp
</h2>

<p>
This comparison shows whether DMAs with positive or negative
BSTS-estimated intervention effects also exhibited positive
or negative year-over-year comp performance.
</p>

{fig_bsts_comp}

</section>


<section>

<h2>
Actual Comp vs Implied Counterfactual
</h2>

<p>
The implied counterfactual represents the theoretical comp
performance after removing the BSTS-estimated intervention
lift from the observed comp result.
</p>

{fig_counterfactual}

</section>


<section>

<h2>
Implied Incremental Comp
</h2>

<p>
This represents the theoretical percentage-point contribution
of the intervention to comp traffic under the translation
assumption described above.
</p>

{fig_incremental}

</section>


<section>

<h2>
Important Analytical Caveat
</h2>

<div class="warning">

<strong>
This is a reconciliation analysis, not a second causal model.
</strong>

<p>
The BSTS model is the causal framework used to estimate
incremental traffic. The comp calculation provides an
additional business interpretation of that result.
</p>

<p>
In particular, the translation assumes that the relative
BSTS intervention lift can be applied proportionally to the
observed comp traffic rate. This assumption allows us to
express the BSTS result in familiar comp-traffic terms, but
it does not prove that the intervention independently caused
the calculated comp percentage-point change.
</p>

</div>

</section>


<section>

<h2>
Management Interpretation
</h2>

<p>
<section>

<h2>
Management Interpretation
</h2>

<p>
The BSTS / comp comparison indicates that response to the
marketing intervention varied materially across DMAs.
The four classifications distinguish observed year-over-year
comp performance from the modeled intervention-associated
traffic effect.
</p>

<p>
<strong>Positive Impact</strong> markets exhibited both a
positive BSTS-estimated intervention effect and positive
observed comp traffic. These represent the most consistently
favorable results.
</p>

<p>
<strong>Mitigated Decline</strong> markets experienced negative
observed comp traffic, but the BSTS counterfactual suggests
performance could have been weaker without the intervention.
The intervention therefore appears to have partially offset
underlying traffic pressure.
</p>

<p>
<strong>Offsetting Drag</strong> markets retained positive
observed comp traffic, but the BSTS estimate indicates that
performance may have been stronger absent the intervention.
These markets warrant closer investigation because positive
headline comp performance may mask an unfavorable modeled
intervention effect.
</p>

<p>
<strong>Negative Impact</strong> markets exhibited both negative
observed comp traffic and a negative BSTS-estimated intervention
effect. These markets provide the clearest unfavorable signal
within the comparison framework.
</p>

</section>


<footer>

Generated from persisted BSTS causal-impact outputs and
DMA-level comp traffic outputs.

</footer>

</div>

</body>

</html>
"""

    report_path = (
        REPORT_DIR
        / "bsts_comp_comparison.html"
    )

    report_path.write_text(
        report_html,
        encoding="utf-8",
    )

    logger.info(
        "Report written to %s",
        report_path,
    )

    return report_path


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    comparison,
):
    """
    Save full DMA comparison and streamlined summary outputs.
    """

    # ========================================================
    # Full analytical comparison
    # ========================================================

    output_file = (
        TABLE_DIR
        / "bsts_comp_dma_comparison.csv"
    )

    comparison.to_csv(
        output_file,
        index=False,
    )

    logger.info(
        "Saved DMA comparison: %s",
        output_file,
    )

    # ========================================================
    # Streamlined business summary
    # ========================================================

    if not comparison.empty:

        summary_columns = [
            "DMA",
            "Post_Treatment_Months",
            "Comp_Months_Available",

            "BSTS_Incremental_Traffic",
            "BSTS_Relative_Lift",
            "BSTS_Probability_Positive",

            "Actual_Comp_Traffic",
            "Implied_Counterfactual_Comp",
            "Implied_Incremental_Comp",

            "BSTS_Direction",
            "Comp_Direction",
            "Comp_Impact_Direction",

            "DMA_Impact_Classification",
            "DMA_Interpretation",
        ]

        summary_table = (
            comparison[
                summary_columns
            ]
            .sort_values(
                "Implied_Incremental_Comp",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        summary_file = (
            TABLE_DIR
            / "bsts_comp_summary.csv"
        )

        summary_table.to_csv(
            summary_file,
            index=False,
        )

        logger.info(
            "Saved summary: %s",
            summary_file,
        )
# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("=" * 70)
    logger.info(
        "BSTS → COMP TRAFFIC COMPARISON"
    )
    logger.info("=" * 70)

    create_output_directories()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    bsts = load_bsts_results()

    comp = load_comp_traffic()

    # --------------------------------------------------------
    # Normalize DMA names
    # --------------------------------------------------------

    bsts, comp = add_dma_keys(
        bsts,
        comp,
    )

    # --------------------------------------------------------
    # Production-valid BSTS universe
    # --------------------------------------------------------

    bsts = bsts[
        bsts["DMA_Key"].notna()
    ].copy()

    logger.info(
        "Production-valid BSTS DMAs: %d",
        bsts["DMA_Key"].nunique(),
    )

    # --------------------------------------------------------
    # Build comparison
    # --------------------------------------------------------

    comparison = build_dma_comparison(
        bsts,
        comp,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        comparison
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = build_summary(
        comparison
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    figures = generate_figures(
        comparison
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_path = generate_html_report(
        comparison,
        summary,
        figures,
    )

    # --------------------------------------------------------
    # Final logging
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info(
        "BSTS → COMP TRAFFIC COMPARISON COMPLETE"
    )
    logger.info("=" * 70)

    logger.info(
        "DMAs compared: %d",
        summary["dma_count"],
    )

    logger.info(
        "BSTS positive DMAs: %d",
        summary["positive_bsts_count"],
    )

    logger.info(
        "BSTS negative DMAs: %d",
        summary["negative_bsts_count"],
    )

    logger.info(
        "Comp positive DMAs: %d",
        summary["positive_comp_count"],
    )

    logger.info(
        "Comp negative DMAs: %d",
        summary["negative_comp_count"],
    )

    # logger.info(
    #     "Directionally aligned DMAs: %d",
    #     summary["aligned_count"],
    # )

    logger.info(
        "Mean BSTS relative lift: %s",
        fmt_pct(
            summary["mean_bsts_lift"]
        ),
    )

    logger.info(
        "Mean actual comp: %s",
        fmt_pct(
            summary["mean_actual_comp"]
        ),
    )

    logger.info(
        "Mean implied counterfactual comp: %s",
        fmt_pct(
            summary["mean_counterfactual_comp"]
        ),
    )

    logger.info(
        "Mean implied incremental comp: %s",
        fmt_pct(
            summary["mean_incremental_comp"]
        ),
    )

    logger.info(
        "HTML report: %s",
        report_path,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()