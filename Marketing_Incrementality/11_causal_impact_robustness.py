"""
11_causal_impact_robustness.py

Causal Impact Robustness, DMA Heterogeneity, and
Marketing Exposure Analysis

Purpose
-------
Evaluate the robustness and heterogeneity of the causal-impact
estimates produced by:

    09_causal_impact.py
    10_causal_impact_validation.py

This script does not modify the validated BSTS estimates.

Robustness Layers
-----------------
1. DMA heterogeneity analysis
2. Placebo intervention analysis
3. Marketing spend / treatment-effect analysis

Inputs
------
Saved outputs from 10_causal_impact_validation.py
Original model panel from the processed data layer

Outputs
-------
Figures
Tables
HTML report
"""

# =========================================================
# Imports
# =========================================================

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from config import (
    FIGURE_DIR,
    TABLE_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
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
# Output Folders
# =========================================================

REPORT_DIR = (
    OUTPUT_DIR /
    "reports"
)


ROBUSTNESS_FIGURE_DIR = (
    FIGURE_DIR /
    "causal_impact_robustness"
)


ROBUSTNESS_TABLE_DIR = (
    TABLE_DIR /
    "causal_impact_robustness"
)


def create_output_folders():

    folders = [

        # -------------------------------------------------
        # Figures
        # -------------------------------------------------

        ROBUSTNESS_FIGURE_DIR,

        ROBUSTNESS_FIGURE_DIR /
        "heterogeneity",

        ROBUSTNESS_FIGURE_DIR /
        "placebo",

        ROBUSTNESS_FIGURE_DIR /
        "marketing_spend",

        ROBUSTNESS_FIGURE_DIR /
        "diagnostics",

        # -------------------------------------------------
        # Tables
        # -------------------------------------------------

        ROBUSTNESS_TABLE_DIR,

        ROBUSTNESS_TABLE_DIR /
        "heterogeneity",

        ROBUSTNESS_TABLE_DIR /
        "placebo",

        ROBUSTNESS_TABLE_DIR /
        "marketing_spend",

        # -------------------------------------------------
        # Reports
        # -------------------------------------------------

        REPORT_DIR,

    ]

    for folder in folders:

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    logger.info(
        "Robustness analysis output folders created."
    )


# =========================================================
# Figure Helper
# =========================================================

def save_plot(
    fig,
    folder,
    filename,
):
    """
    Save a robustness-analysis figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.

    folder : str
        Subfolder under ROBUSTNESS_FIGURE_DIR.

    filename : str
        Output filename without extension.
    """

    output_path = (
        ROBUSTNESS_FIGURE_DIR /
        folder /
        f"{filename}.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        f"Saved figure: {output_path}"
    )


# =========================================================
# Model Panel Loader
# =========================================================

def load_model_panel():
    """
    Load the processed modeling panel.

    The panel is used primarily for:

        - marketing spend exposure
        - DMA/store normalization metrics
        - placebo-period construction
        - independent validation of observed traffic

    Returns
    -------
    pd.DataFrame
        Cleaned model panel.
    """

    logger.info(
        "Loading model panel..."
    )

    panel_path = (
        PROCESSED_DATA_DIR /
        "model_panel.parquet"
    )

    if not panel_path.exists():

        raise FileNotFoundError(
            f"Model panel does not exist: "
            f"{panel_path}"
        )

    panel = pd.read_parquet(
        panel_path
    )

    if "Month" not in panel.columns:

        raise ValueError(
            "Model panel is missing 'Month'."
        )

    if "DMA" not in panel.columns:

        raise ValueError(
            "Model panel is missing 'DMA'."
        )

    if "Traffic" not in panel.columns:

        raise ValueError(
            "Model panel is missing 'Traffic'."
        )

    panel["Month"] = pd.to_datetime(
        panel["Month"]
    )

    panel = panel.loc[
        panel["DMA"].notna()
        &
        panel["Traffic"].notna()
    ].copy()

    if panel.empty:

        raise ValueError(
            "Model panel is empty after filtering "
            "missing DMA and Traffic values."
        )

    if panel.duplicated(
        subset=[
            "DMA",
            "Month",
        ]
    ).any():

        raise ValueError(
            "Model panel contains duplicate "
            "DMA-month observations."
        )

    panel = (
        panel
        .sort_values(
            [
                "DMA",
                "Month",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    logger.info(
        f"Loaded {len(panel):,} rows"
    )

    logger.info(
        f"DMAs: {panel['DMA'].nunique()}"
    )

    logger.info(
        f"Months: {panel['Month'].nunique()}"
    )

    logger.info(
        f"Date range: "
        f"{panel['Month'].min().date()} "
        f"to "
        f"{panel['Month'].max().date()}"
    )

    return panel


# =========================================================
# Load Causal Impact Outputs
# =========================================================

def load_causal_impact_outputs(
    table_dir=None,
):
    """
    Load validated causal-impact outputs produced by
    10_causal_impact_validation.py.

    Returns
    -------
    dict
        Dictionary containing the loaded output tables.
    """

    if table_dir is None:

        table_dir = (
            TABLE_DIR /
            "causal_impact"
        )

    table_dir = Path(
        table_dir
    )

    required_files = {

        "dma_impact":
            table_dir /
            "dma_impact_summary.csv",

        "dma_contribution":
            table_dir /
            "dma_contribution_analysis.csv",

        "dma_contribution_summary":
            table_dir /
            "dma_contribution_summary.csv",

        "aggregate_impact":
            table_dir /
            "aggregate_impact_summary.csv",

        "counterfactual_validation":
            table_dir /
            "counterfactual_validation_summary.csv",

        "bsts_validation":
            table_dir /
            "bsts_model_validation_summary.csv",

    }

    outputs = {}

    logger.info(
        "Loading validated causal-impact outputs..."
    )

    for name, path in required_files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Required causal-impact output "
                f"does not exist: {path}"
            )

        logger.info(
            f"Loading {name}: {path.name}"
        )

        outputs[name] = pd.read_csv(
            path
        )

    logger.info(
        f"Loaded {len(outputs)} causal-impact "
        "output tables."
    )

    return outputs

# =========================================================
# Load DMA Analysis
# =========================================================
def load_actual_dma_universe():
    """
    Load production DMA validation results and return the DMAs
    that passed both model and counterfactual validation.
    """

    model_path = (
        TABLE_DIR /
        "causal_impact" /
        "bsts_model_validation_summary.csv"
    )

    counterfactual_path = (
        TABLE_DIR /
        "causal_impact" /
        "counterfactual_validation_summary.csv"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"BSTS model validation summary not found: "
            f"{model_path}"
        )

    if not counterfactual_path.exists():
        raise FileNotFoundError(
            f"Counterfactual validation summary not found: "
            f"{counterfactual_path}"
        )

    model_validation = pd.read_csv(
        model_path
    )

    counterfactual_validation = pd.read_csv(
        counterfactual_path
    )

    # -------------------------------------------------
    # Normalize DMA column naming
    # -------------------------------------------------

    if "DMA" not in model_validation.columns:

        if "dma" in model_validation.columns:
            model_validation = (
                model_validation.rename(
                    columns={"dma": "DMA"}
                )
            )

        else:
            raise ValueError(
                "BSTS model validation summary does not "
                "contain a DMA column."
            )

    if "DMA" not in counterfactual_validation.columns:

        if "dma" in counterfactual_validation.columns:
            counterfactual_validation = (
                counterfactual_validation.rename(
                    columns={"dma": "DMA"}
                )
            )

        else:
            raise ValueError(
                "Counterfactual validation summary does not "
                "contain a DMA column."
            )

    # -------------------------------------------------
    # Validate required status columns
    # -------------------------------------------------

    if "converged" not in model_validation.columns:

        raise ValueError(
            "BSTS model validation summary does not contain "
            "'converged'."
        )

    if "counterfactual_valid" not in (
        counterfactual_validation.columns
    ):

        raise ValueError(
            "Counterfactual validation summary does not "
            "contain 'counterfactual_valid'."
        )

    # -------------------------------------------------
    # Normalize boolean values
    # -------------------------------------------------

    model_validation["converged"] = (
        model_validation["converged"]
        .astype(bool)
    )

    counterfactual_validation[
        "counterfactual_valid"
    ] = (
        counterfactual_validation[
            "counterfactual_valid"
        ]
        .astype(bool)
    )

    # -------------------------------------------------
    # Build eligible universe
    # -------------------------------------------------

    valid_models = set(
        model_validation.loc[
            model_validation["converged"],
            "DMA",
        ]
    )

    valid_counterfactuals = set(
        counterfactual_validation.loc[
            counterfactual_validation[
                "counterfactual_valid"
            ],
            "DMA",
        ]
    )

    eligible_dmas = sorted(
        valid_models
        &
        valid_counterfactuals
    )

    logger.info(
        f"Production eligible DMA universe: "
        f"{len(eligible_dmas)} DMAs"
    )

    logger.info(
        f"Eligible DMAs: {eligible_dmas}"
    )

    return eligible_dmas

# =========================================================
# Load Aggregate Summary
# =========================================================
def load_actual_aggregate_summary():

    path = (
        TABLE_DIR /
        "causal_impact" /
        "aggregate_impact_summary.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Production aggregate summary not found: {path}"
        )

    return pd.read_csv(
        path
    )

# =========================================================
# Existing BSTS Modeling Functions
# =========================================================

import importlib.util


BSTS_MODELING_SCRIPT_PATH = (
    Path(__file__).resolve().parent /
    "09_causal_impact.py"
)

BSTS_VALIDATION_SCRIPT_PATH = (
    Path(__file__).resolve().parent /
    "10_causal_impact_validation.py"
)


def load_module_from_path(
    module_name,
    script_path,
):
    """
    Dynamically load a Python module from a script path.
    """

    script_path = Path(
        script_path
    )

    if not script_path.exists():

        raise FileNotFoundError(
            f"Python script does not exist: "
            f"{script_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        script_path,
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            f"Unable to create module specification "
            f"for {script_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def load_bsts_modeling_module():

    return load_module_from_path(
        module_name="causal_impact_modeling",
        script_path=BSTS_MODELING_SCRIPT_PATH,
    )


def load_bsts_validation_module():

    return load_module_from_path(
        module_name="causal_impact_validation",
        script_path=BSTS_VALIDATION_SCRIPT_PATH,
    )


def load_actual_dma_impact_summary():
    """
    Load the production DMA-level causal impact summary
    generated by 10_causal_impact_validation.py.
    """

    path = (
        TABLE_DIR /
        "causal_impact" /
        "dma_impact_summary.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Production DMA impact summary not found: "
            f"{path}"
        )

    dma_impact_summary = pd.read_csv(
        path
    )

    required_columns = {
        "DMA",
        "Incremental_Traffic_Mean",
        "Incremental_Traffic_Median",
        "Impact_Lower_95",
        "Impact_Upper_95",
        "Probability_Positive",
        "Probability_Negative",
        "Relative_Lift_Median",
    }

    missing_columns = (
        required_columns
        -
        set(dma_impact_summary.columns)
    )

    if missing_columns:

        raise ValueError(
            "Production DMA impact summary is missing "
            f"required columns: "
            f"{sorted(missing_columns)}"
        )

    logger.info(
        f"Loaded production DMA impact summary: "
        f"{len(dma_impact_summary)} rows"
    )

    return dma_impact_summary

# =========================================================
# Validate Causal Impact Output Schema
# =========================================================

def validate_causal_impact_outputs(
    outputs,
):
    """
    Validate the structure of the saved causal-impact
    results before robustness analysis begins.
    """

    if not isinstance(
        outputs,
        dict,
    ):

        raise TypeError(
            "outputs must be a dictionary."
        )

    # -----------------------------------------------------
    # DMA impact
    # -----------------------------------------------------

    required_dma_impact_columns = {

        "DMA",

        "Incremental_Traffic_Mean",

        "Incremental_Traffic_Median",

        "Impact_Lower_95",

        "Impact_Upper_95",

        "Probability_Positive",

        "Relative_Lift_Median",

    }

    missing = (
        required_dma_impact_columns
        -
        set(
            outputs[
                "dma_impact"
            ].columns
        )
    )

    if missing:

        raise ValueError(
            "dma_impact_summary is missing "
            f"required columns: {sorted(missing)}"
        )

    # -----------------------------------------------------
    # Contribution
    # -----------------------------------------------------

    required_contribution_columns = {

        "DMA",

        "Incremental_Traffic_Mean",

        "Absolute_Impact_Mean",

        "Impact_Direction",

        "Model_Valid",

        "Counterfactual_Valid",

    }

    missing = (
        required_contribution_columns
        -
        set(
            outputs[
                "dma_contribution"
            ].columns
        )
    )

    if missing:

        raise ValueError(
            "dma_contribution_analysis is missing "
            f"required columns: {sorted(missing)}"
        )

    # -----------------------------------------------------
    # Counterfactual validation
    # -----------------------------------------------------

    required_cf_columns = {

        "DMA",

        "pre_r2",

        "pre_rmse_ratio",

        "pre_mae_ratio",

        "pre_coverage_95",

        "pre_mean_bias_ratio",

        "counterfactual_valid",

    }

    missing = (
        required_cf_columns
        -
        set(
            outputs[
                "counterfactual_validation"
            ].columns
        )
    )

    if missing:

        raise ValueError(
            "counterfactual_validation_summary is missing "
            f"required columns: {sorted(missing)}"
        )

    # -----------------------------------------------------
    # Aggregate
    # -----------------------------------------------------

    required_aggregate_columns = {

        "DMAs_Included",

        "DMAs_Excluded",

        "Incremental_Traffic_Mean",

        "Impact_Lower_95",

        "Impact_Upper_95",

        "Probability_Positive",

        "Probability_Negative",

        "Relative_Lift_Median",

    }

    missing = (
        required_aggregate_columns
        -
        set(
            outputs[
                "aggregate_impact"
            ].columns
        )
    )

    if missing:

        raise ValueError(
            "aggregate_impact_summary is missing "
            f"required columns: {sorted(missing)}"
        )

    logger.info(
        "Causal-impact output schema validation passed."
    )



# =========================================================
# Analyze DMA Heterogeneity
# =========================================================
def analyze_dma_heterogeneity(
    dma_impact,
    dma_contribution,
    counterfactual_validation,
    min_probability_threshold=0.95,
):
    """
    Analyze heterogeneity in estimated DMA-level treatment effects.

    Parameters
    ----------
    dma_impact : pd.DataFrame
        DMA-level causal impact summary produced by
        10_causal_impact_validation.py.

    dma_contribution : pd.DataFrame
        DMA contribution analysis produced by
        10_causal_impact_validation.py.

    counterfactual_validation : pd.DataFrame
        Pre-treatment counterfactual validation results.

    min_probability_threshold : float, default=0.95
        Posterior probability threshold used to identify DMAs with
        strong directional evidence.

    Returns
    -------
    dma_heterogeneity : pd.DataFrame
        DMA-level heterogeneity analysis.

    heterogeneity_summary : pd.DataFrame
        Portfolio-level summary of treatment-effect heterogeneity.
    """

    logger.info("=" * 70)
    logger.info(
        "Analyzing DMA-level treatment-effect heterogeneity..."
    )
    logger.info("=" * 70)

    # =====================================================
    # Validate Inputs
    # =====================================================

    if not isinstance(
        dma_impact,
        pd.DataFrame,
    ):
        raise TypeError(
            "dma_impact must be a pandas DataFrame."
        )

    if not isinstance(
        dma_contribution,
        pd.DataFrame,
    ):
        raise TypeError(
            "dma_contribution must be a pandas DataFrame."
        )

    if not isinstance(
        counterfactual_validation,
        pd.DataFrame,
    ):
        raise TypeError(
            "counterfactual_validation must be a "
            "pandas DataFrame."
        )

    if not 0 < min_probability_threshold < 1:
        raise ValueError(
            "min_probability_threshold must be between "
            "0 and 1."
        )

    # =====================================================
    # Required Columns
    # =====================================================

    required_impact_columns = {
        "DMA",
        "Incremental_Traffic_Mean",
        "Incremental_Traffic_Median",
        "Impact_Lower_95",
        "Impact_Upper_95",
        "Probability_Positive",
        "Probability_Negative",
        "Relative_Lift_Mean",
        "Relative_Lift_Median",
        "Relative_Lift_Lower_95",
        "Relative_Lift_Upper_95",
    }

    missing = (
        required_impact_columns
        -
        set(dma_impact.columns)
    )

    if missing:
        raise ValueError(
            "dma_impact is missing required columns: "
            f"{sorted(missing)}"
        )

    required_contribution_columns = {
        "DMA",
        "Absolute_Impact_Mean",
        "Impact_Direction",
    }

    missing = (
        required_contribution_columns
        -
        set(dma_contribution.columns)
    )

    if missing:
        raise ValueError(
            "dma_contribution is missing required columns: "
            f"{sorted(missing)}"
        )

    required_validation_columns = {
        "DMA",
        "pre_r2",
        "pre_rmse_ratio",
        "pre_mae_ratio",
        "pre_coverage_95",
        "pre_mean_bias_ratio",
        "counterfactual_valid",
    }

    missing = (
        required_validation_columns
        -
        set(counterfactual_validation.columns)
    )

    if missing:
        raise ValueError(
            "counterfactual_validation is missing "
            "required columns: "
            f"{sorted(missing)}"
        )

    # =====================================================
    # Copy Inputs
    # =====================================================

    impact = (
        dma_impact
        .copy()
    )

    contribution = (
        dma_contribution
        .copy()
    )

    validation = (
        counterfactual_validation
        .copy()
    )

    # =====================================================
    # Validate DMA Uniqueness
    # =====================================================

    for name, dataframe in [
        ("dma_impact", impact),
        ("dma_contribution", contribution),
        (
            "counterfactual_validation",
            validation,
        ),
    ]:

        if dataframe["DMA"].duplicated().any():

            duplicate_dmas = (
                dataframe.loc[
                    dataframe["DMA"].duplicated(
                        keep=False
                    ),
                    "DMA",
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                f"{name} contains duplicate DMA rows: "
                f"{duplicate_dmas}"
            )

    # =====================================================
    # Restrict to DMAs Eligible for Causal Interpretation
    # =====================================================

    eligible_dmas = (
        contribution.loc[
            contribution["Model_Valid"].astype(bool)
            &
            contribution["Counterfactual_Valid"].astype(bool),
            "DMA",
        ]
        .unique()
        .tolist()
    )

    if not eligible_dmas:

        raise ValueError(
            "No DMAs passed both model and "
            "counterfactual validation."
        )

    logger.info(
        f"Eligible DMAs for heterogeneity analysis: "
        f"{len(eligible_dmas)}"
    )

    excluded_from_heterogeneity = sorted(
        set(impact["DMA"])
        -
        set(eligible_dmas)
    )

    if excluded_from_heterogeneity:

        logger.info(
            "DMAs excluded from heterogeneity analysis:"
        )

        for dma in excluded_from_heterogeneity:

            logger.info(
                f"  {dma}"
            )

    impact = (
        impact[
            impact["DMA"].isin(
                eligible_dmas
            )
        ]
        .copy()
    )

    contribution = (
        contribution[
            contribution["DMA"].isin(
                eligible_dmas
            )
        ]
        .copy()
    )

    validation = (
        validation[
            validation["DMA"].isin(
                eligible_dmas
            )
        ]
        .copy()
    )

    # =====================================================
    # Merge Results
    # =====================================================

    heterogeneity = (
        impact[
            [
                "DMA",
                "Incremental_Traffic_Mean",
                "Incremental_Traffic_Median",
                "Impact_Lower_95",
                "Impact_Upper_95",
                "Probability_Positive",
                "Probability_Negative",
                "Relative_Lift_Mean",
                "Relative_Lift_Median",
                "Relative_Lift_Lower_95",
                "Relative_Lift_Upper_95",
            ]
        ]
        .merge(
            contribution[
                [
                    "DMA",
                    "Absolute_Impact_Mean",
                    "Impact_Direction",
                    "Model_Valid",
                    "Counterfactual_Valid",
                ]
            ],
            on="DMA",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            validation[
                [
                    "DMA",
                    "pre_r2",
                    "pre_rmse_ratio",
                    "pre_mae_ratio",
                    "pre_coverage_95",
                    "pre_mean_bias_ratio",
                    "counterfactual_valid",
                ]
            ],
            on="DMA",
            how="inner",
            validate="one_to_one",
        )
    )

    if heterogeneity.empty:

        raise ValueError(
            "No DMAs remained after merging eligible "
            "heterogeneity inputs."
        )

    # =====================================================
    # Validate Numeric Values
    # =====================================================

    numeric_columns = [
        "Incremental_Traffic_Mean",
        "Incremental_Traffic_Median",
        "Impact_Lower_95",
        "Impact_Upper_95",
        "Probability_Positive",
        "Probability_Negative",
        "Relative_Lift_Mean",
        "Relative_Lift_Median",
        "Relative_Lift_Lower_95",
        "Relative_Lift_Upper_95",
        "Absolute_Impact_Mean",
        "pre_r2",
        "pre_rmse_ratio",
        "pre_mae_ratio",
        "pre_coverage_95",
        "pre_mean_bias_ratio",
    ]

    for column in numeric_columns:

        invalid_mask = ~np.isfinite(
            heterogeneity[column]
        )

        if invalid_mask.any():

            invalid_dmas = (
                heterogeneity.loc[
                    invalid_mask,
                    "DMA",
                ]
                .tolist()
            )

            raise ValueError(
                f"DMA heterogeneity column "
                f"'{column}' contains non-finite "
                f"values for DMAs: {invalid_dmas}"
            )

    # =====================================================
    # Credible Interval Classification
    # =====================================================

    heterogeneity[
        "Impact_Excludes_Zero"
    ] = (

        (
            heterogeneity[
                "Impact_Lower_95"
            ] > 0
        )
        |
        (
            heterogeneity[
                "Impact_Upper_95"
            ] < 0
        )
    )

    heterogeneity[
        "Impact_Credible_Direction"
    ] = np.select(
        [
            heterogeneity[
                "Impact_Lower_95"
            ] > 0,

            heterogeneity[
                "Impact_Upper_95"
            ] < 0,
        ],
        [
            "Positive",
            "Negative",
        ],
        default="Inconclusive",
    )

    # =====================================================
    # Posterior Evidence Classification
    # =====================================================

    heterogeneity[
        "Strong_Positive_Evidence"
    ] = (
        heterogeneity[
            "Probability_Positive"
        ]
        >=
        min_probability_threshold
    )

    heterogeneity[
        "Strong_Negative_Evidence"
    ] = (
        heterogeneity[
            "Probability_Negative"
        ]
        >=
        min_probability_threshold
    )

    heterogeneity[
        "Posterior_Direction"
    ] = np.select(
        [
            heterogeneity[
                "Strong_Positive_Evidence"
            ],

            heterogeneity[
                "Strong_Negative_Evidence"
            ],
        ],
        [
            "Positive",
            "Negative",
        ],
        default="Inconclusive",
    )

    # =====================================================
    # Effect Size Metrics
    # =====================================================

    heterogeneity[
        "Impact_Interval_Width"
    ] = (
        heterogeneity[
            "Impact_Upper_95"
        ]
        -
        heterogeneity[
            "Impact_Lower_95"
        ]
    )

    heterogeneity[
        "Relative_Lift_Interval_Width"
    ] = (
        heterogeneity[
            "Relative_Lift_Upper_95"
        ]
        -
        heterogeneity[
            "Relative_Lift_Lower_95"
        ]
    )

    # =====================================================
    # Absolute Effect Rank
    # =====================================================

    heterogeneity[
        "Absolute_Impact_Rank"
    ] = (
        heterogeneity[
            "Absolute_Impact_Mean"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    # =====================================================
    # Positive / Negative Pool Classification
    # =====================================================

    heterogeneity[
        "Impact_Pool"
    ] = np.select(
        [
            heterogeneity[
                "Incremental_Traffic_Mean"
            ] > 0,

            heterogeneity[
                "Incremental_Traffic_Mean"
            ] < 0,
        ],
        [
            "Positive",
            "Negative",
        ],
        default="Neutral",
    )

    # =====================================================
    # Positive / Negative Pool Shares
    # =====================================================

    positive_total = float(
        heterogeneity.loc[
            heterogeneity[
                "Incremental_Traffic_Mean"
            ] > 0,
            "Incremental_Traffic_Mean",
        ]
        .sum()
    )

    negative_total_abs = float(
        np.abs(
            heterogeneity.loc[
                heterogeneity[
                    "Incremental_Traffic_Mean"
                ] < 0,
                "Incremental_Traffic_Mean",
            ]
        )
        .sum()
    )

    if positive_total > 0:

        heterogeneity[
            "Positive_Pool_Share"
        ] = np.where(

            heterogeneity[
                "Incremental_Traffic_Mean"
            ] > 0,

            heterogeneity[
                "Incremental_Traffic_Mean"
            ]
            /
            positive_total,

            0.0,
        )

    else:

        heterogeneity[
            "Positive_Pool_Share"
        ] = 0.0

    if negative_total_abs > 0:

        heterogeneity[
            "Negative_Pool_Share"
        ] = np.where(

            heterogeneity[
                "Incremental_Traffic_Mean"
            ] < 0,

            np.abs(
                heterogeneity[
                    "Incremental_Traffic_Mean"
                ]
            )
            /
            negative_total_abs,

            0.0,
        )

    else:

        heterogeneity[
            "Negative_Pool_Share"
        ] = 0.0

    # =====================================================
    # Sort by Estimated Impact
    # =====================================================

    heterogeneity = (
        heterogeneity
        .sort_values(
            "Incremental_Traffic_Mean",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # =====================================================
    # Cross-DMA Heterogeneity Statistics
    # =====================================================

    effects = (
        heterogeneity[
            "Incremental_Traffic_Mean"
        ]
        .to_numpy()
    )

    relative_lifts = (
        heterogeneity[
            "Relative_Lift_Mean"
        ]
        .to_numpy()
    )

    effect_mean = float(
        np.mean(effects)
    )

    effect_median = float(
        np.median(effects)
    )

    effect_std = float(
        np.std(
            effects,
            ddof=1,
        )
    ) if len(effects) > 1 else 0.0

    effect_min = float(
        np.min(effects)
    )

    effect_max = float(
        np.max(effects)
    )

    effect_range = (
        effect_max -
        effect_min
    )

    q1 = float(
        np.percentile(
            effects,
            25,
        )
    )

    q3 = float(
        np.percentile(
            effects,
            75,
        )
    )

    effect_iqr = (
        q3 -
        q1
    )

    lift_std = float(
        np.std(
            relative_lifts,
            ddof=1,
        )
    ) if len(relative_lifts) > 1 else 0.0

    # =====================================================
    # Counts
    # =====================================================

    positive_count = int(
        (
            heterogeneity[
                "Incremental_Traffic_Mean"
            ] > 0
        ).sum()
    )

    negative_count = int(
        (
            heterogeneity[
                "Incremental_Traffic_Mean"
            ] < 0
        ).sum()
    )

    inconclusive_count = int(
        (
            heterogeneity[
                "Impact_Credible_Direction"
            ]
            ==
            "Inconclusive"
        ).sum()
    )

    strong_positive_count = int(
        heterogeneity[
            "Strong_Positive_Evidence"
        ].sum()
    )

    strong_negative_count = int(
        heterogeneity[
            "Strong_Negative_Evidence"
        ].sum()
    )

    credible_positive_count = int(
        (
            heterogeneity[
                "Impact_Credible_Direction"
            ]
            ==
            "Positive"
        ).sum()
    )

    credible_negative_count = int(
        (
            heterogeneity[
                "Impact_Credible_Direction"
            ]
            ==
            "Negative"
        ).sum()
    )

    # =====================================================
    # Most Extreme DMAs
    # =====================================================

    strongest_positive_dma = (
        heterogeneity.loc[
            heterogeneity[
                "Incremental_Traffic_Mean"
            ].idxmax(),
            "DMA",
        ]
    )

    strongest_negative_dma = (
        heterogeneity.loc[
            heterogeneity[
                "Incremental_Traffic_Mean"
            ].idxmin(),
            "DMA",
        ]
    )

    largest_uncertainty_dma = (
        heterogeneity.loc[
            heterogeneity[
                "Impact_Interval_Width"
            ].idxmax(),
            "DMA",
        ]
    )

    # =====================================================
    # Summary
    # =====================================================

    heterogeneity_summary = pd.DataFrame([{

        "DMAs_Analyzed":
            len(heterogeneity),

        "Positive_DMA_Count":
            positive_count,

        "Negative_DMA_Count":
            negative_count,

        "Inconclusive_DMA_Count":
            inconclusive_count,

        "Credible_Positive_DMA_Count":
            credible_positive_count,

        "Credible_Negative_DMA_Count":
            credible_negative_count,

        "Strong_Positive_Evidence_Count":
            strong_positive_count,

        "Strong_Negative_Evidence_Count":
            strong_negative_count,

        "Mean_DMA_Impact":
            effect_mean,

        "Median_DMA_Impact":
            effect_median,

        "DMA_Impact_Std":
            effect_std,

        "DMA_Impact_Min":
            effect_min,

        "DMA_Impact_Max":
            effect_max,

        "DMA_Impact_Range":
            effect_range,

        "DMA_Impact_IQR":
            effect_iqr,

        "Mean_Relative_Lift":
            float(
                np.mean(
                    relative_lifts
                )
            ),

        "Median_Relative_Lift":
            float(
                np.median(
                    relative_lifts
                )
            ),

        "Relative_Lift_Std":
            lift_std,

        "Strongest_Positive_DMA":
            strongest_positive_dma,

        "Strongest_Negative_DMA":
            strongest_negative_dma,

        "Largest_Uncertainty_DMA":
            largest_uncertainty_dma,

        "Probability_Threshold":
            min_probability_threshold,

    }])

    # =====================================================
    # Logging
    # =====================================================

    logger.info(
        "DMA heterogeneity analysis complete."
    )

    logger.info(
        f"  DMAs analyzed             : "
        f"{len(heterogeneity)}"
    )

    logger.info(
        f"  Positive DMA estimates   : "
        f"{positive_count}"
    )

    logger.info(
        f"  Negative DMA estimates   : "
        f"{negative_count}"
    )

    logger.info(
        f"  Credible positive effects: "
        f"{credible_positive_count}"
    )

    logger.info(
        f"  Credible negative effects: "
        f"{credible_negative_count}"
    )

    logger.info(
        f"  Strong positive evidence : "
        f"{strong_positive_count}"
    )

    logger.info(
        f"  Strong negative evidence : "
        f"{strong_negative_count}"
    )

    logger.info(
        f"  DMA impact mean          : "
        f"{effect_mean:,.0f}"
    )

    logger.info(
        f"  DMA impact std           : "
        f"{effect_std:,.0f}"
    )

    logger.info(
        f"  Strongest positive DMA   : "
        f"{strongest_positive_dma}"
    )

    logger.info(
        f"  Strongest negative DMA   : "
        f"{strongest_negative_dma}"
    )

    logger.info(
        f"  Largest uncertainty DMA  : "
        f"{largest_uncertainty_dma}"
    )

    return (
        heterogeneity,
        heterogeneity_summary,
    )


def validate_placebo_counterfactual(
    model_results,
    model_data,
    min_pre_r2=0.80,
    max_pre_rmse_ratio=0.20,
    max_pre_mae_ratio=0.15,
    min_pre_coverage=0.80,
    max_mean_bias_ratio=0.10,
):
    """
    Validate the counterfactual generated for a placebo intervention.

    The placebo model should reproduce the observed pre-placebo
    period reasonably well before its apparent post-placebo effect
    is considered interpretable.

    Accuracy metrics are calculated from posterior conditional-mean
    counterfactual draws.

    Uncertainty coverage is evaluated using the posterior predictive
    interval returned by generate_counterfactual().

    Parameters
    ----------
    model_results : dict
        Results returned by the BSTS fitting and counterfactual
        generation workflow.

        Required keys:

            counterfactual_mean_draws
            counterfactual_predictive_draws
            counterfactual_predictive_mean
            counterfactual_predictive_lower
            counterfactual_predictive_upper

    model_data : dict
        BSTS model metadata.

        Required keys:

            target_dma
            y_pre
            n_pre
            dates
            intervention_date

    min_pre_r2 : float, default=0.80
        Minimum acceptable pre-placebo R².

    max_pre_rmse_ratio : float, default=0.20
        Maximum RMSE relative to mean pre-placebo traffic.

    max_pre_mae_ratio : float, default=0.15
        Maximum MAE relative to mean pre-placebo traffic.

    min_pre_coverage : float, default=0.80
        Minimum proportion of observed pre-placebo traffic values
        falling within the 95% posterior predictive interval.

    max_mean_bias_ratio : float, default=0.10
        Maximum absolute mean prediction bias relative to mean
        pre-placebo traffic.

    Returns
    -------
    validation : dict
        Structured placebo counterfactual validation results.

    validation_summary : pd.DataFrame
        One-row summary suitable for consolidation across
        placebo tests.
    """

    logger.info("=" * 70)
    logger.info(
        "Validating placebo counterfactual quality..."
    )
    logger.info("=" * 70)

    # =====================================================
    # Validate Inputs
    # =====================================================

    if not isinstance(
        model_results,
        dict,
    ):
        raise TypeError(
            "model_results must be a dictionary."
        )

    if not isinstance(
        model_data,
        dict,
    ):
        raise TypeError(
            "model_data must be a dictionary."
        )

    # =====================================================
    # Required Model Results
    # =====================================================

    required_result_keys = [
        "counterfactual_mean_draws",
        "counterfactual_predictive_draws",
        "counterfactual_predictive_mean",
        "counterfactual_predictive_lower",
        "counterfactual_predictive_upper",
    ]

    missing_results = [
        key
        for key in required_result_keys
        if key not in model_results
    ]

    if missing_results:

        raise ValueError(
            "model_results is missing required placebo "
            f"counterfactual fields: {missing_results}"
        )

    # =====================================================
    # Required Model Data
    # =====================================================

    required_data_keys = [
        "target_dma",
        "y_pre",
        "n_pre",
        "dates",
        "intervention_date",
    ]

    missing_data = [
        key
        for key in required_data_keys
        if key not in model_data
    ]

    if missing_data:

        raise ValueError(
            "model_data is missing required fields: "
            f"{missing_data}"
        )

    # =====================================================
    # Validate Thresholds
    # =====================================================

    if not 0 <= min_pre_coverage <= 1:

        raise ValueError(
            "min_pre_coverage must be between 0 and 1."
        )

    if min_pre_r2 > 1:

        raise ValueError(
            "min_pre_r2 cannot exceed 1."
        )

    if max_pre_rmse_ratio < 0:

        raise ValueError(
            "max_pre_rmse_ratio must be non-negative."
        )

    if max_pre_mae_ratio < 0:

        raise ValueError(
            "max_pre_mae_ratio must be non-negative."
        )

    if max_mean_bias_ratio < 0:

        raise ValueError(
            "max_mean_bias_ratio must be non-negative."
        )

    # =====================================================
    # Extract Metadata
    # =====================================================

    target_dma = model_data[
        "target_dma"
    ]

    y_pre = np.asarray(
        model_data["y_pre"],
        dtype=float,
    )

    n_pre = int(
        model_data["n_pre"]
    )

    dates = pd.DatetimeIndex(
        model_data["dates"]
    )

    intervention_date = pd.Timestamp(
        model_data["intervention_date"]
    )

    # =====================================================
    # Validate Date Structure
    # =====================================================

    if len(dates) == 0:

        raise ValueError(
            f"{target_dma}: no modeling dates available."
        )

    if not dates.is_monotonic_increasing:

        raise ValueError(
            f"{target_dma}: modeling dates are not "
            "monotonically increasing."
        )

    if dates.has_duplicates:

        raise ValueError(
            f"{target_dma}: modeling dates contain "
            "duplicates."
        )

    if n_pre <= 0:

        raise ValueError(
            f"{target_dma}: n_pre must be greater than zero."
        )

    if n_pre > len(dates):

        raise ValueError(
            f"{target_dma}: n_pre={n_pre} exceeds "
            f"available periods={len(dates)}."
        )

    # =====================================================
    # Validate Pre-Placebo Alignment
    # =====================================================

    pre_dates = dates[
        :n_pre
    ]

    if not np.all(
        pre_dates < intervention_date
    ):

        raise ValueError(
            f"{target_dma}: n_pre={n_pre} includes "
            "dates on or after the placebo intervention."
        )

    if len(dates) > n_pre:

        post_dates = dates[
            n_pre:
        ]

        if not np.all(
            post_dates >= intervention_date
        ):

            raise ValueError(
                f"{target_dma}: dates after n_pre contain "
                "values before the placebo intervention."
            )

    # =====================================================
    # Validate Observed Pre-Placebo Traffic
    # =====================================================

    if len(y_pre) != n_pre:

        raise ValueError(
            f"{target_dma}: y_pre contains "
            f"{len(y_pre)} observations but n_pre="
            f"{n_pre}."
        )

    if not np.isfinite(
        y_pre
    ).all():

        raise ValueError(
            f"{target_dma}: y_pre contains "
            "non-finite values."
        )

    # =====================================================
    # Extract Counterfactual Outputs
    # =====================================================

    counterfactual_mean_draws = np.asarray(
        model_results[
            "counterfactual_mean_draws"
        ],
        dtype=float,
    )

    counterfactual_predictive_draws = np.asarray(
        model_results[
            "counterfactual_predictive_draws"
        ],
        dtype=float,
    )

    counterfactual_predictive_mean = np.asarray(
        model_results[
            "counterfactual_predictive_mean"
        ],
        dtype=float,
    )

    counterfactual_predictive_lower = np.asarray(
        model_results[
            "counterfactual_predictive_lower"
        ],
        dtype=float,
    )

    counterfactual_predictive_upper = np.asarray(
        model_results[
            "counterfactual_predictive_upper"
        ],
        dtype=float,
    )

    # =====================================================
    # Validate Draw Arrays
    # =====================================================

    if counterfactual_mean_draws.ndim != 2:

        raise ValueError(
            f"{target_dma}: counterfactual_mean_draws "
            "must be 2-dimensional. Received shape "
            f"{counterfactual_mean_draws.shape}."
        )

    if counterfactual_predictive_draws.ndim != 2:

        raise ValueError(
            f"{target_dma}: counterfactual_predictive_draws "
            "must be 2-dimensional. Received shape "
            f"{counterfactual_predictive_draws.shape}."
        )

    n_draws, n_full = (
        counterfactual_mean_draws.shape
    )

    predictive_draws, predictive_periods = (
        counterfactual_predictive_draws.shape
    )

    if predictive_draws != n_draws:

        raise ValueError(
            f"{target_dma}: conditional and predictive "
            "counterfactual draws contain different "
            f"numbers of posterior draws: "
            f"{n_draws} vs {predictive_draws}."
        )

    if predictive_periods != n_full:

        raise ValueError(
            f"{target_dma}: conditional and predictive "
            "counterfactual draws contain different "
            f"numbers of periods: "
            f"{n_full} vs {predictive_periods}."
        )

    if n_full != len(dates):

        raise ValueError(
            f"{target_dma}: counterfactual contains "
            f"{n_full} periods but dates contains "
            f"{len(dates)}."
        )

    # =====================================================
    # Validate Summary Shapes
    # =====================================================

    expected_vector_shape = (
        n_full,
    )

    for name, values in {

        "counterfactual_predictive_mean":
            counterfactual_predictive_mean,

        "counterfactual_predictive_lower":
            counterfactual_predictive_lower,

        "counterfactual_predictive_upper":
            counterfactual_predictive_upper,

    }.items():

        if values.shape != expected_vector_shape:

            raise ValueError(
                f"{target_dma}: {name} has shape "
                f"{values.shape}; expected "
                f"{expected_vector_shape}."
            )

    # =====================================================
    # Validate Numeric Values
    # =====================================================

    arrays_to_validate = {

        "counterfactual_mean_draws":
            counterfactual_mean_draws,

        "counterfactual_predictive_draws":
            counterfactual_predictive_draws,

        "counterfactual_predictive_mean":
            counterfactual_predictive_mean,

        "counterfactual_predictive_lower":
            counterfactual_predictive_lower,

        "counterfactual_predictive_upper":
            counterfactual_predictive_upper,

    }

    for name, values in (
        arrays_to_validate.items()
    ):

        if not np.isfinite(
            values
        ).all():

            raise ValueError(
                f"{target_dma}: {name} contains "
                "non-finite values."
            )

    # =====================================================
    # Restrict to Pre-Placebo Period
    # =====================================================

    mean_draws_pre = (
        counterfactual_mean_draws[
            :,
            :n_pre,
        ]
    )

    predictive_draws_pre = (
        counterfactual_predictive_draws[
            :,
            :n_pre,
        ]
    )

    predictive_mean_pre = (
        counterfactual_predictive_mean[
            :n_pre
        ]
    )

    predictive_lower_pre = (
        counterfactual_predictive_lower[
            :n_pre
        ]
    )

    predictive_upper_pre = (
        counterfactual_predictive_upper[
            :n_pre
        ]
    )

    # =====================================================
    # Posterior Conditional Mean
    # =====================================================

    # Mean across posterior draws.
    conditional_mean_pre = np.mean(
        mean_draws_pre,
        axis=0,
    )

    # =====================================================
    # Validate Prediction Interval
    # =====================================================

    if np.any(
        predictive_lower_pre
        >
        predictive_upper_pre
    ):

        raise ValueError(
            f"{target_dma}: posterior predictive "
            "interval lower bounds exceed upper bounds."
        )

    # =====================================================
    # Basic Prediction Errors
    #
    # Accuracy is evaluated against the posterior
    # conditional mean, not observation-noise draws.
    # =====================================================

    errors = (
        conditional_mean_pre
        -
        y_pre
    )

    absolute_errors = np.abs(
        errors
    )

    squared_errors = (
        errors ** 2
    )

    pre_rmse = float(
        np.sqrt(
            np.mean(
                squared_errors
            )
        )
    )

    pre_mae = float(
        np.mean(
            absolute_errors
        )
    )

    mean_observed = float(
        np.mean(
            y_pre
        )
    )

    if (
        not np.isfinite(
            mean_observed
        )
        or
        mean_observed == 0
    ):

        raise ValueError(
            f"{target_dma}: mean pre-placebo traffic "
            "is zero or non-finite."
        )

    # =====================================================
    # Relative Error Metrics
    # =====================================================

    pre_rmse_ratio = (
        pre_rmse
        /
        abs(
            mean_observed
        )
    )

    pre_mae_ratio = (
        pre_mae
        /
        abs(
            mean_observed
        )
    )

    mean_bias = float(
        np.mean(
            errors
        )
    )

    mean_bias_ratio = (
        abs(
            mean_bias
        )
        /
        abs(
            mean_observed
        )
    )

    # =====================================================
    # Pre-Placebo R²
    # =====================================================

    ss_res = float(
        np.sum(
            squared_errors
        )
    )

    ss_tot = float(
        np.sum(
            (
                y_pre
                -
                mean_observed
            )
            ** 2
        )
    )

    if ss_tot > 0:

        pre_r2 = float(
            1.0
            -
            (
                ss_res
                /
                ss_tot
            )
        )

    else:

        pre_r2 = np.nan

    # =====================================================
    # 95% Posterior Predictive Coverage
    # =====================================================

    inside_interval = (
        (
            y_pre
            >=
            predictive_lower_pre
        )
        &
        (
            y_pre
            <=
            predictive_upper_pre
        )
    )

    pre_coverage_95 = float(
        np.mean(
            inside_interval
        )
    )

    # =====================================================
    # Interval Width
    # =====================================================

    interval_width = (
        predictive_upper_pre
        -
        predictive_lower_pre
    )

    mean_interval_width = float(
        np.mean(
            interval_width
        )
    )

    mean_interval_width_ratio = (
        mean_interval_width
        /
        abs(
            mean_observed
        )
    )

    # =====================================================
    # Prediction Correlation
    # =====================================================

    observed_std = float(
        np.std(
            y_pre
        )
    )

    predicted_std = float(
        np.std(
            conditional_mean_pre
        )
    )

    if (
        observed_std > 0
        and
        predicted_std > 0
    ):

        pre_correlation = float(
            np.corrcoef(
                y_pre,
                conditional_mean_pre,
            )[0, 1]
        )

    else:

        pre_correlation = np.nan

    # =====================================================
    # Validation Criteria
    # =====================================================

    r2_pass = (
        np.isfinite(
            pre_r2
        )
        and
        pre_r2 >= min_pre_r2
    )

    rmse_pass = (
        np.isfinite(
            pre_rmse_ratio
        )
        and
        pre_rmse_ratio <= max_pre_rmse_ratio
    )

    mae_pass = (
        np.isfinite(
            pre_mae_ratio
        )
        and
        pre_mae_ratio <= max_pre_mae_ratio
    )

    coverage_pass = (
        np.isfinite(
            pre_coverage_95
        )
        and
        pre_coverage_95 >= min_pre_coverage
    )

    bias_pass = (
        np.isfinite(
            mean_bias_ratio
        )
        and
        mean_bias_ratio <= max_mean_bias_ratio
    )

    # =====================================================
    # Overall Validation
    # =====================================================

    counterfactual_valid = all(
        [
            r2_pass,
            rmse_pass,
            mae_pass,
            coverage_pass,
            bias_pass,
        ]
    )

    # =====================================================
    # Structured Result
    # =====================================================

    validation = {

        "DMA":
            target_dma,

        "Placebo_Date":
            intervention_date,

        "n_pre":
            n_pre,

        "posterior_draws":
            n_draws,

        "pre_rmse":
            pre_rmse,

        "pre_mae":
            pre_mae,

        "pre_r2":
            pre_r2,

        "pre_rmse_ratio":
            pre_rmse_ratio,

        "pre_mae_ratio":
            pre_mae_ratio,

        "pre_mean_bias":
            mean_bias,

        "pre_mean_bias_ratio":
            mean_bias_ratio,

        "pre_correlation":
            pre_correlation,

        "pre_coverage_95":
            pre_coverage_95,

        "mean_interval_width":
            mean_interval_width,

        "mean_interval_width_ratio":
            mean_interval_width_ratio,

        "r2_pass":
            r2_pass,

        "rmse_pass":
            rmse_pass,

        "mae_pass":
            mae_pass,

        "coverage_pass":
            coverage_pass,

        "bias_pass":
            bias_pass,

        "counterfactual_valid":
            counterfactual_valid,

        "min_pre_r2":
            min_pre_r2,

        "max_pre_rmse_ratio":
            max_pre_rmse_ratio,

        "max_pre_mae_ratio":
            max_pre_mae_ratio,

        "min_pre_coverage":
            min_pre_coverage,

        "max_mean_bias_ratio":
            max_mean_bias_ratio,

        "coverage_interval_type":
            "posterior_predictive",

    }

    # =====================================================
    # One-Row Summary
    # =====================================================

    validation_summary = pd.DataFrame(
        [validation]
    )

    # =====================================================
    # Logging
    # =====================================================

    if counterfactual_valid:

        logger.info(
            f"{target_dma}: placebo counterfactual "
            "validation PASSED."
        )

    else:

        logger.warning(
            f"{target_dma}: placebo counterfactual "
            "validation FAILED."
        )

    logger.info(
        f"  Placebo date          : "
        f"{intervention_date.date()}"
    )

    logger.info(
        f"  Pre-treatment periods : "
        f"{n_pre}"
    )

    logger.info(
        f"  Posterior draws       : "
        f"{n_draws:,}"
    )

    logger.info(
        f"  Pre-treatment R²     : "
        f"{pre_r2:.4f}"
    )

    logger.info(
        f"  Pre-treatment RMSE   : "
        f"{pre_rmse:,.2f} "
        f"({pre_rmse_ratio:.2%})"
    )

    logger.info(
        f"  Pre-treatment MAE    : "
        f"{pre_mae:,.2f} "
        f"({pre_mae_ratio:.2%})"
    )

    logger.info(
        f"  Mean bias            : "
        f"{mean_bias:,.2f} "
        f"({mean_bias_ratio:.2%})"
    )

    logger.info(
        f"  95% coverage         : "
        f"{pre_coverage_95:.2%}"
    )

    logger.info(
        f"  Mean interval width  : "
        f"{mean_interval_width:,.2f}"
    )

    logger.info(
        f"  Correlation          : "
        f"{pre_correlation:.4f}"
    )

    logger.info(
        f"  Overall validation   : "
        f"{counterfactual_valid}"
    )

    return (
        validation,
        validation_summary,
    )

# =========================================================
# DMA-Level Placebo Analysis
# =========================================================
def run_placebo_analysis(
    panel,
    bsts_modeling,
    bsts_validation,
    actual_intervention_date,
    placebo_dates=None,
    min_pre_periods=6,
    min_post_periods=3,
    include_trend=True,
    include_seasonality=False,
    draws=1000,
    tune=1000,
    chains=4,
    cores=None,
    target_accept=0.95,
    random_seed=42,
):
    """
    Run placebo intervention tests using the same BSTS framework
    used for the primary causal-impact analysis.

    The placebo analysis is explicitly restricted to information
    available before the real intervention.

    For a placebo date T_p and actual intervention date T_a:

        Training period:
            Month < T_p

        Placebo post-period:
            T_p <= Month < T_a

        Data on or after T_a are never used.

    This prevents the actual treatment period from contaminating
    the placebo test.

    Parameters
    ----------
    panel : pd.DataFrame
        Original model panel.

    bsts_modeling : module
        Module containing the BSTS model-building functions from
        09_causal_impact.py.

    bsts_validation : module
        Module containing counterfactual and impact functions from
        10_causal_impact_validation.py.

    actual_intervention_date : str or pd.Timestamp
        Actual intervention date. Placebo dates must occur strictly
        before this date.

    placebo_dates : sequence of timestamps, optional
        Explicit placebo dates to evaluate.

        If None, valid placebo dates are automatically generated
        from the pre-intervention data.

    min_pre_periods : int, default=6
        Minimum number of observations before a placebo date.

    min_post_periods : int, default=3
        Minimum number of untreated observations between the
        placebo date and the actual intervention.

    include_trend : bool, default=True
        Whether to include the deterministic trend.

    include_seasonality : bool, default=False
        Whether to include Fourier seasonality.

    draws : int, default=1000
        Posterior draws per chain.

    tune : int, default=1000
        Tuning draws per chain.

    chains : int, default=4
        Number of posterior chains.

    cores : int, optional
        Number of sampling cores.

    target_accept : float, default=0.95
        NUTS target acceptance rate.

    random_seed : int, default=42
        Base random seed.

    Returns
    -------
    placebo_results : pd.DataFrame
        DMA-by-placebo-date placebo impact results.

    placebo_summary : pd.DataFrame
        Portfolio-level placebo summary.

    placebo_draws : pd.DataFrame
        Posterior cumulative placebo impact draws.

    placebo_validation : pd.DataFrame
        Counterfactual validation results for each placebo model.
    """

    logger.info("=" * 70)
    logger.info(
        "Running placebo intervention analysis..."
    )
    logger.info("=" * 70)

    # =====================================================
    # Validate Panel
    # =====================================================

    if not isinstance(
        panel,
        pd.DataFrame,
    ):
        raise TypeError(
            "panel must be a pandas DataFrame."
        )

    required_columns = {
        "DMA",
        "Month",
        "Traffic",
    }

    missing_columns = (
        required_columns
        -
        set(panel.columns)
    )

    if missing_columns:
        raise ValueError(
            "Panel is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    panel = panel.copy()

    panel["Month"] = pd.to_datetime(
        panel["Month"]
    )

    panel = (
        panel.loc[
            panel["DMA"].notna()
            &
            panel["Traffic"].notna()
        ]
        .sort_values(
            [
                "DMA",
                "Month",
            ]
        )
        .copy()
    )

    if panel.empty:
        raise ValueError(
            "Panel is empty after validation."
        )

    # =====================================================
    # Validate Dates
    # =====================================================

    actual_intervention_date = pd.Timestamp(
        actual_intervention_date
    )

    if actual_intervention_date not in set(
        panel["Month"]
    ):

        logger.warning(
            f"Actual intervention date "
            f"{actual_intervention_date.date()} "
            "is not present in the panel."
        )

    # =====================================================
    # Validate Configuration
    # =====================================================

    if (
        not isinstance(
            min_pre_periods,
            int,
        )
        or
        min_pre_periods < 3
    ):

        raise ValueError(
            "min_pre_periods must be an integer >= 3."
        )

    if (
        not isinstance(
            min_post_periods,
            int,
        )
        or
        min_post_periods < 1
    ):

        raise ValueError(
            "min_post_periods must be a positive integer."
        )

    # =====================================================
    # Truncate Panel at Actual Intervention
    #
    # IMPORTANT:
    # No observations on or after the real intervention
    # are allowed into placebo model construction.
    # =====================================================

    placebo_panel = (
        panel.loc[
            panel["Month"]
            <
            actual_intervention_date
        ]
        .copy()
    )

    if placebo_panel.empty:

        raise ValueError(
            "No pre-intervention observations are available "
            "for placebo analysis."
        )

    # =====================================================
    # Determine Available Months
    # =====================================================

    pre_intervention_months = (
        pd.DatetimeIndex(
            placebo_panel["Month"].unique()
        )
        .sort_values()
    )

    if len(pre_intervention_months) < (
        min_pre_periods
        +
        min_post_periods
    ):

        raise ValueError(
            "Insufficient pre-intervention history for "
            "the requested placebo configuration."
        )

    # =====================================================
    # Determine Placebo Dates
    #
    # Valid placebo date must:
    #
    #   1. occur before actual intervention
    #   2. have >= min_pre_periods before it
    #   3. have >= min_post_periods before actual intervention
    # =====================================================

    if placebo_dates is None:

        candidate_dates = []

        for placebo_date in (
            pre_intervention_months
        ):

            placebo_date = pd.Timestamp(
                placebo_date
            )

            pre_count = int(
                (
                    pre_intervention_months
                    <
                    placebo_date
                ).sum()
            )

            post_count = int(
                (
                    (
                        pre_intervention_months
                        >=
                        placebo_date
                    )
                    &
                    (
                        pre_intervention_months
                        <
                        actual_intervention_date
                    )
                ).sum()
            )

            if (
                pre_count >= min_pre_periods
                and
                post_count >= min_post_periods
            ):

                candidate_dates.append(
                    placebo_date
                )

        placebo_dates = pd.DatetimeIndex(
            candidate_dates
        )

    else:

        placebo_dates = pd.DatetimeIndex(
            pd.to_datetime(
                placebo_dates
            )
        )

        # -------------------------------------------------
        # Strictly exclude actual/future intervention dates
        # -------------------------------------------------

        placebo_dates = (
            placebo_dates[
                placebo_dates
                <
                actual_intervention_date
            ]
            .sort_values()
            .unique()
        )

    if len(placebo_dates) == 0:

        raise ValueError(
            "No valid placebo dates were identified."
        )

    # =====================================================
    # Identify DMAs
    # =====================================================

    dmas = (
        placebo_panel["DMA"]
        .dropna()
        .unique()
        .tolist()
    )

    logger.info(
        f"Actual intervention date: "
        f"{actual_intervention_date.date()}"
    )

    logger.info(
        f"Placebo dates: "
        f"{list(placebo_dates)}"
    )

    logger.info(
        f"DMAs available for placebo analysis: "
        f"{len(dmas)}"
    )

    # =====================================================
    # Result Containers
    # =====================================================

    placebo_rows = []

    placebo_draw_rows = []

    placebo_validation_rows = []

    # =====================================================
    # Run Placebo Tests
    # =====================================================

    for placebo_index, placebo_date in enumerate(
        placebo_dates
    ):

        placebo_date = pd.Timestamp(
            placebo_date
        )

        logger.info("=" * 70)
        logger.info(
            f"Placebo intervention: "
            f"{placebo_date.date()}"
        )
        logger.info("=" * 70)

        # -------------------------------------------------
        # Calendar-Level Placebo Periods
        # -------------------------------------------------

        calendar_pre_months = (
            pre_intervention_months[
                pre_intervention_months
                <
                placebo_date
            ]
        )

        calendar_post_months = (
            pre_intervention_months[
                (
                    pre_intervention_months
                    >=
                    placebo_date
                )
                &
                (
                    pre_intervention_months
                    <
                    actual_intervention_date
                )
            ]
        )

        global_pre_periods = len(
            calendar_pre_months
        )

        global_post_periods = len(
            calendar_post_months
        )

        logger.info(
            f"  Calendar pre-periods  : "
            f"{global_pre_periods}"
        )

        logger.info(
            f"  Calendar placebo-post : "
            f"{global_post_periods}"
        )

        if global_pre_periods < min_pre_periods:

            logger.warning(
                f"Skipping placebo "
                f"{placebo_date.date()}: "
                f"only {global_pre_periods} "
                "pre-periods."
            )

            continue

        if global_post_periods < min_post_periods:

            logger.warning(
                f"Skipping placebo "
                f"{placebo_date.date()}: "
                f"only {global_post_periods} "
                "untreated post-periods."
            )

            continue

        # -------------------------------------------------
        # Every DMA
        # -------------------------------------------------

        for dma_index, target_dma in enumerate(
            dmas
        ):

            logger.info(
                f"Running placebo for "
                f"{target_dma}"
            )

            # -------------------------------------------------
            # DMA-specific data
            # -------------------------------------------------

            dma_panel = (
                placebo_panel.loc[
                    placebo_panel["DMA"].eq(
                        target_dma
                    )
                ]
                .sort_values(
                    "Month"
                )
                .copy()
            )

            if dma_panel.empty:

                logger.warning(
                    f"{target_dma}: no observations. "
                    "Skipping."
                )

                continue

            pre_dma = dma_panel.loc[
                dma_panel["Month"]
                <
                placebo_date
            ]

            placebo_post_dma = dma_panel.loc[
                (
                    dma_panel["Month"]
                    >=
                    placebo_date
                )
                &
                (
                    dma_panel["Month"]
                    <
                    actual_intervention_date
                )
            ]

            dma_pre_periods = len(
                pre_dma
            )

            dma_post_periods = len(
                placebo_post_dma
            )

            if dma_pre_periods < min_pre_periods:

                logger.warning(
                    f"{target_dma}: placebo "
                    f"{placebo_date.date()} has "
                    f"{dma_pre_periods} pre-periods. "
                    "Skipping."
                )

                continue

            if dma_post_periods < min_post_periods:

                logger.warning(
                    f"{target_dma}: placebo "
                    f"{placebo_date.date()} has "
                    f"{dma_post_periods} untreated "
                    "post-periods. Skipping."
                )

                continue

            # -------------------------------------------------
            # Prepare Controls
            #
            # IMPORTANT:
            # We pass placebo_panel, which ends before the
            # actual intervention.
            # -------------------------------------------------

            try:

                dma_timeseries = (
                    bsts_modeling.prepare_dma_timeseries(
                        panel=placebo_panel,
                        intervention_date=placebo_date,
                    )
                )

                treated_dmas = [
                    target_dma
                ]

                (
                    candidate_controls,
                    control_summary,
                ) = (
                    bsts_modeling.get_candidate_controls(
                        placebo_panel,
                        treated_dmas,
                    )
                )

                control_matrix = (
                    bsts_modeling.build_control_matrix(
                        dma_timeseries=dma_timeseries,
                        candidate_controls=candidate_controls,
                        intervention_date=placebo_date,
                    )
                )

                bsts_modeling.validate_control_matrix(
                    control_matrix
                )

                (
                    selected_controls,
                    control_selection_summary,
                ) = (
                    bsts_modeling.select_controls(
                        treated_dmas,
                        dma_timeseries,
                        control_matrix=control_matrix,
                    )
                )

            except Exception as exc:

                logger.warning(
                    f"{target_dma}: placebo "
                    f"{placebo_date.date()} control "
                    f"selection failed: {exc}"
                )

                continue

            if not selected_controls:

                logger.warning(
                    f"{target_dma}: placebo "
                    f"{placebo_date.date()} produced "
                    "no selected controls."
                )

                continue

            # -------------------------------------------------
            # Prepare BSTS Inputs
            # -------------------------------------------------

            try:

                prepared_inputs = (
                    bsts_modeling.prepare_causal_impact_inputs(
                        dma_timeseries=dma_timeseries,
                        target_dma=target_dma,
                        selected_controls=selected_controls,
                        intervention_date=placebo_date,
                    )
                )

                prepared_pre_count = len(
                    prepared_inputs[
                        "y_pre"
                    ]
                )

                prepared_full_count = len(
                    prepared_inputs[
                        "y_full"
                    ]
                )

                if prepared_pre_count < min_pre_periods:

                    logger.warning(
                        f"{target_dma}: placebo "
                        f"{placebo_date.date()} model "
                        f"contains only "
                        f"{prepared_pre_count} "
                        "pre-treatment observations."
                    )

                    continue

                # -------------------------------------------------
                # Ensure the generated model period ends before
                # the actual intervention.
                # -------------------------------------------------

                prepared_dates = pd.DatetimeIndex(
                    prepared_inputs[
                        "dates"
                    ]
                )

                if (
                    len(prepared_dates) > 0
                    and
                    prepared_dates.max()
                    >=
                    actual_intervention_date
                ):

                    raise ValueError(
                        f"{target_dma}: placebo "
                        f"{placebo_date.date()} model "
                        "contains observations on or after "
                        "the actual intervention date."
                    )

                # -------------------------------------------------
                # Configure Model
                # -------------------------------------------------

                model, model_data = (
                    bsts_modeling.configure_bsts_model(
                        prepared_inputs=prepared_inputs,
                        include_trend=include_trend,
                        include_seasonality=include_seasonality,
                    )
                )

                # -------------------------------------------------
                # Fit Model
                # -------------------------------------------------

                placebo_seed = (
                    random_seed
                    +
                    placebo_index * 1000
                    +
                    dma_index
                )

                model_results = (
                    bsts_modeling.fit_bsts_model(
                        model=model,
                        model_data=model_data,
                        draws=draws,
                        tune=tune,
                        chains=chains,
                        cores=cores,
                        target_accept=target_accept,
                        random_seed=placebo_seed,
                    )
                )

                # -------------------------------------------------
                # Optional Posterior Diagnostics
                #
                # We record divergences here. A placebo model
                # with divergent transitions should not silently
                # become a successful placebo.
                # -------------------------------------------------

                divergences = int(
                    model_results.get(
                        "divergences",
                        0,
                    )
                )

                divergence_rate = float(
                    model_results.get(
                        "divergence_rate",
                        np.nan,
                    )
                )

                # -------------------------------------------------
                # Generate Conditional Counterfactual
                # -------------------------------------------------

                counterfactual_results = (
                    bsts_validation.generate_counterfactual(
                        idata=model_results["idata"],
                        model_data=model_data,
                        include_observation_noise=False,
                        random_seed=placebo_seed,
                    )
                )

                model_results.update(
                    counterfactual_results
                )

                # -------------------------------------------------
                # Validate Placebo Counterfactual
                # -------------------------------------------------

                (
                    cf_validation,
                    cf_summary,
                ) = validate_placebo_counterfactual(
                    model_results=model_results,
                    model_data=model_data,
                )

                # -------------------------------------------------
                # Save Validation Information
                # -------------------------------------------------

                placebo_validation_row = (
                    cf_validation.copy()
                )

                placebo_validation_row[
                    "Divergences"
                ] = divergences

                placebo_validation_row[
                    "Divergence_Rate"
                ] = divergence_rate

                placebo_validation_row[
                    "Selected_Controls"
                ] = len(
                    selected_controls
                )

                placebo_validation_row[
                    "Prepared_Model_Periods"
                ] = prepared_full_count

                placebo_validation_rows.append(
                    placebo_validation_row
                )

                # -------------------------------------------------
                # Require Counterfactual Validation
                # -------------------------------------------------

                if not cf_validation[
                    "counterfactual_valid"
                ]:

                    logger.warning(
                        f"{target_dma}: placebo "
                        f"{placebo_date.date()} "
                        "counterfactual validation failed."
                    )

                    continue

                # -------------------------------------------------
                # Require Clean Posterior
                #
                # This prevents a divergent placebo model from
                # contaminating the placebo distribution.
                # -------------------------------------------------

                if divergences > 0:

                    logger.warning(
                        f"{target_dma}: placebo "
                        f"{placebo_date.date()} has "
                        f"{divergences} divergent transitions. "
                        "Excluding from placebo impact results."
                    )

                    continue

                # -------------------------------------------------
                # Extract Placebo Impact
                #
                # extract_dma_impact() will use only dates from
                # placebo_date through the final pre-intervention
                # month because model_data stops before the actual
                # intervention.
                # -------------------------------------------------

                (
                    placebo_summary,
                    placebo_monthly,
                    placebo_impact_draws,
                    placebo_counterfactual_post,
                ) = bsts_validation.extract_dma_impact(
                    model_results=model_results,
                    model_data=model_data,
                    panel=placebo_panel,
                )

                # -------------------------------------------------
                # Extract Point Estimates
                # -------------------------------------------------

                result_row = (
                    placebo_summary.iloc[0]
                )

                impact_mean = float(
                    result_row[
                        "Incremental_Traffic_Mean"
                    ]
                )

                impact_median = float(
                    result_row[
                        "Incremental_Traffic_Median"
                    ]
                )

                impact_lower = float(
                    result_row[
                        "Impact_Lower_95"
                    ]
                )

                impact_upper = float(
                    result_row[
                        "Impact_Upper_95"
                    ]
                )

                probability_positive = float(
                    result_row[
                        "Probability_Positive"
                    ]
                )

                probability_negative = float(
                    result_row[
                        "Probability_Negative"
                    ]
                )

                relative_lift = float(
                    result_row[
                        "Relative_Lift_Median"
                    ]
                )

                # -------------------------------------------------
                # Store Placebo Result
                # -------------------------------------------------

                placebo_rows.append({

                    "DMA":
                        target_dma,

                    "Placebo_Date":
                        placebo_date,

                    "Actual_Intervention_Date":
                        actual_intervention_date,

                    "Pre_Treatment_Months":
                        dma_pre_periods,

                    "Placebo_Post_Months":
                        dma_post_periods,

                    "Selected_Controls":
                        len(
                            selected_controls
                        ),

                    "Incremental_Traffic_Mean":
                        impact_mean,

                    "Incremental_Traffic_Median":
                        impact_median,

                    "Impact_Lower_95":
                        impact_lower,

                    "Impact_Upper_95":
                        impact_upper,

                    "Probability_Positive":
                        probability_positive,

                    "Probability_Negative":
                        probability_negative,

                    "Relative_Lift_Median":
                        relative_lift,

                    "Counterfactual_Valid":
                        True,

                    "Model_Divergences":
                        divergences,

                    "Model_Divergence_Rate":
                        divergence_rate,

                })

                # -------------------------------------------------
                # Store Posterior Cumulative Impact Draws
                # -------------------------------------------------

                impact_draws = np.asarray(
                    placebo_impact_draws,
                    dtype=float,
                )

                if impact_draws.ndim != 2:

                    raise ValueError(
                        f"{target_dma}: placebo impact "
                        "draws must be 2-dimensional."
                    )

                cumulative_placebo_draws = (
                    impact_draws.sum(
                        axis=1
                    )
                )

                for draw_index, draw_value in enumerate(
                    cumulative_placebo_draws
                ):

                    placebo_draw_rows.append({

                        "DMA":
                            target_dma,

                        "Placebo_Date":
                            placebo_date,

                        "Actual_Intervention_Date":
                            actual_intervention_date,

                        "Draw":
                            draw_index,

                        "Cumulative_Impact":
                            float(
                                draw_value
                            ),

                    })

                logger.info(
                    f"{target_dma}: placebo "
                    f"{placebo_date.date()} "
                    "PASSED."
                )

            except Exception as exc:

                logger.exception(
                    f"{target_dma}: placebo "
                    f"{placebo_date.date()} failed: "
                    f"{exc}"
                )

                continue

    # =====================================================
    # Build Result Tables
    # =====================================================

    placebo_results = pd.DataFrame(
        placebo_rows
    )

    placebo_draws = pd.DataFrame(
        placebo_draw_rows
    )

    placebo_validation = pd.DataFrame(
        placebo_validation_rows
    )

    if placebo_results.empty:

        raise ValueError(
            "No valid placebo impact results were generated."
        )

    # =====================================================
    # Placebo Summary Metrics
    # =====================================================

    impact_values = (
        placebo_results[
            "Incremental_Traffic_Mean"
        ]
        .to_numpy(
            dtype=float
        )
    )

    relative_lifts = (
        placebo_results[
            "Relative_Lift_Median"
        ]
        .to_numpy(
            dtype=float
        )
    )

    placebo_summary = pd.DataFrame([{

        "Actual_Intervention_Date":
            actual_intervention_date,

        "Placebo_Tests":
            len(placebo_results),

        "DMA_Count":
            placebo_results[
                "DMA"
            ]
            .nunique(),

        "Placebo_Date_Count":
            placebo_results[
                "Placebo_Date"
            ]
            .nunique(),

        "Mean_Placebo_Impact":
            float(
                np.mean(
                    impact_values
                )
            ),

        "Median_Placebo_Impact":
            float(
                np.median(
                    impact_values
                )
            ),

        "Placebo_Impact_Std":
            float(
                np.std(
                    impact_values,
                    ddof=1,
                )
            )
            if len(
                impact_values
            ) > 1
            else 0.0,

        "Placebo_Impact_Min":
            float(
                np.min(
                    impact_values
                )
            ),

        "Placebo_Impact_Max":
            float(
                np.max(
                    impact_values
                )
            ),

        "Mean_Placebo_Relative_Lift":
            float(
                np.mean(
                    relative_lifts
                )
            ),

        "Median_Placebo_Relative_Lift":
            float(
                np.median(
                    relative_lifts
                )
            ),

        "Placebo_Positive_Rate":
            float(
                np.mean(
                    impact_values > 0
                )
            ),

        "Placebo_Negative_Rate":
            float(
                np.mean(
                    impact_values < 0
                )
            ),

        "Placebo_Zero_Crossing_Rate":
            float(
                np.mean(
                    (
                        placebo_results[
                            "Impact_Lower_95"
                        ]
                        <= 0
                    )
                    &
                    (
                        placebo_results[
                            "Impact_Upper_95"
                        ]
                        >= 0
                    )
                )
            ),

        "Strong_Positive_Posterior_Rate":
            float(
                np.mean(
                    placebo_results[
                        "Probability_Positive"
                    ]
                    >= 0.95
                )
            ),

        "Strong_Negative_Posterior_Rate":
            float(
                np.mean(
                    placebo_results[
                        "Probability_Negative"
                    ]
                    >= 0.95
                )
            ),

        "Mean_Selected_Controls":
            float(
                placebo_results[
                    "Selected_Controls"
                ]
                .mean()
            ),

        "Median_Selected_Controls":
            float(
                placebo_results[
                    "Selected_Controls"
                ]
                .median()
            ),

        "Mean_Pre_Treatment_Months":
            float(
                placebo_results[
                    "Pre_Treatment_Months"
                ]
                .mean()
            ),

        "Mean_Placebo_Post_Months":
            float(
                placebo_results[
                    "Placebo_Post_Months"
                ]
                .mean()
            ),

    }])

    # =====================================================
    # Logging
    # =====================================================

    logger.info("=" * 70)
    logger.info(
        "Placebo intervention analysis complete."
    )
    logger.info("=" * 70)

    logger.info(
        f"  Actual intervention   : "
        f"{actual_intervention_date.date()}"
    )

    logger.info(
        f"  Valid placebo tests   : "
        f"{len(placebo_results)}"
    )

    logger.info(
        f"  DMAs tested           : "
        f"{placebo_results['DMA'].nunique()}"
    )

    logger.info(
        f"  Placebo dates tested  : "
        f"{placebo_results['Placebo_Date'].nunique()}"
    )

    logger.info(
        f"  Mean placebo impact   : "
        f"{placebo_summary.iloc[0]['Mean_Placebo_Impact']:,.0f}"
    )

    logger.info(
        f"  Median placebo impact : "
        f"{placebo_summary.iloc[0]['Median_Placebo_Impact']:,.0f}"
    )

    logger.info(
        f"  Positive rate         : "
        f"{placebo_summary.iloc[0]['Placebo_Positive_Rate']:.2%}"
    )

    logger.info(
        f"  95% intervals crossing "
        f"zero                   : "
        f"{placebo_summary.iloc[0]['Placebo_Zero_Crossing_Rate']:.2%}"
    )

    logger.info(
        f"  Strong positive rate  : "
        f"{placebo_summary.iloc[0]['Strong_Positive_Posterior_Rate']:.2%}"
    )

    logger.info(
        f"  Strong negative rate  : "
        f"{placebo_summary.iloc[0]['Strong_Negative_Posterior_Rate']:.2%}"
    )

    return (
        placebo_results,
        placebo_summary,
        placebo_draws,
        placebo_validation,
    )


# =========================================================
# Market-Spend Analysis
# =========================================================
def analyze_marketing_spend_effects(
    panel,
    dma_impact_summary,
    intervention_date,
    eligible_dmas,
    expected_dma_count=None,
):
    """
    Analyze the relationship between DMA-level marketing spend
    changes and estimated BSTS traffic treatment effects.

    This is a dose-response / association analysis. It does not
    establish that marketing spend itself caused the estimated
    BSTS treatment effects.

    Parameters
    ----------
    panel : pd.DataFrame
        Model panel containing DMA, Month, and Spend.

    dma_impact_summary : pd.DataFrame
        Production DMA-level causal-impact summary generated by
        10_causal_impact_validation.py.

    intervention_date : str or pd.Timestamp
        Actual intervention date.

    eligible_dmas : sequence of str
        Production-valid DMA universe to analyze.

        This should be the same DMA universe used by the primary
        aggregate causal-impact analysis.

    expected_dma_count : int, default=8
        Expected number of production-valid DMAs.

    Returns
    -------
    spend_impact_analysis : pd.DataFrame
        DMA-level marketing spend and causal-impact comparison.

    spend_relationship_summary : pd.DataFrame
        Pearson/Spearman association summary.

    spend_direction_summary : pd.DataFrame
        Summary by spend-change direction.

    spend_aggregate_summary : pd.DataFrame
        Aggregate spend and impact metrics across the eligible
        DMA universe.

    Notes
    -----
    Incremental_Post_Spend is calculated as:

        Monthly_Spend_Change * Post_Month_Count

    Impact_Per_Incremental_Spend_Dollar is:

        Incremental_Traffic_Mean / Incremental_Post_Spend

    These are descriptive efficiency metrics and should not be
    interpreted as causal ROAS or incremental traffic caused per
    dollar of marketing spend.
    """

    logger.info("=" * 70)
    logger.info(
        "Analyzing marketing spend effects..."
    )
    logger.info("=" * 70)

    # =====================================================
    # Validate Inputs
    # =====================================================

    if not isinstance(
        panel,
        pd.DataFrame,
    ):
        raise TypeError(
            "panel must be a pandas DataFrame."
        )

    if not isinstance(
        dma_impact_summary,
        pd.DataFrame,
    ):
        raise TypeError(
            "dma_impact_summary must be a pandas DataFrame."
        )

    if eligible_dmas is None:
        raise ValueError(
            "eligible_dmas must be provided explicitly."
        )

    eligible_dmas = list(
        dict.fromkeys(
            str(dma)
            for dma in eligible_dmas
            if pd.notna(dma)
        )
    )

    if expected_dma_count is not None:

        if len(eligible_dmas) != expected_dma_count:

            raise ValueError(
                f"Expected exactly "
                f"{expected_dma_count} eligible DMAs, "
                f"but received {len(eligible_dmas)}: "
                f"{eligible_dmas}"
            )

    if not eligible_dmas:

        raise ValueError(
            "eligible_dmas is empty."
        )

    required_panel_columns = {
        "DMA",
        "Month",
        "Spend",
    }

    missing_panel_columns = (
        required_panel_columns
        -
        set(panel.columns)
    )

    if missing_panel_columns:

        raise ValueError(
            "Panel is missing required columns: "
            f"{sorted(missing_panel_columns)}"
        )

    required_impact_columns = {
        "DMA",
        "Incremental_Traffic_Mean",
        "Incremental_Traffic_Median",
        "Impact_Lower_95",
        "Impact_Upper_95",
        "Probability_Positive",
        "Relative_Lift_Median",
    }

    missing_impact_columns = (
        required_impact_columns
        -
        set(dma_impact_summary.columns)
    )

    if missing_impact_columns:

        raise ValueError(
            "dma_impact_summary is missing required "
            f"columns: {sorted(missing_impact_columns)}"
        )

    # =====================================================
    # Normalize Data
    # =====================================================

    panel = panel.copy()

    panel["Month"] = pd.to_datetime(
        panel["Month"]
    )

    panel["Spend"] = pd.to_numeric(
        panel["Spend"],
        errors="coerce",
    )

    panel = panel.loc[
        panel["DMA"].notna()
        &
        panel["Month"].notna()
        &
        panel["Spend"].notna()
    ].copy()

    dma_impact_summary = (
        dma_impact_summary.copy()
    )

    intervention_date = pd.Timestamp(
        intervention_date
    )

    # =====================================================
    # Restrict Impact Results to Eligible DMA Universe
    # =====================================================

    dma_impact_summary = dma_impact_summary.loc[
        dma_impact_summary["DMA"].isin(
            eligible_dmas
        )
    ].copy()

    matched_impact_dmas = set(
        dma_impact_summary["DMA"]
    )

    missing_impact_dmas = sorted(
        set(eligible_dmas)
        -
        matched_impact_dmas
    )

    if missing_impact_dmas:

        raise ValueError(
            "Eligible DMAs are missing from "
            "dma_impact_summary: "
            f"{missing_impact_dmas}"
        )

    if (
        dma_impact_summary["DMA"]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "dma_impact_summary contains duplicate "
            "DMA rows within the eligible universe."
        )

    # =====================================================
    # Determine Pre/Post Periods
    # =====================================================

    pre_panel = panel.loc[
        panel["Month"] < intervention_date
    ].copy()

    post_panel = panel.loc[
        panel["Month"] >= intervention_date
    ].copy()

    if pre_panel.empty:

        raise ValueError(
            "No pre-intervention spend observations found."
        )

    if post_panel.empty:

        raise ValueError(
            "No post-intervention spend observations found."
        )

    # =====================================================
    # Restrict Spend Panel to Eligible DMAs
    # =====================================================

    pre_panel = pre_panel.loc[
        pre_panel["DMA"].isin(
            eligible_dmas
        )
    ].copy()

    post_panel = post_panel.loc[
        post_panel["DMA"].isin(
            eligible_dmas
        )
    ].copy()

    if pre_panel.empty:

        raise ValueError(
            "No pre-intervention spend observations remain "
            "for the eligible DMA universe."
        )

    if post_panel.empty:

        raise ValueError(
            "No post-intervention spend observations remain "
            "for the eligible DMA universe."
        )

    # =====================================================
    # DMA-Level Spend Aggregation
    #
    # Average monthly spend is used because the pre/post
    # periods have different numbers of observations.
    # =====================================================

    pre_spend = (
        pre_panel
        .groupby("DMA")["Spend"]
        .agg(
            Pre_Monthly_Spend_Mean="mean",
            Pre_Total_Spend="sum",
            Pre_Month_Count="count",
        )
        .reset_index()
    )

    post_spend = (
        post_panel
        .groupby("DMA")["Spend"]
        .agg(
            Post_Monthly_Spend_Mean="mean",
            Post_Total_Spend="sum",
            Post_Month_Count="count",
        )
        .reset_index()
    )

    # =====================================================
    # Validate Spend Coverage
    # =====================================================

    pre_spend_dmas = set(
        pre_spend["DMA"]
    )

    post_spend_dmas = set(
        post_spend["DMA"]
    )

    missing_pre_spend = sorted(
        set(eligible_dmas)
        -
        pre_spend_dmas
    )

    missing_post_spend = sorted(
        set(eligible_dmas)
        -
        post_spend_dmas
    )

    if missing_pre_spend:

        raise ValueError(
            "Eligible DMAs are missing pre-treatment "
            f"spend data: {missing_pre_spend}"
        )

    if missing_post_spend:

        raise ValueError(
            "Eligible DMAs are missing post-treatment "
            f"spend data: {missing_post_spend}"
        )

    # =====================================================
    # Merge Spend Metrics
    # =====================================================

    spend_analysis = (
        pd.DataFrame(
            {
                "DMA":
                    eligible_dmas
            }
        )
        .merge(
            pre_spend,
            on="DMA",
            how="left",
        )
        .merge(
            post_spend,
            on="DMA",
            how="left",
        )
    )

    # =====================================================
    # Spend Changes
    # =====================================================

    spend_analysis[
        "Monthly_Spend_Change"
    ] = (
        spend_analysis[
            "Post_Monthly_Spend_Mean"
        ]
        -
        spend_analysis[
            "Pre_Monthly_Spend_Mean"
        ]
    )

    spend_analysis[
        "Monthly_Spend_Change_Pct"
    ] = np.divide(
        spend_analysis[
            "Monthly_Spend_Change"
        ],
        spend_analysis[
            "Pre_Monthly_Spend_Mean"
        ],
        out=np.full(
            len(spend_analysis),
            np.nan,
            dtype=float,
        ),
        where=(
            spend_analysis[
                "Pre_Monthly_Spend_Mean"
            ]
            != 0
        ),
    )

    # =====================================================
    # Incremental Post-Treatment Spend
    #
    # Approximate additional spend during the post period
    # relative to maintaining the pre-treatment monthly
    # average.
    #
    # IMPORTANT:
    # This is a descriptive spend-change measure, not a
    # causal incremental-spend estimate.
    # =====================================================

    spend_analysis[
        "Incremental_Post_Spend"
    ] = (
        spend_analysis[
            "Monthly_Spend_Change"
        ]
        *
        spend_analysis[
            "Post_Month_Count"
        ]
    )

    # =====================================================
    # Spend Direction
    # =====================================================

    spend_analysis[
        "Spend_Direction"
    ] = np.select(
        [
            spend_analysis[
                "Monthly_Spend_Change"
            ] > 0,

            spend_analysis[
                "Monthly_Spend_Change"
            ] < 0,
        ],
        [
            "Increased",
            "Decreased",
        ],
        default="Unchanged",
    )

    # =====================================================
    # Post Spend Share
    # =====================================================

    total_post_spend = float(
        spend_analysis[
            "Post_Total_Spend"
        ]
        .sum()
    )

    if total_post_spend > 0:

        spend_analysis[
            "Post_Spend_Share"
        ] = (
            spend_analysis[
                "Post_Total_Spend"
            ]
            /
            total_post_spend
        )

    else:

        spend_analysis[
            "Post_Spend_Share"
        ] = np.nan

    # =====================================================
    # Merge Causal Impact Results
    # =====================================================

    spend_impact_analysis = (
        spend_analysis
        .merge(
            dma_impact_summary[
                [
                    "DMA",
                    "Incremental_Traffic_Mean",
                    "Incremental_Traffic_Median",
                    "Impact_Lower_95",
                    "Impact_Upper_95",
                    "Probability_Positive",
                    "Probability_Negative",
                    "Relative_Lift_Median",
                ]
            ],
            on="DMA",
            how="inner",
        )
    )

    if len(
        spend_impact_analysis
    ) != len(
        eligible_dmas
    ):

        raise ValueError(
            "Marketing spend analysis did not retain all "
            "eligible DMAs. "
            f"Expected {len(eligible_dmas)}, "
            f"received {len(spend_impact_analysis)}."
        )

    # =====================================================
    # Absolute Impact
    # =====================================================

    spend_impact_analysis[
        "Absolute_Impact"
    ] = (
        spend_impact_analysis[
            "Incremental_Traffic_Mean"
        ]
        .abs()
    )

    # =====================================================
    # Impact Direction
    # =====================================================

    spend_impact_analysis[
        "Impact_Direction"
    ] = np.select(
        [
            spend_impact_analysis[
                "Incremental_Traffic_Mean"
            ] > 0,

            spend_impact_analysis[
                "Incremental_Traffic_Mean"
            ] < 0,
        ],
        [
            "Positive",
            "Negative",
        ],
        default="Neutral",
    )

    # =====================================================
    # Impact Efficiency
    #
    # Cumulative post-period incremental traffic divided
    # by cumulative incremental spend.
    #
    # This is descriptive only and is NOT causal ROAS.
    # =====================================================

    spend_impact_analysis[
        "Impact_Per_Incremental_Spend_Dollar"
    ] = np.divide(
        spend_impact_analysis[
            "Incremental_Traffic_Mean"
        ],
        spend_impact_analysis[
            "Incremental_Post_Spend"
        ],
        out=np.full(
            len(spend_impact_analysis),
            np.nan,
            dtype=float,
        ),
        where=(
            spend_impact_analysis[
                "Incremental_Post_Spend"
            ]
            != 0
        ),
    )

    spend_impact_analysis[
        "Impact_Per_Incremental_Spend_1K"
    ] = (
        spend_impact_analysis[
            "Impact_Per_Incremental_Spend_Dollar"
        ]
        *
        1000.0
    )

    # =====================================================
    # Incremental Spend Share
    # =====================================================

    total_incremental_spend = float(
        spend_impact_analysis[
            "Incremental_Post_Spend"
        ]
        .sum()
    )

    if total_incremental_spend > 0:

        spend_impact_analysis[
            "Incremental_Spend_Share"
        ] = (
            spend_impact_analysis[
                "Incremental_Post_Spend"
            ]
            /
            total_incremental_spend
        )

    else:

        spend_impact_analysis[
            "Incremental_Spend_Share"
        ] = np.nan

    # =====================================================
    # Spend-Weighted Impact
    #
    # DMA impact weighted by post-treatment spend share.
    #
    # This is a descriptive portfolio statistic, not a
    # causal estimator.
    # =====================================================

    spend_impact_analysis[
        "Spend_Weighted_Impact"
    ] = (
        spend_impact_analysis[
            "Incremental_Traffic_Mean"
        ]
        *
        spend_impact_analysis[
            "Post_Spend_Share"
        ]
    )

    spend_impact_analysis[
        "Spend_Weighted_Relative_Lift"
    ] = (
        spend_impact_analysis[
            "Relative_Lift_Median"
        ]
        *
        spend_impact_analysis[
            "Post_Spend_Share"
        ]
    )

    # =====================================================
    # Rank DMAs by Spend Change
    # =====================================================

    spend_impact_analysis[
        "Spend_Change_Rank"
    ] = (
        spend_impact_analysis[
            "Monthly_Spend_Change"
        ]
        .rank(
            ascending=False,
            method="min",
        )
    )

    spend_impact_analysis[
        "Spend_Change_Pct_Rank"
    ] = (
        spend_impact_analysis[
            "Monthly_Spend_Change_Pct"
        ]
        .rank(
            ascending=False,
            method="min",
        )
    )

    spend_impact_analysis[
        "Impact_Rank"
    ] = (
        spend_impact_analysis[
            "Incremental_Traffic_Mean"
        ]
        .rank(
            ascending=False,
            method="min",
        )
    )

    spend_impact_analysis[
        "Efficiency_Rank"
    ] = (
        spend_impact_analysis[
            "Impact_Per_Incremental_Spend_Dollar"
        ]
        .rank(
            ascending=False,
            method="min",
        )
    )

    # =====================================================
    # Validate Numeric Analysis Columns
    # =====================================================

    numeric_columns = [
        "Monthly_Spend_Change",
        "Monthly_Spend_Change_Pct",
        "Incremental_Post_Spend",
        "Incremental_Traffic_Mean",
        "Incremental_Traffic_Median",
        "Relative_Lift_Median",
        "Impact_Per_Incremental_Spend_Dollar",
        "Impact_Per_Incremental_Spend_1K",
    ]

    for column in numeric_columns:

        values = (
            spend_impact_analysis[
                column
            ]
            .to_numpy(
                dtype=float
            )
        )

        finite_values = (
            values[
                np.isfinite(
                    values
                )
            ]
        )

        if len(finite_values) < len(values):

            logger.warning(
                f"Marketing spend analysis column "
                f"'{column}' contains "
                f"{len(values) - len(finite_values)} "
                "non-finite values."
            )

    # =====================================================
    # Association Analysis
    # =====================================================

    correlation_rows = []

    def add_relationship(
        spend_column,
        impact_column,
        relationship_name,
    ):
        """
        Calculate Pearson and Spearman relationships
        while safely handling missing values.
        """

        from scipy.stats import (
            pearsonr,
            spearmanr,
        )

        values = (
            spend_impact_analysis[
                [
                    spend_column,
                    impact_column,
                ]
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if len(values) < 3:

            logger.warning(
                f"Insufficient observations for "
                f"{relationship_name}."
            )

            return

        x = values[
            spend_column
        ].to_numpy(
            dtype=float
        )

        y = values[
            impact_column
        ].to_numpy(
            dtype=float
        )

        if (
            np.std(x) == 0
            or
            np.std(y) == 0
        ):

            pearson_r = np.nan
            pearson_p = np.nan

        else:

            pearson_r, pearson_p = (
                pearsonr(
                    x,
                    y,
                )
            )

        if np.std(x) == 0:

            spearman_rho = np.nan
            spearman_p = np.nan

        else:

            spearman_rho, spearman_p = (
                spearmanr(
                    x,
                    y,
                )
            )

        correlation_rows.append({

            "Relationship":
                relationship_name,

            "Spend_Metric":
                spend_column,

            "Impact_Metric":
                impact_column,

            "N":
                len(values),

            "Pearson_R":
                float(
                    pearson_r
                )
                if np.isfinite(
                    pearson_r
                )
                else np.nan,

            "Pearson_P_Value":
                float(
                    pearson_p
                )
                if np.isfinite(
                    pearson_p
                )
                else np.nan,

            "Spearman_Rho":
                float(
                    spearman_rho
                )
                if np.isfinite(
                    spearman_rho
                )
                else np.nan,

            "Spearman_P_Value":
                float(
                    spearman_p
                )
                if np.isfinite(
                    spearman_p
                )
                else np.nan,

        })

    # =====================================================
    # Primary Relationships
    # =====================================================

    add_relationship(
        spend_column="Monthly_Spend_Change",
        impact_column="Incremental_Traffic_Mean",
        relationship_name=(
            "Spend Change vs Incremental Traffic"
        ),
    )

    add_relationship(
        spend_column="Monthly_Spend_Change_Pct",
        impact_column="Incremental_Traffic_Mean",
        relationship_name=(
            "Spend % Change vs Incremental Traffic"
        ),
    )

    add_relationship(
        spend_column="Monthly_Spend_Change_Pct",
        impact_column="Relative_Lift_Median",
        relationship_name=(
            "Spend % Change vs Relative Lift"
        ),
    )

    add_relationship(
        spend_column="Post_Monthly_Spend_Mean",
        impact_column="Incremental_Traffic_Mean",
        relationship_name=(
            "Post Spend Level vs Incremental Traffic"
        ),
    )

    add_relationship(
        spend_column="Incremental_Post_Spend",
        impact_column="Incremental_Traffic_Mean",
        relationship_name=(
            "Incremental Post Spend vs Incremental Traffic"
        ),
    )

    add_relationship(
        spend_column="Incremental_Post_Spend",
        impact_column="Relative_Lift_Median",
        relationship_name=(
            "Incremental Post Spend vs Relative Lift"
        ),
    )

    spend_relationship_summary = pd.DataFrame(
        correlation_rows
    )

    # =====================================================
    # Spend Direction Summary
    # =====================================================

    direction_summary = (
        spend_impact_analysis
        .groupby(
            "Spend_Direction",
            dropna=False,
        )
        .agg(
            DMA_Count=(
                "DMA",
                "nunique",
            ),

            Mean_Spend_Change=(
                "Monthly_Spend_Change",
                "mean",
            ),

            Mean_Spend_Change_Pct=(
                "Monthly_Spend_Change_Pct",
                "mean",
            ),

            Mean_Incremental_Post_Spend=(
                "Incremental_Post_Spend",
                "mean",
            ),

            Mean_Incremental_Traffic=(
                "Incremental_Traffic_Mean",
                "mean",
            ),

            Median_Incremental_Traffic=(
                "Incremental_Traffic_Median",
                "median",
            ),

            Mean_Relative_Lift=(
                "Relative_Lift_Median",
                "mean",
            ),

            Positive_Impact_Rate=(
                "Incremental_Traffic_Mean",
                lambda x: np.mean(
                    x > 0
                ),
            ),

            Mean_Probability_Positive=(
                "Probability_Positive",
                "mean",
            ),

            Mean_Impact_Per_Incremental_Spend_1K=(
                "Impact_Per_Incremental_Spend_1K",
                "mean",
            ),
        )
        .reset_index()
    )

    # =====================================================
    # Aggregate Spend / Impact Summary
    # =====================================================

    total_incremental_traffic = float(
        spend_impact_analysis[
            "Incremental_Traffic_Mean"
        ]
        .sum()
    )

    total_incremental_post_spend = float(
        spend_impact_analysis[
            "Incremental_Post_Spend"
        ]
        .sum()
    )

    aggregate_impact_per_spend = (
        total_incremental_traffic
        /
        total_incremental_post_spend
        if total_incremental_post_spend != 0
        else np.nan
    )

    weighted_relative_lift = float(
        spend_impact_analysis[
            "Spend_Weighted_Relative_Lift"
        ]
        .sum()
    )

    spend_weighted_impact = float(
        spend_impact_analysis[
            "Spend_Weighted_Impact"
        ]
        .sum()
    )

    positive_dma_count = int(
        np.sum(
            spend_impact_analysis[
                "Incremental_Traffic_Mean"
            ]
            > 0
        )
    )

    negative_dma_count = int(
        np.sum(
            spend_impact_analysis[
                "Incremental_Traffic_Mean"
            ]
            < 0
        )
    )

    spend_aggregate_summary = pd.DataFrame([{

        "Eligible_DMA_Count":
            len(
                eligible_dmas
            ),

        "Total_Pre_Treatment_Spend":
            float(
                spend_impact_analysis[
                    "Pre_Total_Spend"
                ]
                .sum()
            ),

        "Total_Post_Treatment_Spend":
            float(
                spend_impact_analysis[
                    "Post_Total_Spend"
                ]
                .sum()
            ),

        "Total_Incremental_Post_Spend":
            total_incremental_post_spend,

        "Total_Incremental_Traffic_Mean":
            total_incremental_traffic,

        "Total_Absolute_Impact":
            float(
                spend_impact_analysis[
                    "Absolute_Impact"
                ]
                .sum()
            ),

        "Aggregate_Impact_Per_Incremental_Spend_Dollar":
            aggregate_impact_per_spend,

        "Aggregate_Impact_Per_Incremental_Spend_1K":
            aggregate_impact_per_spend
            *
            1000.0
            if np.isfinite(
                aggregate_impact_per_spend
            )
            else np.nan,

        "Spend_Weighted_Impact":
            spend_weighted_impact,

        "Spend_Weighted_Relative_Lift":
            weighted_relative_lift,

        "Positive_DMA_Count":
            positive_dma_count,

        "Negative_DMA_Count":
            negative_dma_count,

        "Positive_DMA_Share":
            positive_dma_count
            /
            len(
                eligible_dmas
            ),

        "Negative_DMA_Share":
            negative_dma_count
            /
            len(
                eligible_dmas
            ),

    }])

    # =====================================================
    # Sort Output
    # =====================================================

    spend_impact_analysis = (
        spend_impact_analysis
        .sort_values(
            "Monthly_Spend_Change",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # =====================================================
    # Logging
    # =====================================================

    logger.info("=" * 70)
    logger.info(
        "Marketing spend analysis complete."
    )
    logger.info("=" * 70)

    logger.info(
        f"  Eligible DMAs          : "
        f"{len(eligible_dmas)}"
    )

    logger.info(
        f"  DMAs analyzed          : "
        f"{len(spend_impact_analysis)}"
    )

    logger.info(
        f"  Spend increased        : "
        f"{(
            spend_impact_analysis[
                'Spend_Direction'
            ]
            == 'Increased'
        ).sum()}"
    )

    logger.info(
        f"  Spend decreased        : "
        f"{(
            spend_impact_analysis[
                'Spend_Direction'
            ]
            == 'Decreased'
        ).sum()}"
    )

    logger.info(
        f"  Total incremental spend: "
        f"${total_incremental_post_spend:,.0f}"
    )

    logger.info(
        f"  Total incremental traffic: "
        f"{total_incremental_traffic:,.0f}"
    )

    if np.isfinite(
        aggregate_impact_per_spend
    ):

        logger.info(
            f"  Incremental traffic / "
            f"$1 incremental spend: "
            f"{aggregate_impact_per_spend:.3f}"
        )

        logger.info(
            f"  Incremental traffic / "
            f"$1K incremental spend: "
            f"{aggregate_impact_per_spend * 1000:,.1f}"
        )

    logger.info(
        f"  Spend-weighted impact : "
        f"{spend_weighted_impact:,.0f}"
    )

    logger.info(
        f"  Spend-weighted lift   : "
        f"{weighted_relative_lift:.2%}"
    )

    logger.info(
        f"  Positive-impact DMAs  : "
        f"{positive_dma_count}"
    )

    logger.info(
        f"  Negative-impact DMAs  : "
        f"{negative_dma_count}"
    )

    if not spend_relationship_summary.empty:

        logger.info(
            "Spend / impact relationships:"
        )

        for _, row in (
            spend_relationship_summary
            .iterrows()
        ):

            logger.info(
                f"  {row['Relationship']}: "
                f"Pearson r="
                f"{row['Pearson_R']:.3f}, "
                f"Spearman rho="
                f"{row['Spearman_Rho']:.3f}"
            )

    return (
        spend_impact_analysis,
        spend_relationship_summary,
        direction_summary,
        spend_aggregate_summary,
    )

# =========================================================
# Main
# =========================================================

def main():

    logger.info("=" * 70)
    logger.info(
        "Begin Causal Impact Robustness Analysis"
    )
    logger.info("=" * 70)

    # =====================================================
    # Configuration
    # =====================================================

    create_output_folders()

    # =====================================================
    # Load Original Model Panel
    # =====================================================

    panel = load_model_panel()

    # =====================================================
    # Load Validated Causal Impact Outputs
    # =====================================================

    outputs = load_causal_impact_outputs()

    validate_causal_impact_outputs(
        outputs
    )

    # =====================================================
    # Load BSTS Pipeline Modules
    # =====================================================

    logger.info("=" * 70)
    logger.info(
        "Loading BSTS modeling and validation modules..."
    )
    logger.info("=" * 70)

    bsts_modeling = load_bsts_modeling_module()

    bsts_validation = load_bsts_validation_module()

    logger.info(
        "BSTS pipeline modules loaded successfully."
    )

    # =====================================================
    # Load Production DMA Impact Results
    # =====================================================

    dma_impact_summary = (
        load_actual_dma_impact_summary()
    )

    # =====================================================
    # Load Production-Valid DMA Universe
    # =====================================================

    eligible_dmas = load_actual_dma_universe()

    logger.info(
        f"Production eligible DMA universe: "
        f"{len(eligible_dmas)} DMAs"
    )
    # =====================================================
    # Final Loading Summary
    # =====================================================

    logger.info("=" * 70)
    logger.info(
        "Causal Impact Robustness Inputs Loaded"
    )
    logger.info("=" * 70)

    logger.info(
        f"Model panel rows: "
        f"{len(panel):,}"
    )

    logger.info(
        f"DMA impact rows: "
        f"{len(outputs['dma_impact']):,}"
    )

    logger.info(
        f"DMA contribution rows: "
        f"{len(outputs['dma_contribution']):,}"
    )

    logger.info(
        f"Counterfactual validation rows: "
        f"{len(outputs['counterfactual_validation']):,}"
    )

    logger.info(
        f"Aggregate impact rows: "
        f"{len(outputs['aggregate_impact']):,}"
    )

    # =====================================================
    # DMA Heterogeneity Analysis
    # =====================================================

    logger.info("=" * 70)
    logger.info(
        "Running DMA heterogeneity analysis..."
    )
    logger.info("=" * 70)

    (
        dma_heterogeneity,
        heterogeneity_summary,
    ) = analyze_dma_heterogeneity(
        dma_impact=outputs["dma_impact"],
        dma_contribution=outputs["dma_contribution"],
        counterfactual_validation=(
            outputs[
                "counterfactual_validation"
            ]
        ),
        min_probability_threshold=0.95,
    )

    # =====================================================
    # Save DMA Heterogeneity Results
    # =====================================================

    heterogeneity_table_dir = (
        ROBUSTNESS_TABLE_DIR /
        "heterogeneity"
    )

    heterogeneity_table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dma_heterogeneity_path = (
        heterogeneity_table_dir /
        "dma_heterogeneity_analysis.csv"
    )

    dma_heterogeneity.to_csv(
        dma_heterogeneity_path,
        index=False,
    )

    logger.info(
        f"Saved DMA heterogeneity analysis to "
        f"{dma_heterogeneity_path}"
    )

    heterogeneity_summary_path = (
        heterogeneity_table_dir /
        "dma_heterogeneity_summary.csv"
    )

    heterogeneity_summary.to_csv(
        heterogeneity_summary_path,
        index=False,
    )

    logger.info(
        f"Saved DMA heterogeneity summary to "
        f"{heterogeneity_summary_path}"
    )

    # -----------------------------------------------------
    # Placebo Intervention
    # -----------------------------------------------------
    
    logger.info("=" * 70)
    logger.info(
        "Running placebo intervention analysis..."
    )
    logger.info("=" * 70)

    (
        placebo_results,
        placebo_summary,
        placebo_draws,
        placebo_validation,
    ) = run_placebo_analysis(
        panel=panel,
        bsts_modeling=bsts_modeling,
        bsts_validation=bsts_validation,
        actual_intervention_date=pd.Timestamp(
            "2026-01-01"
        ),
        placebo_dates=None,
        min_pre_periods=6,
        min_post_periods=3,
        include_trend=True,
        include_seasonality=False,
        draws=1000,
        tune=1000,
        chains=4,
        cores=None,
        target_accept=0.95,
        random_seed=42,
    )

    # =====================================================
    # Save Placebo Results
    # =====================================================

    placebo_table_dir = (
        ROBUSTNESS_TABLE_DIR /
        "placebo"
    )

    placebo_table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    placebo_results_path = (
        placebo_table_dir /
        "placebo_results.csv"
    )

    placebo_results.to_csv(
        placebo_results_path,
        index=False,
    )

    logger.info(
        f"Saved placebo results to "
        f"{placebo_results_path}"
    )

    placebo_summary_path = (
        placebo_table_dir /
        "placebo_summary.csv"
    )

    placebo_summary.to_csv(
        placebo_summary_path,
        index=False,
    )

    logger.info(
        f"Saved placebo summary to "
        f"{placebo_summary_path}"
    )

    placebo_draws_path = (
        placebo_table_dir /
        "placebo_draws.csv"
    )

    placebo_draws.to_csv(
        placebo_draws_path,
        index=False,
    )

    logger.info(
        f"Saved placebo posterior draws to "
        f"{placebo_draws_path}"
    )

    placebo_validation_path = (
        placebo_table_dir /
        "placebo_counterfactual_validation.csv"
    )

    placebo_validation.to_csv(
        placebo_validation_path,
        index=False,
    )

    logger.info(
        f"Saved placebo counterfactual validation to "
        f"{placebo_validation_path}"
    )


    # =====================================================
    # Marketing Spend Analysis
    # =====================================================

    logger.info("=" * 70)
    logger.info(
        "Analyzing marketing spend effects..."
    )
    logger.info("=" * 70)

    (
        spend_impact_analysis,
        spend_relationship_summary,
        spend_direction_summary,
        spend_aggregate_summary,
    ) = analyze_marketing_spend_effects(
        panel=panel,
        dma_impact_summary=dma_impact_summary,
        intervention_date=pd.Timestamp(
            "2026-01-01"
        ),
        eligible_dmas=eligible_dmas,
        expected_dma_count=None,
    )

    marketing_spend_table_dir = (
        ROBUSTNESS_TABLE_DIR /
        "marketing_spend"
    )

    marketing_spend_table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    spend_impact_analysis.to_csv(
        marketing_spend_table_dir /
        "dma_spend_impact_analysis.csv",
        index=False,
    )

    spend_relationship_summary.to_csv(
        marketing_spend_table_dir /
        "spend_impact_relationships.csv",
        index=False,
    )

    spend_direction_summary.to_csv(
        marketing_spend_table_dir /
        "spend_direction_summary.csv",
        index=False,
    )

    spend_aggregate_summary.to_csv(
        marketing_spend_table_dir /
        "spend_aggregate_summary.csv",
        index=False,
    )

    # =====================================================
    # Final Return
    # =====================================================

    logger.info("=" * 70)
    logger.info(
        "Causal Impact Robustness Analysis Inputs Ready"
    )
    logger.info("=" * 70)

    return {
        "panel":
            panel,

        "outputs":
            outputs,

        "dma_heterogeneity":
            dma_heterogeneity,

        "heterogeneity_summary":
            heterogeneity_summary,

        "placebo_results":
            placebo_results,

        "placebo_summary":
            placebo_summary,

        "placebo_draws":
            placebo_draws,
    }

    logger.info("=" * 70)
    logger.info(
        "Causal Impact Robustness Analysis Complete"
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()