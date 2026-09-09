"""
10_causal_impact_validation.py

Causal Impact Validation and Impact Analysis

Purpose
-------
Validate saved BSTS models and evaluate the resulting
counterfactuals and treatment effects.

The BSTS models are fitted and persisted by
09_causal_impact.py.

This script does not fit new BSTS models.

Validation Layers
-----------------
1. Posterior convergence diagnostics
2. Posterior / parameter diagnostics
3. Pre-treatment counterfactual validation
4. Holdout validation
5. Treatment impact estimation

Outputs
-------
Figures
Tables
HTML report
"""

# =========================================================
# Imports
# =========================================================

import json
import logging
from pathlib import Path

import arviz as az
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


def create_output_folders():

    folders = [

        FIGURE_DIR / "causal_impact",

        FIGURE_DIR /
        "causal_impact" /
        "parallel_trends",

        FIGURE_DIR /
        "causal_impact" /
        "effects",

        FIGURE_DIR /
        "causal_impact" /
        "diagnostics",

        TABLE_DIR /
        "causal_impact",

        REPORT_DIR,

    ]

    for folder in folders:

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )


def load_model_panel():

    logger.info("Loading model panel...")

    panel = pd.read_parquet(
        PROCESSED_DATA_DIR /
        "model_panel.parquet"
    )

    panel["Month"] = pd.to_datetime(
        panel["Month"]
    )

    panel = panel[
        panel["DMA"].notna()
    ].copy()

    panel = panel[
        panel["Traffic"].notna()
    ].copy()

    logger.info(
        f"Loaded {len(panel):,} rows"
    )

    logger.info(
        f"{panel['DMA'].nunique()} DMAs"
    )

    return panel

# =========================================================
# Discover Artifacts
# =========================================================
def discover_bsts_artifacts(
    model_dir=None,
):
    """
    Discover canonical saved BSTS model artifacts.

    Each DMA is expected to have:

        <safe_dma>_bsts.nc
        <safe_dma>_inputs.npz
        <safe_dma>_metadata.json

    Parameters
    ----------
    model_dir : Path, optional
        Directory containing saved BSTS artifacts.
        Defaults to MODEL_DIR / "bsts".

    Returns
    -------
    artifacts : list of dict
        Discovered artifact paths.
    """

    if model_dir is None:

        model_dir = (
            MODEL_DIR /
            "bsts"
        )

    model_dir = Path(
        model_dir
    )

    if not model_dir.exists():

        raise FileNotFoundError(
            f"BSTS model directory does not exist: "
            f"{model_dir}"
        )

    # =====================================================
    # Discover metadata files
    #
    # Metadata is the anchor because it is explicitly part
    # of the new artifact naming convention.
    # =====================================================

    metadata_paths = sorted(
        model_dir.glob(
            "*_metadata.json"
        )
    )

    if not metadata_paths:

        raise FileNotFoundError(
            f"No BSTS metadata files found in "
            f"{model_dir}"
        )

    artifacts = []

    for metadata_path in metadata_paths:

        filename = metadata_path.name

        safe_dma = filename[
            :-len("_metadata.json")
        ]

        posterior_path = (
            model_dir /
            f"{safe_dma}_bsts.nc"
        )

        inputs_path = (
            model_dir /
            f"{safe_dma}_inputs.npz"
        )

        artifacts.append(
            {
                "safe_dma": safe_dma,
                "metadata_path": metadata_path,
                "posterior_path": posterior_path,
                "inputs_path": inputs_path,
            }
        )

    logger.info(
        f"Discovered {len(artifacts)} BSTS artifact sets."
    )

    return artifacts


# =========================================================
# Reconstruct Data
# =========================================================
def reconstruct_bsts_model_data(
    metadata,
    inputs,
    idata,
):
    """
    Reconstruct the minimal model_data dictionary required
    for posterior counterfactual generation.
    """

    required_metadata = {
        "target_dma",
        "dates",
        "intervention_date",
        "y_mean",
        "y_std",
        "include_trend",
        "include_seasonality",
        "seasonal_period",
        "seasonal_harmonics",
    }

    missing = sorted(
        required_metadata - set(metadata.keys())
    )

    if missing:
        raise ValueError(
            f"{metadata.get('target_dma', 'Unknown DMA')}: "
            f"metadata missing required fields: {missing}"
        )

    if "controls_full_scaled" not in inputs:
        raise ValueError(
            f"{metadata['target_dma']}: "
            "inputs.npz missing 'controls_full_scaled'."
        )

    controls_full_scaled = np.asarray(
        inputs["controls_full_scaled"],
        dtype=float,
    )

    y_pre = np.asarray(
        inputs["y_pre"],
        dtype=float,
    )

    n_pre = len(y_pre)
    n_full = controls_full_scaled.shape[0]

    n_controls = (
        controls_full_scaled.shape[1]
        if controls_full_scaled.ndim > 1
        else 1
    )

    dates = pd.DatetimeIndex(
        metadata["dates"]
    )

    if len(dates) != n_full:
        raise ValueError(
            f"{metadata['target_dma']}: dates length "
            f"{len(dates)} != controls_full_scaled "
            f"length {n_full}."
        )

    time_index_pre = np.arange(
        n_pre,
        dtype=float,
    )

    time_mean = float(
        time_index_pre.mean()
    )

    time_std = float(
        time_index_pre.std()
    )

    if time_std <= 0:
        raise ValueError(
            f"{metadata['target_dma']}: invalid "
            f"time_std={time_std}."
        )

    return {
        "target_dma":
            metadata["target_dma"],

        "dates":
            dates,

        "intervention_date":
            pd.Timestamp(
                metadata["intervention_date"]
            ),

        "y_pre":
            y_pre,

        "y_mean":
            float(metadata["y_mean"]),

        "y_std":
            float(metadata["y_std"]),

        "n_pre":
            n_pre,

        "n_controls":
            n_controls,

        "controls_full_scaled":
            controls_full_scaled,

        "include_trend":
            bool(metadata["include_trend"]),

        "include_seasonality":
            bool(metadata["include_seasonality"]),

        "seasonal_period":
            int(metadata["seasonal_period"]),

        "seasonal_harmonics":
            int(metadata["seasonal_harmonics"]),

        "time_index_pre":
            time_index_pre,

        "time_mean":
            time_mean,

        "time_std":
            time_std,
    }# =========================================================
# Load Artifacts
# =========================================================
def load_bsts_artifact(
    posterior_path,
    inputs_path,
    metadata_path,
):
    """
    Load a complete saved BSTS artifact set.

    Parameters
    ----------
    posterior_path : Path
        Saved ArviZ BSTS posterior.

    inputs_path : Path
        Saved NumPy model-input archive.

    metadata_path : Path
        Saved JSON model metadata.

    Returns
    -------
    artifact : dict
        Loaded BSTS artifact.
    """

    posterior_path = Path(
        posterior_path
    )

    inputs_path = Path(
        inputs_path
    )

    metadata_path = Path(
        metadata_path
    )

    # =====================================================
    # Validate Files
    # =====================================================

    for path, label in [
        (posterior_path, "posterior"),
        (inputs_path, "inputs"),
        (metadata_path, "metadata"),
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"BSTS {label} file does not exist: "
                f"{path}"
            )

    # =====================================================
    # Load Posterior
    # =====================================================

    logger.info(
        f"Loading posterior: "
        f"{posterior_path.name}"
    )

    idata = az.from_netcdf(
        posterior_path
    )

    if not hasattr(
        idata,
        "posterior",
    ):

        raise ValueError(
            f"Posterior file does not contain "
            f"a posterior group: {posterior_path}"
        )

    # =====================================================
    # Load Inputs
    # =====================================================

    logger.info(
        f"Loading model inputs: "
        f"{inputs_path.name}"
    )

    inputs = np.load(
        inputs_path,
        allow_pickle=False,
    )

    inputs = {
        key: inputs[key]
        for key in inputs.files
    }

    logger.info(
        f"Saved input arrays: "
        f"{list(inputs.keys())}"
    )

    for name, values in inputs.items():

        logger.info(
            f"  {name}: "
            f"shape={values.shape}, "
            f"dtype={values.dtype}"
        )

    # =====================================================
    # Load Metadata
    # =====================================================

    logger.info(
        f"Loading metadata: "
        f"{metadata_path.name}"
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    if not isinstance(
        metadata,
        dict,
    ):

        raise ValueError(
            f"BSTS metadata must contain a JSON object: "
            f"{metadata_path}"
        )

    # =====================================================
    # Return Artifact
    # =====================================================

    return {
        "idata": idata,
        "inputs": inputs,
        "metadata": metadata,
        "posterior_path": posterior_path,
        "inputs_path": inputs_path,
        "metadata_path": metadata_path,
    }


def load_all_bsts_artifacts(
    model_dir=None,
):
    """
    Discover, load, validate, and reconstruct all saved BSTS
    artifacts.

    Returns
    -------
    artifacts : dict
        Dictionary keyed by canonical DMA name.
    """

    artifact_paths = discover_bsts_artifacts(
        model_dir=model_dir
    )

    artifacts = {}

    logger.info(
        f"Discovered {len(artifact_paths)} BSTS artifact sets."
    )

    for artifact_info in artifact_paths:

        # =================================================
        # Artifact Paths
        # =================================================

        safe_dma = artifact_info["safe_dma"]

        metadata_path = artifact_info[
            "metadata_path"
        ]

        posterior_path = artifact_info[
            "posterior_path"
        ]

        inputs_path = artifact_info[
            "inputs_path"
        ]

        logger.info(
            f"Artifact info keys: "
            f"{list(artifact_info.keys())}"
        )

        logger.info(
            f"Artifact info: "
            f"{artifact_info}"
        )

        # =================================================
        # Load Metadata
        # =================================================

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as f:

            metadata = json.load(f)

        target_dma = metadata.get(
            "target_dma"
        )

        if target_dma is None:

            raise ValueError(
                f"BSTS metadata for {safe_dma} "
                "is missing 'target_dma'."
            )

        logger.info(
            f"Loading BSTS artifacts for "
            f"{target_dma} "
            f"(safe name: {safe_dma})"
        )

        # =================================================
        # Load BSTS Artifact
        # =================================================

        artifact = load_bsts_artifact(
            posterior_path=posterior_path,
            inputs_path=inputs_path,
            metadata_path=metadata_path,
        )

        # =================================================
        # Reconstruct Model Data
        # =================================================

        logger.info(
            f"Reconstructing model_data for {target_dma}"
        )

        artifact["model_data"] = reconstruct_bsts_model_data(
            metadata=artifact["metadata"],
            inputs=artifact["inputs"],
            idata=artifact["idata"],
        )

        logger.info(
            f"Reconstructed model_data for {target_dma}"
        )

        logger.info(
            f"  Dates               : "
            f"{len(artifact['model_data']['dates'])}"
        )

        logger.info(
            f"  Pre-treatment       : "
            f"{artifact['model_data']['n_pre']}"
        )

        logger.info(
            f"  Controls            : "
            f"{artifact['model_data']['n_controls']}"
        )

        logger.info(
            f"  y_mean              : "
            f"{artifact['model_data']['y_mean']:.4f}"
        )

        logger.info(
            f"  y_std               : "
            f"{artifact['model_data']['y_std']:.4f}"
        )

        logger.info(
            f"  time_mean           : "
            f"{artifact['model_data']['time_mean']:.4f}"
        )

        logger.info(
            f"  time_std            : "
            f"{artifact['model_data']['time_std']:.4f}"
        )
        # =================================================
        # Validate Artifact
        # =================================================

        validation = validate_bsts_artifact(
            artifact
        )

        artifact[
            "artifact_validation"
        ] = validation

        # =================================================
        # Store By Canonical DMA
        # =================================================

        artifacts[
            target_dma
        ] = artifact

    # =====================================================
    # Final Logging
    # =====================================================

    logger.info("=" * 70)

    logger.info(
        f"Loaded and validated "
        f"{len(artifacts)} BSTS artifacts."
    )

    logger.info(
        f"DMAs: {list(artifacts.keys())}"
    )

    logger.info("=" * 70)

    return artifacts

# =========================================================
# Validate Artifacts
# =========================================================
def validate_bsts_artifact(
    artifact,
):
    """
    Validate the structure and consistency of a saved BSTS artifact.

    Parameters
    ----------
    artifact : dict
        Output from load_bsts_artifact().

    Returns
    -------
    validation : dict
        Structured artifact validation results.
    """

    if artifact is None:

        raise ValueError(
            "artifact is None."
        )

    required_keys = [
        "idata",
        "metadata",
        "posterior_path",
        "metadata_path",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in artifact
    ]

    if missing_keys:

        raise ValueError(
            "BSTS artifact is missing required fields: "
            f"{missing_keys}"
        )

    idata = artifact[
        "idata"
    ]

    metadata = artifact[
        "metadata"
    ]

    # =====================================================
    # Posterior Validation
    # =====================================================

    if not hasattr(
        idata,
        "posterior",
    ):

        raise ValueError(
            "BSTS artifact does not contain posterior."
        )

    posterior = idata.posterior

    chains = int(
        posterior.sizes.get(
            "chain",
            0,
        )
    )

    draws = int(
        posterior.sizes.get(
            "draw",
            0,
        )
    )

    if chains <= 0:

        raise ValueError(
            "BSTS posterior contains no chains."
        )

    if draws <= 0:

        raise ValueError(
            "BSTS posterior contains no draws."
        )

    # =====================================================
    # Metadata Validation
    # =====================================================

    required_metadata = [
        "target_dma",
        "dates",
        "intervention_date",
        "y_mean",
        "y_std",
        "n_pre",
        "include_trend",
        "include_seasonality",
        "seasonal_period",
        "seasonal_harmonics",
    ]

    missing_metadata = [
        key
        for key in required_metadata
        if key not in metadata
    ]

    if missing_metadata:

        raise ValueError(
            "BSTS metadata is missing required fields: "
            f"{missing_metadata}"
        )

    target_dma = metadata[
        "target_dma"
    ]

    dates = pd.DatetimeIndex(
        metadata["dates"]
    )

    intervention_date = pd.Timestamp(
        metadata["intervention_date"]
    )

    n_pre = int(
        metadata["n_pre"]
    )

    y_mean = float(
        metadata["y_mean"]
    )

    y_std = float(
        metadata["y_std"]
    )

    # =====================================================
    # Metadata Consistency
    # =====================================================

    if len(dates) == 0:

        raise ValueError(
            f"{target_dma}: metadata contains no dates."
        )

    if n_pre <= 0:

        raise ValueError(
            f"{target_dma}: n_pre must be positive."
        )

    if n_pre > len(dates):

        raise ValueError(
            f"{target_dma}: n_pre={n_pre} exceeds "
            f"number of modeling periods={len(dates)}."
        )

    if not np.isfinite(
        y_mean
    ):

        raise ValueError(
            f"{target_dma}: y_mean is not finite."
        )

    if not np.isfinite(
        y_std
    ) or y_std <= 0:

        raise ValueError(
            f"{target_dma}: y_std must be finite "
            "and greater than zero."
        )

    if intervention_date not in dates:

        raise ValueError(
            f"{target_dma}: intervention date "
            f"{intervention_date.date()} is not present "
            "in modeling dates."
        )

    # =====================================================
    # Posterior Parameter Validation
    # =====================================================

    required_parameters = [
        "intercept",
        "sigma_obs",
    ]

    if metadata[
        "include_trend"
    ]:

        required_parameters.append(
            "trend_beta"
        )

    if metadata[
        "include_seasonality"
    ]:

        required_parameters.append(
            "seasonal_beta"
        )

    if metadata.get(
        "n_controls",
        0,
    ) > 0:

        required_parameters.append(
            "beta"
        )

    missing_parameters = [
        parameter
        for parameter in required_parameters
        if parameter not in posterior
    ]

    if missing_parameters:

        raise ValueError(
            f"{target_dma}: posterior is missing required "
            f"parameters: {missing_parameters}"
        )

    # =====================================================
    # Validation Result
    # =====================================================

    validation = {

        "dma":
            target_dma,

        "chains":
            chains,

        "draws_per_chain":
            draws,

        "total_draws":
            chains * draws,

        "n_periods":
            len(dates),

        "n_pre":
            n_pre,

        "n_post":
            len(dates) - n_pre,

        "intervention_date":
            intervention_date,

        "include_trend":
            bool(
                metadata["include_trend"]
            ),

        "include_seasonality":
            bool(
                metadata["include_seasonality"]
            ),

        "posterior_path":
            str(
                artifact["posterior_path"]
            ),

        "metadata_path":
            str(
                artifact["metadata_path"]
            ),

        "valid":
            True,
    }

    logger.info(
        f"{target_dma}: BSTS artifact validation PASSED."
    )

    logger.info(
        f"  Chains             : {chains}"
    )

    logger.info(
        f"  Draws per chain    : {draws}"
    )

    logger.info(
        f"  Total draws        : {chains * draws}"
    )

    logger.info(
        f"  Modeling periods   : {len(dates)}"
    )

    logger.info(
        f"  Pre-treatment      : {n_pre}"
    )

    logger.info(
        f"  Post-treatment     : "
        f"{len(dates) - n_pre}"
    )

    return validation
# =========================================================
# Figure Helper
# =========================================================

def save_plot(
    fig,
    folder,
    filename,
):

    path = (
        FIGURE_DIR
        / "causal_impact"
        / folder
        / f"{filename}.png"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        f"Saved figure: {path}"
    )

    return path



#=========================================================
# Validate BSTS Model
#=========================================================
def validate_bsts_model(
    model_results,
    model_data,
    max_rhat=1.01,
    min_ess_bulk=400,
    min_ess_tail=400,
    max_divergence_rate=0.0,
    max_pre_rmse=None,
    max_pre_mae=None,
):
    """
    Validate a fitted deterministic BSTS model.

    Validation includes:

        1. Sampling diagnostics
        2. Posterior parameter diagnostics
        3. Model structure
        4. Pre-treatment fitted performance
        5. Overall validation status

    Parameters
    ----------
    model_results : dict
        Dictionary containing the fitted ArviZ InferenceData under
        the "idata" key.

    model_data : dict
        Metadata returned by configure_bsts_model() and saved with
        the fitted model.

    max_rhat : float, default=1.01
        Maximum acceptable R-hat.

    min_ess_bulk : float, default=400
        Minimum acceptable bulk effective sample size.

    min_ess_tail : float, default=400
        Minimum acceptable tail effective sample size.

    max_divergence_rate : float, default=0.0
        Maximum acceptable divergence rate.

    max_pre_rmse : float or None, optional
        Optional maximum acceptable pre-treatment RMSE on the
        original traffic scale.

    max_pre_mae : float or None, optional
        Optional maximum acceptable pre-treatment MAE on the
        original traffic scale.

    Returns
    -------
    validation : dict
        Structured validation results.

    validation_summary : pd.DataFrame
        One-row validation summary suitable for consolidation
        across DMAs.
    """

    logger.info("=" * 70)
    logger.info("Validating BSTS model...")
    logger.info("=" * 70)

    # =====================================================
    # Validate Inputs
    # =====================================================

    if not isinstance(model_results, dict):

        raise TypeError(
            "model_results must be a dictionary."
        )

    if not isinstance(model_data, dict):

        raise TypeError(
            "model_data must be a dictionary."
        )

    if "idata" not in model_results:

        raise ValueError(
            "model_results is missing 'idata'."
        )

    if not hasattr(
        model_results["idata"],
        "posterior",
    ):

        raise ValueError(
            "model_results['idata'] does not contain "
            "a posterior group."
        )

    idata = model_results["idata"]

    target_dma = model_data.get(
        "target_dma",
        "Unknown DMA",
    )

    # =====================================================
    # Validate Thresholds
    # =====================================================

    if max_rhat <= 0:

        raise ValueError(
            "max_rhat must be greater than zero."
        )

    if min_ess_bulk <= 0:

        raise ValueError(
            "min_ess_bulk must be greater than zero."
        )

    if min_ess_tail <= 0:

        raise ValueError(
            "min_ess_tail must be greater than zero."
        )

    if max_divergence_rate < 0:

        raise ValueError(
            "max_divergence_rate cannot be negative."
        )

    if max_pre_rmse is not None and max_pre_rmse < 0:

        raise ValueError(
            "max_pre_rmse cannot be negative."
        )

    if max_pre_mae is not None and max_pre_mae < 0:

        raise ValueError(
            "max_pre_mae cannot be negative."
        )

    # =====================================================
    # Sampling Dimensions
    # =====================================================

    posterior = idata.posterior

    chains = int(
        posterior.sizes.get(
            "chain",
            0,
        )
    )

    draws_per_chain = int(
        posterior.sizes.get(
            "draw",
            0,
        )
    )

    total_draws = (
        chains *
        draws_per_chain
    )

    if chains < 1:

        raise ValueError(
            f"{target_dma}: posterior contains no chains."
        )

    if draws_per_chain < 1:

        raise ValueError(
            f"{target_dma}: posterior contains no draws."
        )

    # =====================================================
    # Divergence Diagnostics
    #
    # Calculate directly from saved InferenceData rather
    # than relying on transient model_results fields.
    # =====================================================

    divergences = 0

    if (
        hasattr(idata, "sample_stats")
        and
        "diverging" in idata.sample_stats
    ):

        divergences = int(
            idata.sample_stats["diverging"]
            .sum()
            .item()
        )

    divergence_rate = (
        divergences / total_draws
    )

    divergence_pass = (
        divergence_rate <=
        max_divergence_rate
    )

    # =====================================================
    # ArviZ Posterior Diagnostics
    # =====================================================

    posterior_summary = az.summary(
        idata,
        round_to=None,
    )

    if posterior_summary.empty:

        raise ValueError(
            f"{target_dma}: posterior summary is empty."
        )

    # -----------------------------------------------------
    # R-hat
    # -----------------------------------------------------

    if "r_hat" not in posterior_summary.columns:

        raise ValueError(
            f"{target_dma}: ArviZ summary does not contain "
            "'r_hat'."
        )

    rhat_values = posterior_summary["r_hat"]

    max_rhat_value = float(
        rhat_values.max()
    )

    worst_rhat_parameter = (
        rhat_values.idxmax()
    )

    rhat_pass = (
        np.isfinite(max_rhat_value)
        and
        max_rhat_value <= max_rhat
    )

    # -----------------------------------------------------
    # Bulk ESS
    # -----------------------------------------------------

    if "ess_bulk" not in posterior_summary.columns:

        raise ValueError(
            f"{target_dma}: ArviZ summary does not contain "
            "'ess_bulk'."
        )

    ess_bulk_values = (
        posterior_summary["ess_bulk"]
    )

    min_ess_bulk_value = float(
        ess_bulk_values.min()
    )

    worst_bulk_ess_parameter = (
        ess_bulk_values.idxmin()
    )

    ess_bulk_pass = (
        np.isfinite(min_ess_bulk_value)
        and
        min_ess_bulk_value >= min_ess_bulk
    )

    # -----------------------------------------------------
    # Tail ESS
    # -----------------------------------------------------

    if "ess_tail" not in posterior_summary.columns:

        raise ValueError(
            f"{target_dma}: ArviZ summary does not contain "
            "'ess_tail'."
        )

    ess_tail_values = (
        posterior_summary["ess_tail"]
    )

    min_ess_tail_value = float(
        ess_tail_values.min()
    )

    worst_tail_ess_parameter = (
        ess_tail_values.idxmin()
    )

    ess_tail_pass = (
        np.isfinite(min_ess_tail_value)
        and
        min_ess_tail_value >= min_ess_tail
    )

    # =====================================================
    # Identify Problematic Parameters
    # =====================================================

    problematic_mask = (
        (rhat_values > max_rhat)
        |
        (ess_bulk_values < min_ess_bulk)
        |
        (ess_tail_values < min_ess_tail)
    )

    problematic_parameters = (
        posterior_summary.loc[
            problematic_mask
        ]
        .index
        .tolist()
    )

    parameter_count = len(
        posterior_summary
    )

    # =====================================================
    # Model Structure
    # =====================================================

    n_pre = int(
        model_data.get(
            "n_pre",
            len(
                model_data.get(
                    "y_pre",
                    [],
                )
            ),
        )
    )

    n_controls = int(
        model_data.get(
            "n_controls",
            0,
        )
    )

    include_trend = bool(
        model_data.get(
            "include_trend",
            False,
        )
    )

    include_seasonality = bool(
        model_data.get(
            "include_seasonality",
            False,
        )
    )

    seasonal_period = model_data.get(
        "seasonal_period"
    )

    seasonal_harmonics = model_data.get(
        "seasonal_harmonics"
    )

    seasonal_matrix = model_data.get(
        "seasonal_matrix"
    )

    if seasonal_matrix is not None:

        seasonal_matrix = np.asarray(
            seasonal_matrix
        )

        seasonal_terms = (
            seasonal_matrix.shape[1]
            if seasonal_matrix.ndim == 2
            else 0
        )

    else:

        seasonal_terms = 0

    # =====================================================
    # Expected Posterior Parameters
    # =====================================================

    expected_parameters = {
        "intercept",
        "sigma_obs",
    }

    if include_trend:

        expected_parameters.add(
            "trend_beta"
        )

    if n_controls > 0:

        expected_parameters.add(
            "beta"
        )

    if include_seasonality:

        expected_parameters.add(
            "seasonal_beta"
        )

    actual_parameters = set(
        posterior.data_vars
    )

    missing_parameters = sorted(
        expected_parameters -
        actual_parameters
    )

    unexpected_parameters = sorted(
        actual_parameters -
        expected_parameters
    )

    structure_pass = (
        len(missing_parameters) == 0
    )

    # =====================================================
    # Pre-treatment Fitted Performance
    # =====================================================

    pre_rmse = np.nan
    pre_mae = np.nan
    pre_r2 = np.nan

    predictive_fit_available = False

    if (
        "y_pre_scaled" in model_data
        and
        "y_mean" in model_data
        and
        "y_std" in model_data
    ):

        y_pre_scaled = np.asarray(
            model_data["y_pre_scaled"],
            dtype=float,
        )

        if len(y_pre_scaled) != n_pre:

            raise ValueError(
                f"{target_dma}: y_pre_scaled contains "
                f"{len(y_pre_scaled)} observations but "
                f"n_pre={n_pre}."
            )

        # -------------------------------------------------
        # Posterior Fitted Values
        # -------------------------------------------------

        intercept_draws = (
            posterior["intercept"]
            .values
            .reshape(-1)
        )

        mu_draws = (
            intercept_draws[:, None]
        )

        # -------------------------------------------------
        # Deterministic Trend
        # -------------------------------------------------

        if include_trend:

            if "trend_beta" not in posterior:

                raise ValueError(
                    f"{target_dma}: trend is enabled but "
                    "'trend_beta' is missing from posterior."
                )

            trend_beta_draws = (
                posterior["trend_beta"]
                .values
                .reshape(-1)
            )

            time_scaled = np.asarray(
                model_data["time_scaled"],
                dtype=float,
            )

            if len(time_scaled) != n_pre:

                raise ValueError(
                    f"{target_dma}: time_scaled contains "
                    f"{len(time_scaled)} observations but "
                    f"n_pre={n_pre}."
                )

            mu_draws = (
                mu_draws
                +
                trend_beta_draws[:, None]
                *
                time_scaled[None, :]
            )

        # -------------------------------------------------
        # Control DMA Effects
        # -------------------------------------------------

        if n_controls > 0:

            if "beta" not in posterior:

                raise ValueError(
                    f"{target_dma}: controls are enabled but "
                    "'beta' is missing from posterior."
                )

            beta_draws = (
                posterior["beta"]
                .values
            )

            beta_draws = beta_draws.reshape(
                -1,
                n_controls,
            )

            X_pre_scaled = np.asarray(
                model_data["controls_pre_scaled"],
                dtype=float,
            )

            if X_pre_scaled.shape != (
                n_pre,
                n_controls,
            ):

                raise ValueError(
                    f"{target_dma}: controls_pre_scaled "
                    f"has shape {X_pre_scaled.shape}; "
                    f"expected {(n_pre, n_controls)}."
                )

            mu_draws = (
                mu_draws
                +
                beta_draws
                @ X_pre_scaled.T
            )

        # -------------------------------------------------
        # Fourier Seasonality
        # -------------------------------------------------

        if include_seasonality:

            if seasonal_terms == 0:

                raise ValueError(
                    f"{target_dma}: seasonality is enabled "
                    "but seasonal_matrix is empty."
                )

            if "seasonal_beta" not in posterior:

                raise ValueError(
                    f"{target_dma}: seasonality is enabled "
                    "but 'seasonal_beta' is missing."
                )

            seasonal_beta_draws = (
                posterior["seasonal_beta"]
                .values
                .reshape(
                    -1,
                    seasonal_terms,
                )
            )

            if seasonal_matrix.shape != (
                n_pre,
                seasonal_terms,
            ):

                raise ValueError(
                    f"{target_dma}: seasonal_matrix has "
                    f"shape {seasonal_matrix.shape}; "
                    f"expected "
                    f"{(n_pre, seasonal_terms)}."
                )

            mu_draws = (
                mu_draws
                +
                seasonal_beta_draws
                @ seasonal_matrix.T
            )

        # -------------------------------------------------
        # Posterior Mean Fitted Values
        # -------------------------------------------------

        fitted_scaled = np.mean(
            mu_draws,
            axis=0,
        )

        y_mean = float(
            model_data["y_mean"]
        )

        y_std = float(
            model_data["y_std"]
        )

        if not np.isfinite(y_std) or y_std <= 0:

            raise ValueError(
                f"{target_dma}: invalid y_std "
                f"({y_std})."
            )

        fitted_original = (
            fitted_scaled *
            y_std
            +
            y_mean
        )

        y_pre_original = (
            y_pre_scaled *
            y_std
            +
            y_mean
        )

        residuals = (
            y_pre_original -
            fitted_original
        )

        pre_rmse = float(
            np.sqrt(
                np.mean(
                    residuals ** 2
                )
            )
        )

        pre_mae = float(
            np.mean(
                np.abs(residuals)
            )
        )

        ss_res = float(
            np.sum(
                residuals ** 2
            )
        )

        ss_tot = float(
            np.sum(
                (
                    y_pre_original -
                    np.mean(y_pre_original)
                ) ** 2
            )
        )

        if ss_tot > 0:

            pre_r2 = float(
                1 -
                ss_res /
                ss_tot
            )

        predictive_fit_available = True

    # =====================================================
    # Pre-treatment Fit Thresholds
    # =====================================================

    if max_pre_rmse is None:

        rmse_pass = True

    else:

        rmse_pass = (
            predictive_fit_available
            and
            np.isfinite(pre_rmse)
            and
            pre_rmse <= max_pre_rmse
        )

    if max_pre_mae is None:

        mae_pass = True

    else:

        mae_pass = (
            predictive_fit_available
            and
            np.isfinite(pre_mae)
            and
            pre_mae <= max_pre_mae
        )

    # =====================================================
    # Overall Validation
    # =====================================================

    converged = all(
        [
            rhat_pass,
            ess_bulk_pass,
            ess_tail_pass,
            divergence_pass,
            structure_pass,
            rmse_pass,
            mae_pass,
        ]
    )

    # =====================================================
    # Logging
    # =====================================================

    logger.info(
        f"{target_dma}: model validation complete."
    )

    logger.info(
        f"  Chains                 : {chains}"
    )

    logger.info(
        f"  Draws / chain          : {draws_per_chain}"
    )

    logger.info(
        f"  Total posterior draws : {total_draws}"
    )

    logger.info(
        f"  Parameters             : {parameter_count}"
    )

    logger.info(
        f"  Maximum R-hat          : "
        f"{max_rhat_value:.4f} "
        f"(threshold={max_rhat:.4f})"
    )

    logger.info(
        f"  Minimum bulk ESS       : "
        f"{min_ess_bulk_value:.1f} "
        f"(threshold={min_ess_bulk:.1f})"
    )

    logger.info(
        f"  Minimum tail ESS       : "
        f"{min_ess_tail_value:.1f} "
        f"(threshold={min_ess_tail:.1f})"
    )

    logger.info(
        f"  Divergences            : "
        f"{divergences}"
    )

    logger.info(
        f"  Divergence rate        : "
        f"{divergence_rate:.2%}"
    )

    logger.info(
        f"  Controls               : "
        f"{n_controls}"
    )

    logger.info(
        f"  Trend enabled          : "
        f"{include_trend}"
    )

    logger.info(
        f"  Seasonal terms         : "
        f"{seasonal_terms}"
    )

    if predictive_fit_available:

        logger.info(
            f"  Pre-treatment RMSE    : "
            f"{pre_rmse:,.2f}"
        )

        logger.info(
            f"  Pre-treatment MAE     : "
            f"{pre_mae:,.2f}"
        )

        logger.info(
            f"  Pre-treatment R²      : "
            f"{pre_r2:.4f}"
        )

    logger.info(
        f"  R-hat validation       : "
        f"{rhat_pass}"
    )

    logger.info(
        f"  Bulk ESS validation    : "
        f"{ess_bulk_pass}"
    )

    logger.info(
        f"  Tail ESS validation    : "
        f"{ess_tail_pass}"
    )

    logger.info(
        f"  Divergence validation  : "
        f"{divergence_pass}"
    )

    logger.info(
        f"  Structure validation   : "
        f"{structure_pass}"
    )

    logger.info(
        f"  Overall validation     : "
        f"{converged}"
    )

    if problematic_parameters:

        logger.warning(
            f"{target_dma}: problematic parameters: "
            f"{problematic_parameters}"
        )

    if missing_parameters:

        logger.error(
            f"{target_dma}: missing expected parameters: "
            f"{missing_parameters}"
        )

    if unexpected_parameters:

        logger.warning(
            f"{target_dma}: unexpected posterior "
            f"parameters: {unexpected_parameters}"
        )

    # =====================================================
    # Structured Result
    # =====================================================

    validation = {

        "DMA":
            target_dma,

        "chains":
            chains,

        "draws_per_chain":
            draws_per_chain,

        "total_draws":
            total_draws,

        "parameter_count":
            parameter_count,

        "max_rhat":
            max_rhat_value,

        "min_ess_bulk":
            min_ess_bulk_value,

        "min_ess_tail":
            min_ess_tail_value,

        "worst_rhat_parameter":
            worst_rhat_parameter,

        "worst_bulk_ess_parameter":
            worst_bulk_ess_parameter,

        "worst_tail_ess_parameter":
            worst_tail_ess_parameter,

        "divergences":
            divergences,

        "divergence_rate":
            divergence_rate,

        "n_pre":
            n_pre,

        "n_controls":
            n_controls,

        "include_trend":
            include_trend,

        "include_seasonality":
            include_seasonality,

        "seasonal_period":
            seasonal_period,

        "seasonal_harmonics":
            seasonal_harmonics,

        "seasonal_terms":
            seasonal_terms,

        "pre_rmse":
            pre_rmse,

        "pre_mae":
            pre_mae,

        "pre_r2":
            pre_r2,

        "predictive_fit_available":
            predictive_fit_available,

        "problematic_parameters":
            problematic_parameters,

        "missing_parameters":
            missing_parameters,

        "unexpected_parameters":
            unexpected_parameters,

        "rhat_pass":
            rhat_pass,

        "ess_bulk_pass":
            ess_bulk_pass,

        "ess_tail_pass":
            ess_tail_pass,

        "divergence_pass":
            divergence_pass,

        "structure_pass":
            structure_pass,

        "rmse_pass":
            rmse_pass,

        "mae_pass":
            mae_pass,

        "converged":
            converged,

        "model_type":
            model_data.get(
                "model_type"
            ),

        "model_version":
            model_data.get(
                "model_version"
            ),
    }

    # =====================================================
    # One-row DataFrame
    # =====================================================

    validation_summary = pd.DataFrame(
        [validation]
    )

    return (
        validation,
        validation_summary,
    )#=========================================================
# Evaluate Posterior
#=========================================================


def evaluate_posterior(
    idata,
    dma=None,
    output_name="posterior",
    rhat_threshold=1.01,
    ess_threshold=400,
):
    """
    Evaluate and document posterior sampling diagnostics
    for a fitted PyMC BSTS model.

    This function provides detailed posterior diagnostics and
    parameter-level reporting. Formal model PASS/FAIL validation
    is handled separately by validate_bsts_model().

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior samples returned by the PyMC model.

    dma : str, optional
        DMA associated with the model. Used for logging and
        output naming.

    output_name : str, default="posterior"
        Base name used for exported diagnostic files.

    rhat_threshold : float, default=1.01
        Threshold used to flag problematic R-hat values.

    ess_threshold : float, default=400
        Threshold used to flag low bulk or tail ESS.

    Returns
    -------
    posterior_summary : pd.DataFrame
        ArviZ posterior summary with parameter-level diagnostics.

    diagnostics : dict
        Detailed high-level posterior diagnostics.

    problematic_parameters : pd.DataFrame
        Parameters failing one or more diagnostic thresholds.
    """

    logger.info("=" * 70)
    logger.info("Evaluating posterior...")
    logger.info("=" * 70)

    # =====================================================
    # Validate Input
    # =====================================================

    if idata is None:

        raise ValueError(
            "idata is None. Posterior evaluation cannot proceed."
        )

    if not hasattr(idata, "posterior"):

        raise ValueError(
            "Input does not contain a posterior group."
        )

    if idata.posterior is None:

        raise ValueError(
            "Posterior group is empty."
        )

    # =====================================================
    # DMA Label
    # =====================================================

    dma_label = (
        dma
        if dma is not None
        else "Unknown DMA"
    )

    logger.info(
        f"Evaluating posterior for {dma_label}..."
    )

    # =====================================================
    # Posterior Dimensions
    # =====================================================

    posterior = idata.posterior

    chains = int(
        posterior.sizes.get(
            "chain",
            0,
        )
    )

    draws = int(
        posterior.sizes.get(
            "draw",
            0,
        )
    )

    total_draws = (
        chains *
        draws
    )

    logger.info(
        f"Posterior chains: {chains}"
    )

    logger.info(
        f"Posterior draws per chain: {draws}"
    )

    logger.info(
        f"Total posterior draws: {total_draws}"
    )

    # =====================================================
    # ArviZ Posterior Summary
    # =====================================================

    posterior_summary = az.summary(
        idata,
        round_to=4,
    )

    if posterior_summary.empty:

        raise ValueError(
            f"{dma_label}: ArviZ posterior summary is empty."
        )

    required_diagnostic_columns = {
        "r_hat",
        "ess_bulk",
        "ess_tail",
    }

    missing_columns = (
        required_diagnostic_columns
        -
        set(posterior_summary.columns)
    )

    if missing_columns:

        raise ValueError(
            f"{dma_label}: posterior summary is missing "
            f"diagnostic columns: "
            f"{sorted(missing_columns)}"
        )

    # =====================================================
    # Parameter-Level Diagnostic Flags
    # =====================================================

    posterior_summary["Rhat_OK"] = (
        posterior_summary["r_hat"]
        <= rhat_threshold
    )

    posterior_summary["ESS_Bulk_OK"] = (
        posterior_summary["ess_bulk"]
        >= ess_threshold
    )

    posterior_summary["ESS_Tail_OK"] = (
        posterior_summary["ess_tail"]
        >= ess_threshold
    )

    posterior_summary["Parameter_OK"] = (
        posterior_summary["Rhat_OK"]
        &
        posterior_summary["ESS_Bulk_OK"]
        &
        posterior_summary["ESS_Tail_OK"]
    )

    # =====================================================
    # Identify Problematic Parameters
    # =====================================================

    problematic_parameters = (
        posterior_summary[
            ~posterior_summary["Parameter_OK"]
        ]
        .copy()
    )

    if not problematic_parameters.empty:

        failure_reasons = []

        for parameter in (
            problematic_parameters.index
        ):

            row = problematic_parameters.loc[
                parameter
            ]

            reasons = []

            if (
                pd.notna(row["r_hat"])
                and
                row["r_hat"] > rhat_threshold
            ):

                reasons.append(
                    f"R-hat > {rhat_threshold}"
                )

            if (
                pd.notna(row["ess_bulk"])
                and
                row["ess_bulk"] < ess_threshold
            ):

                reasons.append(
                    f"bulk ESS < {ess_threshold}"
                )

            if (
                pd.notna(row["ess_tail"])
                and
                row["ess_tail"] < ess_threshold
            ):

                reasons.append(
                    f"tail ESS < {ess_threshold}"
                )

            failure_reasons.append(
                "; ".join(reasons)
            )

        problematic_parameters[
            "Failure_Reasons"
        ] = failure_reasons

    # =====================================================
    # High-Level Diagnostics
    # =====================================================

    max_rhat = float(
        posterior_summary["r_hat"].max()
    )

    min_ess_bulk = float(
        posterior_summary["ess_bulk"].min()
    )

    min_ess_tail = float(
        posterior_summary["ess_tail"].min()
    )

    # =====================================================
    # Worst Parameters
    # =====================================================

    worst_rhat_parameter = (
        posterior_summary["r_hat"]
        .idxmax()
    )

    worst_rhat_value = float(
        posterior_summary["r_hat"]
        .max()
    )

    worst_bulk_ess_parameter = (
        posterior_summary["ess_bulk"]
        .idxmin()
    )

    worst_bulk_ess_value = float(
        posterior_summary["ess_bulk"]
        .min()
    )

    worst_tail_ess_parameter = (
        posterior_summary["ess_tail"]
        .idxmin()
    )

    worst_tail_ess_value = float(
        posterior_summary["ess_tail"]
        .min()
    )

    # =====================================================
    # Divergence Diagnostics
    # =====================================================

    divergences = 0

    if (
        hasattr(idata, "sample_stats")
        and
        idata.sample_stats is not None
        and
        "diverging" in idata.sample_stats
    ):

        divergences = int(
            idata.sample_stats["diverging"]
            .sum()
            .item()
        )

    divergence_rate = (
        divergences / total_draws
        if total_draws > 0
        else np.nan
    )

    # =====================================================
    # Diagnostic Status
    #
    # These are diagnostic flags only.
    # Formal model validation remains in
    # validate_bsts_model().
    # =====================================================

    rhat_ok = (
        np.isfinite(max_rhat)
        and
        max_rhat <= rhat_threshold
    )

    ess_bulk_ok = (
        np.isfinite(min_ess_bulk)
        and
        min_ess_bulk >= ess_threshold
    )

    ess_tail_ok = (
        np.isfinite(min_ess_tail)
        and
        min_ess_tail >= ess_threshold
    )

    divergences_ok = (
        divergences == 0
    )

    parameter_diagnostics_ok = (
        problematic_parameters.empty
    )

    # =====================================================
    # Structured Diagnostics
    # =====================================================

    diagnostics = {

        "DMA":
            dma_label,

        "Chains":
            chains,

        "Draws_Per_Chain":
            draws,

        "Total_Draws":
            total_draws,

        "Max_Rhat":
            max_rhat,

        "Min_ESS_Bulk":
            min_ess_bulk,

        "Min_ESS_Tail":
            min_ess_tail,

        "Divergences":
            divergences,

        "Divergence_Rate":
            divergence_rate,

        "Parameters_Evaluated":
            len(posterior_summary),

        "Problematic_Parameters":
            len(problematic_parameters),

        "Worst_Rhat_Parameter":
            worst_rhat_parameter,

        "Worst_Rhat":
            worst_rhat_value,

        "Worst_Bulk_ESS_Parameter":
            worst_bulk_ess_parameter,

        "Worst_Bulk_ESS":
            worst_bulk_ess_value,

        "Worst_Tail_ESS_Parameter":
            worst_tail_ess_parameter,

        "Worst_Tail_ESS":
            worst_tail_ess_value,

        "Rhat_OK":
            rhat_ok,

        "ESS_Bulk_OK":
            ess_bulk_ok,

        "ESS_Tail_OK":
            ess_tail_ok,

        "Divergences_OK":
            divergences_ok,

        "Parameter_Diagnostics_OK":
            parameter_diagnostics_ok,
    }

    # =====================================================
    # Logging
    # =====================================================

    logger.info(
        f"{dma_label}: posterior evaluation complete."
    )

    logger.info(
        f"  Maximum R-hat       : "
        f"{max_rhat:.4f}"
    )

    logger.info(
        f"  Minimum bulk ESS    : "
        f"{min_ess_bulk:.1f}"
    )

    logger.info(
        f"  Minimum tail ESS    : "
        f"{min_ess_tail:.1f}"
    )

    logger.info(
        f"  Divergences         : "
        f"{divergences}"
    )

    logger.info(
        f"  Divergence rate     : "
        f"{divergence_rate:.2%}"
    )

    logger.info(
        f"  Parameters evaluated: "
        f"{len(posterior_summary)}"
    )

    logger.info(
        f"  Problematic params  : "
        f"{len(problematic_parameters)}"
    )

    logger.info(
        f"  R-hat diagnostics   : "
        f"{rhat_ok}"
    )

    logger.info(
        f"  Bulk ESS diagnostics: "
        f"{ess_bulk_ok}"
    )

    logger.info(
        f"  Tail ESS diagnostics: "
        f"{ess_tail_ok}"
    )

    logger.info(
        f"  Divergence check    : "
        f"{divergences_ok}"
    )

    if problematic_parameters.empty:

        logger.info(
            "No parameters failed the R-hat / ESS "
            "diagnostic thresholds."
        )

    else:

        logger.warning(
            f"{len(problematic_parameters)} parameters "
            "failed posterior diagnostic thresholds."
        )

        for parameter in (
            problematic_parameters.index
        ):

            row = problematic_parameters.loc[
                parameter
            ]

            logger.warning(
                f"  {parameter}: "
                f"R-hat={row['r_hat']:.4f}, "
                f"bulk ESS={row['ess_bulk']:.1f}, "
                f"tail ESS={row['ess_tail']:.1f}, "
                f"reason={row['Failure_Reasons']}"
            )

    if divergences > 0:

        logger.warning(
            f"{dma_label}: {divergences} divergent "
            f"transitions detected "
            f"({divergence_rate:.2%})."
        )

    # =====================================================
    # Export Posterior Summary
    # =====================================================

    bsts_table_dir = (
        TABLE_DIR /
        "bsts"
    )

    bsts_table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        bsts_table_dir /
        f"{output_name}_summary.csv"
    )

    posterior_summary.to_csv(
        summary_path
    )

    logger.info(
        f"Saved posterior summary to "
        f"{summary_path}"
    )

    # =====================================================
    # Export Problematic Parameters
    # =====================================================

    problematic_path = (
        bsts_table_dir /
        f"{output_name}_problematic_parameters.csv"
    )

    problematic_parameters.to_csv(
        problematic_path
    )

    logger.info(
        f"Saved problematic parameter diagnostics "
        f"to {problematic_path}"
    )

    # =====================================================
    # Export High-Level Diagnostics
    # =====================================================

    diagnostics_df = pd.DataFrame(
        [diagnostics]
    )

    diagnostics_path = (
        bsts_table_dir /
        f"{output_name}_diagnostics.csv"
    )

    diagnostics_df.to_csv(
        diagnostics_path,
        index=False,
    )

    logger.info(
        f"Saved posterior diagnostics to "
        f"{diagnostics_path}"
    )

    # =====================================================
    # Return
    # =====================================================

    return (
        posterior_summary,
        diagnostics,
        problematic_parameters,
    )#=========================================================

#=========================================================
# Generate Counterfactual Predictions
#=========================================================

def generate_counterfactual(
    idata,
    model_data,
    include_observation_noise=False,
    random_seed=42,
):
    """
    Generate posterior counterfactual traffic predictions for
    the full modeling period from a fitted deterministic BSTS
    model.

    The counterfactual represents expected traffic for the
    treated DMA in the absence of treatment.

    Posterior draws are propagated through the fitted model
    using the observed control-DMA values over the full
    modeling period.

    Parameters
    ----------
    idata : arviz.InferenceData
        Saved posterior samples.

    model_data : dict
        Saved model metadata and input arrays required to
        reconstruct the model over the full modeling period.

    include_observation_noise : bool, default=False
        If False, return posterior uncertainty in the
        conditional counterfactual mean.

        If True, add posterior observation noise to generate
        posterior predictive counterfactual draws.

    random_seed : int, default=42
        Random seed used when observation noise is generated.

    Returns
    -------
    counterfactual : np.ndarray
        Posterior counterfactual draws in original traffic units.

        Shape:
            (posterior_draws, full_periods)

    counterfactual_mean : np.ndarray
        Posterior mean conditional counterfactual.

    counterfactual_lower : np.ndarray
        Lower 95% posterior interval.

    counterfactual_upper : np.ndarray
        Upper 95% posterior interval.
    """

    logger.info("=" * 70)
    logger.info("Generating posterior counterfactual...")
    logger.info("=" * 70)

    # =====================================================
    # Validate Inputs
    # =====================================================

    if idata is None:

        raise ValueError(
            "idata is None. Cannot generate counterfactual."
        )

    if not hasattr(idata, "posterior"):

        raise ValueError(
            "idata does not contain a posterior group."
        )

    if model_data is None:

        raise ValueError(
            "model_data is None. Cannot generate counterfactual."
        )

    if not isinstance(model_data, dict):

        raise TypeError(
            "model_data must be a dictionary."
        )

    # =====================================================
    # Required Metadata
    # =====================================================

    required_keys = {
        "target_dma",
        "dates",
        "intervention_date",
        "y_mean",
        "y_std",
        "n_pre",
        "n_controls",
        "include_trend",
        "include_seasonality",
        "seasonal_period",
        "seasonal_harmonics",
        "time_mean",
        "time_std",
    }

    missing_keys = sorted(
        required_keys -
        set(model_data.keys())
    )

    if missing_keys:

        raise ValueError(
            "model_data is missing required fields: "
            f"{missing_keys}"
        )

    # =====================================================
    # Extract Metadata
    # =====================================================

    target_dma = model_data[
        "target_dma"
    ]

    dates = pd.DatetimeIndex(
        model_data["dates"]
    )

    intervention_date = pd.Timestamp(
        model_data["intervention_date"]
    )

    y_mean = float(
        model_data["y_mean"]
    )

    y_std = float(
        model_data["y_std"]
    )

    n_pre = int(
        model_data["n_pre"]
    )

    n_controls = int(
        model_data["n_controls"]
    )

    include_trend = bool(
        model_data["include_trend"]
    )

    include_seasonality = bool(
        model_data["include_seasonality"]
    )

    seasonal_period = int(
        model_data["seasonal_period"]
    )

    seasonal_harmonics = int(
        model_data["seasonal_harmonics"]
    )

    time_mean = float(
        model_data["time_mean"]
    )

    time_std = float(
        model_data["time_std"]
    )

    # =====================================================
    # Validate Scaling Metadata
    # =====================================================

    if not np.isfinite(y_mean):

        raise ValueError(
            f"{target_dma}: y_mean is not finite."
        )

    if (
        not np.isfinite(y_std)
        or
        y_std <= 0
    ):

        raise ValueError(
            f"{target_dma}: y_std must be finite "
            "and greater than zero."
        )

    if include_trend:

        if (
            not np.isfinite(time_mean)
            or
            not np.isfinite(time_std)
            or
            time_std <= 0
        ):

            raise ValueError(
                f"{target_dma}: invalid saved trend "
                "scaling parameters."
            )

    # =====================================================
    # Validate Dates
    # =====================================================

    n_full = len(dates)

    if n_full == 0:

        raise ValueError(
            f"{target_dma}: no modeling dates available."
        )

    if (
        n_pre <= 0
        or
        n_pre > n_full
    ):

        raise ValueError(
            f"{target_dma}: invalid n_pre={n_pre} "
            f"for {n_full} total periods."
        )

    # =====================================================
    # Validate Posterior
    # =====================================================

    posterior = idata.posterior

    chains = int(
        posterior.sizes.get(
            "chain",
            0,
        )
    )

    draws = int(
        posterior.sizes.get(
            "draw",
            0,
        )
    )

    if chains < 1 or draws < 1:

        raise ValueError(
            f"{target_dma}: posterior contains no draws."
        )

    n_posterior = (
        chains *
        draws
    )

    # =====================================================
    # Posterior Variable Helper
    # =====================================================

    def flatten_posterior_variable(
        variable_name
    ):
        """
        Flatten chain/draw dimensions while preserving
        parameter dimensions.
        """

        if variable_name not in posterior:

            raise ValueError(
                f"{target_dma}: posterior variable "
                f"'{variable_name}' is missing."
            )

        values = np.asarray(
            posterior[
                variable_name
            ].values,
            dtype=float,
        )

        if values.ndim < 2:

            raise ValueError(
                f"{target_dma}: posterior variable "
                f"'{variable_name}' has unexpected "
                f"dimensions: {values.shape}"
            )

        return values.reshape(
            chains * draws,
            *values.shape[2:],
        )

    # =====================================================
    # Intercept
    # =====================================================

    intercept = flatten_posterior_variable(
        "intercept"
    )

    if intercept.ndim != 1:

        raise ValueError(
            f"{target_dma}: intercept has unexpected "
            f"shape {intercept.shape}."
        )

    # =====================================================
    # Deterministic Trend
    #
    # IMPORTANT:
    # Use the saved pre-treatment scaling parameters.
    # Do not recompute them here.
    # =====================================================

    if include_trend:

        trend_beta = flatten_posterior_variable(
            "trend_beta"
        )

        if trend_beta.ndim != 1:

            raise ValueError(
                f"{target_dma}: trend_beta has unexpected "
                f"shape {trend_beta.shape}."
            )

        full_time_index = np.arange(
            n_full,
            dtype=float,
        )

        time_scaled_full = (
            full_time_index -
            time_mean
        ) / time_std

    else:

        trend_beta = np.zeros(
            n_posterior,
            dtype=float,
        )

        time_scaled_full = np.zeros(
            n_full,
            dtype=float,
        )

    # =====================================================
    # Full Control Matrix
    # =====================================================

    X_full_scaled = model_data.get(
        "controls_full_scaled"
    )

    if n_controls > 0:

        if X_full_scaled is None:

            raise ValueError(
                f"{target_dma}: model_data specifies "
                f"{n_controls} controls but "
                "controls_full_scaled is missing."
            )

        X_full_scaled = np.asarray(
            X_full_scaled,
            dtype=float,
        )

        if X_full_scaled.ndim == 1:

            X_full_scaled = (
                X_full_scaled.reshape(
                    -1,
                    1,
                )
            )

        if X_full_scaled.shape != (
            n_full,
            n_controls,
        ):

            raise ValueError(
                f"{target_dma}: controls_full_scaled "
                f"has shape {X_full_scaled.shape}; "
                f"expected "
                f"({n_full}, {n_controls})."
            )

        if not np.isfinite(
            X_full_scaled
        ).all():

            raise ValueError(
                f"{target_dma}: controls_full_scaled "
                "contains non-finite values."
            )

        beta = flatten_posterior_variable(
            "beta"
        )

        if beta.ndim != 2:

            raise ValueError(
                f"{target_dma}: beta has unexpected "
                f"shape {beta.shape}."
            )

        if beta.shape != (
            n_posterior,
            n_controls,
        ):

            raise ValueError(
                f"{target_dma}: beta has shape "
                f"{beta.shape}; expected "
                f"({n_posterior}, {n_controls})."
            )

        control_component = (
            beta @
            X_full_scaled.T
        )

    else:

        control_component = np.zeros(
            (
                n_posterior,
                n_full,
            ),
            dtype=float,
        )

    # =====================================================
    # Fourier Seasonality
    # =====================================================

    if include_seasonality:

        seasonal_terms = []

        for harmonic in range(
            1,
            seasonal_harmonics + 1,
        ):

            seasonal_terms.append(
                np.sin(
                    2
                    * np.pi
                    * harmonic
                    * np.arange(n_full)
                    / seasonal_period
                )
            )

            seasonal_terms.append(
                np.cos(
                    2
                    * np.pi
                    * harmonic
                    * np.arange(n_full)
                    / seasonal_period
                )
            )

        seasonal_matrix_full = (
            np.column_stack(
                seasonal_terms
            )
        )

        n_seasonal_terms = (
            seasonal_matrix_full.shape[1]
        )

        seasonal_beta = (
            flatten_posterior_variable(
                "seasonal_beta"
            )
        )

        if seasonal_beta.ndim != 2:

            raise ValueError(
                f"{target_dma}: seasonal_beta has "
                f"unexpected shape "
                f"{seasonal_beta.shape}."
            )

        if seasonal_beta.shape != (
            n_posterior,
            n_seasonal_terms,
        ):

            raise ValueError(
                f"{target_dma}: seasonal_beta has shape "
                f"{seasonal_beta.shape}; expected "
                f"({n_posterior}, "
                f"{n_seasonal_terms})."
            )

        seasonal_component = (
            seasonal_beta @
            seasonal_matrix_full.T
        )

    else:

        n_seasonal_terms = 0

        seasonal_component = np.zeros(
            (
                n_posterior,
                n_full,
            ),
            dtype=float,
        )

    # =====================================================
    # Conditional Counterfactual Mean
    # =====================================================

    counterfactual_mean_scaled = (
        intercept[:, np.newaxis]
        +
        trend_beta[:, np.newaxis]
        *
        time_scaled_full[np.newaxis, :]
        +
        control_component
        +
        seasonal_component
    )

    # =====================================================
    # Transform Conditional Mean Back to Original Units
    # =====================================================

    counterfactual_mean_draws = (
        counterfactual_mean_scaled *
        y_std
        +
        y_mean
    )

    if not np.isfinite(
        counterfactual_mean_draws
    ).all():

        raise ValueError(
            f"{target_dma}: counterfactual mean draws "
            "contain non-finite values."
        )

    # =====================================================
    # Posterior Predictive Draws
    #
    # Without observation noise:
    #
    #     predictive draws = conditional mean draws
    #
    # With observation noise:
    #
    #     predictive draws = conditional mean draws
    #                         + observation noise
    #
    # sigma_obs is expressed in standardized target units.
    # =====================================================

    counterfactual_predictive_draws = (
        counterfactual_mean_draws.copy()
    )

    if include_observation_noise:

        sigma_obs = flatten_posterior_variable(
            "sigma_obs"
        )

        if sigma_obs.ndim != 1:

            raise ValueError(
                f"{target_dma}: sigma_obs has unexpected "
                f"shape {sigma_obs.shape}."
            )

        if not np.isfinite(
            sigma_obs
        ).all():

            raise ValueError(
                f"{target_dma}: sigma_obs contains "
                "non-finite values."
            )

        if np.any(
            sigma_obs < 0
        ):

            raise ValueError(
                f"{target_dma}: sigma_obs contains "
                "negative values."
            )

        rng = np.random.default_rng(
            random_seed
        )

        observation_noise = rng.normal(
            loc=0.0,
            scale=sigma_obs[:, np.newaxis],
            size=(
                n_posterior,
                n_full,
            ),
        )

        counterfactual_predictive_draws = (
            counterfactual_mean_draws
            +
            observation_noise * y_std
        )

    if not np.isfinite(
        counterfactual_predictive_draws
    ).all():

        raise ValueError(
            f"{target_dma}: posterior predictive "
            "counterfactual draws contain non-finite "
            "values."
        )

    # =====================================================
    # Posterior Predictive Summaries
    # =====================================================

    counterfactual_predictive_mean = (
        np.mean(
            counterfactual_predictive_draws,
            axis=0,
        )
    )

    counterfactual_predictive_lower = (
        np.percentile(
            counterfactual_predictive_draws,
            2.5,
            axis=0,
        )
    )

    counterfactual_predictive_upper = (
        np.percentile(
            counterfactual_predictive_draws,
            97.5,
            axis=0,
        )
    )

    # =====================================================
    # Validate Output
    # =====================================================

    expected_shape = (
        n_posterior,
        n_full,
    )

    if counterfactual_mean_draws.shape != expected_shape:

        raise ValueError(
            f"{target_dma}: counterfactual_mean_draws "
            f"has shape {counterfactual_mean_draws.shape}; "
            f"expected {expected_shape}."
        )

    if counterfactual_predictive_draws.shape != expected_shape:

        raise ValueError(
            f"{target_dma}: counterfactual_predictive_draws "
            f"has shape {counterfactual_predictive_draws.shape}; "
            f"expected {expected_shape}."
        )

    if not (
        np.isfinite(
            counterfactual_predictive_mean
        ).all()
        and
        np.isfinite(
            counterfactual_predictive_lower
        ).all()
        and
        np.isfinite(
            counterfactual_predictive_upper
        ).all()
    ):

        raise ValueError(
            f"{target_dma}: counterfactual predictive "
            "summaries contain non-finite values."
        )

    # =====================================================
    # Identify Post-Treatment Period
    # =====================================================

    post_mask = (
        dates >= intervention_date
    )

    n_post = int(
        post_mask.sum()
    )

    if n_post == 0:

        raise ValueError(
            f"{target_dma}: no post-treatment periods "
            "were found."
        )

    # =====================================================
    # Logging
    # =====================================================

    logger.info(
        f"Counterfactual generated for {target_dma}."
    )

    logger.info(
        f"  Posterior draws      : {n_posterior}"
    )

    logger.info(
        f"  Modeling periods     : {n_full}"
    )

    logger.info(
        f"  Pre-treatment       : {n_pre}"
    )

    logger.info(
        f"  Post-treatment      : {n_post}"
    )

    logger.info(
        f"  Control predictors  : {n_controls}"
    )

    logger.info(
        f"  Seasonal terms      : "
        f"{n_seasonal_terms}"
    )

    logger.info(
        f"  Observation noise   : "
        f"{include_observation_noise}"
    )

    logger.info(
        f"  Counterfactual shape: "
        f"{counterfactual_predictive_draws.shape}"
    )

    # =====================================================
    # Return
    # =====================================================

    return {
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

#=========================================================
# Validate Counterfactual Predictions
#=========================================================
def validate_counterfactual(
    model_results,
    model_data,
    min_pre_r2=0.80,
    max_pre_rmse_ratio=0.20,
    max_pre_mae_ratio=0.15,
    min_pre_coverage=0.80,
    max_mean_bias_ratio=0.10,
):
    """
    Validate posterior counterfactual quality using the
    pre-treatment period.

    The counterfactual is considered useful for causal-impact
    analysis only if the fitted model reproduces observed
    pre-treatment traffic with acceptable accuracy, calibration,
    and uncertainty coverage.

    Parameters
    ----------
    model_results : dict
        Output from fit_bsts_model() containing posterior
        counterfactual predictions.

    model_data : dict
        Metadata returned by configure_bsts_model().

    min_pre_r2 : float, default=0.80
        Minimum acceptable pre-treatment R².

    max_pre_rmse_ratio : float, default=0.20
        Maximum acceptable RMSE relative to mean observed
        pre-treatment traffic.

    max_pre_mae_ratio : float, default=0.15
        Maximum acceptable MAE relative to mean observed
        pre-treatment traffic.

    min_pre_coverage : float, default=0.80
        Minimum acceptable proportion of observed pre-treatment
        values falling within the 95% posterior predictive interval.

    max_mean_bias_ratio : float, default=0.10
        Maximum acceptable absolute mean prediction bias relative
        to mean observed pre-treatment traffic.

    Returns
    -------
    validation : dict
        Structured validation results.

    validation_summary : pd.DataFrame
        One-row DataFrame suitable for consolidation across DMAs.
    """

    logger.info("=" * 70)
    logger.info("Validating posterior counterfactual quality...")
    logger.info("=" * 70)

    # =====================================================
    # Validate Inputs
    # =====================================================

    if not isinstance(model_results, dict):
        raise TypeError(
            "model_results must be a dictionary."
        )

    if not isinstance(model_data, dict):
        raise TypeError(
            "model_data must be a dictionary."
        )

    required_model_result_keys = [
        "counterfactual_mean_draws",
        "counterfactual_predictive_draws",
        "counterfactual_predictive_mean",
        "counterfactual_predictive_lower",
        "counterfactual_predictive_upper",
    ]

    missing_results = [
        key
        for key in required_model_result_keys
        if key not in model_results
    ]

    if missing_results:
        raise ValueError(
            "model_results is missing required fields: "
            f"{missing_results}"
        )

    required_model_data_keys = [
        "target_dma",
        "y_pre",
        "n_pre",
        "dates",
        "intervention_date",
    ]

    missing_data = [
        key
        for key in required_model_data_keys
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

    target_dma = model_data["target_dma"]

    n_pre = int(
        model_data["n_pre"]
    )

    dates = pd.DatetimeIndex(
        model_data["dates"]
    )

    intervention_date = pd.Timestamp(
        model_data["intervention_date"]
    )

    y_pre = np.asarray(
        model_data["y_pre"],
        dtype=float,
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
            f"{target_dma}: modeling dates contain duplicates."
        )

    if n_pre <= 0:
        raise ValueError(
            f"{target_dma}: n_pre must be greater than zero."
        )

    if n_pre > len(dates):
        raise ValueError(
            f"{target_dma}: n_pre={n_pre} exceeds "
            f"available modeling periods={len(dates)}."
        )

    # =====================================================
    # Validate Pre-treatment Alignment
    # =====================================================

    pre_dates = dates[:n_pre]

    if not np.all(
        pre_dates < intervention_date
    ):
        raise ValueError(
            f"{target_dma}: n_pre={n_pre} includes "
            "one or more dates on or after the intervention date."
        )

    if len(dates) > n_pre:

        post_dates = dates[n_pre:]

        if not np.all(
            post_dates >= intervention_date
        ):
            raise ValueError(
                f"{target_dma}: dates after n_pre contain "
                "values before the intervention date."
            )

    # =====================================================
    # Validate Observed Pre-treatment Series
    # =====================================================

    if len(y_pre) != n_pre:
        raise ValueError(
            f"{target_dma}: y_pre contains "
            f"{len(y_pre)} observations but n_pre="
            f"{n_pre}."
        )

    if not np.isfinite(y_pre).all():
        raise ValueError(
            f"{target_dma}: y_pre contains "
            "non-finite values."
        )

    # =====================================================
    # Extract Counterfactual Outputs
    # =====================================================

    counterfactual_mean_draws = np.asarray(
        model_results["counterfactual_mean_draws"],
        dtype=float,
    )

    counterfactual_predictive_draws = np.asarray(
        model_results["counterfactual_predictive_draws"],
        dtype=float,
    )

    counterfactual_predictive_mean = np.asarray(
        model_results["counterfactual_predictive_mean"],
        dtype=float,
    )

    counterfactual_predictive_lower = np.asarray(
        model_results["counterfactual_predictive_lower"],
        dtype=float,
    )

    counterfactual_predictive_upper = np.asarray(
        model_results["counterfactual_predictive_upper"],
        dtype=float,
    )

    # =====================================================
    # Validate Counterfactual Draw Shapes
    # =====================================================

    if counterfactual_mean_draws.ndim != 2:

        raise ValueError(
            f"{target_dma}: counterfactual_mean_draws must "
            f"be 2-dimensional. Received "
            f"{counterfactual_mean_draws.shape}."
        )

    if counterfactual_predictive_draws.ndim != 2:

        raise ValueError(
            f"{target_dma}: counterfactual_predictive_draws "
            f"must be 2-dimensional. Received "
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
            f"counterfactual draws have different numbers "
            f"of posterior draws: "
            f"{n_draws} vs {predictive_draws}."
        )

    if predictive_periods != n_full:

        raise ValueError(
            f"{target_dma}: conditional and predictive "
            f"counterfactual draws have different time "
            f"dimensions: "
            f"{n_full} vs {predictive_periods}."
        )

    if n_full != len(dates):

        raise ValueError(
            f"{target_dma}: counterfactual contains "
            f"{n_full} periods but dates contains "
            f"{len(dates)}."
        )

    # =====================================================
    # Validate Predictive Summary Shapes
    # =====================================================

    expected_vector_shape = (n_full,)

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
    # Validate Counterfactual Numeric Values
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

    for name, values in arrays_to_validate.items():

        if not np.isfinite(values).all():

            raise ValueError(
                f"{target_dma}: {name} contains "
                "non-finite values."
            )
    # =====================================================
    # Restrict to Pre-treatment Period
    # =====================================================

    y_pre_cf = y_pre

    cf_mean_pre = (
        counterfactual_predictive_mean[
            :n_pre
        ]
    )

    cf_lower_pre = (
        counterfactual_predictive_lower[
            :n_pre
        ]
    )

    cf_upper_pre = (
        counterfactual_predictive_upper[
            :n_pre
        ]
    )

    # =====================================================
    # Validate Prediction Interval
    # =====================================================

    if np.any(
        cf_lower_pre > cf_upper_pre
    ):
        raise ValueError(
            f"{target_dma}: counterfactual interval "
            "lower bounds exceed upper bounds."
        )

    # =====================================================
    # Basic Prediction Errors
    # =====================================================

    errors = (
        cf_mean_pre -
        y_pre_cf
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
        np.mean(y_pre_cf)
    )

    if (
        not np.isfinite(mean_observed)
        or
        mean_observed == 0
    ):
        raise ValueError(
            f"{target_dma}: mean pre-treatment traffic "
            "is zero or non-finite."
        )

    # =====================================================
    # Relative Error Metrics
    # =====================================================

    pre_rmse_ratio = (
        pre_rmse /
        abs(mean_observed)
    )

    pre_mae_ratio = (
        pre_mae /
        abs(mean_observed)
    )

    mean_bias = float(
        np.mean(errors)
    )

    mean_bias_ratio = (
        abs(mean_bias) /
        abs(mean_observed)
    )

    # =====================================================
    # Pre-treatment R²
    # =====================================================

    ss_res = float(
        np.sum(
            squared_errors
        )
    )

    ss_tot = float(
        np.sum(
            (
                y_pre_cf -
                mean_observed
            ) ** 2
        )
    )

    if ss_tot > 0:

        pre_r2 = float(
            1.0 -
            (
                ss_res /
                ss_tot
            )
        )

    else:

        pre_r2 = np.nan

    # =====================================================
    # 95% Posterior Predictive Coverage
    # =====================================================

    inside_interval = (
        (y_pre_cf >= cf_lower_pre)
        &
        (y_pre_cf <= cf_upper_pre)
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
        cf_upper_pre -
        cf_lower_pre
    )

    mean_interval_width = float(
        np.mean(
            interval_width
        )
    )

    mean_interval_width_ratio = (
        mean_interval_width /
        abs(mean_observed)
    )

    # =====================================================
    # Prediction Correlation
    # =====================================================

    observed_std = float(
        np.std(y_pre_cf)
    )

    predicted_std = float(
        np.std(cf_mean_pre)
    )

    if (
        observed_std > 0
        and
        predicted_std > 0
    ):

        pre_correlation = float(
            np.corrcoef(
                y_pre_cf,
                cf_mean_pre,
            )[0, 1]
        )

    else:

        pre_correlation = np.nan

    # =====================================================
    # Validation Criteria
    # =====================================================

    r2_pass = (
        np.isfinite(pre_r2)
        and
        pre_r2 >= min_pre_r2
    )

    rmse_pass = (
        np.isfinite(pre_rmse_ratio)
        and
        pre_rmse_ratio <= max_pre_rmse_ratio
    )

    mae_pass = (
        np.isfinite(pre_mae_ratio)
        and
        pre_mae_ratio <= max_pre_mae_ratio
    )

    coverage_pass = (
        np.isfinite(pre_coverage_95)
        and
        pre_coverage_95 >= min_pre_coverage
    )

    bias_pass = (
        np.isfinite(mean_bias_ratio)
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
    }

    # =====================================================
    # One-row Summary
    # =====================================================

    validation_summary = pd.DataFrame(
        [validation]
    )

    # =====================================================
    # Logging
    # =====================================================

    if counterfactual_valid:

        logger.info(
            f"{target_dma}: counterfactual validation PASSED."
        )

    else:

        logger.warning(
            f"{target_dma}: counterfactual validation FAILED."
        )

    logger.info(
        f"  Pre-treatment R²       : "
        f"{pre_r2:.4f}"
    )

    logger.info(
        f"  Pre-treatment RMSE     : "
        f"{pre_rmse:,.2f} "
        f"({pre_rmse_ratio:.2%} of mean traffic)"
    )

    logger.info(
        f"  Pre-treatment MAE      : "
        f"{pre_mae:,.2f} "
        f"({pre_mae_ratio:.2%} of mean traffic)"
    )

    logger.info(
        f"  Mean bias              : "
        f"{mean_bias:,.2f} "
        f"({mean_bias_ratio:.2%})"
    )

    logger.info(
        f"  95% interval coverage  : "
        f"{pre_coverage_95:.2%}"
    )

    logger.info(
        f"  Mean interval width    : "
        f"{mean_interval_width:,.2f}"
    )

    logger.info(
        f"  Prediction correlation : "
        f"{pre_correlation:.4f}"
    )

    logger.info(
        f"  R² validation          : "
        f"{r2_pass}"
    )

    logger.info(
        f"  RMSE validation        : "
        f"{rmse_pass}"
    )

    logger.info(
        f"  MAE validation         : "
        f"{mae_pass}"
    )

    logger.info(
        f"  Coverage validation    : "
        f"{coverage_pass}"
    )

    logger.info(
        f"  Bias validation        : "
        f"{bias_pass}"
    )

    # =====================================================
    # Return
    # =====================================================

    return (
        validation,
        validation_summary,
    )
#=========================================================
# Extract DMA Impact
#=========================================================

def extract_dma_impact(
    model_results,
    model_data,
    panel,
):
    """
    Extract posterior DMA-level causal impact estimates.

    Uses posterior conditional-mean counterfactual draws,
    not posterior predictive draws containing observation noise.

    The observed traffic series is reconstructed from the
    model panel using the saved modeling dates. This avoids
    requiring y_full to be persisted inside model_data.

    Parameters
    ----------
    model_results : dict
        Output from generate_counterfactual().

    model_data : dict
        Reconstructed BSTS model metadata.

    panel : pd.DataFrame
        Full model panel containing at minimum:
            DMA
            Month
            Traffic

    Returns
    -------
    dma_summary : pd.DataFrame
        One-row cumulative DMA impact summary.

    monthly_impact : pd.DataFrame
        Monthly posterior impact estimates.
    """

    logger.info("=" * 70)
    logger.info("Extracting DMA-level causal impact...")
    logger.info("=" * 70)

    # =====================================================
    # Validate Inputs
    # =====================================================

    if not isinstance(model_results, dict):
        raise TypeError(
            "model_results must be a dictionary."
        )

    if not isinstance(model_data, dict):
        raise TypeError(
            "model_data must be a dictionary."
        )

    if not isinstance(panel, pd.DataFrame):
        raise TypeError(
            "panel must be a pandas DataFrame."
        )

    required_results = [
        "counterfactual_mean_draws",
    ]

    missing_results = [
        key
        for key in required_results
        if key not in model_results
    ]

    if missing_results:
        raise ValueError(
            "model_results is missing required fields: "
            f"{missing_results}"
        )

    required_data = [
        "target_dma",
        "dates",
        "intervention_date",
    ]

    missing_data = [
        key
        for key in required_data
        if key not in model_data
    ]

    if missing_data:
        raise ValueError(
            "model_data is missing required fields: "
            f"{missing_data}"
        )

    required_panel_columns = {
        "DMA",
        "Month",
        "Traffic",
    }

    missing_panel_columns = (
        required_panel_columns
        - set(panel.columns)
    )

    if missing_panel_columns:
        raise ValueError(
            "panel is missing required columns: "
            f"{sorted(missing_panel_columns)}"
        )

    # =====================================================
    # Metadata
    # =====================================================

    target_dma = model_data[
        "target_dma"
    ]

    dates = pd.DatetimeIndex(
        pd.to_datetime(
            model_data["dates"]
        )
    )

    intervention_date = pd.Timestamp(
        model_data["intervention_date"]
    )

    # =====================================================
    # Validate Modeling Dates
    # =====================================================

    if len(dates) == 0:
        raise ValueError(
            f"{target_dma}: no modeling dates available."
        )

    if dates.has_duplicates:
        raise ValueError(
            f"{target_dma}: modeling dates contain "
            "duplicates."
        )

    if not dates.is_monotonic_increasing:
        raise ValueError(
            f"{target_dma}: modeling dates are not "
            "monotonically increasing."
        )

    # =====================================================
    # Prepare Panel
    # =====================================================

    panel_work = panel.copy()

    panel_work["Month"] = pd.to_datetime(
        panel_work["Month"]
    )

    # -----------------------------------------------------
    # Restrict to target DMA
    # -----------------------------------------------------

    dma_panel = panel_work.loc[
        panel_work["DMA"].eq(target_dma),
        [
            "Month",
            "Traffic",
        ],
    ].copy()

    if dma_panel.empty:
        raise ValueError(
            f"{target_dma}: no observations found in "
            "model panel."
        )

    # -----------------------------------------------------
    # Check for duplicate DMA-month observations
    # -----------------------------------------------------

    duplicate_mask = (
        dma_panel.duplicated(
            subset=["Month"],
            keep=False,
        )
    )

    if duplicate_mask.any():

        duplicate_months = (
            dma_panel.loc[
                duplicate_mask,
                "Month",
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        raise ValueError(
            f"{target_dma}: duplicate DMA-month "
            f"observations found in panel: "
            f"{duplicate_months}"
        )

    # =====================================================
    # Align Observed Traffic to Saved Modeling Dates
    # =====================================================

    dma_panel = (
        dma_panel
        .set_index("Month")
        .sort_index()
        .reindex(dates)
    )

    observed_full = (
        dma_panel["Traffic"]
        .to_numpy(
            dtype=float
        )
    )

    # =====================================================
    # Validate Observed Series
    # =====================================================

    if observed_full.ndim != 1:
        raise ValueError(
            f"{target_dma}: observed traffic must be "
            "one-dimensional."
        )

    if len(observed_full) != len(dates):
        raise ValueError(
            f"{target_dma}: observed traffic contains "
            f"{len(observed_full)} observations but "
            f"dates contains {len(dates)} periods."
        )

    if not np.isfinite(
        observed_full
    ).all():

        missing_dates = (
            dates[
                ~np.isfinite(observed_full)
            ]
            .strftime("%Y-%m-%d")
            .tolist()
        )

        raise ValueError(
            f"{target_dma}: observed Traffic contains "
            "missing or non-finite values after "
            "alignment to modeling dates. "
            f"Problem dates: {missing_dates}"
        )

    # =====================================================
    # Treatment Periods
    # =====================================================

    pre_mask = (
        dates < intervention_date
    )

    post_mask = (
        dates >= intervention_date
    )

    pre_period = dates[
        pre_mask
    ]

    post_period = dates[
        post_mask
    ]

    if len(pre_period) == 0:
        raise ValueError(
            f"{target_dma}: no pre-treatment periods "
            "found."
        )

    if len(post_period) == 0:
        raise ValueError(
            f"{target_dma}: no post-treatment periods "
            "found."
        )

    observed_post = (
        observed_full[
            post_mask
        ]
    )

    # =====================================================
    # Counterfactual Draws
    # =====================================================

    counterfactual_draws = np.asarray(
        model_results[
            "counterfactual_mean_draws"
        ],
        dtype=float,
    )

    if counterfactual_draws.ndim != 2:

        raise ValueError(
            f"{target_dma}: "
            "counterfactual_mean_draws must have "
            "shape (posterior_draws, periods). "
            f"Received {counterfactual_draws.shape}."
        )

    posterior_draws, n_periods = (
        counterfactual_draws.shape
    )

    if n_periods != len(dates):

        raise ValueError(
            f"{target_dma}: counterfactual contains "
            f"{n_periods} periods but dates contains "
            f"{len(dates)}."
        )

    if posterior_draws == 0:

        raise ValueError(
            f"{target_dma}: counterfactual contains "
            "zero posterior draws."
        )

    if not np.isfinite(
        counterfactual_draws
    ).all():

        raise ValueError(
            f"{target_dma}: counterfactual draws "
            "contain non-finite values."
        )

    # =====================================================
    # Post-Treatment Counterfactual
    # =====================================================

    counterfactual_post = (
        counterfactual_draws[
            :,
            post_mask,
        ]
    )

    # =====================================================
    # Posterior Impact
    #
    # Impact = Observed - Counterfactual
    # =====================================================

    impact_draws = (
        observed_post[
            np.newaxis,
            :
        ]
        -
        counterfactual_post
    )

    if impact_draws.shape != (
        posterior_draws,
        len(post_period),
    ):

        raise ValueError(
            f"{target_dma}: impact_draws has unexpected "
            f"shape {impact_draws.shape}."
        )

    # =====================================================
    # Monthly Impact Summaries
    # =====================================================

    monthly_mean = np.mean(
        impact_draws,
        axis=0,
    )

    monthly_median = np.median(
        impact_draws,
        axis=0,
    )

    monthly_lower = np.percentile(
        impact_draws,
        2.5,
        axis=0,
    )

    monthly_upper = np.percentile(
        impact_draws,
        97.5,
        axis=0,
    )

    monthly_probability_positive = (
        np.mean(
            impact_draws > 0,
            axis=0,
        )
    )

    monthly_probability_negative = (
        np.mean(
            impact_draws < 0,
            axis=0,
        )
    )

    # =====================================================
    # Monthly Counterfactual Summaries
    # =====================================================

    counterfactual_mean = np.mean(
        counterfactual_post,
        axis=0,
    )

    counterfactual_median = np.median(
        counterfactual_post,
        axis=0,
    )

    counterfactual_lower = np.percentile(
        counterfactual_post,
        2.5,
        axis=0,
    )

    counterfactual_upper = np.percentile(
        counterfactual_post,
        97.5,
        axis=0,
    )

    # =====================================================
    # Monthly Relative Lift
    # =====================================================

    monthly_relative_lift = np.divide(
        monthly_mean,
        counterfactual_mean,
        out=np.full_like(
            monthly_mean,
            np.nan,
        ),
        where=counterfactual_mean != 0,
    )

    # =====================================================
    # Monthly Results
    # =====================================================

    monthly_impact = pd.DataFrame({

        "DMA":
            target_dma,

        "Month":
            post_period,

        "Observed_Traffic":
            observed_post,

        "Counterfactual_Traffic_Mean":
            counterfactual_mean,

        "Counterfactual_Traffic_Median":
            counterfactual_median,

        "Counterfactual_Traffic_Lower_95":
            counterfactual_lower,

        "Counterfactual_Traffic_Upper_95":
            counterfactual_upper,

        "Incremental_Traffic_Mean":
            monthly_mean,

        "Incremental_Traffic_Median":
            monthly_median,

        "Impact_Lower_95":
            monthly_lower,

        "Impact_Upper_95":
            monthly_upper,

        "Probability_Positive":
            monthly_probability_positive,

        "Probability_Negative":
            monthly_probability_negative,

        "Relative_Lift":
            monthly_relative_lift,
    })

    # =====================================================
    # Cumulative Posterior Distributions
    # =====================================================

    cumulative_impact_draws = (
        impact_draws.sum(
            axis=1
        )
    )

    cumulative_counterfactual_draws = (
        counterfactual_post.sum(
            axis=1
        )
    )

    cumulative_observed = float(
        observed_post.sum()
    )

    # -----------------------------------------------------
    # Counterfactual
    # -----------------------------------------------------

    cumulative_counterfactual_mean = float(
        np.mean(
            cumulative_counterfactual_draws
        )
    )

    cumulative_counterfactual_median = float(
        np.median(
            cumulative_counterfactual_draws
        )
    )

    cumulative_counterfactual_lower = float(
        np.percentile(
            cumulative_counterfactual_draws,
            2.5,
        )
    )

    cumulative_counterfactual_upper = float(
        np.percentile(
            cumulative_counterfactual_draws,
            97.5,
        )
    )

    # -----------------------------------------------------
    # Impact
    # -----------------------------------------------------

    cumulative_impact_mean = float(
        np.mean(
            cumulative_impact_draws
        )
    )

    cumulative_impact_median = float(
        np.median(
            cumulative_impact_draws
        )
    )

    cumulative_impact_lower = float(
        np.percentile(
            cumulative_impact_draws,
            2.5,
        )
    )

    cumulative_impact_upper = float(
        np.percentile(
            cumulative_impact_draws,
            97.5,
        )
    )

    cumulative_probability_positive = float(
        np.mean(
            cumulative_impact_draws > 0
        )
    )

    cumulative_probability_negative = float(
        np.mean(
            cumulative_impact_draws < 0
        )
    )

    # =====================================================
    # Posterior Relative Lift
    # =====================================================

    relative_lift_draws = np.divide(
        cumulative_impact_draws,
        cumulative_counterfactual_draws,
        out=np.full_like(
            cumulative_impact_draws,
            np.nan,
        ),
        where=cumulative_counterfactual_draws != 0,
    )

    valid_relative_lift = (
        relative_lift_draws[
            np.isfinite(
                relative_lift_draws
            )
        ]
    )

    relative_lift_valid_fraction = (
        len(valid_relative_lift)
        /
        len(relative_lift_draws)
    )

    if len(valid_relative_lift) > 0:

        relative_lift_mean = float(
            np.mean(
                valid_relative_lift
            )
        )

        relative_lift_median = float(
            np.median(
                valid_relative_lift
            )
        )

        relative_lift_lower = float(
            np.percentile(
                valid_relative_lift,
                2.5,
            )
        )

        relative_lift_upper = float(
            np.percentile(
                valid_relative_lift,
                97.5,
            )
        )

    else:

        relative_lift_mean = np.nan
        relative_lift_median = np.nan
        relative_lift_lower = np.nan
        relative_lift_upper = np.nan

    # =====================================================
    # DMA Summary
    # =====================================================

    dma_summary = pd.DataFrame([{

        "DMA":
            target_dma,

        "Intervention_Date":
            intervention_date,

        "Pre_Treatment_Months":
            len(pre_period),

        "Post_Treatment_Months":
            len(post_period),

        "Posterior_Draws":
            posterior_draws,

        "Observed_Traffic":
            cumulative_observed,

        "Counterfactual_Traffic_Mean":
            cumulative_counterfactual_mean,

        "Counterfactual_Traffic_Median":
            cumulative_counterfactual_median,

        "Counterfactual_Traffic_Lower_95":
            cumulative_counterfactual_lower,

        "Counterfactual_Traffic_Upper_95":
            cumulative_counterfactual_upper,

        "Incremental_Traffic_Mean":
            cumulative_impact_mean,

        "Incremental_Traffic_Median":
            cumulative_impact_median,

        "Impact_Lower_95":
            cumulative_impact_lower,

        "Impact_Upper_95":
            cumulative_impact_upper,

        "Probability_Positive":
            cumulative_probability_positive,

        "Probability_Negative":
            cumulative_probability_negative,

        "Relative_Lift_Mean":
            relative_lift_mean,

        "Relative_Lift_Median":
            relative_lift_median,

        "Relative_Lift_Lower_95":
            relative_lift_lower,

        "Relative_Lift_Upper_95":
            relative_lift_upper,

        "Relative_Lift_Valid_Fraction":
            relative_lift_valid_fraction,
    }])

    # =====================================================
    # Logging
    # =====================================================

    logger.info(
        f"{target_dma}: DMA impact extraction complete."
    )

    logger.info(
        f"  Modeling periods      : "
        f"{len(dates)}"
    )

    logger.info(
        f"  Pre-treatment months  : "
        f"{len(pre_period)}"
    )

    logger.info(
        f"  Post-treatment months : "
        f"{len(post_period)}"
    )

    logger.info(
        f"  Posterior draws       : "
        f"{posterior_draws:,}"
    )

    logger.info(
        f"  Observed traffic      : "
        f"{cumulative_observed:,.0f}"
    )

    logger.info(
        f"  Counterfactual traffic: "
        f"{cumulative_counterfactual_mean:,.0f}"
    )

    logger.info(
        f"  Incremental traffic   : "
        f"{cumulative_impact_mean:,.0f}"
    )

    logger.info(
        f"  95% impact interval   : "
        f"[{cumulative_impact_lower:,.0f}, "
        f"{cumulative_impact_upper:,.0f}]"
    )

    logger.info(
        f"  P(Impact > 0)         : "
        f"{cumulative_probability_positive:.3f}"
    )

    if np.isfinite(
        relative_lift_median
    ):

        logger.info(
            f"  Posterior relative lift: "
            f"{relative_lift_median:.2%}"
        )

    logger.info("=" * 70)

    return (
        dma_summary,
        monthly_impact,
        impact_draws,
        counterfactual_post,
    )

#=========================================================
# Aggregate DMA Impact
#=========================================================
def aggregate_dma_impact(
    dma_results,
    model_validation=None,
    counterfactual_validation=None,
    require_valid_models=True,
):
    """
    Aggregate posterior DMA-level causal impact estimates.

    The aggregation is performed across DMA-level posterior impact
    distributions rather than simply summing DMA point estimates.

    Parameters
    ----------
    dma_results : dict
        Dictionary keyed by DMA containing the output from
        extract_dma_impact() and the posterior impact draws.

        Expected structure:

            {
                "DMA Name": {
                    "dma_summary": pd.DataFrame,
                    "monthly_impact": pd.DataFrame,
                    "impact_draws": np.ndarray,
                    "counterfactual_draws": np.ndarray,
                }
            }

        `impact_draws` must have shape:

            (posterior_draws, post_treatment_periods)

        `counterfactual_draws` must have the same shape.

    model_validation : pd.DataFrame, optional
        Model validation results.

    counterfactual_validation : pd.DataFrame, optional
        Counterfactual validation results.

    require_valid_models : bool, default=True
        If True, DMAs failing model/counterfactual validation are excluded
        from the aggregate.

    Returns
    -------
    aggregate_summary : pd.DataFrame
        One-row portfolio-level posterior impact summary.

    dma_analysis : pd.DataFrame
        DMA-level contribution/ranking table.

    aggregate_draws : np.ndarray
        Posterior aggregate incremental traffic draws.
    """

    logger.info("=" * 70)
    logger.info("Aggregating DMA-level causal impact...")
    logger.info("=" * 70)

    # =====================================================
    # Validate Inputs
    # =====================================================

    if not isinstance(dma_results, dict):
        raise TypeError(
            "dma_results must be a dictionary."
        )

    if len(dma_results) == 0:
        raise ValueError(
            "dma_results is empty."
        )

    # =====================================================
    # Validation Lookup Tables
    # =====================================================

    model_validation_lookup = {}

    if model_validation is not None:

        if not isinstance(
            model_validation,
            pd.DataFrame,
        ):
            raise TypeError(
                "model_validation must be a DataFrame."
            )

        if "DMA" not in model_validation.columns:
            raise ValueError(
                "model_validation must contain a 'DMA' column."
            )

        model_validation_lookup = (
            model_validation
            .set_index("DMA")
            .to_dict(
                orient="index"
            )
        )

    counterfactual_validation_lookup = {}

    if counterfactual_validation is not None:

        if not isinstance(
            counterfactual_validation,
            pd.DataFrame,
        ):
            raise TypeError(
                "counterfactual_validation must be a DataFrame."
            )

        if "DMA" not in counterfactual_validation.columns:
            raise ValueError(
                "counterfactual_validation must contain "
                "a 'DMA' column."
            )

        counterfactual_validation_lookup = (
            counterfactual_validation
            .set_index("DMA")
            .to_dict(
                orient="index"
            )
        )

    # =====================================================
    # Determine Valid DMAs
    # =====================================================

    valid_dmas = []
    excluded_dmas = []

    for dma, result in dma_results.items():

        is_valid = True
        exclusion_reasons = []

        # -------------------------------------------------
        # Model Validation
        # -------------------------------------------------

        if (
            require_valid_models
            and
            model_validation_lookup
        ):

            validation = (
                model_validation_lookup
                .get(dma)
            )

            if validation is None:

                is_valid = False

                exclusion_reasons.append(
                    "missing_model_validation"
                )

            else:

                if "converged" in validation:

                    if not bool(
                        validation["converged"]
                    ):

                        is_valid = False

                        exclusion_reasons.append(
                            "model_not_converged"
                        )

        # -------------------------------------------------
        # Counterfactual Validation
        # -------------------------------------------------

        if (
            require_valid_models
            and
            counterfactual_validation_lookup
        ):

            validation = (
                counterfactual_validation_lookup
                .get(dma)
            )

            if validation is None:

                is_valid = False

                exclusion_reasons.append(
                    "missing_counterfactual_validation"
                )

            else:

                if "counterfactual_valid" in validation:

                    if not bool(
                        validation[
                            "counterfactual_valid"
                        ]
                    ):

                        is_valid = False

                        exclusion_reasons.append(
                            "counterfactual_invalid"
                        )

        if is_valid:

            valid_dmas.append(dma)

        else:

            excluded_dmas.append({
                "DMA":
                    dma,

                "Reasons":
                    "; ".join(
                        exclusion_reasons
                    ),
            })

    # =====================================================
    # Validation Summary
    # =====================================================

    logger.info(
        f"Total DMAs available       : "
        f"{len(dma_results)}"
    )

    logger.info(
        f"DMAs included in aggregate : "
        f"{len(valid_dmas)}"
    )

    logger.info(
        f"DMAs excluded              : "
        f"{len(excluded_dmas)}"
    )

    if excluded_dmas:

        logger.warning(
            "Excluded DMAs:"
        )

        for item in excluded_dmas:

            logger.warning(
                f"  {item['DMA']}: "
                f"{item['Reasons']}"
            )

    if len(valid_dmas) == 0:

        raise ValueError(
            "No valid DMAs remain for aggregation."
        )

    # =====================================================
    # Extract DMA Posterior Draws
    # =====================================================

    aggregate_dma_rows = []

    aggregate_draws_by_dma = {}

    aggregate_counterfactual_by_dma = {}

    for dma in valid_dmas:

        result = dma_results[dma]

        if not isinstance(
            result,
            dict,
        ):
            raise TypeError(
                f"{dma}: DMA result must be a dictionary."
            )

        if "impact_draws" not in result:

            raise ValueError(
                f"{dma}: DMA result is missing "
                "'impact_draws'."
            )

        if "counterfactual_draws" not in result:

            raise ValueError(
                f"{dma}: DMA result is missing "
                "'counterfactual_draws'."
            )

        impact_draws = np.asarray(
            result["impact_draws"],
            dtype=float,
        )

        counterfactual_draws = np.asarray(
            result["counterfactual_draws"],
            dtype=float,
        )

        if impact_draws.ndim != 2:

            raise ValueError(
                f"{dma}: impact_draws must be 2-dimensional. "
                f"Received {impact_draws.shape}."
            )

        if counterfactual_draws.shape != (
            impact_draws.shape
        ):

            raise ValueError(
                f"{dma}: counterfactual_draws shape "
                f"{counterfactual_draws.shape} does not "
                f"match impact_draws shape "
                f"{impact_draws.shape}."
            )

        if not np.isfinite(
            impact_draws
        ).all():

            raise ValueError(
                f"{dma}: impact_draws contain "
                "non-finite values."
            )

        if not np.isfinite(
            counterfactual_draws
        ).all():

            raise ValueError(
                f"{dma}: counterfactual_draws contain "
                "non-finite values."
            )

        # -------------------------------------------------
        # Collapse Monthly Draws Into Cumulative DMA Draws
        # -------------------------------------------------

        dma_impact_draws = (
            impact_draws.sum(
                axis=1
            )
        )

        dma_counterfactual_draws = (
            counterfactual_draws.sum(
                axis=1
            )
        )

        aggregate_draws_by_dma[
            dma
        ] = dma_impact_draws

        aggregate_counterfactual_by_dma[
            dma
        ] = dma_counterfactual_draws

        # -------------------------------------------------
        # DMA Summary Statistics
        # -------------------------------------------------

        observed = float(
            np.mean(
                dma_counterfactual_draws
                +
                dma_impact_draws
            )
        )

        counterfactual_mean = float(
            np.mean(
                dma_counterfactual_draws
            )
        )

        impact_mean = float(
            np.mean(
                dma_impact_draws
            )
        )

        impact_lower = float(
            np.percentile(
                dma_impact_draws,
                2.5,
            )
        )

        impact_upper = float(
            np.percentile(
                dma_impact_draws,
                97.5,
            )
        )

        probability_positive = float(
            np.mean(
                dma_impact_draws > 0
            )
        )

        relative_lift = (
            impact_mean
            /
            counterfactual_mean
            if counterfactual_mean != 0
            else np.nan
        )

        aggregate_dma_rows.append({

            "DMA":
                dma,

            "Observed_Traffic":
                observed,

            "Counterfactual_Traffic":
                counterfactual_mean,

            "Incremental_Traffic":
                impact_mean,

            "Impact_Lower_95":
                impact_lower,

            "Impact_Upper_95":
                impact_upper,

            "Probability_Positive":
                probability_positive,

            "Relative_Lift":
                relative_lift,

            "Posterior_Draws":
                len(dma_impact_draws),
        })

    # =====================================================
    # Align Posterior Draws
    # =====================================================
    #
    # Each DMA should have the same number of posterior
    # draws because all models were fitted with the same
    # sampling configuration.
    #
    # We validate this rather than silently resampling.

    posterior_counts = {
        dma:
            len(draws)
        for dma, draws
        in aggregate_draws_by_dma.items()
    }

    unique_draw_counts = set(
        posterior_counts.values()
    )

    if len(unique_draw_counts) != 1:

        raise ValueError(
            "DMAs have inconsistent posterior draw counts: "
            f"{posterior_counts}"
        )

    # =====================================================
    # Portfolio Posterior
    # =====================================================

    aggregate_draws = np.sum(
        np.column_stack([
            aggregate_draws_by_dma[dma]
            for dma in valid_dmas
        ]),
        axis=1,
    )

    aggregate_counterfactual_draws = np.sum(
        np.column_stack([
            aggregate_counterfactual_by_dma[dma]
            for dma in valid_dmas
        ]),
        axis=1,
    )

    # =====================================================
    # Portfolio Observed Traffic
    # =====================================================

    aggregate_observed = float(
        np.mean(
            aggregate_counterfactual_draws
            +
            aggregate_draws
        )
    )

    # =====================================================
    # Portfolio Counterfactual
    # =====================================================

    aggregate_counterfactual_mean = float(
        np.mean(
            aggregate_counterfactual_draws
        )
    )

    aggregate_counterfactual_median = float(
        np.median(
            aggregate_counterfactual_draws
        )
    )

    aggregate_counterfactual_lower = float(
        np.percentile(
            aggregate_counterfactual_draws,
            2.5,
        )
    )

    aggregate_counterfactual_upper = float(
        np.percentile(
            aggregate_counterfactual_draws,
            97.5,
        )
    )

    # =====================================================
    # Portfolio Impact
    # =====================================================

    aggregate_impact_mean = float(
        np.mean(
            aggregate_draws
        )
    )

    aggregate_impact_median = float(
        np.median(
            aggregate_draws
        )
    )

    aggregate_impact_lower = float(
        np.percentile(
            aggregate_draws,
            2.5,
        )
    )

    aggregate_impact_upper = float(
        np.percentile(
            aggregate_draws,
            97.5,
        )
    )

    aggregate_probability_positive = float(
        np.mean(
            aggregate_draws > 0
        )
    )

    aggregate_probability_negative = float(
        np.mean(
            aggregate_draws < 0
        )
    )

    # =====================================================
    # Portfolio Relative Lift
    # =====================================================

    aggregate_relative_lift_draws = np.divide(
        aggregate_draws,
        aggregate_counterfactual_draws,
        out=np.full_like(
            aggregate_draws,
            np.nan,
        ),
        where=(
            aggregate_counterfactual_draws
            != 0
        ),
    )

    valid_lift = (
        aggregate_relative_lift_draws[
            np.isfinite(
                aggregate_relative_lift_draws
            )
        ]
    )

    if len(valid_lift) > 0:

        aggregate_relative_lift_mean = float(
            np.mean(
                valid_lift
            )
        )

        aggregate_relative_lift_median = float(
            np.median(
                valid_lift
            )
        )

        aggregate_relative_lift_lower = float(
            np.percentile(
                valid_lift,
                2.5,
            )
        )

        aggregate_relative_lift_upper = float(
            np.percentile(
                valid_lift,
                97.5,
            )
        )

    else:

        aggregate_relative_lift_mean = np.nan
        aggregate_relative_lift_median = np.nan
        aggregate_relative_lift_lower = np.nan
        aggregate_relative_lift_upper = np.nan

    # =====================================================
    # DMA Analysis
    # =====================================================

    dma_analysis = pd.DataFrame(
        aggregate_dma_rows
    )

    dma_analysis[
        "Impact_Rank"
    ] = (
        dma_analysis[
            "Incremental_Traffic"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    dma_analysis[
        "Absolute_Impact"
    ] = np.abs(
        dma_analysis[
            "Incremental_Traffic"
        ]
    )

    total_positive_impact = (
        dma_analysis.loc[
            dma_analysis[
                "Incremental_Traffic"
            ] > 0,
            "Incremental_Traffic",
        ]
        .sum()
    )

    total_negative_impact = (
        dma_analysis.loc[
            dma_analysis[
                "Incremental_Traffic"
            ] < 0,
            "Incremental_Traffic",
        ]
        .sum()
    )

    if total_positive_impact != 0:

        dma_analysis[
            "Positive_Impact_Share"
        ] = np.where(
            dma_analysis[
                "Incremental_Traffic"
            ] > 0,

            dma_analysis[
                "Incremental_Traffic"
            ]
            /
            total_positive_impact,

            0.0,
        )

    else:

        dma_analysis[
            "Positive_Impact_Share"
        ] = 0.0

    if total_negative_impact != 0:

        dma_analysis[
            "Negative_Impact_Share"
        ] = np.where(
            dma_analysis[
                "Incremental_Traffic"
            ] < 0,

            dma_analysis[
                "Incremental_Traffic"
            ]
            /
            total_negative_impact,

            0.0,
        )

    else:

        dma_analysis[
            "Negative_Impact_Share"
        ] = 0.0

    dma_analysis[
        "Impact_Interval_Excludes_Zero"
    ] = (
        (
            dma_analysis[
                "Impact_Lower_95"
            ] > 0
        )
        |
        (
            dma_analysis[
                "Impact_Upper_95"
            ] < 0
        )
    )

    dma_analysis = (
        dma_analysis
        .sort_values(
            "Incremental_Traffic",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # =====================================================
    # Aggregate Summary
    # =====================================================

    aggregate_summary = pd.DataFrame([{

        "DMAs_Included":
            len(valid_dmas),

        "DMAs_Excluded":
            len(excluded_dmas),

        "Posterior_Draws":
            len(aggregate_draws),

        "Observed_Traffic":
            aggregate_observed,

        "Counterfactual_Traffic_Mean":
            aggregate_counterfactual_mean,

        "Counterfactual_Traffic_Median":
            aggregate_counterfactual_median,

        "Counterfactual_Traffic_Lower_95":
            aggregate_counterfactual_lower,

        "Counterfactual_Traffic_Upper_95":
            aggregate_counterfactual_upper,

        "Incremental_Traffic_Mean":
            aggregate_impact_mean,

        "Incremental_Traffic_Median":
            aggregate_impact_median,

        "Impact_Lower_95":
            aggregate_impact_lower,

        "Impact_Upper_95":
            aggregate_impact_upper,

        "Probability_Positive":
            aggregate_probability_positive,

        "Probability_Negative":
            aggregate_probability_negative,

        "Relative_Lift_Mean":
            aggregate_relative_lift_mean,

        "Relative_Lift_Median":
            aggregate_relative_lift_median,

        "Relative_Lift_Lower_95":
            aggregate_relative_lift_lower,

        "Relative_Lift_Upper_95":
            aggregate_relative_lift_upper,

        "Relative_Lift_Valid_Fraction":
            (
                len(valid_lift)
                /
                len(aggregate_relative_lift_draws)
            ),

    }])

    # =====================================================
    # Logging
    # =====================================================

    logger.info("")
    logger.info(
        "Aggregate DMA impact:"
    )

    logger.info(
        f"  DMAs included          : "
        f"{len(valid_dmas)}"
    )

    logger.info(
        f"  DMAs excluded          : "
        f"{len(excluded_dmas)}"
    )

    logger.info(
        f"  Observed traffic       : "
        f"{aggregate_observed:,.0f}"
    )

    logger.info(
        f"  Counterfactual traffic : "
        f"{aggregate_counterfactual_mean:,.0f}"
    )

    logger.info(
        f"  Incremental traffic    : "
        f"{aggregate_impact_mean:,.0f}"
    )

    logger.info(
        f"  95% impact interval    : "
        f"[{aggregate_impact_lower:,.0f}, "
        f"{aggregate_impact_upper:,.0f}]"
    )

    logger.info(
        f"  P(Impact > 0)          : "
        f"{aggregate_probability_positive:.3f}"
    )

    logger.info(
        f"  Relative lift          : "
        f"{aggregate_relative_lift_median:.2%}"
    )

    logger.info("")
    logger.info(
        "Top positive DMA contributors:"
    )

    logger.info(
        "\n%s",
        dma_analysis[
            [
                "DMA",
                "Incremental_Traffic",
                "Relative_Lift",
                "Probability_Positive",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    logger.info("")
    logger.info(
        "Top negative DMA contributors:"
    )

    logger.info(
        "\n%s",
        dma_analysis[
            [
                "DMA",
                "Incremental_Traffic",
                "Relative_Lift",
                "Probability_Positive",
            ]
        ]
        .tail(10)
        .sort_values(
            "Incremental_Traffic"
        )
        .to_string(
            index=False
        )
    )

    logger.info("=" * 70)

    return (
        aggregate_summary,
        dma_analysis,
        aggregate_draws,
    )

#=========================================================
# DMA Contribution Analysis
#=========================================================
def analyze_dma_contribution(
    dma_results,
    aggregate_draws,
    require_valid_models=True,
):
    """
    Analyze each DMA's contribution to aggregate incremental traffic.

    Parameters
    ----------
    dma_results : dict
        Dictionary keyed by DMA containing the outputs generated during
        the main validation loop. Each DMA entry should contain:

            - dma_summary
            - monthly_impact
            - impact_draws
            - counterfactual_draws
            - model_validation
            - counterfactual_validation

    aggregate_draws : np.ndarray
        Posterior draws of aggregate incremental traffic.

        Expected shape:
            (posterior_draws,)

    require_valid_models : bool, default=True
        If True, only DMAs that pass both BSTS model validation and
        counterfactual validation are included.

    Returns
    -------
    dma_contribution : pd.DataFrame
        One row per included DMA containing contribution metrics.

    contribution_summary : pd.DataFrame
        Portfolio-level summary of the DMA contribution analysis.

    dma_contribution_draws : pd.DataFrame
        Posterior contribution draws by DMA.
    """

    logger.info("=" * 70)
    logger.info("Analyzing DMA contribution to aggregate impact...")
    logger.info("=" * 70)

    # =====================================================
    # Validate Inputs
    # =====================================================

    if not isinstance(dma_results, dict):
        raise TypeError(
            "dma_results must be a dictionary."
        )

    if not dma_results:
        raise ValueError(
            "dma_results is empty."
        )

    aggregate_draws = np.asarray(
        aggregate_draws,
        dtype=float,
    )

    if aggregate_draws.ndim != 1:
        raise ValueError(
            "aggregate_draws must be one-dimensional."
        )

    if len(aggregate_draws) == 0:
        raise ValueError(
            "aggregate_draws is empty."
        )

    if not np.isfinite(aggregate_draws).all():
        raise ValueError(
            "aggregate_draws contains non-finite values."
        )

    # =====================================================
    # Identify Valid DMAs
    # =====================================================

    included_dmas = []
    excluded_dmas = []

    for dma, result in dma_results.items():

        if not isinstance(result, dict):
            excluded_dmas.append(
                (dma, "invalid result structure")
            )
            continue

        model_validation = result.get(
            "model_validation"
        )

        counterfactual_validation = result.get(
            "counterfactual_validation"
        )

        if (
            model_validation is None
            or counterfactual_validation is None
        ):

            if require_valid_models:

                excluded_dmas.append(
                    (
                        dma,
                        "missing validation results",
                    )
                )

                continue

        if require_valid_models:

            model_valid = bool(
                model_validation.get(
                    "converged",
                    False,
                )
            )

            counterfactual_valid = bool(
                counterfactual_validation.get(
                    "counterfactual_valid",
                    False,
                )
            )

            if not model_valid:

                excluded_dmas.append(
                    (
                        dma,
                        "BSTS model validation failed",
                    )
                )

                continue

            if not counterfactual_valid:

                excluded_dmas.append(
                    (
                        dma,
                        "counterfactual validation failed",
                    )
                )

                continue

        if "impact_draws" not in result:

            excluded_dmas.append(
                (
                    dma,
                    "impact_draws missing",
                )
            )

            continue

        impact_draws = np.asarray(
            result["impact_draws"],
            dtype=float,
        )

        if impact_draws.ndim != 2:

            excluded_dmas.append(
                (
                    dma,
                    "impact_draws is not 2-dimensional",
                )
            )

            continue

        if impact_draws.shape[0] != len(
            aggregate_draws
        ):

            excluded_dmas.append(
                (
                    dma,
                    (
                        "posterior draw count mismatch: "
                        f"{impact_draws.shape[0]} vs "
                        f"{len(aggregate_draws)}"
                    ),
                )
            )

            continue

        if not np.isfinite(
            impact_draws
        ).all():

            excluded_dmas.append(
                (
                    dma,
                    "impact_draws contains non-finite values",
                )
            )

            continue

        included_dmas.append(dma)

    if not included_dmas:

        raise ValueError(
            "No valid DMAs available for contribution analysis."
        )

    logger.info(
        f"Included DMAs: {len(included_dmas)}"
    )

    if excluded_dmas:

        logger.warning(
            f"Excluded DMAs: {len(excluded_dmas)}"
        )

        for dma, reason in excluded_dmas:

            logger.warning(
                f"  {dma}: {reason}"
            )

    # =====================================================
    # Extract Cumulative DMA Impact Draws
    # =====================================================

    dma_impact_draws = {}

    for dma in included_dmas:

        impact_draws = np.asarray(
            dma_results[dma]["impact_draws"],
            dtype=float,
        )

        dma_impact_draws[dma] = (
            impact_draws.sum(axis=1)
        )

    dma_impact_matrix = np.column_stack(
        [
            dma_impact_draws[dma]
            for dma in included_dmas
        ]
    )

    # Shape:
    #
    # posterior_draws x DMAs
    #
    # Each row represents one posterior draw.

    # =====================================================
    # Validate Against Aggregate Draws
    # =====================================================

    reconstructed_aggregate_draws = (
        dma_impact_matrix.sum(axis=1)
    )

    if not np.allclose(
        reconstructed_aggregate_draws,
        aggregate_draws,
        rtol=1e-5,
        atol=1e-5,
    ):

        logger.warning(
            "DMA impact draws do not exactly reproduce "
            "aggregate_draws."
        )

        logger.warning(
            "This may occur when aggregate_dma_impact() "
            "uses a different DMA inclusion set."
        )

    # =====================================================
    # DMA Contribution Draws
    # =====================================================

    dma_contribution_draws = {}

    for i, dma in enumerate(
        included_dmas
    ):

        dma_draws = (
            dma_impact_matrix[:, i]
        )

        dma_contribution_draws[dma] = (
            dma_draws
        )

    # =====================================================
    # Contribution Share Draws
    # =====================================================
    #
    # Important:
    #
    # We calculate contribution using the signed
    # incremental impact rather than absolute impact.
    #
    # This allows the contribution shares to preserve
    # the direction of each DMA's effect.
    #
    # However, when aggregate impact is near zero,
    # contribution ratios become unstable.
    #

    aggregate_abs = np.abs(
        aggregate_draws
    )

    contribution_share_matrix = np.divide(
        dma_impact_matrix,
        aggregate_draws[:, np.newaxis],
        out=np.full_like(
            dma_impact_matrix,
            np.nan,
        ),
        where=np.abs(
            aggregate_draws[:, np.newaxis]
        ) > 1e-12,
    )

    # =====================================================
    # Build DMA Contribution Summary
    # =====================================================

    contribution_rows = []

    for i, dma in enumerate(
        included_dmas
    ):

        impact_draws = (
            dma_impact_matrix[:, i]
        )

        share_draws = (
            contribution_share_matrix[:, i]
        )

        valid_share_draws = share_draws[
            np.isfinite(share_draws)
        ]

        # -------------------------------------------------
        # Cumulative Impact
        # -------------------------------------------------

        impact_mean = float(
            np.mean(impact_draws)
        )

        impact_median = float(
            np.median(impact_draws)
        )

        impact_lower = float(
            np.percentile(
                impact_draws,
                2.5,
            )
        )

        impact_upper = float(
            np.percentile(
                impact_draws,
                97.5,
            )
        )

        probability_positive = float(
            np.mean(
                impact_draws > 0
            )
        )

        probability_negative = float(
            np.mean(
                impact_draws < 0
            )
        )

        # -------------------------------------------------
        # Contribution Share
        # -------------------------------------------------

        if len(valid_share_draws) > 0:

            contribution_share_mean = float(
                np.mean(
                    valid_share_draws
                )
            )

            contribution_share_median = float(
                np.median(
                    valid_share_draws
                )
            )

            contribution_share_lower = float(
                np.percentile(
                    valid_share_draws,
                    2.5,
                )
            )

            contribution_share_upper = float(
                np.percentile(
                    valid_share_draws,
                    97.5,
                )
            )

            contribution_share_valid_fraction = (
                len(valid_share_draws)
                /
                len(share_draws)
            )

        else:

            contribution_share_mean = np.nan
            contribution_share_median = np.nan
            contribution_share_lower = np.nan
            contribution_share_upper = np.nan
            contribution_share_valid_fraction = 0.0

        # -------------------------------------------------
        # Absolute Contribution
        # -------------------------------------------------

        absolute_impact_mean = float(
            np.mean(
                np.abs(
                    impact_draws
                )
            )
        )

        # -------------------------------------------------
        # DMA Summary Information
        # -------------------------------------------------

        dma_summary = dma_results[dma].get(
            "dma_summary"
        )

        relative_lift_median = np.nan

        if (
            isinstance(
                dma_summary,
                pd.DataFrame,
            )
            and not dma_summary.empty
            and "Relative_Lift_Median"
            in dma_summary.columns
        ):

            relative_lift_median = float(
                dma_summary.iloc[0][
                    "Relative_Lift_Median"
                ]
            )

        # -------------------------------------------------
        # Append Row
        # -------------------------------------------------

        contribution_rows.append({

            "DMA":
                dma,

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

            "Contribution_Share_Mean":
                contribution_share_mean,

            "Contribution_Share_Median":
                contribution_share_median,

            "Contribution_Share_Lower_95":
                contribution_share_lower,

            "Contribution_Share_Upper_95":
                contribution_share_upper,

            "Contribution_Share_Valid_Fraction":
                contribution_share_valid_fraction,

            "Absolute_Impact_Mean":
                absolute_impact_mean,

            "Relative_Lift_Median":
                relative_lift_median,

            "Impact_Direction":
                (
                    "Positive"
                    if impact_mean > 0
                    else "Negative"
                    if impact_mean < 0
                    else "Neutral"
                ),

            "Model_Valid":
                bool(
                    dma_results[dma]
                    ["model_validation"]
                    .get(
                        "converged",
                        False,
                    )
                ),

            "Counterfactual_Valid":
                bool(
                    dma_results[dma]
                    ["counterfactual_validation"]
                    .get(
                        "counterfactual_valid",
                        False,
                    )
                ),
        })

    dma_contribution = pd.DataFrame(
        contribution_rows
    )

    # =====================================================
    # Rank DMAs
    # =====================================================

    dma_contribution[
        "Impact_Rank"
    ] = (
        dma_contribution[
            "Incremental_Traffic_Mean"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    dma_contribution[
        "Absolute_Impact_Rank"
    ] = (
        dma_contribution[
            "Absolute_Impact_Mean"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    dma_contribution[
        "Contribution_Rank"
    ] = (
        dma_contribution[
            "Contribution_Share_Mean"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    # -----------------------------------------------------
    # Sort by Contribution
    # -----------------------------------------------------

    dma_contribution = (
        dma_contribution
        .sort_values(
            "Contribution_Share_Mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # =====================================================
    # Posterior Contribution Draw DataFrame
    # =====================================================

    dma_contribution_draws = pd.DataFrame(
        dma_impact_matrix,
        columns=included_dmas,
    )

    dma_contribution_draws.insert(
        0,
        "Draw",
        np.arange(
            len(
                dma_contribution_draws
            )
        ),
    )

    # =====================================================
    # Contribution Summary
    # =====================================================

    total_positive_mean = float(
        dma_contribution.loc[
            dma_contribution[
                "Incremental_Traffic_Mean"
            ] > 0,
            "Incremental_Traffic_Mean",
        ].sum()
    )

    total_negative_mean = float(
        dma_contribution.loc[
            dma_contribution[
                "Incremental_Traffic_Mean"
            ] < 0,
            "Incremental_Traffic_Mean",
        ].sum()
    )

    aggregate_mean = float(
        np.mean(
            aggregate_draws
        )
    )

    contribution_summary = pd.DataFrame([{

        "Included_DMAs":
            len(included_dmas),

        "Excluded_DMAs":
            len(excluded_dmas),

        "Posterior_Draws":
            len(aggregate_draws),

        "Aggregate_Incremental_Traffic_Mean":
            aggregate_mean,

        "Aggregate_Incremental_Traffic_Median":
            float(
                np.median(
                    aggregate_draws
                )
            ),

        "Aggregate_Impact_Lower_95":
            float(
                np.percentile(
                    aggregate_draws,
                    2.5,
                )
            ),

        "Aggregate_Impact_Upper_95":
            float(
                np.percentile(
                    aggregate_draws,
                    97.5,
                )
            ),

        "Total_Positive_DMA_Impact":
            total_positive_mean,

        "Total_Negative_DMA_Impact":
            total_negative_mean,

        "Positive_DMA_Count":
            int(
                (
                    dma_contribution[
                        "Incremental_Traffic_Mean"
                    ] > 0
                ).sum()
            ),

        "Negative_DMA_Count":
            int(
                (
                    dma_contribution[
                        "Incremental_Traffic_Mean"
                    ] < 0
                ).sum()
            ),

        "Aggregate_Probability_Positive":
            float(
                np.mean(
                    aggregate_draws > 0
                )
            ),

        "Aggregate_Probability_Negative":
            float(
                np.mean(
                    aggregate_draws < 0
                )
            ),
    }])

    # =====================================================
    # Logging
    # =====================================================

    logger.info(
        "DMA contribution analysis complete."
    )

    logger.info(
        f"  Included DMAs       : "
        f"{len(included_dmas)}"
    )

    logger.info(
        f"  Positive DMAs       : "
        f"{(
            dma_contribution[
                'Incremental_Traffic_Mean'
            ] > 0
        ).sum()}"
    )

    logger.info(
        f"  Negative DMAs       : "
        f"{(
            dma_contribution[
                'Incremental_Traffic_Mean'
            ] < 0
        ).sum()}"
    )

    logger.info(
        f"  Aggregate impact    : "
        f"{aggregate_mean:,.0f}"
    )

    logger.info(
        "Top positive contributors:"
    )

    top_positive = (
        dma_contribution
        .loc[
            dma_contribution[
                "Incremental_Traffic_Mean"
            ] > 0
        ]
        .sort_values(
            "Incremental_Traffic_Mean",
            ascending=False,
        )
        .head(5)
    )

    for _, row in top_positive.iterrows():

        logger.info(
            f"  {row['DMA']}: "
            f"{row['Incremental_Traffic_Mean']:,.0f} "
            f"({row['Contribution_Share_Mean']:.2%})"
        )

    logger.info(
        "Top negative contributors:"
    )

    top_negative = (
        dma_contribution
        .loc[
            dma_contribution[
                "Incremental_Traffic_Mean"
            ] < 0
        ]
        .sort_values(
            "Incremental_Traffic_Mean"
        )
        .head(5)
    )

    for _, row in top_negative.iterrows():

        logger.info(
            f"  {row['DMA']}: "
            f"{row['Incremental_Traffic_Mean']:,.0f} "
            f"({row['Contribution_Share_Mean']:.2%})"
        )

    if excluded_dmas:

        logger.info(
            "Excluded DMAs:"
        )

        for dma, reason in excluded_dmas:

            logger.info(
                f"  {dma}: {reason}"
            )

    return (
        dma_contribution,
        contribution_summary,
        dma_contribution_draws,
    )


def main():

    logger.info("=" * 70)
    logger.info("Begin BSTS Causal Impact Validation")
    logger.info("=" * 70)

    # =====================================================
    # Configuration
    # =====================================================

    create_output_folders()

    # =====================================================
    # Load Model Panel
    # =====================================================

    panel = load_model_panel()

    # =====================================================
    # Load Saved BSTS Artifacts
    # =====================================================

    logger.info("=" * 70)
    logger.info("Loading saved BSTS artifacts")
    logger.info("=" * 70)

    bsts_artifacts = load_all_bsts_artifacts()

    if not bsts_artifacts:
        raise ValueError(
            "No BSTS artifacts were loaded."
        )

    logger.info(
        f"BSTS artifacts loaded: "
        f"{len(bsts_artifacts)}"
    )

    logger.info(
        f"DMAs: "
        f"{list(bsts_artifacts.keys())}"
    )

    # =====================================================
    # Initialize Result Containers
    # =====================================================

    bsts_validation_summaries = []

    counterfactual_validation_summaries = []

    dma_impacts = []

    validation_results = {}

    # =====================================================
    # Validate Each Saved BSTS Model
    # =====================================================

    for target_dma, artifacts in bsts_artifacts.items():

        logger.info("=" * 70)
        logger.info(
            f"Beginning validation for {target_dma}"
        )
        logger.info("=" * 70)

        # =================================================
        # Extract Saved Artifacts
        # =================================================

        idata = artifacts["idata"]

        model_data = artifacts["model_data"]

        model_results = {
            "idata": idata,
            "model_data": model_data,
            "target_dma": target_dma,
        }

        # =================================================
        # Validate Fitted BSTS Model
        # =================================================

        logger.info("-" * 70)
        logger.info(
            f"Validating fitted BSTS model for {target_dma}"
        )
        logger.info("-" * 70)

        (
            validation,
            validation_summary,
        ) = validate_bsts_model(
            model_results=model_results,
            model_data=model_data,
            max_rhat=1.01,
            min_ess_bulk=400,
            min_ess_tail=400,
            max_divergence_rate=0.0,
        )

        model_results["validation"] = validation

        bsts_validation_summaries.append(
            validation_summary
        )

        # -------------------------------------------------
        # Log BSTS Validation Status
        # -------------------------------------------------

        if validation["converged"]:

            logger.info(
                f"{target_dma}: "
                "BSTS model validation PASSED."
            )

        else:

            logger.warning(
                f"{target_dma}: "
                "BSTS model validation FAILED."
            )

            logger.warning(
                f"  Max R-hat       : "
                f"{validation['max_rhat']:.4f}"
            )

            logger.warning(
                f"  Min bulk ESS    : "
                f"{validation['min_ess_bulk']:.1f}"
            )

            logger.warning(
                f"  Min tail ESS    : "
                f"{validation['min_ess_tail']:.1f}"
            )

            logger.warning(
                f"  Divergences     : "
                f"{validation['divergences']}"
            )

        # =================================================
        # Evaluate Posterior
        # =================================================

        logger.info("-" * 70)
        logger.info(
            f"Evaluating posterior for {target_dma}"
        )
        logger.info("-" * 70)

        (
            posterior_summary,
            diagnostics,
            problematic_parameters,
        ) = evaluate_posterior(
            idata=idata,
            dma=target_dma,
            output_name=f"{target_dma}_posterior",
        )

        model_results["posterior_summary"] = (
            posterior_summary
        )

        model_results["posterior_diagnostics"] = (
            diagnostics
        )

        model_results["problematic_parameters"] = (
            problematic_parameters
        )

        # =================================================
        # Generate Posterior Counterfactual
        # =================================================

        logger.info("-" * 70)
        logger.info(
            f"Generating counterfactual for {target_dma}"
        )
        logger.info("-" * 70)

        counterfactual_results = (
            generate_counterfactual(
                idata=idata,
                model_data=model_data,
                include_observation_noise=False,
                random_seed=42,
            )
        )

        model_results.update(
            counterfactual_results
        )

        # =================================================
        # Validate Counterfactual Quality
        # =================================================

        logger.info("-" * 70)
        logger.info(
            f"Validating counterfactual for {target_dma}"
        )
        logger.info("-" * 70)

        (
            counterfactual_validation,
            counterfactual_validation_summary,
        ) = validate_counterfactual(
            model_results=model_results,
            model_data=model_data,
            min_pre_r2=0.80,
            max_pre_rmse_ratio=0.20,
            max_pre_mae_ratio=0.15,
            min_pre_coverage=0.80,
            max_mean_bias_ratio=0.10,
        )

        model_results[
            "counterfactual_validation"
        ] = counterfactual_validation

        counterfactual_validation_summaries.append(
            counterfactual_validation_summary
        )

        # -------------------------------------------------
        # Log Counterfactual Validation Status
        # -------------------------------------------------

        if counterfactual_validation[
            "counterfactual_valid"
        ]:

            logger.info(
                f"{target_dma}: "
                "counterfactual validation PASSED."
            )

        else:

            logger.warning(
                f"{target_dma}: "
                "counterfactual validation FAILED."
            )

            logger.warning(
                f"  Pre R²          : "
                f"{counterfactual_validation['pre_r2']:.4f}"
            )

            logger.warning(
                f"  RMSE ratio      : "
                f"{counterfactual_validation['pre_rmse_ratio']:.2%}"
            )

            logger.warning(
                f"  MAE ratio       : "
                f"{counterfactual_validation['pre_mae_ratio']:.2%}"
            )

            logger.warning(
                f"  95% coverage    : "
                f"{counterfactual_validation['pre_coverage_95']:.2%}"
            )

            logger.warning(
                f"  Bias ratio      : "
                f"{counterfactual_validation['pre_mean_bias_ratio']:.2%}"
            )

        # =================================================
        # Extract DMA Impact
        # =================================================

        logger.info("-" * 70)
        logger.info(
            f"Extracting DMA impact for {target_dma}"
        )
        logger.info("-" * 70)

        (
            dma_summary,
            monthly_impact,
            impact_draws,
            counterfactual_post,
        ) = extract_dma_impact(
            model_results=model_results,
            model_data=model_data,
            panel=panel,
        )

        # -------------------------------------------------
        # Store DMA Impact Results
        # -------------------------------------------------

        model_results["dma_summary"] = (
            dma_summary
        )

        model_results["monthly_impact"] = (
            monthly_impact
        )

        model_results["impact_draws"] = (
            impact_draws
        )

        model_results["counterfactual_post"] = (
            counterfactual_post
        )

        dma_impacts.append(
            dma_summary
        )

        # =================================================
        # Store Complete DMA Results
        # =================================================

        validation_results[target_dma] = (
            model_results
        )

        logger.info(
            f"Completed validation for {target_dma}"
        )

    # =====================================================
    # Combine Validation Results
    # =====================================================

    if bsts_validation_summaries:

        bsts_validation_summary = pd.concat(
            bsts_validation_summaries,
            ignore_index=True,
        )

    else:

        bsts_validation_summary = pd.DataFrame()

    # -----------------------------------------------------

    if counterfactual_validation_summaries:

        counterfactual_validation_summary = pd.concat(
            counterfactual_validation_summaries,
            ignore_index=True,
        )

    else:

        counterfactual_validation_summary = pd.DataFrame()

    # -----------------------------------------------------

    if dma_impacts:

        dma_impact_summary = pd.concat(
            dma_impacts,
            ignore_index=True,
        )

    else:

        dma_impact_summary = pd.DataFrame()

    # =====================================================
    # Validate DMA Results
    # =====================================================

    if not validation_results:

        raise ValueError(
            "No DMA validation results were generated."
        )

    # =====================================================
    # Build DMA Results for Aggregate Analysis
    # =====================================================

    logger.info("=" * 70)
    logger.info(
        "Preparing DMA results for aggregate impact analysis"
    )
    logger.info("=" * 70)

    dma_results = {}

    for target_dma, model_results in validation_results.items():

        required_keys = [
            "dma_summary",
            "monthly_impact",
            "impact_draws",
            "counterfactual_post",
            "validation",
            "counterfactual_validation",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in model_results
        ]

        if missing_keys:

            logger.warning(
                f"{target_dma}: skipping aggregate analysis. "
                f"Missing results: {missing_keys}"
            )

            continue

        dma_results[target_dma] = {

            "dma_summary":
                model_results[
                    "dma_summary"
                ],

            "monthly_impact":
                model_results[
                    "monthly_impact"
                ],

            "impact_draws":
                model_results[
                    "impact_draws"
                ],

            "counterfactual_draws":
                model_results[
                    "counterfactual_post"
                ],

            "model_validation":
                model_results[
                    "validation"
                ],

            "counterfactual_validation":
                model_results[
                    "counterfactual_validation"
                ],
        }

    logger.info(
        f"Prepared {len(dma_results)} DMAs "
        "for aggregate impact analysis."
    )

    if not dma_results:

        raise ValueError(
            "No valid DMA results available for "
            "aggregate impact analysis."
        )

    # =====================================================
    # Aggregate DMA Impact
    # =====================================================

    logger.info("=" * 70)
    logger.info(
        "Aggregating DMA-level causal impact..."
    )
    logger.info("=" * 70)

    (
        aggregate_summary,
        dma_analysis,
        aggregate_draws,
    ) = aggregate_dma_impact(
        dma_results=dma_results,
        model_validation=bsts_validation_summary,
        counterfactual_validation=counterfactual_validation_summary,
        require_valid_models=True,
    )

    # =====================================================
    # DMA Contribution Analysis
    # =====================================================

    logger.info("=" * 70)
    logger.info(
        "Analyzing DMA contribution to aggregate impact..."
    )
    logger.info("=" * 70)

    (
        dma_contribution,
        contribution_summary,
        dma_contribution_draws,
    ) = analyze_dma_contribution(
        dma_results=dma_results,
        aggregate_draws=aggregate_draws,
        require_valid_models=True,
    )


    # =====================================================
    # Output Directory
    # =====================================================

    causal_impact_table_dir = (
        TABLE_DIR /
        "causal_impact"
    )

    causal_impact_table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # Save BSTS Model Validation Summary
    # =====================================================

    bsts_validation_output_path = (
        causal_impact_table_dir /
        "bsts_model_validation_summary.csv"
    )

    bsts_validation_summary.to_csv(
        bsts_validation_output_path,
        index=False,
    )

    logger.info(
        f"Saved BSTS model validation summary to "
        f"{bsts_validation_output_path}"
    )

    # =====================================================
    # Save Counterfactual Validation Summary
    # =====================================================

    counterfactual_validation_output_path = (
        causal_impact_table_dir /
        "counterfactual_validation_summary.csv"
    )

    counterfactual_validation_summary.to_csv(
        counterfactual_validation_output_path,
        index=False,
    )

    logger.info(
        f"Saved counterfactual validation summary to "
        f"{counterfactual_validation_output_path}"
    )

    # =====================================================
    # Save DMA Impact Summary
    # =====================================================

    dma_impact_output_path = (
        causal_impact_table_dir /
        "dma_impact_summary.csv"
    )

    dma_impact_summary.to_csv(
        dma_impact_output_path,
        index=False,
    )

    logger.info(
        f"Saved DMA impact summary to "
        f"{dma_impact_output_path}"
    )

    # =====================================================
    # Save Aggregate Impact Summary
    # =====================================================

    aggregate_summary_path = (
        causal_impact_table_dir /
        "aggregate_impact_summary.csv"
    )

    aggregate_summary.to_csv(
        aggregate_summary_path,
        index=False,
    )

    logger.info(
        f"Saved aggregate impact summary to "
        f"{aggregate_summary_path}"
    )

    # =====================================================
    # Save DMA Impact Analysis
    # =====================================================

    dma_analysis_path = (
        causal_impact_table_dir /
        "dma_impact_analysis.csv"
    )

    dma_analysis.to_csv(
        dma_analysis_path,
        index=False,
    )

    logger.info(
        f"Saved DMA impact analysis to "
        f"{dma_analysis_path}"
    )

    # =====================================================
    # Save DMA Contribution Analysis
    # =====================================================

    dma_contribution_path = (
        causal_impact_table_dir /
        "dma_contribution_analysis.csv"
    )

    dma_contribution.to_csv(
        dma_contribution_path,
        index=False,
    )

    logger.info(
        f"Saved DMA contribution analysis to "
        f"{dma_contribution_path}"
    )


    contribution_summary_path = (
        causal_impact_table_dir /
        "dma_contribution_summary.csv"
    )

    contribution_summary.to_csv(
        contribution_summary_path,
        index=False,
    )

    logger.info(
        f"Saved DMA contribution summary to "
        f"{contribution_summary_path}"
    )


    dma_contribution_draws_path = (
        causal_impact_table_dir /
        "dma_contribution_draws.csv"
    )

    dma_contribution_draws.to_csv(
        dma_contribution_draws_path,
        index=False,
    )

    logger.info(
        f"Saved DMA contribution posterior draws to "
        f"{dma_contribution_draws_path}"
    )


    # =====================================================
    # Final Summary
    # =====================================================

    logger.info("=" * 70)
    logger.info(
        "BSTS Causal Impact Validation Complete"
    )
    logger.info("=" * 70)

    logger.info(
        f"DMAs analyzed: "
        f"{len(validation_results)}"
    )

    logger.info(
        f"DMAs included in aggregate analysis: "
        f"{len(dma_results)}"
    )

    logger.info(
        f"DMA impact results: "
        f"{len(dma_impact_summary)} rows"
    )

    if not dma_impact_summary.empty:

        logger.info(
            f"DMAs with impact estimates: "
            f"{dma_impact_summary['DMA'].nunique()}"
        )

    logger.info(
        f"BSTS validation results: "
        f"{len(bsts_validation_summary)} rows"
    )

    logger.info(
        f"Counterfactual validation results: "
        f"{len(counterfactual_validation_summary)} rows"
    )

    # =====================================================
    # Return
    # =====================================================

    return {
        "bsts_artifacts":
            bsts_artifacts,

        "validation_results":
            validation_results,

        "bsts_validation_summary":
            bsts_validation_summary,

        "counterfactual_validation_summary":
            counterfactual_validation_summary,

        "dma_impact_summary":
            dma_impact_summary,

        "aggregate_summary":
            aggregate_summary,

        "dma_analysis":
            dma_analysis,

        "dma_contribution":
            dma_contribution,

        "contribution_summary":
            contribution_summary,

        "dma_contribution_draws":
            dma_contribution_draws,
    }


if __name__ == "__main__":
    main()



    main()