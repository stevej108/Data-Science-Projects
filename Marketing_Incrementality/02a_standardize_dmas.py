"""
02a_standardize_dmas.py

Standardize DMA names across the marketing datasets.

Workflow
--------
1. Load cleaned datasets
2. Build a suggested DMA crosswalk (first run only)
3. Stop so the analyst can review the suggestions
4. Apply the approved crosswalk
5. Validate DMA alignment
6. Save standardized datasets

Stores.csv is treated as the source of truth for DMA names.
"""

import logging
from difflib import get_close_matches
from pathlib import Path

import pandas as pd

from config import (
    INTERIM_DATA_DIR,
)

# -------------------------------------------------------
# Logging
# -------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# File locations
# -------------------------------------------------------

SPEND_FILE = INTERIM_DATA_DIR / "spend_clean.parquet"
TRAFFIC_FILE = INTERIM_DATA_DIR / "traffic_dma.parquet"
STORE_FILE = INTERIM_DATA_DIR / "stores_clean.parquet"

CROSSWALK_FILE = INTERIM_DATA_DIR / "dma_crosswalk.csv"

# -------------------------------------------------------
# Load data
# -------------------------------------------------------

def load_data():

    logger.info("Loading cleaned datasets...")

    spend = pd.read_parquet(SPEND_FILE)
    traffic = pd.read_parquet(TRAFFIC_FILE)
    stores = pd.read_parquet(STORE_FILE)

    return spend, traffic, stores


# -------------------------------------------------------
# Generate suggested matches
# -------------------------------------------------------

import re

def extract_primary_city(dma):
    """
    Extract the primary city name from a DMA.

    This is used ONLY for matching DMAs during
    crosswalk creation.
    """

    if pd.isna(dma):
        return ""

    dma = str(dma).strip().lower()

    # Remove anything in parentheses
    dma = re.sub(r"\(.*?\)", "", dma)

    # Remove everything after first comma
    dma = dma.split(",")[0]

    # Remove secondary metro names
    dma = dma.split("-")[0]

    # Collapse whitespace
    dma = re.sub(r"\s+", " ", dma).strip()

    return dma

INVALID_DMA_PATTERNS = [
    "TOTAL",
    "NOTE",
    "FORECAST",
    "INCLUDES",
    "EXCLUDES"
]


def is_valid_dma(dma):

    if pd.isna(dma):
        return False

    dma = str(dma).upper()

    return not any(
        pattern in dma
        for pattern in INVALID_DMA_PATTERNS
    )

from difflib import get_close_matches
def create_crosswalk(spend, stores):

    logger.info("Creating DMA crosswalk...")

    # -------------------------------------------------------
    # Unique Spend DMAs
    # -------------------------------------------------------

    spend_dmas = (
        spend.loc[
            spend["DMA"].apply(is_valid_dma)
        ]
        ["DMA"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .sort_values()
    )

    # -------------------------------------------------------
    # Build Store Lookup
    # -------------------------------------------------------

    store_lookup = (
        stores[["DMA"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .rename(columns={"DMA": "Store_DMA"})
        .copy()
    )

    store_lookup["Store_DMA"] = (
        store_lookup["Store_DMA"]
        .str.strip()
    )

    store_lookup["PrimaryCity"] = (
        store_lookup["Store_DMA"]
        .apply(extract_primary_city)
    )

    # -------------------------------------------------------
    # Debugging
    # -------------------------------------------------------

    logger.info("=" * 60)
    logger.info("STORE LOOKUP SAMPLE")
    print(store_lookup.head(20))

    logger.info("=" * 60)
    logger.info("STORE LOOKUP MATCH COUNTS")

    match_counts = (
        store_lookup["PrimaryCity"]
        .value_counts()
        .sort_index()
    )

    print(match_counts.head(30))

    duplicate_cities = match_counts[match_counts > 1]

    if not duplicate_cities.empty:

        logger.info("Cities with multiple DMAs:")

        for city in duplicate_cities.index:

            logger.info(f"\n{city}")

            print(
                store_lookup.loc[
                    store_lookup["PrimaryCity"] == city,
                    "Store_DMA"
                ].tolist()
            )

    # -------------------------------------------------------
    # Build Crosswalk
    # -------------------------------------------------------

    rows = []

    for spend_dma in spend_dmas:

        spend_city = extract_primary_city(spend_dma)

        # -------------------------------------------------------
        # Exact Match
        # -------------------------------------------------------

        matches = (
            store_lookup.loc[
                store_lookup["PrimaryCity"] == spend_city,
                "Store_DMA"
            ]
            .tolist()
        )

        # -------------------------------------------------------
        # Fuzzy Fallback
        # -------------------------------------------------------

        match_method = "EXACT"

        if len(matches) == 0:

            fuzzy = get_close_matches(
                spend_city,
                store_lookup["PrimaryCity"].unique(),
                n=1,
                cutoff=0.80
            )

            if fuzzy:

                logger.info(
                    f"Fuzzy matched '{spend_city}' -> '{fuzzy[0]}'"
                )

                matches = (
                    store_lookup.loc[
                        store_lookup["PrimaryCity"] == fuzzy[0],
                        "Store_DMA"
                    ]
                    .tolist()
                )

                match_method = "FUZZY"

        # Debug every lookup
        logger.info(
            f"{spend_dma:25s} ({match_method}) --> {matches}"
                )


        if len(matches) == 1:

            suggested = matches[0]
            approved = matches[0]
            status = (
                "AUTO"
                if match_method == "EXACT"
                else "AUTO_FUZZY"
            )

        elif len(matches) > 1:

            suggested = " | ".join(matches)
            approved = ""
            status = "REVIEW"

        else:

            suggested = ""
            approved = ""
            status = "MISSING"

        rows.append(
            {
                "Spend_DMA": spend_dma,
                "Suggested_DMA": suggested,
                "Approved_DMA": approved,
                "Status": status
            }
        )

    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------

    crosswalk = (
        pd.DataFrame(rows)
        .sort_values("Spend_DMA")
    )

    logger.info("=" * 60)
    logger.info("GENERATED CROSSWALK")
    print(crosswalk.head(20))

    crosswalk.to_csv(
        CROSSWALK_FILE,
        index=False
    )

    logger.info(f"Created {CROSSWALK_FILE}")
    logger.info(f"Auto matched : {(crosswalk.Status == 'AUTO').sum()}")
    logger.info(f"Needs review : {(crosswalk.Status == 'REVIEW').sum()}")
    logger.info(f"No match     : {(crosswalk.Status == 'MISSING').sum()}")

# -------------------------------------------------------
# Apply crosswalk
# -------------------------------------------------------

def apply_crosswalk(
    spend,
    traffic,
    stores
):

    logger.info("Applying DMA crosswalk...")


    crosswalk = pd.read_csv(
        CROSSWALK_FILE,
        dtype=str,
        keep_default_na=False
    )


    for col in crosswalk.columns:
        crosswalk[col] = (
            crosswalk[col]
            .astype(str)
            .str.strip()
        )


    approved = crosswalk[
        crosswalk["Approved_DMA"] != ""
    ].copy()


    mapping = dict(
        zip(
            approved["Spend_DMA"],
            approved["Approved_DMA"]
        )
    )


    # Keep originals for audit
    spend["DMA_Original"] = spend["DMA"]
    traffic["DMA_Original"] = traffic["DMA"]
    stores["DMA_Original"] = stores["DMA"]


    # ONLY standardize spend
    spend["DMA"] = (
        spend["DMA"]
        .replace(mapping)
    )


    return (
        spend,
        traffic,
        stores
    )


# -------------------------------------------------------
# Validation
# -------------------------------------------------------

def validate(
    spend,
    traffic,
    stores
):

    logger.info("=" * 60)

    spend_set = set(
        spend["DMA"]
        .dropna()
        .astype(str)
    )

    traffic_set = set(
        traffic["DMA"]
        .dropna()
        .astype(str)
    )

    store_set = set(
        stores["DMA"]
        .dropna()
        .astype(str)
    )

    logger.info(f"Spend Markets : {len(spend_set)}")
    logger.info(f"Traffic DMAs : {len(traffic_set)}")
    logger.info(f"Store DMAs   : {len(store_set)}")

    logger.info("")

    logger.info("Spend not found in Traffic")

    missing = sorted(
        spend_set - traffic_set
    )

    for dma in missing:

        logger.info(f"   {dma}")

    logger.info("")

    logger.info("Spend not found in Stores")

    missing = sorted(
        spend_set - store_set
    )

    for dma in missing:

        logger.info(f"   {dma}")

    logger.info("=" * 60)

def validate_unique_panel(df, name):

    dupes = (
        df.groupby(["DMA","Month"])
        .size()
        .reset_index(name="count")
    )

    dupes = dupes[
        dupes["count"] > 1
    ]

    if len(dupes):

        logger.error(
            f"{name} contains duplicate DMA-Month combinations"
        )

        logger.error(dupes.head())

        raise ValueError(
            f"{name} failed DMA-Month uniqueness"
        )


def report_spend_dma_traffic_coverage(spend, traffic):

    spend_dmas = sorted(
        spend["DMA"]
        .dropna()
        .unique()
    )

    coverage = (
        traffic[
            traffic["DMA"].isin(spend_dmas)
        ]
        .groupby("Month")["DMA"]
        .nunique()
    )

    expected = len(spend_dmas)

    report = pd.DataFrame({
        "Traffic_Observed": coverage,
        "Traffic_Missing": expected - coverage,
        "Expected": expected,
        "Coverage": coverage / expected
    })

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAFFIC COVERAGE FOR SPEND DMAs")
    logger.info("=" * 70)
    logger.info(report.to_string())

    return report    
    
# -------------------------------------------------------
# Save
# -------------------------------------------------------

def save_data(
    spend,
    traffic,
    stores
):

    logger.info("Saving standardized datasets...")

    logger.info(f"Spend output: {SPEND_FILE}")
    logger.info(f"Traffic output: {TRAFFIC_FILE}")
    logger.info(f"Stores output: {STORE_FILE}")

    logger.info(f"Spend exists before save: {SPEND_FILE.exists()}")
    logger.info(f"Traffic exists before save: {TRAFFIC_FILE.exists()}")
    logger.info(f"Stores exists before save: {STORE_FILE.exists()}")

    spend.to_parquet(
        SPEND_FILE,
        index=False
    )

    traffic.to_parquet(
        TRAFFIC_FILE,
        index=False
    )

    stores.to_parquet(
        STORE_FILE,
        index=False
    )


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    spend, traffic, stores = load_data()

    if not CROSSWALK_FILE.exists():

        create_crosswalk(
            spend,
            stores
        )

        logger.info("")
        logger.info("=" * 70)
        logger.info("FIRST RUN COMPLETE")
        logger.info("Review dma_crosswalk.csv")
        logger.info("Fill Approved_DMA")
        logger.info("Run this script again.")
        logger.info("=" * 70)

        return

    spend, traffic, stores = apply_crosswalk(
        spend,
        traffic,
        stores
    )

    validate(
        spend,
        traffic,
        stores
    )


    validate_unique_panel(
        traffic,
        "Traffic"
    )

    report_spend_dma_traffic_coverage(
        spend,
        traffic
    )

    save_data(
        spend,
        traffic,
        stores
    )

    

    logger.info("DMA standardization complete.")


if __name__ == "__main__":
    main()