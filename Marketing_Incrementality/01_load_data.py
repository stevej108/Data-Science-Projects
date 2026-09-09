"""
01_load_data.py

Purpose:
    Load all raw source data into pandas DataFrames.

Outputs:
    spend_df
    traffic_df
    stores_df
"""

import logging
from pathlib import Path

import pandas as pd

from config import (
    SPEND_FILE,
    TRAFFIC_FILE,
    STORE_FILE,
    LOG_LEVEL,
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
# Helper Functions
# ---------------------------------------------------------------------

def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Parameters
    ----------
    file_path : Path
        Location of csv file.

    Returns
    -------
    pd.DataFrame
    """

    logger.info(f"Loading {file_path.name}")

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}")

    df = pd.read_csv(
        file_path,
        engine='python')

    logger.info(
        f"Loaded {file_path.name}: "
        f"{df.shape[0]:,} rows x {df.shape[1]} columns"
    )

    return df


def summarize_dataframe(name: str, df: pd.DataFrame):
    """
    Log basic dataframe information.
    """

    logger.info("-" * 60)
    logger.info(f"{name}")
    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Columns: {len(df.columns)}")
    logger.info(f"Missing Values: {df.isna().sum().sum():,}")
    logger.info("-" * 60)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    spend_df = load_csv(SPEND_FILE)
    traffic_df = load_csv(TRAFFIC_FILE)
    stores_df = load_csv(STORE_FILE)

    summarize_dataframe("Marketing Spend", spend_df)
    summarize_dataframe("Traffic", traffic_df)
    summarize_dataframe("Stores", stores_df)

    logger.info("Finished loading all datasets.")

    return spend_df, traffic_df, stores_df


if __name__ == "__main__":

    spend_df, traffic_df, stores_df = main()