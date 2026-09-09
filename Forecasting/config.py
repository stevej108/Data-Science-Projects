from pathlib import Path
import pandas as pd

PROJECT_NAME = "Forecasting"

VERSION = "1.0"

LOG_LEVEL = "INFO"



PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "data"

INTERIM_DATA_DIR = DATA_DIR / "interim"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR = OUTPUT_DIR / "tables"

MODEL_DIR = OUTPUT_DIR / "models"

REPORT_DIR = OUTPUT_DIR / "reports"

STYLE = "tableau-colorblind10"

#-----------------------------
# File Naming
#-----------------------------
SPEND_FILE = DATA_DIR / "spend.csv"

TRAFFIC_FILE = DATA_DIR / "traffic.csv"

STORE_FILE = DATA_DIR / "stores.csv"

CX_FILE = DATA_DIR / "cx.csv"

WEATHER_FILE = DATA_DIR / "weather_master.csv"

#-----------------------------
# Column Names
#-----------------------------
COL_DMA = "DMA"

COL_DATE = "Month"

COL_SPEND = "Spend"

COL_TRAFFIC = "Traffic"

COL_STORE_COUNT = "Store_Count"

COL_YEAR = "Year"

COL_MONTH = "Month_Num"

COL_QUARTER = "Quarter"

COL_LAG = "Lag"

COL_ROLLING = "Rolling"

COL_TREATMENT = "Treatment"

COL_POST = "Post"

COL_EVENT_TIME = "Event_Time"

#-----------------------------
# Model Parameters
#-----------------------------
# Random Seed
RANDOM_SEED = 42

# Lag periods (months)
MAX_LAG = 6
LAG_PERIODS = [1, 2, 3, 6]

# Rolling windows
ROLLING_WINDOWS = [3, 6, 12]

# Seasonality
SEASONAL_PERIOD = 12

# Train/Test Split
TRAIN_END = pd.Timestamp("2026-01-01")

# Significance
ALPHA = 0.05

# Date Frequency
DATE_FREQ = "MS"

#-----------------------------
# Validation
#-----------------------------

MAX_MISSING_PCT = 0.10

ALLOW_NEGATIVE_SPEND = False

ALLOW_DUPLICATES = False

#-----------------------------
# Plotting Parameters
#-----------------------------
FIG_SIZE = (12, 6)

DPI = 300

#-----------------------------
# Output File Naming
#-----------------------------
CLEAN_SPEND_FILE = INTERIM_DATA_DIR / "spend_clean.parquet"

CLEAN_TRAFFIC_FILE = INTERIM_DATA_DIR / "traffic_clean.parquet"

PANEL_FILE = PROCESSED_DATA_DIR / "panel_data.parquet"



