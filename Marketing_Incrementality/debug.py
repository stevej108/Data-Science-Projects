import os
import pandas as pd
import pathlib

# CONFIGURATION
# =============================================================================

from config import (
    DATA_DIR,
    SPEND_FILE,
    TRAFFIC_FILE,
    STORE_FILE,
    LOG_LEVEL,
    OUTPUT_DIR
)


#read processed model_panel file

comp_path = COMP_FILE = DATA_DIR /"processed"/ "model_panel.parquet"

model=pd.read_parquet(comp_path)

print(model.head(60))

print(model.columns.to_list())