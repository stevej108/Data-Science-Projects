import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pptx import Presentation
from typing import TypedDict
import json
import re
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langgraph.graph import END

import warnings
warnings.filterwarnings('ignore')




# ==========================================================
# Configuration
# ==========================================================

from pathlib import Path

# Root directory of synced OneDrive
ONEDRIVE_ROOT = Path(
    r"C:\Users\jsteve\OneDrive - Burlington"
)

# File types to search (update as needed with different extenstions)
SUPPORTED_DOCUMENTS = {

    ".pptx": "powerpoint",
    ".ppt": "powerpoint",

    ".pdf": "pdf",

    ".docx": "word",
    ".doc": "word",

    ".xlsx": "excel",
    ".xls": "excel"
}

# Number of retrieved slides to send to the LLM
TOP_K = 25

EXCEL_ROWS_PER_CHUNK = 200

print(f"OneDrive Root: {ONEDRIVE_ROOT}")