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
# Initialize LLM for AI Agentic Nodes
# ==========================================================

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.2,
)