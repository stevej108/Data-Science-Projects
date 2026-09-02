# core/logger.py

import logging
import sys
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parents[1] / "OneDrive_Search.log"


def get_logger(name: str) -> logging.Logger:

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        "%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    logfile = logging.FileHandler(LOG_FILE, encoding="utf-8")
    logfile.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(logfile)

    logger.propagate = False

    return logger