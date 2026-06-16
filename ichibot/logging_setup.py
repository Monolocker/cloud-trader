"""Logging setup: writes to the console and to a rotating file in logs/."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "ichibot.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return the 'ichibot' logger. Safe to call more than once."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ichibot")
    logger.setLevel(level)
    logger.propagate = False

    # Avoid adding duplicate handlers if called twice.
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        Path(log_dir) / log_file, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger