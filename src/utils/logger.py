"""
logger.py
---------
Centralized logging setup for the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_loggers: dict = {}

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Get (or create) a named logger.

    Parameters
    ----------
    name : str
        Logger name (typically __name__).
    level : int
        Logging level (default: INFO).
    log_file : str, optional
        If provided, also write logs to this file.

    Returns
    -------
    logging.Logger
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
            logger.addHandler(file_handler)

    logger.propagate = False
    _loggers[name] = logger
    return logger


def setup_root_logger(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure the root logger. Call once at application startup.

    Parameters
    ----------
    level : int
    log_file : str, optional
    """
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logging.getLogger().addHandler(file_handler)
