"""Utility functions for the causal pricing project.

Provides common utilities including logging configuration and JSON persistence.
"""

import json
import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Creates or retrieves a configured logger instance.

    Args:
        name: Name for the logger, typically the module or experiment name.

    Returns:
        Configured Logger instance with INFO level and timestamp formatting.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_json(data: dict, filepath: str) -> None:
    """Saves a dictionary to a JSON file, creating parent directories as needed.

    Args:
        data: Dictionary to serialise as JSON.
        filepath: Absolute or relative path for the output JSON file.
    """

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
