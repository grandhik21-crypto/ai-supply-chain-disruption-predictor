"""
Sets up text messages that print while the app runs.

When data is loading or cleaning, this controls
what info and warnings show up in the terminal.
"""

from __future__ import annotations  # Modern type hint support

import logging  # Python's built-in logging library
import sys  # Used to send log messages to the terminal (stdout)
from typing import Final  # Marks constants that never change

DEFAULT_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)  # How each log line looks: time | level | module | message
DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"  # Date format in log timestamps

_configured = False  # Tracks whether logging has already been set up


def setup_logging(level: int | None = None) -> None:
    """
    Configure root logging once for the application.

    The level comes from the SC_LOG_LEVEL environment variable unless one
    is passed in directly.
    """
    global _configured
    if _configured:
        return

    if level is None:
        # Read the configured level name (INFO, DEBUG, ...) and convert it
        from config.settings import LOG_LEVEL

        level = getattr(logging, LOG_LEVEL, logging.INFO)  # Don't set up logging twice

    # Send log messages to the terminal screen
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    )

    root = logging.getLogger()  # Get the root (top-level) logger
    root.setLevel(level)  # Show messages at this level and above (INFO, WARNING, ERROR)
    if not root.handlers:
        root.addHandler(handler)  # Attach our terminal handler

    _configured = True  # Mark logging as ready


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring logging is configured."""
    setup_logging()  # Make sure logging is set up first
    return logging.getLogger(name)  # Return a logger for this module name
