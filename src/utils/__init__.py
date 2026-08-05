"""
Shared utilities package for the web application.

Exports logging helpers used by the data ingestion pipeline and backend services.
"""

from src.utils.logging_config import get_logger, setup_logging

__all__ = ["get_logger", "setup_logging"]
