"""
Data access layer for the web application.

Exports CSV loaders, preprocessing utilities, and data providers that
supply cleaned DataFrames and metrics to the dashboard pages.
"""

from src.data.csv_provider import CsvDataProvider
from src.data.data_loader import SupplyChainDataLoader
from src.data.placeholder_provider import PlaceholderDataProvider
from src.data.preprocessing import (
    ColumnValidationError,
    MissingValueStrategy,
    SupplyChainSchema,
    preprocess_supply_chain_dataframe,
)

__all__ = [
    "ColumnValidationError",
    "CsvDataProvider",
    "MissingValueStrategy",
    "PlaceholderDataProvider",
    "SupplyChainDataLoader",
    "SupplyChainSchema",
    "preprocess_supply_chain_dataframe",
]
