"""
All the data-related code.

Loads CSV files, cleans the data, and provides it
to the website pages.
"""

# Export CSV-backed data provider
from src.data.csv_provider import CsvDataProvider
# Export the CSV loader class
from src.data.data_loader import SupplyChainDataLoader
# Export demo/fake data provider
from src.data.placeholder_provider import PlaceholderDataProvider
# Export cleaning tools and types
from src.data.preprocessing import (
    ColumnValidationError,
    MissingValueStrategy,
    SupplyChainSchema,
    preprocess_supply_chain_dataframe,
)

# Names other files can import from this package
__all__ = [
    "ColumnValidationError",
    "CsvDataProvider",
    "MissingValueStrategy",
    "PlaceholderDataProvider",
    "SupplyChainDataLoader",
    "SupplyChainSchema",
    "preprocess_supply_chain_dataframe",
]
