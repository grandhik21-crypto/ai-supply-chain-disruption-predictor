"""
Reads a CSV file and returns clean data.

Opens the supply chain spreadsheet, checks it looks right,
fixes missing values, and hands back ready-to-use data.
"""

from __future__ import annotations  # Modern type hint support

from dataclasses import dataclass, field  # For the loader settings class
from pathlib import Path  # For working with file paths

import pandas as pd  # Reads CSV files into tables

from config.settings import SUPPLY_CHAIN_CSV  # Configurable input path
# Import cleaning tools and settings from preprocessing.py
from src.data.preprocessing import (
    DEFAULT_SCHEMA,
    MissingValueStrategy,
    SupplyChainSchema,
    preprocess_supply_chain_dataframe,
)
# Import logger so we can print progress messages
from src.utils.logging_config import get_logger

logger = get_logger(__name__)  # Create a logger for this file

# Default location of the supplier CSV (override with SC_SUPPLY_CHAIN_CSV)
DEFAULT_DATA_PATH = SUPPLY_CHAIN_CSV


@dataclass
class SupplyChainDataLoader:
    """
    Loads and cleans supply chain CSV data.

    Example:
        loader = SupplyChainDataLoader()
        df = loader.load()
    """

    file_path: Path = field(default_factory=lambda: DEFAULT_DATA_PATH)  # Which CSV to read
    encoding: str = "utf-8"  # Text encoding of the CSV file
    schema: SupplyChainSchema = field(default_factory=lambda: DEFAULT_SCHEMA)  # Required columns
    missing_value_strategy: MissingValueStrategy = field(
        default_factory=MissingValueStrategy
    )  # How to fill in missing data

    def load(self, file_path: Path | str | None = None) -> pd.DataFrame:
        """
        Load a CSV file, preprocess it, and return a cleaned DataFrame.

        Args:
            file_path: Optional override path to the CSV file.

        Returns:
            Cleaned pandas DataFrame ready for analytics.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ColumnValidationError: If required columns are missing.
        """
        # Use the path passed in, or fall back to the default path
        path = Path(file_path) if file_path is not None else self.file_path
        logger.info("Loading supply chain data from '%s'", path)

        if not path.exists():
            # Stop and raise an error if the file is missing
            msg = f"CSV file not found: {path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        # Read the raw CSV into a pandas DataFrame (table)
        raw_df = pd.read_csv(path, encoding=self.encoding)
        logger.info("Loaded %d rows and %d columns from CSV", len(raw_df), len(raw_df.columns))

        # Run the full cleaning pipeline on the raw data
        cleaned_df = preprocess_supply_chain_dataframe(
            raw_df,
            schema=self.schema,
            strategy=self.missing_value_strategy,
        )
        logger.info("Data ingestion complete: %d cleaned rows", len(cleaned_df))
        return cleaned_df  # Return the cleaned table

    def load_raw(self, file_path: Path | str | None = None) -> pd.DataFrame:
        """Load CSV without preprocessing (useful for debugging)."""
        path = Path(file_path) if file_path is not None else self.file_path
        logger.info("Loading raw CSV from '%s' (no preprocessing)", path)
        return pd.read_csv(path, encoding=self.encoding)  # Return uncleaned data
