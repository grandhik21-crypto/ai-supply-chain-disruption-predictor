"""CSV data ingestion for supply chain datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.preprocessing import (
    DEFAULT_SCHEMA,
    MissingValueStrategy,
    SupplyChainSchema,
    preprocess_supply_chain_dataframe,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "supply_chain.csv"


@dataclass
class SupplyChainDataLoader:
    """
    Loads and cleans supply chain CSV data.

    Example:
        loader = SupplyChainDataLoader()
        df = loader.load()
    """

    file_path: Path = field(default_factory=lambda: DEFAULT_DATA_PATH)
    encoding: str = "utf-8"
    schema: SupplyChainSchema = field(default_factory=lambda: DEFAULT_SCHEMA)
    missing_value_strategy: MissingValueStrategy = field(
        default_factory=MissingValueStrategy
    )

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
        path = Path(file_path) if file_path is not None else self.file_path
        logger.info("Loading supply chain data from '%s'", path)

        if not path.exists():
            msg = f"CSV file not found: {path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        raw_df = pd.read_csv(path, encoding=self.encoding)
        logger.info("Loaded %d rows and %d columns from CSV", len(raw_df), len(raw_df.columns))

        cleaned_df = preprocess_supply_chain_dataframe(
            raw_df,
            schema=self.schema,
            strategy=self.missing_value_strategy,
        )
        logger.info("Data ingestion complete: %d cleaned rows", len(cleaned_df))
        return cleaned_df

    def load_raw(self, file_path: Path | str | None = None) -> pd.DataFrame:
        """Load CSV without preprocessing (useful for debugging)."""
        path = Path(file_path) if file_path is not None else self.file_path
        logger.info("Loading raw CSV from '%s' (no preprocessing)", path)
        return pd.read_csv(path, encoding=self.encoding)
