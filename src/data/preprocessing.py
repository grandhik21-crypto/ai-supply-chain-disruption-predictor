"""
Cleans up raw spreadsheet data.

Checks that required columns exist, fixes dates,
fills in missing numbers, and removes bad rows.
"""

from __future__ import annotations  # Modern type hint support

from dataclasses import dataclass, field  # For schema and strategy classes
from typing import Final  # Marks constants

import pandas as pd  # Table data library

from src.utils.logging_config import get_logger  # Logger for progress messages

logger = get_logger(__name__)  # Create logger for this file


class ColumnValidationError(ValueError):
    """Raised when a dataset is missing required columns."""
    # Custom error type when CSV is missing required column names


@dataclass(frozen=True)
class SupplyChainSchema:
    """Canonical schema for ingested supply chain CSV files."""

    # Column names that MUST exist in every CSV file
    required_columns: tuple[str, ...] = (
        "supplier_id",
        "supplier_name",
        "region",
        "category",
        "risk_score",
        "lead_time_days",
        "inventory_coverage_days",
        "on_time_delivery_pct",
        "sentiment_score",
        "order_date",
        "last_disruption",
    )
    # Columns that should be parsed as dates
    date_columns: tuple[str, ...] = ("order_date", "last_disruption")
    # Columns that should be numbers
    numeric_columns: tuple[str, ...] = (
        "risk_score",
        "lead_time_days",
        "inventory_coverage_days",
        "on_time_delivery_pct",
        "sentiment_score",
    )
    # Text columns (supplier name, region, etc.)
    categorical_columns: tuple[str, ...] = (
        "supplier_id",
        "supplier_name",
        "region",
        "category",
    )


DEFAULT_SCHEMA: Final[SupplyChainSchema] = SupplyChainSchema()  # Default column rules


@dataclass
class MissingValueStrategy:
    """Configuration for imputing missing values during preprocessing."""

    numeric_strategy: str = "median"  # Fill missing numbers with median (middle value)
    categorical_fill_value: str = "Unknown"  # Fill missing text with "Unknown"
    drop_rows_missing_required_dates: bool = True  # Remove rows with no order_date


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase snake_case."""
    cleaned = df.copy()  # Work on a copy so we don't change the original
    # Strip spaces, lowercase, replace spaces with underscores
    cleaned.columns = (
        cleaned.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    logger.debug("Normalized column names: %s", list(cleaned.columns))
    return cleaned


def validate_required_columns(
    df: pd.DataFrame,
    schema: SupplyChainSchema = DEFAULT_SCHEMA,
) -> None:
    """Validate that all required columns are present."""
    # Find any required columns that are missing from the table
    missing = [col for col in schema.required_columns if col not in df.columns]
    if missing:
        msg = f"Missing required columns: {missing}"
        logger.error(msg)
        raise ColumnValidationError(msg)  # Stop if columns are missing
    logger.info("Column validation passed (%d required columns)", len(schema.required_columns))


def convert_date_columns(
    df: pd.DataFrame,
    schema: SupplyChainSchema = DEFAULT_SCHEMA,
) -> pd.DataFrame:
    """Parse configured date columns to timezone-naive datetime64."""
    converted = df.copy()
    for column in schema.date_columns:
        if column not in converted.columns:
            continue  # Skip if this date column doesn't exist
        # Try to turn text dates into real datetime objects; bad values become NaT (null)
        converted[column] = pd.to_datetime(
            converted[column],
            errors="coerce",
            utc=False,
        )
        invalid_count = int(converted[column].isna().sum())
        logger.info(
            "Converted date column '%s' (%d null/invalid values)",
            column,
            invalid_count,
        )
    return converted


def coerce_numeric_columns(
    df: pd.DataFrame,
    schema: SupplyChainSchema = DEFAULT_SCHEMA,
) -> pd.DataFrame:
    """Coerce numeric columns and log coercion issues."""
    coerced = df.copy()
    for column in schema.numeric_columns:
        if column not in coerced.columns:
            continue
        original_nulls = int(coerced[column].isna().sum())
        # Try to convert to numbers; text that can't convert becomes NaN
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
        new_nulls = int(coerced[column].isna().sum())
        if new_nulls > original_nulls:
            logger.warning(
                "Column '%s': %d values could not be parsed as numeric",
                column,
                new_nulls - original_nulls,
            )
    return coerced


def handle_missing_values(
    df: pd.DataFrame,
    schema: SupplyChainSchema = DEFAULT_SCHEMA,
    strategy: MissingValueStrategy | None = None,
) -> pd.DataFrame:
    """Impute or drop missing values according to the configured strategy."""
    strategy = strategy or MissingValueStrategy()
    filled = df.copy()

    missing_before = int(filled.isna().sum().sum())  # Count all empty cells
    logger.info("Handling missing values (%d total null cells before)", missing_before)

    if strategy.drop_rows_missing_required_dates and "order_date" in filled.columns:
        before_rows = len(filled)
        filled = filled.dropna(subset=["order_date"])  # Remove rows with no order date
        dropped = before_rows - len(filled)
        if dropped:
            logger.warning("Dropped %d rows with missing order_date", dropped)

    for column in schema.categorical_columns:
        if column not in filled.columns:
            continue
        null_count = int(filled[column].isna().sum())
        if null_count:
            filled[column] = filled[column].fillna(strategy.categorical_fill_value)
            logger.debug("Filled %d missing values in '%s'", null_count, column)

    for column in schema.numeric_columns:
        if column not in filled.columns:
            continue
        null_count = int(filled[column].isna().sum())
        if null_count:
            if strategy.numeric_strategy == "median":
                fill_value = filled[column].median()  # Use middle value
            elif strategy.numeric_strategy == "mean":
                fill_value = filled[column].mean()  # Use average
            else:
                fill_value = filled[column].median()  # Default to median
            filled[column] = filled[column].fillna(fill_value)
            logger.debug(
                "Imputed %d missing values in '%s' using %s (%.4f)",
                null_count,
                column,
                strategy.numeric_strategy,
                fill_value,
            )

    missing_after = int(filled.isna().sum().sum())
    logger.info(
        "Missing value handling complete (%d null cells remaining)",
        missing_after,
    )
    return filled


def preprocess_supply_chain_dataframe(
    df: pd.DataFrame,
    schema: SupplyChainSchema = DEFAULT_SCHEMA,
    strategy: MissingValueStrategy | None = None,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline on a raw DataFrame."""
    logger.info("Starting preprocessing pipeline (%d rows)", len(df))

    result = normalize_column_names(df)  # Step 1: fix column names
    validate_required_columns(result, schema)  # Step 2: check required columns exist
    result = coerce_numeric_columns(result, schema)  # Step 3: convert to numbers
    result = convert_date_columns(result, schema)  # Step 4: convert to dates
    result = handle_missing_values(result, schema, strategy)  # Step 5: fill gaps

    result = result.reset_index(drop=True)  # Reset row numbers to 0, 1, 2, ...
    logger.info("Preprocessing complete (%d rows, %d columns)", len(result), len(result.columns))
    return result
