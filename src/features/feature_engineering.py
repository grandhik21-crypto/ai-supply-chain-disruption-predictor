"""
Feature engineering pipeline for machine learning.

Merges supply chain data with daily supplier sentiment, then engineers
rolling features using vectorized pandas operations.

Steps:
  1. Load cleaned supply chain CSV
  2. Load daily sentiment CSV
  3. Build a daily panel per supplier (join key: supplier_id + date)
  4. Engineer rolling features (lead time, sentiment, negative news, etc.)
  5. Return a clean ML-ready dataset and optionally save to CSV
"""

from __future__ import annotations  # Modern type hint support

from dataclasses import dataclass, field  # For pipeline settings class
from pathlib import Path  # For file paths

import numpy as np  # Vectorized math
import pandas as pd  # Tables and vectorized group/rolling operations

from config.settings import (  # Configurable paths and window size
    DAILY_SENTIMENT_CSV,
    ML_FEATURES_CSV,
    ROLLING_WINDOW_DAYS,
)
from src.data.data_loader import DEFAULT_DATA_PATH, SupplyChainDataLoader
from src.utils.logging_config import get_logger  # Logging helper

logger = get_logger(__name__)

# Default output path for the ML feature dataset (override with SC_PROCESSED_DIR)
DEFAULT_ML_OUTPUT_PATH = ML_FEATURES_CSV

# Where daily sentiment is read from
DEFAULT_DAILY_OUTPUT_PATH = DAILY_SENTIMENT_CSV

# Rolling window size in days (override with SC_ROLLING_WINDOW)
ROLLING_WINDOW = ROLLING_WINDOW_DAYS

# Columns kept in the final ML dataset
FEATURE_COLUMNS: tuple[str, ...] = (
    "supplier_id",
    "date",
    "region",
    "category",
    "risk_score",
    "rolling_lead_time_7d",
    "lead_time_variance_7d",
    "inventory_coverage",
    "supplier_reliability",
    "rolling_sentiment_7d",
    "sentiment_velocity",
    "negative_news_count_7d",
    "sentiment_score_daily",
    "article_count",
)


@dataclass
class FeatureEngineeringPipeline:
    """
    Merges supply chain + sentiment data and engineers ML features.

    Example:
        pipeline = FeatureEngineeringPipeline()
        ml_df = pipeline.run_pipeline()
    """

    supply_chain_path: Path = field(default_factory=lambda: DEFAULT_DATA_PATH)
    sentiment_path: Path = field(default_factory=lambda: DEFAULT_DAILY_OUTPUT_PATH)
    output_path: Path = field(default_factory=lambda: DEFAULT_ML_OUTPUT_PATH)
    rolling_window: int = ROLLING_WINDOW
    _loader: SupplyChainDataLoader = field(
        default_factory=SupplyChainDataLoader, init=False, repr=False
    )

    def load_supply_chain_data(self) -> pd.DataFrame:
        """Step 1 — Load and clean supply chain data."""
        logger.info("Loading supply chain data from '%s'", self.supply_chain_path)
        loader = SupplyChainDataLoader(file_path=self.supply_chain_path)
        df = loader.load()
        # Normalize order_date to a plain date column for joining
        df = df.copy()
        df["date"] = pd.to_datetime(df["order_date"]).dt.normalize()
        logger.info("Loaded %d supply chain rows", len(df))
        return df

    def load_sentiment_data(self) -> pd.DataFrame:
        """Step 2 — Load daily aggregated supplier sentiment."""
        if not self.sentiment_path.exists():
            logger.warning(
                "Sentiment file not found at '%s'. Run scripts/run_sentiment.py first.",
                self.sentiment_path,
            )
            return pd.DataFrame(
                columns=[
                    "supplier_id",
                    "date",
                    "sentiment_label",
                    "avg_confidence",
                    "sentiment_score",
                    "article_count",
                ]
            )

        logger.info("Loading daily sentiment from '%s'", self.sentiment_path)
        df = pd.read_csv(self.sentiment_path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        # Rename for clarity in merged dataset
        df = df.rename(
            columns={
                "sentiment_score": "sentiment_score_daily",
            }
        )
        logger.info("Loaded %d daily sentiment rows", len(df))
        return df

    @staticmethod
    def _build_daily_panel(
        supply_df: pd.DataFrame, sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Step 3a — Create one row per supplier per day.

        Builds a daily calendar for each supplier from the earliest to latest
        date seen in either supply chain or sentiment data.
        """
        supply_bounds = (
            supply_df.groupby("supplier_id")["date"]
            .agg(["min", "max"])
            .rename(columns={"min": "start_date", "max": "end_date"})
            .reset_index()
        )

        if sentiment_df.empty:
            bounds = supply_bounds
        else:
            sentiment_bounds = (
                sentiment_df.groupby("supplier_id")["date"]
                .agg(["min", "max"])
                .rename(columns={"min": "sent_start", "max": "sent_end"})
                .reset_index()
            )
            bounds = supply_bounds.merge(
                sentiment_bounds, on="supplier_id", how="outer"
            )
            bounds["start_date"] = bounds[["start_date", "sent_start"]].min(axis=1)
            bounds["end_date"] = bounds[["end_date", "sent_end"]].max(axis=1)
            bounds = bounds.drop(columns=["sent_start", "sent_end"])

        # One date range per supplier, then explode to daily rows
        bounds["date"] = bounds.apply(
            lambda row: pd.date_range(row["start_date"], row["end_date"], freq="D"),
            axis=1,
        )
        panel = bounds.explode("date", ignore_index=True)[["supplier_id", "date"]]
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        logger.info("Built daily panel with %d supplier-day rows", len(panel))
        return panel

    def merge_datasets(
        self,
        supply_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Step 3 — Merge supply chain and sentiment on supplier_id + date.

        Supply chain metrics are forward/backward-filled across days per supplier.
        Sentiment is joined on exact date match.
        """
        panel = self._build_daily_panel(supply_df, sentiment_df)

        # Columns from supply chain used for features and context
        supply_cols = [
            "supplier_id",
            "date",
            "supplier_name",
            "region",
            "category",
            "risk_score",
            "lead_time_days",
            "inventory_coverage_days",
            "on_time_delivery_pct",
        ]
        supply_ts = (
            supply_df[supply_cols]
            .drop_duplicates(subset=["supplier_id", "date"])
            .sort_values(["supplier_id", "date"])
        )

        panel = panel.sort_values(["supplier_id", "date"]).reset_index(drop=True)

        # Exact match first, then forward/backward fill supply metrics per supplier
        merged = panel.merge(
            supply_ts,
            on=["supplier_id", "date"],
            how="left",
        )

        fill_cols = [
            "supplier_name",
            "region",
            "category",
            "risk_score",
            "lead_time_days",
            "inventory_coverage_days",
            "on_time_delivery_pct",
        ]
        merged[fill_cols] = (
            merged.groupby("supplier_id", sort=False)[fill_cols]
            .ffill()
            .bfill()
        )

        # Exact join for sentiment on supplier + date
        if not sentiment_df.empty:
            sentiment_cols = [
                "supplier_id",
                "date",
                "sentiment_label",
                "avg_confidence",
                "sentiment_score_daily",
                "article_count",
            ]
            merged = merged.merge(
                sentiment_df[sentiment_cols],
                on=["supplier_id", "date"],
                how="left",
            )
        else:
            merged["sentiment_label"] = np.nan
            merged["avg_confidence"] = np.nan
            merged["sentiment_score_daily"] = np.nan
            merged["article_count"] = 0.0

        # Flag days with negative news (used for rolling negative count)
        merged["negative_article_count"] = np.where(
            merged["sentiment_label"].astype(str).str.lower().eq("negative"),
            merged["article_count"].fillna(0),
            0.0,
        )

        logger.info("Merged dataset: %d rows, %d columns", len(merged), len(merged.columns))
        return merged.sort_values(["supplier_id", "date"]).reset_index(drop=True)

    def engineer_features(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4 — Engineer rolling features with vectorized pandas operations.

        All rolling stats use groupby('supplier_id') + rolling(window=7).
        """
        df = merged_df.copy()
        window = self.rolling_window

        logger.info("Engineering features with %d-day rolling window...", window)

        # 7-day rolling mean of lead time (vectorized groupby + rolling)
        df["rolling_lead_time_7d"] = (
            df.groupby("supplier_id", sort=False)["lead_time_days"]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # 7-day rolling variance of lead time
        df["lead_time_variance_7d"] = (
            df.groupby("supplier_id", sort=False)["lead_time_days"]
            .rolling(window=window, min_periods=1)
            .var()
            .reset_index(level=0, drop=True)
        )

        # Inventory coverage (from supply chain snapshot, carried on daily panel)
        df["inventory_coverage"] = df["inventory_coverage_days"]

        # Supplier reliability: on-time delivery % scaled to 0–1
        df["supplier_reliability"] = df["on_time_delivery_pct"] / 100.0

        # 7-day rolling mean sentiment
        df["rolling_sentiment_7d"] = (
            df.groupby("supplier_id", sort=False)["sentiment_score_daily"]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Sentiment velocity: day-over-day change in daily sentiment score
        df["sentiment_velocity"] = df.groupby("supplier_id", sort=False)[
            "sentiment_score_daily"
        ].diff()

        # 7-day rolling sum of negative news article counts
        df["negative_news_count_7d"] = (
            df.groupby("supplier_id", sort=False)["negative_article_count"]
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        # Single-day windows have 0 variance — fill NaN for a clean ML dataset
        df["lead_time_variance_7d"] = df["lead_time_variance_7d"].fillna(0.0)

        # Round numeric features for a clean ML table
        numeric_cols = [
            "rolling_lead_time_7d",
            "lead_time_variance_7d",
            "inventory_coverage",
            "supplier_reliability",
            "rolling_sentiment_7d",
            "sentiment_velocity",
            "negative_news_count_7d",
            "sentiment_score_daily",
            "risk_score",
        ]
        df[numeric_cols] = df[numeric_cols].round(4)

        logger.info("Feature engineering complete")
        return df

    def build_ml_dataset(self, featured_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 5 — Select and clean columns for machine learning.

        Drops helper columns and rows missing critical identifiers.
        """
        available = [col for col in FEATURE_COLUMNS if col in featured_df.columns]
        ml_df = featured_df[available].copy()

        # Ensure correct dtypes for ML frameworks
        ml_df["date"] = pd.to_datetime(ml_df["date"]).dt.date
        ml_df["supplier_id"] = ml_df["supplier_id"].astype(str)
        ml_df["article_count"] = ml_df["article_count"].fillna(0).astype(int)
        ml_df["negative_news_count_7d"] = (
            ml_df["negative_news_count_7d"].fillna(0).astype(int)
        )

        # Drop rows with no supplier id
        ml_df = ml_df.dropna(subset=["supplier_id"]).reset_index(drop=True)

        logger.info(
            "ML dataset ready: %d rows x %d features", len(ml_df), len(ml_df.columns)
        )
        return ml_df

    def save_to_csv(
        self, ml_df: pd.DataFrame, output_path: Path | str | None = None
    ) -> Path:
        """Save the ML dataset to CSV."""
        path = Path(output_path) if output_path is not None else self.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        ml_df.to_csv(path, index=False)
        logger.info("Saved ML features to '%s'", path)
        return path

    def run_pipeline(self, *, save: bool = True) -> pd.DataFrame:
        """Run the full feature engineering pipeline end-to-end."""
        logger.info("=== Starting feature engineering pipeline ===")

        supply_df = self.load_supply_chain_data()
        sentiment_df = self.load_sentiment_data()
        merged_df = self.merge_datasets(supply_df, sentiment_df)
        featured_df = self.engineer_features(merged_df)
        ml_df = self.build_ml_dataset(featured_df)

        if save:
            self.save_to_csv(ml_df)

        logger.info("=== Feature engineering pipeline complete ===")
        return ml_df
