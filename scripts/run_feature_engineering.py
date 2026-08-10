#!/usr/bin/env python3
"""
Run the feature engineering pipeline from the command line.

Steps:
  1. Load supply chain + daily sentiment data
  2. Merge on supplier_id and date
  3. Engineer rolling ML features
  4. Save to data/processed/ml_features.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_engineering import (  # noqa: E402
    DEFAULT_ML_OUTPUT_PATH,
    FeatureEngineeringPipeline,
)


def main() -> None:
    print("Running feature engineering pipeline...")
    pipeline = FeatureEngineeringPipeline()
    ml_df = pipeline.run_pipeline()

    print(f"\nML dataset shape: {ml_df.shape[0]} rows x {ml_df.shape[1]} columns")
    print("\nSample rows:")
    print(ml_df.head(10).to_string(index=False))
    print(f"\nSaved to: {DEFAULT_ML_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
