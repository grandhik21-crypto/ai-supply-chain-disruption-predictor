#!/usr/bin/env python3
"""
Prepare everything the dashboard needs to run.

Useful on a fresh deployment (or a fresh clone) where the generated files
are missing. It builds only what is absent, so it is safe to run repeatedly.

Steps:
  1. Build the ML feature table  (data/processed/ml_features.csv)
  2. Train the disruption model  (models/disruption_xgb.joblib)

Sentiment analysis is NOT run here because it needs the optional FinBERT
dependencies (PyTorch + Transformers). If the sentiment CSV is missing, the
feature pipeline still works — it just omits the news-based columns.
To generate sentiment:  python3 scripts/run_sentiment.py

Usage:
    python3 scripts/bootstrap.py            # build only what's missing
    python3 scripts/bootstrap.py --force    # rebuild everything
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    DAILY_SENTIMENT_CSV,
    ML_FEATURES_CSV,
    MODEL_PATH,
    SUPPLY_CHAIN_CSV,
)
from src.utils.logging_config import get_logger  # noqa: E402

logger = get_logger("bootstrap")


def ensure_features(force: bool = False) -> Path:
    """Build the ML feature table if it does not exist yet."""
    if ML_FEATURES_CSV.exists() and not force:
        logger.info("Features already present: %s", ML_FEATURES_CSV)
        return ML_FEATURES_CSV

    if not SUPPLY_CHAIN_CSV.exists():
        raise FileNotFoundError(
            f"Missing input data: {SUPPLY_CHAIN_CSV}. "
            "Add your supplier CSV before bootstrapping."
        )

    if not DAILY_SENTIMENT_CSV.exists():
        logger.warning(
            "No sentiment file at %s — features will be built without news "
            "columns. Run scripts/run_sentiment.py to add them.",
            DAILY_SENTIMENT_CSV,
        )

    from src.features.feature_engineering import FeatureEngineeringPipeline

    logger.info("Building ML features...")
    FeatureEngineeringPipeline().run_pipeline(save=True)
    return ML_FEATURES_CSV


def ensure_model(force: bool = False) -> Path:
    """Train and save the disruption model if it does not exist yet."""
    if MODEL_PATH.exists() and not force:
        logger.info("Model already present: %s", MODEL_PATH)
        return MODEL_PATH

    from src.ml.model import DisruptionPredictor

    logger.info("Training disruption model...")
    predictor = DisruptionPredictor()
    metrics = predictor.run_pipeline(tune=True, save=True)
    logger.info(
        "Training complete — accuracy=%.3f f1=%.3f roc_auc=%.3f",
        metrics.accuracy,
        metrics.f1,
        metrics.roc_auc,
    )
    return MODEL_PATH


def bootstrap(force: bool = False) -> dict[str, str]:
    """Run every preparation step and return the resulting paths."""
    logger.info("=== Bootstrap starting ===")
    features = ensure_features(force=force)
    model = ensure_model(force=force)
    logger.info("=== Bootstrap complete ===")
    return {"features": str(features), "model": str(model)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data and model files.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild features and retrain the model even if files exist.",
    )
    args = parser.parse_args()

    result = bootstrap(force=args.force)
    print("\nReady to launch:")
    print(f"  Features: {result['features']}")
    print(f"  Model:    {result['model']}")
    print("\nStart the dashboard with:")
    print("  python3 -m streamlit run app/main.py --server.port 8501")


if __name__ == "__main__":
    main()
