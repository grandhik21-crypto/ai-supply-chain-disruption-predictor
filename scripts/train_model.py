#!/usr/bin/env python3
"""
Train the XGBoost shipment disruption model from the command line.

Steps:
  1. Load engineered features (data/processed/ml_features.csv)
  2. Split train/test
  3. Cross-validate
  4. Tune hyperparameters
  5. Evaluate Accuracy / Precision / Recall / F1 / ROC-AUC
  6. Save model to models/disruption_xgb.joblib
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.model import DEFAULT_MODEL_PATH, DisruptionPredictor  # noqa: E402


def main() -> None:
    print("Training shipment disruption model (XGBoost)...")
    predictor = DisruptionPredictor()
    metrics = predictor.run_pipeline(tune=True, save=True)

    print("\n=== Test metrics ===")
    print(f"Accuracy : {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall   : {metrics.recall:.4f}")
    print(f"F1       : {metrics.f1:.4f}")
    print(f"ROC-AUC  : {metrics.roc_auc:.4f}")
    print(
        f"CV Acc   : {metrics.cv_accuracy_mean:.4f} ± {metrics.cv_accuracy_std:.4f}"
    )
    print(f"\nBest params: {metrics.best_params}")
    print(f"\nModel saved to: {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
