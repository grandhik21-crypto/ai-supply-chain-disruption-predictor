"""Machine-learning model for supplier disruption risk.

Wraps a scikit-learn pipeline (standardisation + logistic regression) behind a
small, testable API. The model is trained on synthetic supplier metrics and
exposes calibrated-ish probabilities plus feature-level contributions for
explainability.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES: list[str] = [
    "lead_time_days",
    "on_time_delivery_rate",
    "inventory_days_of_supply",
    "supplier_financial_health",
    "geopolitical_risk_index",
    "demand_volatility",
    "single_source",
]


class DisruptionModel:
    """Logistic-regression disruption classifier over supplier metrics."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _as_matrix(self, features) -> np.ndarray:
        if hasattr(features, "loc"):  # pandas DataFrame
            return features[FEATURE_NAMES].to_numpy(dtype=float)
        arr = np.asarray(features, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} features, got {arr.shape[1]}"
            )
        return arr

    def fit(self, features, labels) -> DisruptionModel:
        x = self._as_matrix(features)
        y = np.asarray(labels, dtype=int).ravel()
        self.pipeline.fit(x, y)
        self._fitted = True
        return self

    def predict_proba(self, features) -> np.ndarray:
        """Return P(disruption) for each row, shape (n_samples,)."""
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        x = self._as_matrix(features)
        return self.pipeline.predict_proba(x)[:, 1]

    def feature_contributions(self, features) -> dict[str, float]:
        """Approximate signed contribution of each feature for a single sample.

        Uses standardised inputs times the logistic-regression coefficients,
        which gives a directional, comparable importance per feature.
        """
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        x = self._as_matrix(features)
        if x.shape[0] != 1:
            raise ValueError("feature_contributions expects a single sample")
        scaler: StandardScaler = self.pipeline.named_steps["scaler"]
        clf: LogisticRegression = self.pipeline.named_steps["clf"]
        standardised = scaler.transform(x)[0]
        contributions = standardised * clf.coef_[0]
        return dict(
            zip(FEATURE_NAMES, (round(float(c), 4) for c in contributions), strict=True)
        )
