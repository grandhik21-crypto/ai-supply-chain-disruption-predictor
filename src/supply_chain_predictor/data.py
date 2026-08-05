"""Synthetic supplier datasets used to train and demo the disruption model.

The generator produces industrial-engineering style supplier metrics along with
a label indicating whether the supplier experienced a disruption in the next
period. The relationship is intentionally deterministic-with-noise so the model
learns a sensible decision boundary and tests stay reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import FEATURE_NAMES


def _disruption_logit(df: pd.DataFrame) -> np.ndarray:
    """A hand-crafted risk signal used to generate labels for the synthetic set."""
    return (
        0.9 * df["geopolitical_risk_index"]
        + 0.8 * (1.0 - df["on_time_delivery_rate"])
        + 0.7 * df["demand_volatility"]
        + 0.6 * (1.0 - df["supplier_financial_health"])
        + 0.5 * df["single_source"]
        + 0.4 * (df["lead_time_days"] / 60.0)
        - 0.5 * (df["inventory_days_of_supply"] / 60.0)
        - 1.4
    )


def generate_supplier_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Return a reproducible synthetic dataset of supplier metrics + labels."""
    rng = np.random.default_rng(seed)

    data = {
        "lead_time_days": rng.uniform(2, 60, n_samples),
        "on_time_delivery_rate": rng.uniform(0.5, 1.0, n_samples),
        "inventory_days_of_supply": rng.uniform(3, 60, n_samples),
        "supplier_financial_health": rng.uniform(0.1, 1.0, n_samples),
        "geopolitical_risk_index": rng.uniform(0.0, 1.0, n_samples),
        "demand_volatility": rng.uniform(0.0, 1.0, n_samples),
        "single_source": rng.integers(0, 2, n_samples).astype(float),
    }
    df = pd.DataFrame(data, columns=FEATURE_NAMES)

    logit = _disruption_logit(df) + rng.normal(0.0, 0.35, n_samples)
    probability = 1.0 / (1.0 + np.exp(-logit))
    df["disrupted"] = (rng.uniform(0.0, 1.0, n_samples) < probability).astype(int)
    return df


def sample_suppliers() -> pd.DataFrame:
    """A small hand-written set of example suppliers for the UI and demos."""
    rows = [
        {
            "name": "Reliable Components Co. (stable)",
            "lead_time_days": 7,
            "on_time_delivery_rate": 0.98,
            "inventory_days_of_supply": 45,
            "supplier_financial_health": 0.9,
            "geopolitical_risk_index": 0.1,
            "demand_volatility": 0.15,
            "single_source": 0,
        },
        {
            "name": "Offshore Sole-Source Ltd. (fragile)",
            "lead_time_days": 48,
            "on_time_delivery_rate": 0.62,
            "inventory_days_of_supply": 8,
            "supplier_financial_health": 0.3,
            "geopolitical_risk_index": 0.85,
            "demand_volatility": 0.8,
            "single_source": 1,
        },
    ]
    return pd.DataFrame(rows)
