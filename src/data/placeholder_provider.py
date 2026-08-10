"""
Makes fake demo data for the website.

When no real CSV file is loaded, this creates sample suppliers,
scores, and charts so you can still explore the app.
"""

from __future__ import annotations  # Modern type hint support

from datetime import date, timedelta  # For generating dates

import numpy as np  # Random number generator for fake data
import pandas as pd  # Table data library

from config.settings import get_settings  # App settings (includes random seed)
from src.models.metrics import KPIMetrics, SupplierRecord


class PlaceholderDataProvider:
    """Provides synthetic supply chain data with a fixed random seed."""

    # List of fake company names used in demo data
    SUPPLIER_NAMES: tuple[str, ...] = (
        "Nova Components Ltd.",
        "Pacific Steel Co.",
        "EuroLogistics GmbH",
        "Atlas Semiconductors",
        "GreenField Packaging",
        "Summit Chemicals",
        "Horizon Electronics",
        "BlueRiver Logistics",
    )

    # World regions assigned to suppliers
    REGIONS: tuple[str, ...] = (
        "North America",
        "Europe",
        "Asia-Pacific",
        "Latin America",
    )

    # Product categories assigned to suppliers
    CATEGORIES: tuple[str, ...] = (
        "Raw Materials",
        "Electronics",
        "Packaging",
        "Logistics",
        "Chemicals",
    )

    def __init__(self, seed: int | None = None) -> None:
        settings = get_settings()
        self._seed = seed if seed is not None else settings.random_seed
        self._rng = np.random.default_rng(self._seed)  # Random generator (same seed = same data)

    def get_kpi_metrics(self) -> KPIMetrics:
        """Return aggregate KPI metrics for the executive dashboard."""
        return KPIMetrics(
            risk_score=round(float(self._rng.uniform(55, 78)), 1),
            lead_time=round(float(self._rng.uniform(18, 42)), 1),
            inventory_coverage=round(float(self._rng.uniform(12, 35)), 1),
            sentiment_score=round(float(self._rng.uniform(0.45, 0.82)), 2),
        )

    def get_suppliers(self) -> list[SupplierRecord]:
        """Return a list of supplier records with varied risk profiles."""
        suppliers: list[SupplierRecord] = []
        base_date = date.today()

        for index, name in enumerate(self.SUPPLIER_NAMES):
            has_disruption = bool(self._rng.integers(0, 2))  # Randomly decide if they had a disruption
            suppliers.append(
                SupplierRecord(
                    supplier_id=f"SUP-{1000 + index}",
                    name=name,
                    region=self.REGIONS[index % len(self.REGIONS)],
                    category=self.CATEGORIES[index % len(self.CATEGORIES)],
                    risk_score=round(float(self._rng.uniform(25, 92)), 1),
                    lead_time_days=round(float(self._rng.uniform(10, 55)), 1),
                    on_time_delivery_pct=round(float(self._rng.uniform(72, 99)), 1),
                    sentiment_score=round(float(self._rng.uniform(0.2, 0.95)), 2),
                    last_disruption=(
                        base_date - timedelta(days=int(self._rng.integers(30, 365)))
                        if has_disruption
                        else None
                    ),
                )
            )
        return suppliers

    def get_suppliers_dataframe(self) -> pd.DataFrame:
        """Return supplier records as a pandas DataFrame."""
        records = self.get_suppliers()
        return pd.DataFrame(
            [
                {
                    "Supplier ID": record.supplier_id,
                    "Name": record.name,
                    "Region": record.region,
                    "Category": record.category,
                    "Risk Score": record.risk_score,
                    "Lead Time (days)": record.lead_time_days,
                    "On-Time Delivery (%)": record.on_time_delivery_pct,
                    "Sentiment Score": record.sentiment_score,
                    "Last Disruption": record.last_disruption,
                }
                for record in records
            ]
        )

    def get_risk_trend(self, periods: int = 12) -> pd.DataFrame:
        """Return monthly risk score trend data."""
        months = pd.date_range(end=date.today(), periods=periods, freq="MS")
        base = float(self._rng.uniform(50, 65))
        noise = self._rng.normal(0, 3, periods)
        trend = np.linspace(0, 8, periods)
        values = np.clip(base + noise + trend, 0, 100)

        return pd.DataFrame({"Month": months, "Risk Score": values.round(1)})

    def get_lead_time_by_region(self) -> pd.DataFrame:
        """Return average lead time grouped by region."""
        df = self.get_suppliers_dataframe()
        grouped = (
            df.groupby("Region", as_index=False)["Lead Time (days)"]
            .mean()
            .round(1)
            .sort_values("Lead Time (days)", ascending=False)
        )
        return grouped.rename(columns={"Lead Time (days)": "Avg Lead Time (days)"})

    def get_inventory_coverage_trend(self, periods: int = 12) -> pd.DataFrame:
        """Return weekly inventory coverage trend data."""
        weeks = pd.date_range(end=date.today(), periods=periods, freq="W")
        base = float(self._rng.uniform(20, 28))
        seasonal = 4 * np.sin(np.linspace(0, 2 * np.pi, periods))
        noise = self._rng.normal(0, 1.5, periods)
        values = np.clip(base + seasonal + noise, 5, 45)

        return pd.DataFrame(
            {"Week": weeks, "Inventory Coverage (days)": values.round(1)}
        )

    def get_sentiment_timeline(self, periods: int = 30) -> pd.DataFrame:
        """Return daily sentiment score timeline."""
        days = pd.date_range(end=date.today(), periods=periods, freq="D")
        base = float(self._rng.uniform(0.55, 0.7))
        noise = self._rng.normal(0, 0.05, periods)
        values = np.clip(base + noise, 0, 1)

        return pd.DataFrame({"Date": days, "Sentiment Score": values.round(2)})

    def get_disruption_probability_forecast(self, horizons: int = 6) -> pd.DataFrame:
        """Return forecasted disruption probability by month."""
        months = pd.date_range(start=date.today(), periods=horizons, freq="MS")
        probabilities = np.clip(
            self._rng.uniform(0.15, 0.55, horizons) + np.linspace(0, 0.1, horizons),
            0,
            1,
        )
        return pd.DataFrame(
            {
                "Month": months,
                "Disruption Probability": (probabilities * 100).round(1),
            }
        )

    def get_feature_importance(self) -> pd.DataFrame:
        """Return model feature importance scores for explainability."""
        features = [
            "Supplier Lead Time",
            "Geopolitical Risk Index",
            "News Sentiment",
            "Port Congestion",
            "Inventory Buffer",
            "Order Volatility",
            "Weather Anomalies",
            "FX Volatility",
        ]
        importance = self._rng.uniform(0.05, 0.22, len(features))
        importance = importance / importance.sum()

        df = pd.DataFrame({"Feature": features, "Importance": importance})
        return df.sort_values("Importance", ascending=True)

    def get_confusion_matrix(self) -> pd.DataFrame:
        """Return a placeholder confusion matrix for model evaluation."""
        return pd.DataFrame(
            {
                "Predicted: No Disruption": [142, 18],
                "Predicted: Disruption": [12, 28],
            },
            index=["Actual: No Disruption", "Actual: Disruption"],
        )
