"""
CSV-backed data provider for the web application.

Uses the ingestion pipeline to load real CSV data and exposes the same
interface as PlaceholderDataProvider so dashboard pages can consume it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.data_loader import DEFAULT_DATA_PATH, SupplyChainDataLoader
from src.models.metrics import KPIMetrics, SupplierRecord


class CsvDataProvider:
    """Loads cleaned supply chain data from CSV via the ingestion pipeline."""

    _DISPLAY_COLUMN_MAP: dict[str, str] = {
        "supplier_id": "Supplier ID",
        "supplier_name": "Name",
        "region": "Region",
        "category": "Category",
        "risk_score": "Risk Score",
        "lead_time_days": "Lead Time (days)",
        "on_time_delivery_pct": "On-Time Delivery (%)",
        "sentiment_score": "Sentiment Score",
        "last_disruption": "Last Disruption",
    }

    def __init__(
        self,
        file_path: Path | str | None = None,
        loader: SupplyChainDataLoader | None = None,
    ) -> None:
        self._loader = loader or SupplyChainDataLoader(
            file_path=Path(file_path) if file_path else DEFAULT_DATA_PATH
        )
        self._df: pd.DataFrame | None = None

    @property
    def dataframe(self) -> pd.DataFrame:
        """Lazy-load and cache the cleaned DataFrame."""
        if self._df is None:
            self._df = self._loader.load()
        return self._df

    def reload(self) -> pd.DataFrame:
        """Force reload from disk."""
        self._df = self._loader.load()
        return self._df

    def get_kpi_metrics(self) -> KPIMetrics:
        """Compute aggregate KPI metrics from ingested data."""
        df = self.dataframe
        return KPIMetrics(
            risk_score=round(float(df["risk_score"].mean()), 1),
            lead_time=round(float(df["lead_time_days"].mean()), 1),
            inventory_coverage=round(float(df["inventory_coverage_days"].mean()), 1),
            sentiment_score=round(float(df["sentiment_score"].mean()), 2),
        )

    def get_suppliers(self) -> list[SupplierRecord]:
        """Convert ingested rows to SupplierRecord domain objects."""
        records: list[SupplierRecord] = []
        for row in self.dataframe.itertuples(index=False):
            last_disruption = row.last_disruption
            disruption_date = (
                last_disruption.date()
                if pd.notna(last_disruption)
                else None
            )
            records.append(
                SupplierRecord(
                    supplier_id=str(row.supplier_id),
                    name=str(row.supplier_name),
                    region=str(row.region),
                    category=str(row.category),
                    risk_score=round(float(row.risk_score), 1),
                    lead_time_days=round(float(row.lead_time_days), 1),
                    on_time_delivery_pct=round(float(row.on_time_delivery_pct), 1),
                    sentiment_score=round(float(row.sentiment_score), 2),
                    last_disruption=disruption_date,
                )
            )
        return records

    def get_suppliers_dataframe(self) -> pd.DataFrame:
        """Return suppliers formatted for dashboard display."""
        df = self.dataframe.copy()
        display_df = df[list(self._DISPLAY_COLUMN_MAP.keys())].rename(
            columns=self._DISPLAY_COLUMN_MAP
        )
        return display_df

    def get_risk_trend(self, periods: int = 12) -> pd.DataFrame:
        """Derive monthly risk trend from order dates."""
        df = self.dataframe.copy()
        df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
        grouped = (
            df.groupby("month", as_index=False)["risk_score"]
            .mean()
            .rename(columns={"month": "Month", "risk_score": "Risk Score"})
            .tail(periods)
        )
        return grouped.round(1)

    def get_lead_time_by_region(self) -> pd.DataFrame:
        """Return average lead time grouped by region."""
        grouped = (
            self.dataframe.groupby("region", as_index=False)["lead_time_days"]
            .mean()
            .rename(columns={"region": "Region", "lead_time_days": "Avg Lead Time (days)"})
            .sort_values("Avg Lead Time (days)", ascending=False)
        )
        return grouped.round(1)

    def get_inventory_coverage_trend(self, periods: int = 12) -> pd.DataFrame:
        """Derive inventory coverage trend from order dates."""
        df = self.dataframe.copy()
        df["week"] = df["order_date"].dt.to_period("W").dt.start_time
        grouped = (
            df.groupby("week", as_index=False)["inventory_coverage_days"]
            .mean()
            .rename(
                columns={
                    "week": "Week",
                    "inventory_coverage_days": "Inventory Coverage (days)",
                }
            )
            .tail(periods)
        )
        return grouped.round(1)

    def get_sentiment_timeline(self, periods: int = 30) -> pd.DataFrame:
        """Return sentiment scores indexed by order date."""
        df = (
            self.dataframe[["order_date", "sentiment_score"]]
            .dropna(subset=["order_date"])
            .sort_values("order_date")
            .rename(columns={"order_date": "Date", "sentiment_score": "Sentiment Score"})
            .tail(periods)
        )
        return df.reset_index(drop=True)

    def get_disruption_probability_forecast(self, horizons: int = 6) -> pd.DataFrame:
        """Generate a simple forecast from high-risk supplier share."""
        high_risk_share = (self.dataframe["risk_score"] >= 70).mean()
        base_prob = min(max(high_risk_share * 100, 15), 55)
        months = pd.date_range(start=date.today(), periods=horizons, freq="MS")
        probabilities = np.linspace(base_prob, base_prob + 8, horizons)
        return pd.DataFrame(
            {"Month": months, "Disruption Probability": probabilities.round(1)}
        )

    def get_feature_importance(self) -> pd.DataFrame:
        """Return heuristic feature importance from column variance."""
        numeric_cols = [
            "risk_score",
            "lead_time_days",
            "inventory_coverage_days",
            "on_time_delivery_pct",
            "sentiment_score",
        ]
        df = self.dataframe[numeric_cols]
        importance = df.var() / df.var().sum()
        feature_names = {
            "risk_score": "Supplier Risk Score",
            "lead_time_days": "Supplier Lead Time",
            "inventory_coverage_days": "Inventory Buffer",
            "on_time_delivery_pct": "On-Time Delivery",
            "sentiment_score": "News Sentiment",
        }
        result = pd.DataFrame(
            {
                "Feature": [feature_names[c] for c in importance.index],
                "Importance": importance.values,
            }
        )
        return result.sort_values("Importance", ascending=True)

    def get_confusion_matrix(self) -> pd.DataFrame:
        """Return a placeholder confusion matrix derived from risk thresholds."""
        df = self.dataframe
        predicted = df["risk_score"] >= 70
        actual = df["sentiment_score"] < 0.55
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        tn = int((~predicted & ~actual).sum())
        return pd.DataFrame(
            {
                "Predicted: No Disruption": [tn, fn],
                "Predicted: Disruption": [fp, tp],
            },
            index=["Actual: No Disruption", "Actual: Disruption"],
        )
