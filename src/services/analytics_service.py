"""Orchestrates data access and analytics for the dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.csv_provider import CsvDataProvider
from src.data.data_loader import DEFAULT_DATA_PATH
from src.data.placeholder_provider import PlaceholderDataProvider
from src.models.metrics import KPIMetrics, SupplierRecord

DataProvider = PlaceholderDataProvider | CsvDataProvider


class SupplyChainAnalyticsService:
    """High-level service exposing supply chain analytics to the UI layer."""

    def __init__(
        self,
        data_provider: DataProvider | None = None,
        *,
        use_csv: bool = False,
        csv_path: Path | str | None = None,
    ) -> None:
        if data_provider is not None:
            self._provider = data_provider
        elif use_csv or csv_path is not None:
            path = csv_path or DEFAULT_DATA_PATH
            self._provider = CsvDataProvider(file_path=path)
        else:
            self._provider = PlaceholderDataProvider()

    def get_kpi_metrics(self) -> KPIMetrics:
        """Fetch aggregate KPI metrics."""
        return self._provider.get_kpi_metrics()

    def get_suppliers(self) -> list[SupplierRecord]:
        """Fetch all supplier records."""
        return self._provider.get_suppliers()

    def get_suppliers_dataframe(self) -> pd.DataFrame:
        """Fetch suppliers as a DataFrame."""
        return self._provider.get_suppliers_dataframe()

    def get_risk_trend(self) -> pd.DataFrame:
        """Fetch monthly risk score trend."""
        return self._provider.get_risk_trend()

    def get_lead_time_by_region(self) -> pd.DataFrame:
        """Fetch regional lead time averages."""
        return self._provider.get_lead_time_by_region()

    def get_inventory_coverage_trend(self) -> pd.DataFrame:
        """Fetch inventory coverage trend."""
        return self._provider.get_inventory_coverage_trend()

    def get_sentiment_timeline(self) -> pd.DataFrame:
        """Fetch daily sentiment timeline."""
        return self._provider.get_sentiment_timeline()

    def get_disruption_probability_forecast(self) -> pd.DataFrame:
        """Fetch disruption probability forecast."""
        return self._provider.get_disruption_probability_forecast()

    def get_feature_importance(self) -> pd.DataFrame:
        """Fetch model feature importance data."""
        return self._provider.get_feature_importance()

    def get_confusion_matrix(self) -> pd.DataFrame:
        """Fetch model confusion matrix."""
        return self._provider.get_confusion_matrix()

    def get_high_risk_suppliers(self, threshold: float = 70.0) -> pd.DataFrame:
        """Return suppliers exceeding the risk score threshold."""
        df = self.get_suppliers_dataframe()
        return df[df["Risk Score"] >= threshold].sort_values(
            "Risk Score", ascending=False
        )

    def get_risk_summary(self) -> dict[str, float | int]:
        """Return summary statistics for supplier risk."""
        df = self.get_suppliers_dataframe()
        return {
            "supplier_count": len(df),
            "avg_risk_score": round(float(df["Risk Score"].mean()), 1),
            "high_risk_count": int((df["Risk Score"] >= 70).sum()),
            "avg_lead_time": round(float(df["Lead Time (days)"].mean()), 1),
        }
