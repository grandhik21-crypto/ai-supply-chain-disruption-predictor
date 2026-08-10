"""
The main helper pages use to fetch all their data.

Call this to get KPI numbers, supplier lists, and chart data.
It can use either demo data or a real CSV file.
"""

from __future__ import annotations  # Modern type hint support

from pathlib import Path  # For file paths to CSV data

import pandas as pd  # Table data library

# Provider that reads real CSV files
from src.data.csv_provider import CsvDataProvider
# Default path to the sample CSV file
from src.data.data_loader import DEFAULT_DATA_PATH
# Provider that generates fake demo data
from src.data.placeholder_provider import PlaceholderDataProvider
# Data types for KPIs and suppliers
from src.models.metrics import KPIMetrics, SupplierRecord

# Either provider type can be used behind this service
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
            # Use a provider passed in directly (for testing or custom setup)
            self._provider = data_provider
        elif use_csv or csv_path is not None:
            # Load data from a CSV file instead of demo data
            path = csv_path or DEFAULT_DATA_PATH
            self._provider = CsvDataProvider(file_path=path)
        else:
            # Default: use fake demo data
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
        # Filter rows where Risk Score >= threshold, sorted highest first
        return df[df["Risk Score"] >= threshold].sort_values(
            "Risk Score", ascending=False
        )

    def get_risk_summary(self) -> dict[str, float | int]:
        """Return summary statistics for supplier risk."""
        df = self.get_suppliers_dataframe()
        # Build a dictionary of summary numbers for the Dashboard
        return {
            "supplier_count": len(df),
            "avg_risk_score": round(float(df["Risk Score"].mean()), 1),
            "high_risk_count": int((df["Risk Score"] >= 70).sum()),
            "avg_lead_time": round(float(df["Lead Time (days)"].mean()), 1),
        }
