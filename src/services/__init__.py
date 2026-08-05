"""
Analytics services package for the web application.

Exports the service layer that dashboard pages call to fetch KPIs, charts,
and supplier data without knowing the underlying data source.
"""

from src.services.analytics_service import SupplyChainAnalyticsService

__all__ = ["SupplyChainAnalyticsService"]
