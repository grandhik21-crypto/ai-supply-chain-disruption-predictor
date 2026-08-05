"""
The service that pages call to get data.

Pages ask this for KPIs, charts, and supplier tables
instead of talking to the data files directly.
"""

# Re-export the main analytics service class
from src.services.analytics_service import SupplyChainAnalyticsService

__all__ = ["SupplyChainAnalyticsService"]
