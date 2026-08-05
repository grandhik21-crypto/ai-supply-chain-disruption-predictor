"""
Domain models package for the web application.

Exports typed data structures (KPIMetrics, SupplierRecord) shared between
data providers, services, and dashboard UI components.
"""

from src.models.metrics import KPIMetrics, SupplierRecord

__all__ = ["KPIMetrics", "SupplierRecord"]
