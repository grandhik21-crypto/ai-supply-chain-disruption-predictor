"""
Data shape definitions used across the app.

Defines what a KPI and a Supplier look like
so all parts of the code use the same structure.
"""

from src.models.metrics import KPIMetrics, SupplierRecord

__all__ = ["KPIMetrics", "SupplierRecord"]
