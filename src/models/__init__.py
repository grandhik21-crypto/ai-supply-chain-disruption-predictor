"""
Data shape definitions used across the app.

Defines what a KPI and a Supplier look like
so all parts of the code use the same structure.
"""

# Import the two main data types defined in metrics.py
from src.models.metrics import KPIMetrics, SupplierRecord

# Names other files can import from this package
__all__ = ["KPIMetrics", "SupplierRecord"]
