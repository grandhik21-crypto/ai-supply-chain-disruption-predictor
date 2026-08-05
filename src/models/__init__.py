"""
Data shape definitions used across the app.

Defines what a KPI, Supplier, and NewsArticle look like
so all parts of the code use the same structure.
"""

from src.models.metrics import KPIMetrics, SupplierRecord
from src.models.sentiment import ArticleSentimentResult, NewsArticle

__all__ = ["ArticleSentimentResult", "KPIMetrics", "NewsArticle", "SupplierRecord"]
