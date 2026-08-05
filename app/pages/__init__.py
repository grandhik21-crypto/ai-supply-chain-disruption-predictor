"""Streamlit page modules."""

from app.pages.about import AboutPage
from app.pages.dashboard import DashboardPage
from app.pages.model_insights import ModelInsightsPage
from app.pages.supplier_analysis import SupplierAnalysisPage

__all__ = [
    "AboutPage",
    "DashboardPage",
    "ModelInsightsPage",
    "SupplierAnalysisPage",
]
