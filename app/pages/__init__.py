"""
Streamlit page modules for the web dashboard.

Exports the four page classes that render each navigable screen in the app.
"""

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
