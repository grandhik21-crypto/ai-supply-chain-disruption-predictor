"""
Lists all the pages in the website.

Each page is its own file: Dashboard, Suppliers, Model Insights, and About.
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
