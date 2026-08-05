"""
Lists all the pages in the website.

Each page is its own file: Dashboard, Suppliers, Model Insights, and About.
"""

# Import each page class so other files can use them easily
from app.pages.about import AboutPage
from app.pages.dashboard import DashboardPage
from app.pages.model_insights import ModelInsightsPage
from app.pages.supplier_analysis import SupplierAnalysisPage

# Names that are allowed to be imported from this package
__all__ = [
    "AboutPage",
    "DashboardPage",
    "ModelInsightsPage",
    "SupplierAnalysisPage",
]
