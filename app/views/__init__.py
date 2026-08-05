"""
Lists all the pages in the website.

Each page is its own file: Dashboard, Suppliers, Model Insights, and About.
"""

# Import each page class so other files can use them easily
from app.views.about import AboutPage
from app.views.dashboard import DashboardPage
from app.views.model_insights import ModelInsightsPage
from app.views.supplier_analysis import SupplierAnalysisPage

# Names that are allowed to be imported from this package
__all__ = [
    "AboutPage",
    "DashboardPage",
    "ModelInsightsPage",
    "SupplierAnalysisPage",
]
