"""
Starts the web app.

This is the main file you run to open the website in your browser.
It shows the sidebar menu and opens the page you pick (Dashboard, Suppliers, etc.).
Run with: python3 -m streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the Python path for imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.sidebar import SidebarNavigator
from app.pages.about import AboutPage
from app.pages.dashboard import DashboardPage
from app.pages.model_insights import ModelInsightsPage
from app.pages.supplier_analysis import SupplierAnalysisPage
from config.settings import get_settings


class Application:
    """Main application controller that wires navigation to page classes."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._navigator = SidebarNavigator(self._settings)
        self._pages: dict[str, object] = {
            "Dashboard": DashboardPage(),
            "Supplier Analysis": SupplierAnalysisPage(),
            "Model Insights": ModelInsightsPage(),
            "About": AboutPage(),
        }

    def run(self) -> None:
        """Configure Streamlit and render the selected page."""
        st.set_page_config(
            page_title=self._settings.title,
            page_icon=self._settings.icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )

        self._inject_global_styles()
        selected_page = self._navigator.render()
        page = self._pages.get(selected_page, DashboardPage())
        page.render()

    @staticmethod
    def _inject_global_styles() -> None:
        """Inject minimal CSS for a polished dashboard appearance."""
        st.markdown(
            """
            <style>
                .block-container {
                    padding-top: 2rem;
                    padding-bottom: 2rem;
                }
                [data-testid="stMetricValue"] {
                    font-size: 1.75rem;
                    font-weight: 700;
                }
                [data-testid="stSidebar"] {
                    background-color: #f8fafc;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    """Run the Streamlit application."""
    Application().run()


if __name__ == "__main__":
    main()
