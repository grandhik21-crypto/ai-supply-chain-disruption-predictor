"""
Starts the web app.

This is the main file you run to open the website in your browser.
It shows the sidebar menu and opens the page you pick (Dashboard, Suppliers, etc.).
Run with: python3 -m streamlit run app/main.py
"""

from __future__ import annotations  # Lets us use modern type hints in older Python versions

import sys  # Used to update Python's module search path
from pathlib import Path  # Helps work with file and folder paths

# Find the project root folder (one level above the app/ folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Add project root to Python path so imports like "from config..." work
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # The library that builds the website

# Import the sidebar menu builder
from app.components.sidebar import SidebarNavigator
# Import each page class the user can navigate to
from app.pages.about import AboutPage
from app.pages.dashboard import DashboardPage
from app.pages.model_insights import ModelInsightsPage
from app.pages.supplier_analysis import SupplierAnalysisPage
# Import app settings (title, icon, page list)
from config.settings import get_settings


class Application:
    """Main application controller that wires navigation to page classes."""

    def __init__(self) -> None:
        # Load app settings (name, version, pages)
        self._settings = get_settings()
        # Create the sidebar navigator using those settings
        self._navigator = SidebarNavigator(self._settings)
        # Map each menu name to its page class
        self._pages: dict[str, object] = {
            "Dashboard": DashboardPage(),
            "Supplier Analysis": SupplierAnalysisPage(),
            "Model Insights": ModelInsightsPage(),
            "About": AboutPage(),
        }

    def run(self) -> None:
        """Configure Streamlit and render the selected page."""
        # Set browser tab title, icon, wide layout, and open sidebar by default
        st.set_page_config(
            page_title=self._settings.title,
            page_icon=self._settings.icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Add custom CSS styling to the page
        self._inject_global_styles()
        # Show sidebar and get which page the user picked
        selected_page = self._navigator.render()
        # Look up the page object; fall back to Dashboard if not found
        page = self._pages.get(selected_page, DashboardPage())
        # Draw the selected page on screen
        page.render()

    @staticmethod
    def _inject_global_styles() -> None:
        """Inject minimal CSS for a polished dashboard appearance."""
        # Insert CSS rules to style padding, metric numbers, and sidebar color
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
            unsafe_allow_html=True,  # Allow raw HTML/CSS in markdown
        )


def main() -> None:
    """Run the Streamlit application."""
    # Create the app and start it
    Application().run()


if __name__ == "__main__":
    # Only run main() when this file is executed directly (not imported)
    main()
