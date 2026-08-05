"""
The left menu on the website.

Shows the app name and lets you switch between
Dashboard, Supplier Analysis, Model Insights, and About.
"""

from __future__ import annotations  # Modern type hint support

import streamlit as st  # Website UI library

# Import app settings type and default settings getter
from config.settings import AppSettings, get_settings


class SidebarNavigator:
    """Renders the application sidebar with branding and page navigation."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        # Use provided settings, or load the default settings
        self._settings = settings or get_settings()

    def render(self) -> str:
        """Render sidebar and return the selected page name."""
        # Everything inside "with st.sidebar" appears in the left menu
        with st.sidebar:
            # Show app title and version at the top of the sidebar
            st.markdown(
                f"""
                <div style="padding: 0.5rem 0 1.5rem 0;">
                    <h2 style="margin: 0; font-size: 1.4rem;">
                        {self._settings.icon} Supply Chain AI
                    </h2>
                    <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.85rem;">
                        Disruption Predictor v{self._settings.version}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,  # Allow HTML for custom styling
            )

            st.markdown("---")  # Horizontal divider line
            st.caption("Navigation")  # Small label above page buttons

            # Radio buttons let the user pick which page to view
            selected = st.radio(
                label="Pages",  # Internal label (hidden from user)
                options=list(self._settings.pages),  # Dashboard, Supplier Analysis, etc.
                label_visibility="collapsed",  # Hide the "Pages" label text
                key="nav_page",  # Unique ID so Streamlit remembers the choice
            )

            st.markdown("---")  # Another divider
            st.caption("Data Source")  # Section label
            st.info("Placeholder data · Demo mode", icon="ℹ️")  # Info box about demo data

            st.markdown("---")  # Another divider
            st.caption("Filters")  # Section label for filters
            # Dropdown to filter by region (demo only for now)
            st.selectbox("Region", ["All Regions", *self._REGION_OPTIONS], key="filter_region")
            # Dropdown to pick a time range (demo only for now)
            st.selectbox(
                "Time Horizon",
                ["Last 30 days", "Last 90 days", "Last 12 months"],
                key="filter_horizon",
            )

        # Return the page name the user selected
        return selected

    # List of world regions shown in the region filter dropdown
    _REGION_OPTIONS: tuple[str, ...] = (
        "North America",
        "Europe",
        "Asia-Pacific",
        "Latin America",
    )
