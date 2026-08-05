"""
Sidebar navigation component for the web dashboard.

Renders the left sidebar with app branding, page links (Dashboard, Supplier
Analysis, Model Insights, About), and demo filter controls.
"""

from __future__ import annotations

import streamlit as st

from config.settings import AppSettings, get_settings


class SidebarNavigator:
    """Renders the application sidebar with branding and page navigation."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()

    def render(self) -> str:
        """Render sidebar and return the selected page name."""
        with st.sidebar:
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
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.caption("Navigation")

            selected = st.radio(
                label="Pages",
                options=list(self._settings.pages),
                label_visibility="collapsed",
                key="nav_page",
            )

            st.markdown("---")
            st.caption("Data Source")
            st.info("Placeholder data · Demo mode", icon="ℹ️")

            st.markdown("---")
            st.caption("Filters")
            st.selectbox("Region", ["All Regions", *self._REGION_OPTIONS], key="filter_region")
            st.selectbox(
                "Time Horizon",
                ["Last 30 days", "Last 90 days", "Last 12 months"],
                key="filter_horizon",
            )

        return selected

    _REGION_OPTIONS: tuple[str, ...] = (
        "North America",
        "Europe",
        "Asia-Pacific",
        "Latin America",
    )
