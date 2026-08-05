"""
The template all pages share.

Every page uses this for the same title bar and setup,
so they all look consistent.
"""

from __future__ import annotations  # Modern type hint support

from abc import ABC, abstractmethod  # ABC = base class; abstractmethod = must be overridden

import streamlit as st  # Website UI library

# Import chart builder used by all pages
from app.components.charts import ChartFactory
# Import the data helper that pages call for numbers and tables
from src.services.analytics_service import SupplyChainAnalyticsService


class BasePage(ABC):
    """Base interface for all application pages."""

    def __init__(
        self,
        analytics_service: SupplyChainAnalyticsService | None = None,
        chart_factory: ChartFactory | None = None,
    ) -> None:
        # Create analytics service if the page didn't receive one
        self.analytics = analytics_service or SupplyChainAnalyticsService()
        # Create chart factory if the page didn't receive one
        self.charts = chart_factory or ChartFactory()

    @property
    @abstractmethod
    def title(self) -> str:
        """Page title displayed in the header."""
        # Each child page must define its title (e.g. "Dashboard")

    @property
    @abstractmethod
    def icon(self) -> str:
        """Emoji icon for the page header."""
        # Each child page must define its icon (e.g. "📊")

    @abstractmethod
    def render_content(self) -> None:
        """Render the page-specific content."""
        # Each child page must define what appears below the header

    def render(self) -> None:
        """Render the full page with a consistent header."""
        # Big heading with icon and page title
        st.markdown(f"## {self.icon} {self.title}")
        # Gray subtitle under the title
        st.markdown(
            "<p style='color: #64748b; margin-top: -0.5rem;'>"
            "AI-powered supply chain disruption intelligence"
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")  # Divider line before main content
        # Call the child page's content method
        self.render_content()
