"""
Abstract base class for all Streamlit dashboard pages.

Provides a shared page layout (header, subtitle, divider) and injects the
analytics service and chart factory used by every page subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import streamlit as st

from app.components.charts import ChartFactory
from src.services.analytics_service import SupplyChainAnalyticsService


class BasePage(ABC):
    """Base interface for all application pages."""

    def __init__(
        self,
        analytics_service: SupplyChainAnalyticsService | None = None,
        chart_factory: ChartFactory | None = None,
    ) -> None:
        self.analytics = analytics_service or SupplyChainAnalyticsService()
        self.charts = chart_factory or ChartFactory()

    @property
    @abstractmethod
    def title(self) -> str:
        """Page title displayed in the header."""

    @property
    @abstractmethod
    def icon(self) -> str:
        """Emoji icon for the page header."""

    @abstractmethod
    def render_content(self) -> None:
        """Render the page-specific content."""

    def render(self) -> None:
        """Render the full page with a consistent header."""
        st.markdown(f"## {self.icon} {self.title}")
        st.markdown(
            "<p style='color: #64748b; margin-top: -0.5rem;'>"
            "AI-powered supply chain disruption intelligence"
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        self.render_content()
