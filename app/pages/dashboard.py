"""
The home page of the website.

Shows the main numbers (KPIs), trend charts,
forecasts, and a quick supplier summary.
"""

from __future__ import annotations  # Modern type hint support

import streamlit as st  # Website UI library

# Import the class that draws the four KPI number boxes
from app.components.kpi_cards import KPICardRenderer
# Import the shared page template all pages extend
from app.pages.base_page import BasePage


class DashboardPage(BasePage):
    """Main dashboard with KPIs and trend visualizations."""

    @property
    def title(self) -> str:
        return "Dashboard"  # Page name shown in the header

    @property
    def icon(self) -> str:
        return "📊"  # Chart emoji shown next to the title

    def render_content(self) -> None:
        # Ask the analytics service for the four main KPI numbers
        metrics = self.analytics.get_kpi_metrics()
        # Draw the four KPI cards at the top of the page
        KPICardRenderer().render(metrics)

        st.markdown("### Trends & Forecasts")  # Section heading
        col_left, col_right = st.columns(2)  # Two columns for charts

        with col_left:  # Left column charts
            # Line chart: risk score over time
            st.plotly_chart(
                self.charts.risk_trend_line(self.analytics.get_risk_trend()),
                use_container_width=True,  # Stretch chart to fill column width
            )
            # Area chart: inventory coverage over time
            st.plotly_chart(
                self.charts.inventory_area(
                    self.analytics.get_inventory_coverage_trend()
                ),
                use_container_width=True,
            )

        with col_right:  # Right column charts
            # Line chart: sentiment score over time
            st.plotly_chart(
                self.charts.sentiment_line(self.analytics.get_sentiment_timeline()),
                use_container_width=True,
            )
            # Bar chart: predicted disruption probability by month
            st.plotly_chart(
                self.charts.disruption_forecast(
                    self.analytics.get_disruption_probability_forecast()
                ),
                use_container_width=True,
            )

        st.markdown("### Quick Summary")  # Section heading for summary stats
        # Get supplier count, average risk, etc.
        summary = self.analytics.get_risk_summary()
        summary_cols = st.columns(4)  # Four small summary boxes
        # Pairs of (label, value) for each summary metric
        labels = [
            ("Active Suppliers", summary["supplier_count"]),
            ("Avg Risk Score", summary["avg_risk_score"]),
            ("High-Risk Suppliers", summary["high_risk_count"]),
            ("Avg Lead Time (days)", summary["avg_lead_time"]),
        ]
        # Put each summary metric in its own column
        for col, (label, value) in zip(summary_cols, labels):
            with col:
                st.metric(label=label, value=value)
