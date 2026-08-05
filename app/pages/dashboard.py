"""Executive dashboard page."""

from __future__ import annotations

import streamlit as st

from app.components.kpi_cards import KPICardRenderer
from app.pages.base_page import BasePage


class DashboardPage(BasePage):
    """Main dashboard with KPIs and trend visualizations."""

    @property
    def title(self) -> str:
        return "Dashboard"

    @property
    def icon(self) -> str:
        return "📊"

    def render_content(self) -> None:
        metrics = self.analytics.get_kpi_metrics()
        KPICardRenderer().render(metrics)

        st.markdown("### Trends & Forecasts")
        col_left, col_right = st.columns(2)

        with col_left:
            st.plotly_chart(
                self.charts.risk_trend_line(self.analytics.get_risk_trend()),
                use_container_width=True,
            )
            st.plotly_chart(
                self.charts.inventory_area(
                    self.analytics.get_inventory_coverage_trend()
                ),
                use_container_width=True,
            )

        with col_right:
            st.plotly_chart(
                self.charts.sentiment_line(self.analytics.get_sentiment_timeline()),
                use_container_width=True,
            )
            st.plotly_chart(
                self.charts.disruption_forecast(
                    self.analytics.get_disruption_probability_forecast()
                ),
                use_container_width=True,
            )

        st.markdown("### Quick Summary")
        summary = self.analytics.get_risk_summary()
        summary_cols = st.columns(4)
        labels = [
            ("Active Suppliers", summary["supplier_count"]),
            ("Avg Risk Score", summary["avg_risk_score"]),
            ("High-Risk Suppliers", summary["high_risk_count"]),
            ("Avg Lead Time (days)", summary["avg_lead_time"]),
        ]
        for col, (label, value) in zip(summary_cols, labels):
            with col:
                st.metric(label=label, value=value)
