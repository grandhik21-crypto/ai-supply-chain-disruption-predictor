"""
The Suppliers page.

Shows which suppliers are risky, how long they take to deliver,
and a full table of all suppliers.
"""

from __future__ import annotations  # Modern type hint support

import streamlit as st  # Website UI library

from app.components.ui import (
    empty_state,
    render_chart,
    render_dataframe,
    safe_section,
    section_header,
)
# Import the shared page template
from app.views.base_page import BasePage


class SupplierAnalysisPage(BasePage):
    """Detailed supplier risk and performance analysis."""

    @property
    def title(self) -> str:
        return "Supplier Analysis"  # Page name in the header

    @property
    def icon(self) -> str:
        return "🏭"  # Factory emoji for suppliers

    @property
    def subtitle(self) -> str:
        return "Compare suppliers by risk, delivery speed, and reliability"

    def render_content(self) -> None:
        # Spinner while supplier data loads
        with st.spinner("Loading supplier data…"):
            df = self.analytics.get_suppliers_dataframe()

        if df.empty:
            empty_state(
                title="No suppliers found",
                message="No supplier records are available to analyze yet.",
                icon="🏭",
                hint="Add rows to data/raw/supply_chain.csv",
            )
            return

        with safe_section("Supplier portfolio chart"):
            section_header(
                "🗺️",
                "Supplier portfolio",
                "Each dot is a supplier. Higher up = riskier, further right = slower delivery.",
            )
            render_chart(
                self.charts.supplier_risk_scatter(df),
                key="supplier_scatter",
                filename="supplier_risk_vs_lead_time",
            )

        col_left, col_right = st.columns(2)  # Split next section into two columns

        with col_left:
            with safe_section("Lead time by region"):
                section_header("🌍", "Delivery time by region", "Average days to deliver.")
                region_df = self.analytics.get_lead_time_by_region()
                if region_df.empty:
                    empty_state(
                        title="No regional data",
                        message="Region information is missing for these suppliers.",
                        icon="🌍",
                    )
                else:
                    render_chart(
                        self.charts.lead_time_bar(region_df),
                        key="lead_time_region",
                        filename="lead_time_by_region",
                    )

        with col_right:
            with safe_section("High-risk supplier list"):
                section_header(
                    "🚨",
                    "High-risk suppliers",
                    "Suppliers with a risk score of 70 or higher.",
                )
                high_risk = self.analytics.get_high_risk_suppliers(threshold=70.0)
                if high_risk.empty:
                    st.success(
                        "No suppliers above the risk threshold.", icon="✅"
                    )
                else:
                    render_dataframe(
                        high_risk[
                            [
                                "Name",
                                "Region",
                                "Risk Score",
                                "Lead Time (days)",
                                "Sentiment Score",
                            ]
                        ],
                        key="high_risk",
                        filename="high_risk_suppliers",
                    )

        with safe_section("Full supplier registry"):
            section_header("📋", "Full supplier registry", "Every supplier on record.")
            render_dataframe(df, key="all_suppliers_registry", filename="supplier_registry")
