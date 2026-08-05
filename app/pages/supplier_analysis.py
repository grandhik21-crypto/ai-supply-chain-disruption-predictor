"""
The Suppliers page.

Shows which suppliers are risky, how long they take to deliver,
and a full table of all suppliers.
"""

from __future__ import annotations

import streamlit as st

from app.pages.base_page import BasePage


class SupplierAnalysisPage(BasePage):
    """Detailed supplier risk and performance analysis."""

    @property
    def title(self) -> str:
        return "Supplier Analysis"

    @property
    def icon(self) -> str:
        return "🏭"

    def render_content(self) -> None:
        df = self.analytics.get_suppliers_dataframe()

        st.markdown("### Supplier Portfolio")
        st.plotly_chart(
            self.charts.supplier_risk_scatter(df),
            use_container_width=True,
        )

        col_left, col_right = st.columns(2)

        with col_left:
            st.plotly_chart(
                self.charts.lead_time_bar(self.analytics.get_lead_time_by_region()),
                use_container_width=True,
            )

        with col_right:
            high_risk = self.analytics.get_high_risk_suppliers(threshold=70.0)
            st.markdown("### High-Risk Suppliers")
            if high_risk.empty:
                st.success("No suppliers above the risk threshold.")
            else:
                st.dataframe(
                    high_risk[
                        ["Name", "Region", "Risk Score", "Lead Time (days)", "Sentiment Score"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("### Full Supplier Registry")
        st.dataframe(df, use_container_width=True, hide_index=True)
