"""
The Suppliers page.

Shows which suppliers are risky, how long they take to deliver,
and a full table of all suppliers.
"""

from __future__ import annotations  # Modern type hint support

import streamlit as st  # Website UI library

# Import the shared page template
from app.pages.base_page import BasePage


class SupplierAnalysisPage(BasePage):
    """Detailed supplier risk and performance analysis."""

    @property
    def title(self) -> str:
        return "Supplier Analysis"  # Page name in the header

    @property
    def icon(self) -> str:
        return "🏭"  # Factory emoji for suppliers

    def render_content(self) -> None:
        # Get all supplier rows as a table
        df = self.analytics.get_suppliers_dataframe()

        st.markdown("### Supplier Portfolio")  # Section heading
        # Scatter plot: each dot is a supplier (risk vs lead time)
        st.plotly_chart(
            self.charts.supplier_risk_scatter(df),
            use_container_width=True,
        )

        col_left, col_right = st.columns(2)  # Split next section into two columns

        with col_left:
            # Bar chart: average delivery time by world region
            st.plotly_chart(
                self.charts.lead_time_bar(self.analytics.get_lead_time_by_region()),
                use_container_width=True,
            )

        with col_right:
            # Get suppliers with risk score 70 or higher
            high_risk = self.analytics.get_high_risk_suppliers(threshold=70.0)
            st.markdown("### High-Risk Suppliers")  # Section heading
            if high_risk.empty:
                # Show green success message if no risky suppliers
                st.success("No suppliers above the risk threshold.")
            else:
                # Show table of risky suppliers with key columns only
                st.dataframe(
                    high_risk[
                        ["Name", "Region", "Risk Score", "Lead Time (days)", "Sentiment Score"]
                    ],
                    use_container_width=True,
                    hide_index=True,  # Don't show row numbers column
                )

        st.markdown("### Full Supplier Registry")  # Section heading
        # Show complete supplier table with all columns
        st.dataframe(df, use_container_width=True, hide_index=True)
