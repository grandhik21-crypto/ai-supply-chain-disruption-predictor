"""
The home page of the website (live predictions).

Pick a supplier from the dropdown and everything updates:
risk score, delay probability, charts, top risk factors,
prediction confidence, and recommended actions.
"""

from __future__ import annotations  # Modern type hint support

import streamlit as st  # Website UI library

from app.views.base_page import BasePage  # Shared page template
from src.services.prediction_service import (
    SupplierPrediction,
    SupplierPredictionService,
)


@st.cache_resource(show_spinner="Loading model predictions...")
def _load_prediction_service() -> SupplierPredictionService:
    """Load the model + data once, then reuse it (keeps the page fast)."""
    service = SupplierPredictionService()
    service.load()
    return service


class DashboardPage(BasePage):
    """Live dashboard driven by the trained disruption model."""

    @property
    def title(self) -> str:
        return "Dashboard"  # Page name shown in the header

    @property
    def icon(self) -> str:
        return "📊"  # Chart emoji shown next to the title

    def render_content(self) -> None:
        service = SupplierPredictionService()

        # No trained model yet → tell the user exactly what to run
        if not service.model_available:
            st.warning(
                "No trained model found. Run these two commands, then refresh:\n\n"
                "```bash\n"
                "python3 scripts/run_feature_engineering.py\n"
                "python3 scripts/train_model.py\n"
                "```"
            )
            return

        try:
            service = _load_prediction_service()
        except Exception as exc:
            st.error("Could not load live predictions.")
            st.exception(exc)
            return

        supplier_ids = service.get_supplier_ids()
        if not supplier_ids:
            st.info("No supplier data available yet.")
            return

        # ---- Supplier dropdown (everything below reacts to this) ----
        st.markdown("#### Choose a supplier")
        selected_id = st.selectbox(
            "Supplier",
            options=supplier_ids,
            format_func=service.get_supplier_label,
            key="dashboard_supplier",
            label_visibility="collapsed",
        )

        prediction = service.get_supplier_prediction(selected_id)

        # Small caption showing which supplier + date is being shown
        st.caption(
            f"**{prediction.supplier_name}** · {prediction.region} · "
            f"{prediction.category} · data as of {prediction.as_of_date}"
        )

        self._render_headline_banner(prediction)
        self._render_kpi_row(prediction)
        self._render_gauges(service, prediction)
        self._render_history_charts(service, selected_id)
        self._render_risk_factors(service, selected_id)
        self._render_recommended_actions(service, prediction)
        self._render_portfolio_section(service)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _render_headline_banner(self, p: SupplierPrediction) -> None:
        """Big colored message with the model's main answer."""
        message = (
            f"**{p.prediction_label}** — about a "
            f"**{p.delay_probability:.0%} chance** of a shipment disruption "
            f"(model confidence {p.confidence:.0%})."
        )
        if p.delay_probability >= 0.7:
            st.error(message, icon="🚨")
        elif p.delay_probability >= 0.4:
            st.warning(message, icon="⚠️")
        else:
            st.success(message, icon="✅")

    def _render_kpi_row(self, p: SupplierPrediction) -> None:
        """Four live KPI cards for the selected supplier."""
        cols = st.columns(4)

        with cols[0]:
            st.metric("Risk Score", f"{p.risk_score:.0f} /100")
        with cols[1]:
            st.metric("Lead Time", f"{p.lead_time_days:.0f} days")
        with cols[2]:
            st.metric("Inventory Coverage", f"{p.inventory_coverage_days:.0f} days")
        with cols[3]:
            st.metric("Sentiment Score", f"{p.sentiment_score:.2f}")

    def _render_gauges(
        self, service: SupplierPredictionService, p: SupplierPrediction
    ) -> None:
        """Risk gauge, delay probability gauge, and confidence details."""
        st.markdown("### Model Prediction")
        left, middle, right = st.columns([1, 1, 1])

        with left:
            st.plotly_chart(
                self.charts.risk_gauge(p.risk_score),
                use_container_width=True,
            )

        with middle:
            st.plotly_chart(
                self.charts.delay_probability_gauge(p.delay_probability),
                use_container_width=True,
            )

        with right:
            st.markdown("#### Prediction Confidence")
            # Progress bar makes confidence easy to read at a glance
            st.progress(min(max(p.confidence, 0.0), 1.0))
            st.metric("Confidence", f"{p.confidence:.0%}")

            metrics = service.get_model_metrics()
            if metrics:
                st.caption(
                    "Model test scores — "
                    f"Accuracy {metrics.get('accuracy', 0):.0%}, "
                    f"F1 {metrics.get('f1', 0):.2f}, "
                    f"ROC-AUC {metrics.get('roc_auc', 0):.2f}"
                )
            st.caption(
                "Confidence means how sure the model is about this answer, "
                "not how bad the disruption would be."
            )

    def _render_history_charts(
        self, service: SupplierPredictionService, supplier_id: str
    ) -> None:
        """Historical lead time, sentiment, inventory, and delay probability."""
        st.markdown("### History & Trends")

        lead_df = service.get_lead_time_history(supplier_id)
        sentiment_df = service.get_sentiment_history(supplier_id)
        inventory_df = service.get_inventory_history(supplier_id)
        delay_df = service.get_delay_probability_history(supplier_id)

        col_left, col_right = st.columns(2)

        with col_left:
            if lead_df.empty:
                st.info("No lead time history for this supplier yet.")
            else:
                st.plotly_chart(
                    self.charts.lead_time_history(lead_df),
                    use_container_width=True,
                )

            if inventory_df.empty:
                st.info("No inventory history for this supplier yet.")
            else:
                st.plotly_chart(
                    self.charts.inventory_history(inventory_df),
                    use_container_width=True,
                )

        with col_right:
            if sentiment_df.empty:
                st.info(
                    "No news sentiment for this supplier yet. "
                    "Run `python3 scripts/run_sentiment.py` to add news data."
                )
            else:
                st.plotly_chart(
                    self.charts.sentiment_history(sentiment_df),
                    use_container_width=True,
                )

            if delay_df.empty:
                st.info("No prediction history for this supplier yet.")
            else:
                st.plotly_chart(
                    self.charts.delay_probability_history(delay_df),
                    use_container_width=True,
                )

    def _render_risk_factors(
        self, service: SupplierPredictionService, supplier_id: str
    ) -> None:
        """Top factors the model used for this supplier (SHAP based)."""
        st.markdown("### Top Risk Factors")
        st.caption(
            "Red bars pushed the risk up. Blue bars pushed it down. "
            "These come from the model itself (SHAP), not fixed rules."
        )
        try:
            factors_df = service.get_top_risk_factors(supplier_id)
            st.plotly_chart(
                self.charts.top_risk_factors(factors_df),
                use_container_width=True,
            )
        except Exception as exc:
            st.info("Risk factor explanations are unavailable right now.")
            st.caption(str(exc))

    def _render_recommended_actions(
        self, service: SupplierPredictionService, p: SupplierPrediction
    ) -> None:
        """Plain-language next steps based on the prediction."""
        st.markdown("### Recommended Actions")
        actions = service.get_recommended_actions(p)

        if not actions:
            st.success("No action needed right now.")
            return

        for action in actions:
            level = action["level"]
            text = action["text"]
            if level == "urgent":
                st.error(text, icon="🚨")
            elif level == "watch":
                st.warning(text, icon="👀")
            else:
                st.success(text, icon="✅")

    def _render_portfolio_section(self, service: SupplierPredictionService) -> None:
        """All-supplier summary and a ranked table of riskiest suppliers."""
        st.markdown("---")
        st.markdown("### All Suppliers (live model output)")

        summary = service.get_portfolio_summary()
        cols = st.columns(4)
        with cols[0]:
            st.metric("Suppliers Tracked", summary["supplier_count"])
        with cols[1]:
            st.metric(
                "Avg Delay Probability",
                f"{float(summary['avg_delay_probability']):.0%}",
            )
        with cols[2]:
            st.metric("High-Risk Suppliers", summary["high_risk_count"])
        with cols[3]:
            st.metric("Avg Inventory", f"{summary['avg_inventory']:.0f} days")

        table = service.get_all_predictions_dataframe()
        if table.empty:
            return

        display = table[
            [
                "supplier_id",
                "supplier_name",
                "region",
                "risk_score",
                "delay_probability",
                "confidence",
                "lead_time_days",
                "inventory_coverage_days",
            ]
        ].rename(
            columns={
                "supplier_id": "Supplier ID",
                "supplier_name": "Name",
                "region": "Region",
                "risk_score": "Risk Score",
                "delay_probability": "Delay Probability",
                "confidence": "Confidence",
                "lead_time_days": "Lead Time (days)",
                "inventory_coverage_days": "Inventory (days)",
            }
        )
        # Show probabilities as easy-to-read percentages
        display["Delay Probability"] = (display["Delay Probability"] * 100).round(1)
        display["Confidence"] = (display["Confidence"] * 100).round(1)

        st.dataframe(display, use_container_width=True, hide_index=True)
