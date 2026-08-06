"""
The home page of the website (live predictions).

Pick a supplier from the dropdown and everything updates:
risk score, delay probability, charts, top risk factors,
prediction confidence, recommended actions, and a downloadable report.
"""

from __future__ import annotations  # Modern type hint support

from datetime import datetime  # Timestamp for report filenames

import streamlit as st  # Website UI library

from app.components.ui import (
    empty_state,
    render_chart,
    render_dataframe,
    risk_badge,
    safe_section,
    section_header,
)
from app.views.base_page import BasePage  # Shared page template
from src.services.prediction_service import (
    SupplierPrediction,
    SupplierPredictionService,
)
from src.services.report_service import (
    build_all_predictions_csv,
    build_prediction_report_csv,
    build_prediction_report_markdown,
)


@st.cache_resource(show_spinner=False)
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

    @property
    def subtitle(self) -> str:
        return "Live disruption predictions for each supplier"

    def render_content(self) -> None:
        service = SupplierPredictionService()

        # No trained model yet → show a helpful empty state instead of an error
        if not service.model_available:
            empty_state(
                title="No trained model yet",
                message=(
                    "The dashboard needs a trained model before it can show predictions."
                ),
                icon="🤖",
                hint="python3 scripts/run_feature_engineering.py &amp;&amp; python3 scripts/train_model.py",
            )
            return

        # Spinner while the model + SHAP data load (first run takes longest)
        with st.spinner("Loading model predictions…"):
            try:
                service = _load_prediction_service()
            except Exception as exc:
                st.error("Could not load live predictions.", icon="🚨")
                with st.expander("Technical details (for developers)"):
                    st.exception(exc)
                return

        supplier_ids = service.get_supplier_ids()
        if not supplier_ids:
            empty_state(
                title="No supplier data found",
                message="The features file loaded, but it has no supplier rows.",
                icon="📭",
                hint="python3 scripts/run_feature_engineering.py",
            )
            return

        # ---- Supplier dropdown (everything below reacts to this) ----
        section_header("🔍", "Select a supplier", "All panels below update automatically.")
        selected_id = st.selectbox(
            "Supplier",
            options=supplier_ids,
            format_func=service.get_supplier_label,
            key="dashboard_supplier",
            label_visibility="collapsed",
        )

        with st.spinner("Scoring this supplier…"):
            prediction = service.get_supplier_prediction(selected_id)

        # Supplier context line with a colored risk badge
        st.markdown(
            f"**{prediction.supplier_name}** &nbsp;·&nbsp; 🌍 {prediction.region} "
            f"&nbsp;·&nbsp; 📦 {prediction.category} &nbsp;·&nbsp; "
            f"🗓️ data as of {prediction.as_of_date} &nbsp; "
            f"{risk_badge(prediction.delay_probability)}",
            unsafe_allow_html=True,
        )

        self._render_headline_banner(prediction)
        self._render_kpi_row(prediction)

        with safe_section("Model prediction gauges"):
            self._render_gauges(service, prediction)

        with safe_section("History and trends"):
            self._render_history_charts(service, selected_id)

        with safe_section("Top risk factors"):
            self._render_risk_factors(service, selected_id)

        with safe_section("Recommended actions"):
            self._render_recommended_actions(service, prediction)

        with safe_section("Prediction report download"):
            self._render_report_download(service, prediction, selected_id)

        with safe_section("All suppliers overview"):
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
        section_header("📌", "Key numbers", "The four headline measures for this supplier.")
        cols = st.columns(4)

        with cols[0]:
            st.metric("⚠️ Risk Score", f"{p.risk_score:.0f} /100")
        with cols[1]:
            st.metric("🚚 Lead Time", f"{p.lead_time_days:.0f} days")
        with cols[2]:
            st.metric("📦 Inventory Coverage", f"{p.inventory_coverage_days:.0f} days")
        with cols[3]:
            st.metric("📰 Sentiment Score", f"{p.sentiment_score:.2f}")

    def _render_gauges(
        self, service: SupplierPredictionService, p: SupplierPrediction
    ) -> None:
        """Risk gauge, delay probability gauge, and confidence details."""
        section_header("🎯", "Model prediction", "Gauges show the current risk picture.")
        left, middle, right = st.columns([1, 1, 1])

        with left:
            render_chart(
                self.charts.risk_gauge(p.risk_score),
                key=f"risk_gauge_{p.supplier_id}",
                filename=f"{p.supplier_id}_risk_score",
            )

        with middle:
            render_chart(
                self.charts.delay_probability_gauge(p.delay_probability),
                key=f"delay_gauge_{p.supplier_id}",
                filename=f"{p.supplier_id}_delay_probability",
            )

        with right:
            st.markdown("**🎚️ Prediction Confidence**")
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
        section_header("📈", "History & trends", "How this supplier changed over time.")

        with st.spinner("Building history charts…"):
            lead_df = service.get_lead_time_history(supplier_id)
            sentiment_df = service.get_sentiment_history(supplier_id)
            inventory_df = service.get_inventory_history(supplier_id)
            delay_df = service.get_delay_probability_history(supplier_id)

        col_left, col_right = st.columns(2)

        with col_left:
            if lead_df.empty:
                empty_state(
                    title="No lead time history",
                    message="This supplier has no delivery-time records yet.",
                    icon="🚚",
                )
            else:
                render_chart(
                    self.charts.lead_time_history(lead_df),
                    key=f"lead_{supplier_id}",
                    filename=f"{supplier_id}_lead_time_history",
                )

            if inventory_df.empty:
                empty_state(
                    title="No inventory history",
                    message="No stock coverage records for this supplier yet.",
                    icon="📦",
                )
            else:
                render_chart(
                    self.charts.inventory_history(inventory_df),
                    key=f"inv_{supplier_id}",
                    filename=f"{supplier_id}_inventory_coverage",
                )

        with col_right:
            if sentiment_df.empty:
                empty_state(
                    title="No news sentiment yet",
                    message="No news articles have been analyzed for this supplier.",
                    icon="📰",
                    hint="python3 scripts/run_sentiment.py",
                )
            else:
                render_chart(
                    self.charts.sentiment_history(sentiment_df),
                    key=f"sent_{supplier_id}",
                    filename=f"{supplier_id}_sentiment_trend",
                )

            if delay_df.empty:
                empty_state(
                    title="No prediction history",
                    message="Not enough data yet to chart predictions over time.",
                    icon="🔮",
                )
            else:
                render_chart(
                    self.charts.delay_probability_history(delay_df),
                    key=f"delayhist_{supplier_id}",
                    filename=f"{supplier_id}_delay_probability_history",
                )

    def _render_risk_factors(
        self, service: SupplierPredictionService, supplier_id: str
    ) -> None:
        """Top factors the model used for this supplier (SHAP based)."""
        section_header(
            "🧠",
            "Top risk factors",
            "Red bars pushed the risk up. Blue bars pushed it down. "
            "These come from the model itself (SHAP), not fixed rules.",
        )
        with st.spinner("Explaining this prediction…"):
            factors_df = service.get_top_risk_factors(supplier_id)

        if factors_df.empty:
            empty_state(
                title="No explanation available",
                message="The model could not produce risk factors for this row.",
                icon="🧠",
            )
            return

        render_chart(
            self.charts.top_risk_factors(factors_df),
            key=f"factors_{supplier_id}",
            filename=f"{supplier_id}_risk_factors",
        )

    def _render_recommended_actions(
        self, service: SupplierPredictionService, p: SupplierPrediction
    ) -> None:
        """Plain-language next steps based on the prediction."""
        section_header("✅", "Recommended actions", "Suggested next steps for your team.")
        actions = service.get_recommended_actions(p)

        if not actions:
            st.success("No action needed right now.", icon="✅")
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

    def _render_report_download(
        self,
        service: SupplierPredictionService,
        p: SupplierPrediction,
        supplier_id: str,
    ) -> None:
        """Let the user download a shareable report for this supplier."""
        section_header(
            "📥",
            "Download prediction report",
            "Save or share this supplier's prediction as a file.",
        )

        actions = service.get_recommended_actions(p)
        try:
            factors = service.get_top_risk_factors(supplier_id)
        except Exception:
            factors = None
        metrics = service.get_model_metrics()

        report_md = build_prediction_report_markdown(p, actions, factors, metrics)
        report_csv = build_prediction_report_csv(p, factors)
        stamp = datetime.now().strftime("%Y%m%d")

        col_md, col_csv = st.columns(2)
        with col_md:
            st.download_button(
                "📄 Download report (Markdown)",
                data=report_md.encode("utf-8"),
                file_name=f"{supplier_id}_disruption_report_{stamp}.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"report_md_{supplier_id}",
            )
        with col_csv:
            st.download_button(
                "🧾 Download report (CSV)",
                data=report_csv.encode("utf-8"),
                file_name=f"{supplier_id}_disruption_report_{stamp}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"report_csv_{supplier_id}",
            )

        with st.expander("Preview the report"):
            st.markdown(report_md)

    def _render_portfolio_section(self, service: SupplierPredictionService) -> None:
        """All-supplier summary and a ranked table of riskiest suppliers."""
        section_header(
            "🏭",
            "All suppliers",
            "Live model output across every supplier being tracked.",
        )

        with st.spinner("Scoring all suppliers…"):
            summary = service.get_portfolio_summary()
            table = service.get_all_predictions_dataframe()

        cols = st.columns(4)
        with cols[0]:
            st.metric("🏭 Suppliers Tracked", summary["supplier_count"])
        with cols[1]:
            st.metric(
                "🔮 Avg Delay Probability",
                f"{float(summary['avg_delay_probability']):.0%}",
            )
        with cols[2]:
            st.metric("🚨 High-Risk Suppliers", summary["high_risk_count"])
        with cols[3]:
            st.metric("📦 Avg Inventory", f"{summary['avg_inventory']:.0f} days")

        if table.empty:
            empty_state(
                title="No predictions to show",
                message="No supplier predictions were produced.",
                icon="📭",
            )
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

        render_dataframe(
            display,
            key="all_suppliers",
            filename="all_supplier_predictions",
            column_config={
                "Delay Probability": st.column_config.ProgressColumn(
                    "Delay Probability",
                    help="Model's estimated chance of disruption",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Risk Score": st.column_config.NumberColumn(
                    "Risk Score", format="%.0f"
                ),
            },
        )

        # Full export of every column (not just the display columns)
        st.download_button(
            "📥 Download full prediction data (CSV)",
            data=build_all_predictions_csv(table).encode("utf-8"),
            file_name="all_supplier_predictions_full.csv",
            mime="text/csv",
            key="all_predictions_full_csv",
        )
