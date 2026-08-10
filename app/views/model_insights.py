"""
The AI Model page.

Shows how the prediction model works, how accurate it is,
and SHAP explanations in plain language (why the AI decided).
"""

from __future__ import annotations  # Modern type hint support

import streamlit as st  # Website UI library

from app.components.ui import (
    empty_state,
    render_chart,
    safe_section,
    section_header,
)
# Import the shared page template
from app.views.base_page import BasePage
from src.ml.explainability import ShapExplainer
from src.ml.model import DEFAULT_MODEL_PATH


@st.cache_resource(show_spinner=False)
def _load_shap_explainer() -> ShapExplainer:
    """Load the SHAP explainer once (keeps the website fast)."""
    explainer = ShapExplainer()
    explainer.load()
    return explainer


class ModelInsightsPage(BasePage):
    """Machine learning model performance and explainability."""

    @property
    def title(self) -> str:
        return "Model Insights"  # Page name in the header

    @property
    def icon(self) -> str:
        return "🤖"  # Robot emoji for AI model

    @property
    def subtitle(self) -> str:
        return "See how the model decides, explained in plain language"

    def render_content(self) -> None:
        # Show real metrics from the saved model when available
        self._render_model_metrics()

        section_header(
            "🔎",
            "Explainable AI (SHAP)",
            "SHAP shows which factors pushed a prediction toward higher risk (red) "
            "or lower risk (blue).",
        )

        # Helpful tip box for non-technical users
        with st.expander("📚 How to read these charts (simple guide)", expanded=False):
            st.markdown(
                """
                - **Feature Importance:** which inputs matter most overall  
                - **SHAP Summary:** each dot is one example; left = safer, right = riskier  
                - **SHAP Waterfall:** for one supplier/day, which factors raised or lowered risk  

                Colors:
                - **Red** → made the prediction more risky  
                - **Blue** → made the prediction safer  
                """
            )

        if not DEFAULT_MODEL_PATH.exists():
            empty_state(
                title="No trained model yet",
                message="Train the model first, then refresh this page.",
                icon="🤖",
                hint="python3 scripts/train_model.py",
            )
            self._render_fallback_charts()
            return

        with st.spinner("Computing SHAP explanations…"):
            try:
                explainer = _load_shap_explainer()
            except Exception as exc:
                st.error("Could not build SHAP explanations.", icon="🚨")
                with st.expander("Technical details (for developers)"):
                    st.exception(exc)
                self._render_fallback_charts()
                return

        with safe_section("SHAP explanations"):
            self._render_shap_section(explainer)

        with safe_section("Forecast output"):
            section_header("🔮", "Forecast output", "Predicted disruption risk by month.")
            render_chart(
                self.charts.disruption_forecast(
                    self.analytics.get_disruption_probability_forecast()
                ),
                key="forecast",
                filename="disruption_forecast",
            )

        st.info(
            "Explanations come from the trained XGBoost model using SHAP. "
            "Demo labels are used until real disruption history is available.",
            icon="ℹ️",
        )

    def _render_model_metrics(self) -> None:
        """Show accuracy-style metrics from the saved model file if present."""
        section_header("📏", "Model quality", "How well the model scored on test data.")
        metric_cols = st.columns(4)
        defaults = [
            ("🧩 Model", "XGBoost"),
            ("🎯 Accuracy", "—"),
            ("⚖️ F1", "—"),
            ("📈 ROC-AUC", "—"),
        ]

        if DEFAULT_MODEL_PATH.exists():
            try:
                import joblib

                payload = joblib.load(DEFAULT_MODEL_PATH)
                saved = payload.get("metrics", {})
                defaults = [
                    ("🧩 Model", "XGBoost"),
                    ("🎯 Accuracy", f"{saved.get('accuracy', 0):.2f}"),
                    ("⚖️ F1", f"{saved.get('f1', 0):.2f}"),
                    ("📈 ROC-AUC", f"{saved.get('roc_auc', 0):.2f}"),
                ]
            except Exception:
                pass

        for col, (label, value) in zip(metric_cols, defaults):
            with col:
                st.metric(label=label, value=value)

    def _render_shap_section(self, explainer: ShapExplainer) -> None:
        """Draw Feature Importance, Summary, and Waterfall charts."""
        # 1) Feature importance
        section_header(
            "1️⃣",
            "Feature importance",
            "Bigger bar = this factor has a stronger average effect on predictions.",
        )
        render_chart(
            explainer.plot_feature_importance(),
            key="shap_importance",
            filename="shap_feature_importance",
        )

        # 2) SHAP summary
        section_header(
            "2️⃣",
            "SHAP summary",
            "Each dot is one shipment example. Dots to the right made the model "
            "more worried about disruption.",
        )
        render_chart(
            explainer.plot_shap_summary(),
            key="shap_summary",
            filename="shap_summary",
        )

        # 3) SHAP waterfall for one selected example
        section_header(
            "3️⃣",
            "SHAP waterfall (one example)",
            "Pick a supplier/date to see why the model scored it that way.",
        )

        labels = explainer.sample_labels
        if not labels:
            empty_state(
                title="No examples to explain",
                message="No rows were available for a single-prediction explanation.",
                icon="🔎",
            )
            return

        selected = st.selectbox(
            "Choose an example to explain",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            key="shap_example_index",
        )

        summary = explainer.prediction_summary(selected)
        st.success(
            f"**Prediction:** {summary['prediction_label']}  \n"
            f"**Estimated chance of disruption:** {summary['probability']:.0%}  \n"
            f"**Top reasons:** " + "; ".join(summary["top_drivers"]),
            icon="🧠",
        )

        render_chart(
            explainer.plot_shap_waterfall(selected),
            key=f"shap_waterfall_{selected}",
            filename="shap_waterfall",
        )

        # Also keep confusion matrix as a small secondary chart
        section_header(
            "🧮",
            "Confusion matrix",
            "How often the model was right vs wrong on test data.",
        )
        render_chart(
            self.charts.confusion_matrix_heatmap(
                self.analytics.get_confusion_matrix()
            ),
            key="confusion_matrix",
            filename="confusion_matrix",
        )

    def _render_fallback_charts(self) -> None:
        """Show simpler charts if SHAP/model is unavailable."""
        col_left, col_right = st.columns(2)
        with col_left:
            render_chart(
                self.charts.feature_importance(self.analytics.get_feature_importance()),
                key="fallback_importance",
                filename="feature_importance",
            )
        with col_right:
            render_chart(
                self.charts.confusion_matrix_heatmap(
                    self.analytics.get_confusion_matrix()
                ),
                key="fallback_confusion",
                filename="confusion_matrix",
            )
