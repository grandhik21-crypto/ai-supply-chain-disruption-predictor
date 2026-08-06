"""
The AI Model page.

Shows how the prediction model works, how accurate it is,
and SHAP explanations in plain language (why the AI decided).
"""

from __future__ import annotations  # Modern type hint support

import streamlit as st  # Website UI library

# Import the shared page template
from app.views.base_page import BasePage
from src.ml.explainability import ShapExplainer
from src.ml.model import DEFAULT_MODEL_PATH, DisruptionPredictor


@st.cache_resource(show_spinner="Computing SHAP explanations...")
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

    def render_content(self) -> None:
        # Short intro text at the top of the page
        st.markdown(
            "This page shows how the disruption model works — "
            "including **simple explanations** of why it makes a prediction."
        )

        # Show real metrics from the saved model when available
        self._render_model_metrics()

        st.markdown("---")
        st.markdown("### Explainable AI (SHAP)")
        st.caption(
            "SHAP shows which factors pushed a prediction toward "
            "**higher risk** (red) or **lower risk** (blue)."
        )

        # Helpful tip box for non-technical users
        with st.expander("How to read these charts (simple guide)", expanded=False):
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
            st.warning(
                "No trained model found yet. Train it first, then refresh this page:\n\n"
                "`python3 scripts/train_model.py`"
            )
            self._render_fallback_charts()
            return

        try:
            explainer = self._get_explainer()
            self._render_shap_section(explainer)
        except Exception as exc:
            st.error("Could not build SHAP explanations.")
            st.exception(exc)
            self._render_fallback_charts()
            return

        st.markdown("---")
        st.markdown("### Forecast Output")
        st.plotly_chart(
            self.charts.disruption_forecast(
                self.analytics.get_disruption_probability_forecast()
            ),
            use_container_width=True,
        )

        st.markdown("### Model Notes")
        st.info(
            "Explanations below come from the trained XGBoost model using SHAP. "
            "Demo labels are used until real disruption history is available.",
            icon="ℹ️",
        )

    def _render_model_metrics(self) -> None:
        """Show accuracy-style metrics from the saved model file if present."""
        metric_cols = st.columns(4)
        defaults = [
            ("Model", "XGBoost"),
            ("Accuracy", "—"),
            ("F1", "—"),
            ("ROC-AUC", "—"),
        ]

        if DEFAULT_MODEL_PATH.exists():
            try:
                predictor = DisruptionPredictor()
                predictor.load_model()
                import joblib

                payload = joblib.load(DEFAULT_MODEL_PATH)
                saved = payload.get("metrics", {})
                defaults = [
                    ("Model", "XGBoost"),
                    ("Accuracy", f"{saved.get('accuracy', 0):.2f}"),
                    ("F1", f"{saved.get('f1', 0):.2f}"),
                    ("ROC-AUC", f"{saved.get('roc_auc', 0):.2f}"),
                ]
            except Exception:
                pass

        for col, (label, value) in zip(metric_cols, defaults):
            with col:
                st.metric(label=label, value=value)

    @st.cache_resource(show_spinner="Computing SHAP explanations...")
    def _get_explainer(_self) -> ShapExplainer:
        """Load SHAP explainer once and reuse it (faster page reloads)."""
        explainer = ShapExplainer()
        explainer.load()
        return explainer

    def _render_shap_section(self, explainer: ShapExplainer) -> None:
        """Draw Feature Importance, Summary, and Waterfall charts."""
        # 1) Feature importance
        st.markdown("#### 1) Feature Importance")
        st.caption("Bigger bar = this factor has a stronger average effect on predictions.")
        st.plotly_chart(
            explainer.plot_feature_importance(),
            use_container_width=True,
        )

        # 2) SHAP summary
        st.markdown("#### 2) SHAP Summary")
        st.caption(
            "Each dot is one shipment example. "
            "Dots to the right made the model more worried about disruption."
        )
        st.plotly_chart(
            explainer.plot_shap_summary(),
            use_container_width=True,
        )

        # 3) SHAP waterfall for one selected example
        st.markdown("#### 3) SHAP Waterfall (one example)")
        st.caption("Pick a supplier/date to see why the model scored it that way.")

        labels = explainer.sample_labels
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
            f"**Top reasons:** " + "; ".join(summary["top_drivers"])
        )

        st.plotly_chart(
            explainer.plot_shap_waterfall(selected),
            use_container_width=True,
        )

        # Also keep confusion matrix as a small secondary chart
        st.markdown("#### Extra: Confusion Matrix")
        st.plotly_chart(
            self.charts.confusion_matrix_heatmap(
                self.analytics.get_confusion_matrix()
            ),
            use_container_width=True,
        )

    def _render_fallback_charts(self) -> None:
        """Show older placeholder charts if SHAP/model is unavailable."""
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(
                self.charts.feature_importance(self.analytics.get_feature_importance()),
                use_container_width=True,
            )
        with col_right:
            st.plotly_chart(
                self.charts.confusion_matrix_heatmap(
                    self.analytics.get_confusion_matrix()
                ),
                use_container_width=True,
            )
