"""Model insights and explainability page."""

from __future__ import annotations

import streamlit as st

from app.pages.base_page import BasePage


class ModelInsightsPage(BasePage):
    """Machine learning model performance and explainability."""

    @property
    def title(self) -> str:
        return "Model Insights"

    @property
    def icon(self) -> str:
        return "🤖"

    def render_content(self) -> None:
        st.markdown(
            "Explore the disruption prediction model's performance, "
            "key drivers, and forecast outputs."
        )

        metric_cols = st.columns(4)
        model_metrics = [
            ("Model Version", "v0.1.0-placeholder"),
            ("AUC-ROC", "0.87"),
            ("Precision", "0.82"),
            ("Recall", "0.79"),
        ]
        for col, (label, value) in zip(metric_cols, model_metrics):
            with col:
                st.metric(label=label, value=value)

        st.markdown("---")

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

        st.markdown("### Forecast Output")
        forecast_df = self.analytics.get_disruption_probability_forecast()
        st.plotly_chart(
            self.charts.disruption_forecast(forecast_df),
            use_container_width=True,
        )

        st.markdown("### Model Notes")
        st.info(
            "**Placeholder model.** Production deployment will integrate NLP sentiment "
            "analysis, time-series forecasting, and gradient-boosted classifiers trained "
            "on historical disruption events.",
            icon="ℹ️",
        )
