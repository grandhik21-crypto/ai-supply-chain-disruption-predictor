"""
The AI Model page.

Shows how the prediction model works: which factors matter most,
how accurate it is, and future disruption forecasts.
"""

from __future__ import annotations  # Modern type hint support

import streamlit as st  # Website UI library

# Import the shared page template
from app.pages.base_page import BasePage


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
            "Explore the disruption prediction model's performance, "
            "key drivers, and forecast outputs."
        )

        metric_cols = st.columns(4)  # Four columns for model stats
        # Demo model performance numbers (placeholder until real model exists)
        model_metrics = [
            ("Model Version", "v0.1.0-placeholder"),
            ("AUC-ROC", "0.87"),
            ("Precision", "0.82"),
            ("Recall", "0.79"),
        ]
        # Show each model stat in its own column
        for col, (label, value) in zip(metric_cols, model_metrics):
            with col:
                st.metric(label=label, value=value)

        st.markdown("---")  # Divider between stats and charts

        col_left, col_right = st.columns(2)  # Two chart columns

        with col_left:
            # Horizontal bar chart: which inputs matter most to the model
            st.plotly_chart(
                self.charts.feature_importance(self.analytics.get_feature_importance()),
                use_container_width=True,
            )

        with col_right:
            # Heatmap: how often the model was right vs wrong
            st.plotly_chart(
                self.charts.confusion_matrix_heatmap(
                    self.analytics.get_confusion_matrix()
                ),
                use_container_width=True,
            )

        st.markdown("### Forecast Output")  # Section heading
        # Get future disruption probability data
        forecast_df = self.analytics.get_disruption_probability_forecast()
        # Bar chart showing forecast by month
        st.plotly_chart(
            self.charts.disruption_forecast(forecast_df),
            use_container_width=True,
        )

        st.markdown("### Model Notes")  # Section heading
        # Blue info box explaining this is demo/placeholder data
        st.info(
            "**Placeholder model.** Production deployment will integrate NLP sentiment "
            "analysis, time-series forecasting, and gradient-boosted classifiers trained "
            "on historical disruption events.",
            icon="ℹ️",
        )
