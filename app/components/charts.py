"""
Makes all the charts you see on the website.

Builds line charts, bar charts, scatter plots, and heatmaps
used on the Dashboard, Supplier, and Model pages.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ChartFactory:
    """Creates consistent Plotly figures for the application."""

    _COLOR_PRIMARY = "#2563eb"
    _COLOR_SECONDARY = "#7c3aed"
    _COLOR_WARNING = "#f59e0b"
    _COLOR_SUCCESS = "#10b981"
    _TEMPLATE = "plotly_white"

    def risk_trend_line(self, df: pd.DataFrame) -> go.Figure:
        """Line chart for monthly risk score trend."""
        fig = px.line(
            df,
            x="Month",
            y="Risk Score",
            markers=True,
            title="Risk Score Trend (12 Months)",
            color_discrete_sequence=[self._COLOR_PRIMARY],
        )
        return self._apply_layout(fig, y_title="Risk Score")

    def lead_time_bar(self, df: pd.DataFrame) -> go.Figure:
        """Bar chart for average lead time by region."""
        fig = px.bar(
            df,
            x="Region",
            y="Avg Lead Time (days)",
            title="Average Lead Time by Region",
            color="Avg Lead Time (days)",
            color_continuous_scale=["#93c5fd", self._COLOR_PRIMARY],
        )
        fig.update_layout(coloraxis_showscale=False)
        return self._apply_layout(fig, y_title="Days")

    def inventory_area(self, df: pd.DataFrame) -> go.Figure:
        """Area chart for inventory coverage over time."""
        fig = px.area(
            df,
            x="Week",
            y="Inventory Coverage (days)",
            title="Inventory Coverage Trend",
            color_discrete_sequence=[self._COLOR_SUCCESS],
        )
        return self._apply_layout(fig, y_title="Days")

    def sentiment_line(self, df: pd.DataFrame) -> go.Figure:
        """Line chart for sentiment score timeline."""
        fig = px.line(
            df,
            x="Date",
            y="Sentiment Score",
            title="News & Market Sentiment Timeline",
            color_discrete_sequence=[self._COLOR_SECONDARY],
        )
        fig.update_yaxes(range=[0, 1])
        return self._apply_layout(fig, y_title="Score (0–1)")

    def disruption_forecast(self, df: pd.DataFrame) -> go.Figure:
        """Bar chart for disruption probability forecast."""
        fig = px.bar(
            df,
            x="Month",
            y="Disruption Probability",
            title="Disruption Probability Forecast",
            color_discrete_sequence=[self._COLOR_WARNING],
        )
        return self._apply_layout(fig, y_title="Probability (%)")

    def feature_importance(self, df: pd.DataFrame) -> go.Figure:
        """Horizontal bar chart for model feature importance."""
        fig = px.bar(
            df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Model Feature Importance",
            color="Importance",
            color_continuous_scale=["#c4b5fd", self._COLOR_SECONDARY],
        )
        fig.update_layout(coloraxis_showscale=False)
        return self._apply_layout(fig, x_title="Relative Importance")

    def supplier_risk_scatter(self, df: pd.DataFrame) -> go.Figure:
        """Scatter plot of supplier risk vs lead time."""
        fig = px.scatter(
            df,
            x="Lead Time (days)",
            y="Risk Score",
            color="Region",
            size="On-Time Delivery (%)",
            hover_name="Name",
            title="Supplier Risk vs Lead Time",
        )
        return self._apply_layout(fig)

    def confusion_matrix_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Heatmap for model confusion matrix."""
        fig = px.imshow(
            df.values,
            x=df.columns.tolist(),
            y=df.index.tolist(),
            text_auto=True,
            color_continuous_scale="Blues",
            title="Model Confusion Matrix",
        )
        fig.update_layout(
            template=self._TEMPLATE,
            height=380,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    def _apply_layout(
        self,
        fig: go.Figure,
        *,
        x_title: str | None = None,
        y_title: str | None = None,
    ) -> go.Figure:
        """Apply consistent styling to a Plotly figure."""
        fig.update_layout(
            template=self._TEMPLATE,
            height=380,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        if x_title:
            fig.update_xaxes(title_text=x_title)
        if y_title:
            fig.update_yaxes(title_text=y_title)
        return fig
