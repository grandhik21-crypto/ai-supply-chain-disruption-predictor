"""
Makes all the charts you see on the website.

Builds line charts, bar charts, scatter plots, and heatmaps
used on the Dashboard, Supplier, and Model pages.
"""

from __future__ import annotations  # Modern type hint support

import pandas as pd  # Table data library
import plotly.express as px  # Easy chart builder
import plotly.graph_objects as go  # Lower-level chart objects


class ChartFactory:
    """Creates consistent Plotly figures for the application."""

    _COLOR_PRIMARY = "#2563eb"  # Main blue color for charts
    _COLOR_SECONDARY = "#7c3aed"  # Purple accent color
    _COLOR_WARNING = "#f59e0b"  # Orange for warnings/forecasts
    _COLOR_SUCCESS = "#10b981"  # Green for positive metrics
    _TEMPLATE = "plotly_white"  # Clean white chart background style

    def risk_trend_line(self, df: pd.DataFrame) -> go.Figure:
        """Line chart for monthly risk score trend."""
        # Build a line chart with dots at each month
        fig = px.line(
            df,
            x="Month",  # X axis = time
            y="Risk Score",  # Y axis = risk value
            markers=True,  # Show a dot on each data point
            title="Risk Score Trend (12 Months)",
            color_discrete_sequence=[self._COLOR_PRIMARY],
        )
        # Apply shared styling and return the chart
        return self._apply_layout(fig, y_title="Risk Score")

    def lead_time_bar(self, df: pd.DataFrame) -> go.Figure:
        """Bar chart for average lead time by region."""
        fig = px.bar(
            df,
            x="Region",
            y="Avg Lead Time (days)",
            title="Average Lead Time by Region",
            color="Avg Lead Time (days)",  # Bar color based on value
            color_continuous_scale=["#93c5fd", self._COLOR_PRIMARY],
        )
        fig.update_layout(coloraxis_showscale=False)  # Hide color legend bar
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
        fig.update_yaxes(range=[0, 1])  # Sentiment is always between 0 and 1
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
            orientation="h",  # Horizontal bars (feature names on left)
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
            color="Region",  # Dot color = region
            size="On-Time Delivery (%)",  # Bigger dot = better delivery
            hover_name="Name",  # Show supplier name on hover
            title="Supplier Risk vs Lead Time",
        )
        return self._apply_layout(fig)

    def confusion_matrix_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Heatmap for model confusion matrix."""
        fig = px.imshow(
            df.values,  # Numeric grid of correct/wrong predictions
            x=df.columns.tolist(),  # Column labels
            y=df.index.tolist(),  # Row labels
            text_auto=True,  # Show numbers inside each cell
            color_continuous_scale="Blues",
            title="Model Confusion Matrix",
        )
        # Custom layout for heatmap (slightly different from other charts)
        fig.update_layout(
            template=self._TEMPLATE,
            height=380,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    # ------------------------------------------------------------------
    # Live model prediction charts (used by the Dashboard)
    # ------------------------------------------------------------------

    def risk_gauge(self, risk_score: float) -> go.Figure:
        """Speedometer-style gauge showing the supplier's risk score (0–100)."""
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_score,
                number={"suffix": " /100"},
                title={"text": "Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1e293b"},
                    # Green = safe, yellow = watch, red = risky
                    "steps": [
                        {"range": [0, 40], "color": "#bbf7d0"},
                        {"range": [40, 70], "color": "#fde68a"},
                        {"range": [70, 100], "color": "#fecaca"},
                    ],
                },
            )
        )
        fig.update_layout(
            template=self._TEMPLATE,
            height=260,
            margin=dict(l=20, r=20, t=50, b=10),
        )
        return fig

    def delay_probability_gauge(self, probability: float) -> go.Figure:
        """Gauge showing the model's chance of a delay/disruption (0–100%)."""
        pct = float(probability) * 100
        color = "#ef4444" if pct >= 70 else "#f59e0b" if pct >= 40 else "#10b981"
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%"},
                title={"text": "Delay Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "#ecfdf5"},
                        {"range": [40, 70], "color": "#fffbeb"},
                        {"range": [70, 100], "color": "#fef2f2"},
                    ],
                },
            )
        )
        fig.update_layout(
            template=self._TEMPLATE,
            height=260,
            margin=dict(l=20, r=20, t=50, b=10),
        )
        return fig

    def lead_time_history(self, df: pd.DataFrame) -> go.Figure:
        """Line chart of this supplier's historical delivery times."""
        fig = px.line(
            df,
            x="Date",
            y="Lead Time (days)",
            markers=True,
            title="Historical Lead Time",
            color_discrete_sequence=[self._COLOR_PRIMARY],
        )
        return self._apply_layout(fig, y_title="Days to deliver")

    def sentiment_history(self, df: pd.DataFrame) -> go.Figure:
        """Line chart of this supplier's news sentiment over time."""
        fig = px.line(
            df,
            x="Date",
            y="Sentiment Score",
            markers=True,
            title="Sentiment Trend (news mood)",
            color_discrete_sequence=[self._COLOR_SECONDARY],
        )
        fig.update_yaxes(range=[0, 1])
        # Middle line = neutral news
        fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8")
        return self._apply_layout(fig, y_title="0 = bad news, 1 = good news")

    def inventory_history(self, df: pd.DataFrame) -> go.Figure:
        """Area chart of days of inventory left over time."""
        fig = px.area(
            df,
            x="Date",
            y="Inventory Coverage (days)",
            title="Inventory Coverage",
            color_discrete_sequence=[self._COLOR_SUCCESS],
        )
        # Warning line: below 15 days is low stock
        fig.add_hline(
            y=15,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="Low stock (15 days)",
        )
        return self._apply_layout(fig, y_title="Days of stock")

    def delay_probability_history(self, df: pd.DataFrame) -> go.Figure:
        """Line chart of the model's predicted delay probability over time."""
        fig = px.line(
            df,
            x="Date",
            y="Delay Probability (%)",
            title="Predicted Delay Probability Over Time",
            color_discrete_sequence=[self._COLOR_WARNING],
        )
        fig.update_yaxes(range=[0, 100])
        fig.add_hline(y=50, line_dash="dash", line_color="#94a3b8")
        return self._apply_layout(fig, y_title="Chance of delay (%)")

    def top_risk_factors(self, df: pd.DataFrame) -> go.Figure:
        """Bar chart of what pushed this supplier's risk up or down."""
        fig = px.bar(
            df,
            x="Impact",
            y="Factor",
            orientation="h",
            title="Top Risk Factors (why the model decided this)",
            color="Direction",
            color_discrete_map={
                "Increases risk": "#ef4444",  # red
                "Reduces risk": "#3b82f6",  # blue
            },
        )
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#94a3b8")
        return self._apply_layout(
            fig, x_title="← Reduces risk        Increases risk →"
        )

    def _apply_layout(
        self,
        fig: go.Figure,
        *,
        x_title: str | None = None,
        y_title: str | None = None,
    ) -> go.Figure:
        """Apply consistent styling to a Plotly figure."""
        # Set shared size, margins, and legend position on every chart
        fig.update_layout(
            template=self._TEMPLATE,
            height=380,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        if x_title:
            fig.update_xaxes(title_text=x_title)  # Label the X axis if provided
        if y_title:
            fig.update_yaxes(title_text=y_title)  # Label the Y axis if provided
        return fig
