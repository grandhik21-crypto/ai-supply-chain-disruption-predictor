"""
The four number boxes on the Dashboard page.

Shows Risk Score, Lead Time, Inventory Coverage,
and Sentiment Score at the top of the home screen.
"""

from __future__ import annotations  # Modern type hint support

from typing import Callable  # Used to type the number-formatting functions

import streamlit as st  # Website UI library

# Import human-readable labels for each KPI (e.g. "Risk Score")
from config.settings import KPI_LABELS
# Import the KPI data object passed in from the analytics service
from src.models.metrics import KPIMetrics


class KPICardRenderer:
    """Renders a row of KPI metric cards."""

    # Small text shown under each number (change vs last period — demo values)
    _DELTA_HINTS: dict[str, str] = {
        "risk_score": "+3.2 vs last month",
        "lead_time": "-1.5 days vs target",
        "inventory_coverage": "+4 days vs last week",
        "sentiment_score": "+0.06 vs baseline",
    }

    # Functions that format each number before display (1 decimal or 2 for sentiment)
    _FORMATters: dict[str, Callable[[float], str]] = {
        "risk_score": lambda v: f"{v:.1f}",
        "lead_time": lambda v: f"{v:.1f}",
        "inventory_coverage": lambda v: f"{v:.1f}",
        "sentiment_score": lambda v: f"{v:.2f}",
    }

    def render(self, metrics: KPIMetrics) -> None:
        """Render four KPI cards in a responsive column layout."""
        # Turn KPI object into a dictionary: name -> number
        values = metrics.as_dict()
        # Split the row into 4 equal columns
        columns = st.columns(4)

        # Loop through each column and its matching KPI value
        for column, (key, value) in zip(columns, values.items()):
            with column:  # Place the next widget inside this column
                # Format the number using the right formatter for this KPI
                formatted = self._FORMATters[key](value)
                # Show the metric card: label, value, and change hint
                st.metric(
                    label=KPI_LABELS[key],
                    value=formatted,
                    delta=self._DELTA_HINTS[key],
                )
