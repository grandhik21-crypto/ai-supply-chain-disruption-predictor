"""
KPI card component for the web dashboard.

Displays the four headline metrics (Risk Score, Lead Time, Inventory Coverage,
Sentiment Score) as Streamlit metric cards on the Dashboard page.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st

from config.settings import KPI_LABELS
from src.models.metrics import KPIMetrics


class KPICardRenderer:
    """Renders a row of KPI metric cards."""

    _DELTA_HINTS: dict[str, str] = {
        "risk_score": "+3.2 vs last month",
        "lead_time": "-1.5 days vs target",
        "inventory_coverage": "+4 days vs last week",
        "sentiment_score": "+0.06 vs baseline",
    }

    _FORMATters: dict[str, Callable[[float], str]] = {
        "risk_score": lambda v: f"{v:.1f}",
        "lead_time": lambda v: f"{v:.1f}",
        "inventory_coverage": lambda v: f"{v:.1f}",
        "sentiment_score": lambda v: f"{v:.2f}",
    }

    def render(self, metrics: KPIMetrics) -> None:
        """Render four KPI cards in a responsive column layout."""
        values = metrics.as_dict()
        columns = st.columns(4)

        for column, (key, value) in zip(columns, values.items()):
            with column:
                formatted = self._FORMATters[key](value)
                st.metric(
                    label=KPI_LABELS[key],
                    value=formatted,
                    delta=self._DELTA_HINTS[key],
                )
