"""
Reusable pieces of the website UI.

Groups the sidebar, KPI cards, and chart helpers
that multiple pages share.
"""

# Import chart builder class
from app.components.charts import ChartFactory
# Import KPI card display class
from app.components.kpi_cards import KPICardRenderer
# Import sidebar menu class
from app.components.sidebar import SidebarNavigator
# Import shared UI helpers (headers, empty states, chart export)
from app.components.ui import (
    empty_state,
    page_header,
    render_chart,
    render_dataframe,
    safe_section,
    section_header,
)

# List of names other files can import from this package
__all__ = [
    "ChartFactory",
    "KPICardRenderer",
    "SidebarNavigator",
    "empty_state",
    "page_header",
    "render_chart",
    "render_dataframe",
    "safe_section",
    "section_header",
]
