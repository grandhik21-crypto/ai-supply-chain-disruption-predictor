"""
Shared helpers for every Streamlit page script.

Makes sure imports work and draws the shared left sidebar branding/filters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Project root = two levels up from app/pages/*.py OR one level up from app/main.py
# We resolve from this file's location: app/utils_ui.py would be better, but keep
# this helper inside app/ so both main.py and pages/ can import it.


def ensure_project_root_on_path() -> Path:
    """Add the project root folder to Python's import path."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def configure_page(title: str, icon: str) -> None:
    """Set browser tab title/icon and wide layout (safe if already configured)."""
    try:
        st.set_page_config(
            page_title=title,
            page_icon=icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        # Streamlit only allows set_page_config once; ignore if already called
        pass


def render_shared_sidebar() -> None:
    """Draw branding + filters in the left sidebar (page links are automatic)."""
    from config.settings import get_settings

    settings = get_settings()
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 0.5rem 0 1rem 0;">
                <h2 style="margin: 0; font-size: 1.4rem;">
                    {settings.icon} Supply Chain AI
                </h2>
                <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.85rem;">
                    Disruption Predictor v{settings.version}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Use the page links above to switch screens.")
        st.markdown("---")
        st.caption("Data Source")
        st.info("Placeholder data · Demo mode", icon="ℹ️")
        st.markdown("---")
        st.caption("Filters")
        st.selectbox(
            "Region",
            ["All Regions", "North America", "Europe", "Asia-Pacific", "Latin America"],
            key="filter_region",
        )
        st.selectbox(
            "Time Horizon",
            ["Last 30 days", "Last 90 days", "Last 12 months"],
            key="filter_horizon",
        )


def inject_global_styles() -> None:
    """Add light CSS for spacing and KPI number size."""
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.75rem;
                font-weight: 700;
            }
            [data-testid="stSidebar"] {
                background-color: #f8fafc;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
