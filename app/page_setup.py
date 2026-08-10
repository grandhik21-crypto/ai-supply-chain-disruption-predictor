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
            <div style="padding: 0.25rem 0 1rem 0;">
                <h2 style="margin: 0; font-size: 1.35rem; letter-spacing:-0.02em;">
                    {settings.icon} Supply Chain AI
                </h2>
                <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.82rem;">
                    Disruption Predictor · v{settings.version}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("🧭 Use the page links above to switch screens.")

        st.markdown("---")
        st.caption("🔌 Data Source")
        _render_data_source_status()

        st.markdown("---")
        with st.expander("❓ Need help?"):
            st.markdown(
                """
                The demo model ships with the repo. If predictions are missing:

                - Click **Build features and train model** on the Dashboard, or
                - Run `python3 scripts/bootstrap.py` then refresh

                Full hosting notes: see `DEPLOYMENT.md`.
                """
            )


def _render_data_source_status() -> None:
    """Tell the user whether live model predictions are available."""
    from src.features.feature_engineering import DEFAULT_ML_OUTPUT_PATH
    from src.ml.model import DEFAULT_MODEL_PATH

    if DEFAULT_MODEL_PATH.exists():
        st.success("Live model predictions", icon="🤖")
    else:
        st.warning("No trained model yet", icon="⚠️")
        st.caption("Use the Dashboard setup button or `python3 scripts/bootstrap.py`")

    if not DEFAULT_ML_OUTPUT_PATH.exists():
        st.caption("⚠️ Missing features file — run the feature pipeline.")


def inject_global_styles() -> None:
    """
    Add the app's visual style: spacing, fonts, cards, and responsive rules.

    Kept in one place so every page looks the same.
    """
    st.markdown(
        """
        <style>
            /* ---------- Design tokens ---------- */
            :root {
                --sc-primary: #2563eb;
                --sc-text: #0f172a;
                --sc-muted: #64748b;
                --sc-border: #e2e8f0;
                --sc-surface: #ffffff;
                --sc-surface-alt: #f8fafc;
                --sc-danger: #ef4444;
                --sc-warning: #f59e0b;
                --sc-success: #10b981;
                --sc-radius: 12px;
            }

            /* ---------- Page spacing ---------- */
            .block-container {
                padding-top: 2.5rem;
                padding-bottom: 4rem;
                max-width: 1400px;
            }

            /* ---------- Typography ---------- */
            html, body, [class*="css"] {
                font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
                color: var(--sc-text);
            }
            h1, h2, h3, h4 {
                letter-spacing: -0.02em;
                font-weight: 700;
            }

            /* ---------- Page header ---------- */
            .page-header {
                border-bottom: 1px solid var(--sc-border);
                padding-bottom: 1rem;
                margin-bottom: 1.75rem;
            }
            .page-title {
                font-size: 2rem;
                margin: 0;
                line-height: 1.2;
            }
            .page-subtitle {
                color: var(--sc-muted);
                font-size: 0.95rem;
                margin: 0.4rem 0 0 0;
            }

            /* ---------- Section headers ---------- */
            .section-header {
                margin: 2.25rem 0 0.75rem 0;
            }
            .section-title {
                font-size: 1.2rem;
                margin: 0;
            }
            .section-hint {
                color: var(--sc-muted);
                font-size: 0.85rem;
                margin: 0.3rem 0 0 0;
                line-height: 1.5;
            }

            /* ---------- KPI metrics ---------- */
            [data-testid="stMetric"] {
                background: var(--sc-surface);
                border: 1px solid var(--sc-border);
                border-radius: var(--sc-radius);
                padding: 1rem 1.15rem;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            }
            [data-testid="stMetricLabel"] {
                color: var(--sc-muted);
                font-size: 0.82rem !important;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            [data-testid="stMetricValue"] {
                font-size: 1.7rem;
                font-weight: 700;
            }

            /* ---------- Custom metric card ---------- */
            .metric-card {
                background: var(--sc-surface);
                border: 1px solid var(--sc-border);
                border-radius: var(--sc-radius);
                padding: 1rem 1.15rem;
                height: 100%;
            }
            .metric-label {
                color: var(--sc-muted);
                font-size: 0.8rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .metric-value {
                font-size: 1.65rem;
                font-weight: 700;
                margin-top: 0.3rem;
            }
            .metric-caption {
                color: var(--sc-muted);
                font-size: 0.78rem;
                margin-top: 0.3rem;
            }

            /* ---------- Empty state ---------- */
            .empty-state {
                border: 1px dashed var(--sc-border);
                border-radius: var(--sc-radius);
                background: var(--sc-surface-alt);
                padding: 2rem 1.5rem;
                text-align: center;
                margin: 0.5rem 0 1rem 0;
            }
            .empty-icon { font-size: 2rem; }
            .empty-title {
                font-weight: 700;
                margin-top: 0.5rem;
                font-size: 1rem;
            }
            .empty-message {
                color: var(--sc-muted);
                font-size: 0.9rem;
                margin-top: 0.3rem;
            }
            .empty-hint {
                color: var(--sc-muted);
                font-size: 0.8rem;
                margin-top: 0.6rem;
                font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            }

            /* ---------- Badges ---------- */
            .badge {
                display: inline-block;
                padding: 0.2rem 0.6rem;
                border-radius: 999px;
                font-size: 0.75rem;
                font-weight: 700;
            }
            .badge-high { background: #fee2e2; color: #b91c1c; }
            .badge-medium { background: #fef3c7; color: #b45309; }
            .badge-low { background: #d1fae5; color: #047857; }

            /* ---------- Sidebar ---------- */
            [data-testid="stSidebar"] {
                background-color: var(--sc-surface-alt);
                border-right: 1px solid var(--sc-border);
            }

            /* ---------- Tables & charts ---------- */
            [data-testid="stDataFrame"] {
                border: 1px solid var(--sc-border);
                border-radius: var(--sc-radius);
            }

            /* ---------- Expanders (export panels) ---------- */
            [data-testid="stExpander"] details {
                border: 1px solid var(--sc-border);
                border-radius: var(--sc-radius);
            }

            /* ---------- Responsive: stack nicely on small screens ---------- */
            @media (max-width: 900px) {
                .block-container {
                    padding-top: 1.5rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
                .page-title { font-size: 1.55rem; }
                [data-testid="stMetricValue"] { font-size: 1.35rem; }
                .metric-value { font-size: 1.35rem; }
                /* Let Streamlit columns wrap instead of squeezing */
                [data-testid="stHorizontalBlock"] {
                    flex-wrap: wrap;
                    gap: 0.75rem;
                }
                [data-testid="stHorizontalBlock"] > div {
                    min-width: 260px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
