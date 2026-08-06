"""
Shared UI building blocks for a clean, consistent look.

Gives every page the same:
  - section headers with icons
  - empty-state messages
  - error handling (no blank screens)
  - chart display with export buttons
  - loading spinners
"""

from __future__ import annotations  # Modern type hint support

from contextlib import contextmanager  # For the safe_section helper
from typing import Iterator  # Return type for the context manager

import pandas as pd  # Tables (used for download buttons)
import plotly.graph_objects as go  # Chart type used everywhere
import streamlit as st  # Website UI library


# ----------------------------------------------------------------------
# Headings and text
# ----------------------------------------------------------------------


def page_header(icon: str, title: str, subtitle: str) -> None:
    """Big page title with icon and a short description under it."""
    st.markdown(
        f"""
        <div class="page-header">
            <h1 class="page-title">{icon} {title}</h1>
            <p class="page-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(icon: str, title: str, hint: str = "") -> None:
    """Section title with an icon, plus optional small helper text."""
    st.markdown(
        f"""
        <div class="section-header">
            <h3 class="section-title">{icon} {title}</h3>
            {f'<p class="section-hint">{hint}</p>' if hint else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def spacer(height: str = "1rem") -> None:
    """Add vertical breathing room between blocks."""
    st.markdown(f"<div style='height:{height}'></div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Empty states and errors
# ----------------------------------------------------------------------


def empty_state(
    title: str,
    message: str,
    icon: str = "📭",
    hint: str = "",
) -> None:
    """
    Friendly 'nothing here yet' box.

    Used instead of showing a blank space when data is missing.
    """
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-icon">{icon}</div>
            <div class="empty-title">{title}</div>
            <div class="empty-message">{message}</div>
            {f'<div class="empty-hint">{hint}</div>' if hint else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def safe_section(section_name: str, *, icon: str = "⚠️") -> Iterator[None]:
    """
    Run a page section without letting one error break the whole page.

    If something fails, the user sees a clear message plus the technical
    details tucked away in an expander.
    """
    try:
        yield
    except Exception as exc:  # Keep the rest of the page working
        st.warning(
            f"**{section_name}** could not be displayed right now.",
            icon=icon,
        )
        with st.expander("Technical details (for developers)"):
            st.exception(exc)


# ----------------------------------------------------------------------
# Charts (with export)
# ----------------------------------------------------------------------


def render_chart(
    fig: go.Figure,
    *,
    key: str,
    filename: str | None = None,
    show_export: bool = True,
) -> None:
    """
    Show a Plotly chart and offer image / interactive downloads.

    PNG export needs the `kaleido` package; if it's missing we quietly
    fall back to an interactive HTML download.
    """
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{key}")

    if not show_export:
        return

    safe_name = filename or key
    with st.expander("Export this chart", expanded=False):
        col_png, col_html = st.columns(2)

        with col_png:
            try:
                png_bytes = fig.to_image(format="png", scale=2)
                st.download_button(
                    "Download PNG",
                    data=png_bytes,
                    file_name=f"{safe_name}.png",
                    mime="image/png",
                    key=f"png_{key}",
                    use_container_width=True,
                )
            except Exception:
                st.caption("PNG export unavailable (install `kaleido`).")

        with col_html:
            html = fig.to_html(include_plotlyjs="cdn", full_html=True)
            st.download_button(
                "Download interactive HTML",
                data=html,
                file_name=f"{safe_name}.html",
                mime="text/html",
                key=f"html_{key}",
                use_container_width=True,
            )


def render_dataframe(
    df: pd.DataFrame,
    *,
    key: str,
    filename: str | None = None,
    show_export: bool = True,
    **kwargs,
) -> None:
    """Show a table and offer a CSV download."""
    st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)

    if show_export and not df.empty:
        st.download_button(
            "Download table (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{filename or key}.csv",
            mime="text/csv",
            key=f"csv_{key}",
        )


# ----------------------------------------------------------------------
# Status pills / badges
# ----------------------------------------------------------------------


def risk_badge(probability: float) -> str:
    """Return a small colored HTML badge describing a risk level."""
    if probability >= 0.7:
        label, css = "High risk", "badge-high"
    elif probability >= 0.4:
        label, css = "Medium risk", "badge-medium"
    else:
        label, css = "Low risk", "badge-low"
    return f'<span class="badge {css}">{label}</span>'


def metric_card(
    label: str,
    value: str,
    icon: str = "",
    caption: str = "",
) -> None:
    """A styled KPI card (nicer than the default metric box)."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-caption">{caption}</div>' if caption else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )
