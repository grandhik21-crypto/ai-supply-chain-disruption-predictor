"""
The left menu branding and filters.

Page switching is handled by Streamlit's built-in links
from files in app/pages/. This class only draws branding/filters.
"""

from __future__ import annotations

from app.page_setup import render_shared_sidebar


class SidebarNavigator:
    """Draws shared sidebar branding and filters."""

    def __init__(self, settings=None) -> None:
        self._settings = settings

    def render(self) -> None:
        """Show branding + filters in the left sidebar."""
        render_shared_sidebar()
