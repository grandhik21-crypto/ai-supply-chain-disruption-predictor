"""
Starts the web app (Home / Dashboard page).

Streamlit also auto-loads other screens from the app/pages/ folder.
Run with: python3 -m streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.page_setup import (
    configure_page,
    ensure_project_root_on_path,
    inject_global_styles,
    render_shared_sidebar,
)

# Make sure "from config..." and "from src..." imports work
ensure_project_root_on_path()

from app.views.dashboard import DashboardPage  # noqa: E402
from config.settings import get_settings  # noqa: E402


def main() -> None:
    """Show the Dashboard (home) page."""
    settings = get_settings()
    configure_page(settings.title, settings.icon)
    inject_global_styles()
    render_shared_sidebar()
    DashboardPage().render()


# Streamlit executes this whole file when you run: streamlit run app/main.py
main()
