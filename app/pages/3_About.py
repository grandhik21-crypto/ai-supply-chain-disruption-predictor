"""
About screen (Streamlit multipage entry).

This thin file exists so Streamlit shows a page link in the sidebar.
The real UI lives in app/views/about.py.
"""

from app.page_setup import (
    configure_page,
    ensure_project_root_on_path,
    inject_global_styles,
    render_shared_sidebar,
)

ensure_project_root_on_path()

from app.views.about import AboutPage  # noqa: E402
from config.settings import get_settings  # noqa: E402

settings = get_settings()
configure_page(f"{settings.title} · About", "ℹ️")
inject_global_styles()
render_shared_sidebar()
AboutPage().render()
