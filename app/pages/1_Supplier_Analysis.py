"""
Supplier Analysis screen (Streamlit multipage entry).

This thin file exists so Streamlit shows a page link in the sidebar.
The real UI lives in app/views/supplier_analysis.py.
"""

from app.page_setup import (
    configure_page,
    ensure_project_root_on_path,
    inject_global_styles,
    render_shared_sidebar,
)

ensure_project_root_on_path()

from app.views.supplier_analysis import SupplierAnalysisPage  # noqa: E402
from config.settings import get_settings  # noqa: E402

settings = get_settings()
configure_page(f"{settings.title} · Supplier Analysis", "🏭")
inject_global_styles()
render_shared_sidebar()
SupplierAnalysisPage().render()
