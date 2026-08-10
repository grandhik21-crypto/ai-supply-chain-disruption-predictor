"""
Supplier Analysis screen (Streamlit multipage entry).

This thin file exists so Streamlit shows a page link in the sidebar.
The real UI lives in app/views/supplier_analysis.py.
"""

import sys
from pathlib import Path

# Make the project root importable when Streamlit runs this page directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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