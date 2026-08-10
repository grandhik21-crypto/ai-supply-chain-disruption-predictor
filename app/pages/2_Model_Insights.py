"""
Model Insights screen (Streamlit multipage entry).

This thin file exists so Streamlit shows a page link in the sidebar.
The real UI lives in app/views/model_insights.py.
"""

import sys
from pathlib import Path

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

from app.views.model_insights import ModelInsightsPage  # noqa: E402
from config.settings import get_settings  # noqa: E402

settings = get_settings()
configure_page(f"{settings.title} · Model Insights", "🤖")
inject_global_styles()
render_shared_sidebar()
ModelInsightsPage().render()