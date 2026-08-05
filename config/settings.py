"""
App name, version, and labels used across the website.

Stores things like the window title, page list, and KPI labels.
"""

from dataclasses import dataclass, field  # Easy way to create settings classes
from typing import Final  # Marks constants that should never change


APP_TITLE: Final[str] = "AI Supply Chain Disruption Predictor"  # Browser tab title
APP_ICON: Final[str] = "🔗"  # Emoji shown in browser tab
APP_VERSION: Final[str] = "0.1.0"  # App version number

# Page names shown in the sidebar menu (in this order)
NAV_PAGES: Final[list[str]] = [
    "Dashboard",
    "Supplier Analysis",
    "Model Insights",
    "About",
]

# Human-readable names for the four KPI cards on the Dashboard
KPI_LABELS: Final[dict[str, str]] = {
    "risk_score": "Risk Score",
    "lead_time": "Lead Time (days)",
    "inventory_coverage": "Inventory Coverage (days)",
    "sentiment_score": "Sentiment Score",
}


@dataclass(frozen=True)  # frozen=True means settings cannot be changed after creation
class AppSettings:
    """Immutable runtime configuration for the application."""

    title: str = APP_TITLE  # Window title
    icon: str = APP_ICON  # Tab icon
    version: str = APP_VERSION  # Version string
    pages: tuple[str, ...] = field(default_factory=lambda: tuple(NAV_PAGES))  # Menu pages
    random_seed: int = 42  # Seed for reproducible demo/random data


def get_settings() -> AppSettings:
    """Return the default application settings singleton."""
    return AppSettings()  # Create and return a fresh settings object
