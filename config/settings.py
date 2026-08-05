"""
App name, version, and labels used across the website.

Stores things like the window title, menu page names,
and what each KPI card is called.
"""

from dataclasses import dataclass, field
from typing import Final


APP_TITLE: Final[str] = "AI Supply Chain Disruption Predictor"
APP_ICON: Final[str] = "🔗"
APP_VERSION: Final[str] = "0.1.0"

NAV_PAGES: Final[list[str]] = [
    "Dashboard",
    "Supplier Analysis",
    "Model Insights",
    "About",
]

KPI_LABELS: Final[dict[str, str]] = {
    "risk_score": "Risk Score",
    "lead_time": "Lead Time (days)",
    "inventory_coverage": "Inventory Coverage (days)",
    "sentiment_score": "Sentiment Score",
}


@dataclass(frozen=True)
class AppSettings:
    """Immutable runtime configuration for the application."""

    title: str = APP_TITLE
    icon: str = APP_ICON
    version: str = APP_VERSION
    pages: tuple[str, ...] = field(default_factory=lambda: tuple(NAV_PAGES))
    random_seed: int = 42


def get_settings() -> AppSettings:
    """Return the default application settings singleton."""
    return AppSettings()
