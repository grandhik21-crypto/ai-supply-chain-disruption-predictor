"""
App settings, file paths, and environment variables.

Everything configurable lives here. Values can be overridden with
environment variables (useful for deployment), and every setting has a
sensible default so the app also runs with no configuration at all.

Environment variables (all optional):
    APP_TITLE                 Browser tab / header title
    APP_ICON                  Emoji shown in the browser tab
    APP_VERSION               Version string shown in the sidebar
    SC_PROJECT_ROOT           Base folder for all data paths
    SC_SUPPLY_CHAIN_CSV       Path to the raw supplier CSV
    SC_NEWS_CSV               Path to the raw news CSV
    SC_PROCESSED_DIR          Folder for generated CSV files
    SC_MODEL_PATH             Path to the trained model file
    SC_FINBERT_MODEL          Hugging Face model name for sentiment
    SC_ROLLING_WINDOW         Rolling window size in days (default 7)
    SC_RANDOM_SEED            Seed for reproducible results (default 42)
    SC_RISK_THRESHOLD         Risk score that counts as "high risk" (default 70)
    SC_LOG_LEVEL              Logging level: DEBUG / INFO / WARNING / ERROR
    SC_ALLOW_TRAINING_IN_APP  "true" lets the dashboard build the model on demand
"""

from __future__ import annotations

import os  # Reads environment variables
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final


# ----------------------------------------------------------------------
# Small helpers for reading environment variables safely
# ----------------------------------------------------------------------


def _env_str(name: str, default: str) -> str:
    """Read a text setting from the environment."""
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    """Read a whole-number setting; fall back to the default if invalid."""
    try:
        return int(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read a decimal setting; fall back to the default if invalid."""
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read a true/false setting (accepts true, 1, yes, on)."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    """Read a file/folder path setting."""
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default


# ----------------------------------------------------------------------
# Project folders
# ----------------------------------------------------------------------

# Repository root = one level above this config/ folder
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT: Final[Path] = _env_path("SC_PROJECT_ROOT", _DEFAULT_ROOT)

RAW_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Final[Path] = _env_path(
    "SC_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"
)
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"

# Input files
SUPPLY_CHAIN_CSV: Final[Path] = _env_path(
    "SC_SUPPLY_CHAIN_CSV", RAW_DATA_DIR / "supply_chain.csv"
)
NEWS_CSV: Final[Path] = _env_path("SC_NEWS_CSV", RAW_DATA_DIR / "news_articles.csv")

# Generated files
DAILY_SENTIMENT_CSV: Final[Path] = PROCESSED_DATA_DIR / "daily_supplier_sentiment.csv"
ARTICLE_SENTIMENT_CSV: Final[Path] = PROCESSED_DATA_DIR / "article_sentiment.csv"
ML_FEATURES_CSV: Final[Path] = PROCESSED_DATA_DIR / "ml_features.csv"
MODEL_PATH: Final[Path] = _env_path(
    "SC_MODEL_PATH", MODELS_DIR / "disruption_xgb.joblib"
)


# ----------------------------------------------------------------------
# App identity and UI labels
# ----------------------------------------------------------------------

APP_TITLE: Final[str] = _env_str("APP_TITLE", "AI Supply Chain Disruption Predictor")
APP_ICON: Final[str] = _env_str("APP_ICON", "🔗")
APP_VERSION: Final[str] = _env_str("APP_VERSION", "1.0.0")

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


# ----------------------------------------------------------------------
# Model / pipeline behavior
# ----------------------------------------------------------------------

FINBERT_MODEL: Final[str] = _env_str("SC_FINBERT_MODEL", "ProsusAI/finbert")
ROLLING_WINDOW_DAYS: Final[int] = _env_int("SC_ROLLING_WINDOW", 7)
RANDOM_SEED: Final[int] = _env_int("SC_RANDOM_SEED", 42)
HIGH_RISK_THRESHOLD: Final[float] = _env_float("SC_RISK_THRESHOLD", 70.0)
LOG_LEVEL: Final[str] = _env_str("SC_LOG_LEVEL", "INFO").upper()

# Allow the dashboard to build features/model on demand (handy on hosted demos)
ALLOW_TRAINING_IN_APP: Final[bool] = _env_bool("SC_ALLOW_TRAINING_IN_APP", True)


@dataclass(frozen=True)  # frozen=True means settings cannot be changed after creation
class AppSettings:
    """Immutable runtime configuration for the application."""

    title: str = APP_TITLE  # Window title
    icon: str = APP_ICON  # Tab icon
    version: str = APP_VERSION  # Version string
    pages: tuple[str, ...] = field(default_factory=lambda: tuple(NAV_PAGES))  # Menu pages
    random_seed: int = RANDOM_SEED  # Seed for reproducible demo/random data

    # Paths (so services can avoid hardcoding locations)
    supply_chain_csv: Path = SUPPLY_CHAIN_CSV
    news_csv: Path = NEWS_CSV
    processed_dir: Path = PROCESSED_DATA_DIR
    model_path: Path = MODEL_PATH

    # Behavior
    rolling_window_days: int = ROLLING_WINDOW_DAYS
    high_risk_threshold: float = HIGH_RISK_THRESHOLD
    allow_training_in_app: bool = ALLOW_TRAINING_IN_APP


def get_settings() -> AppSettings:
    """Return the default application settings."""
    return AppSettings()  # Create and return a fresh settings object
