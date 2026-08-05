"""
Application configuration package.

Re-exports settings used by the web dashboard (app title, page names, KPI labels).
"""

from config.settings import AppSettings, get_settings

__all__ = ["AppSettings", "get_settings"]
