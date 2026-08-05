"""
App settings folder.

Holds basic info like the app name, page list, and KPI labels.
"""

# Import settings class and helper function from settings.py
from config.settings import AppSettings, get_settings

# Names other files can import from this package
__all__ = ["AppSettings", "get_settings"]
