"""
Defines the main data types.

KPIMetrics = the four headline numbers on the Dashboard.
SupplierRecord = one row of info about a single supplier.
"""

from dataclasses import dataclass  # Creates simple classes to hold data
from datetime import date  # Used for supplier disruption dates


@dataclass(frozen=True)  # frozen=True: values cannot be changed after creation
class KPIMetrics:
    """Key performance indicators for supply chain health."""

    risk_score: float  # Overall risk number (higher = riskier)
    lead_time: float  # Average days to receive goods
    inventory_coverage: float  # How many days of stock are on hand
    sentiment_score: float  # Market/news mood score from 0 to 1

    def as_dict(self) -> dict[str, float]:
        """Return KPI values keyed by metric name."""
        # Convert the four fields into a dictionary for easy looping in the UI
        return {
            "risk_score": self.risk_score,
            "lead_time": self.lead_time,
            "inventory_coverage": self.inventory_coverage,
            "sentiment_score": self.sentiment_score,
        }


@dataclass(frozen=True)
class SupplierRecord:
    """A single supplier entity with risk and performance attributes."""

    supplier_id: str  # Unique ID like "SUP-1000"
    name: str  # Company name
    region: str  # World region (e.g. Europe)
    category: str  # Type of goods (e.g. Electronics)
    risk_score: float  # How risky this supplier is
    lead_time_days: float  # Days they take to deliver
    on_time_delivery_pct: float  # Percent of orders delivered on time
    sentiment_score: float  # News/market sentiment about them
    last_disruption: date | None  # Date of last disruption, or None if none recorded
