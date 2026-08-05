"""
Defines the main data types.

KPIMetrics = the four headline numbers on the Dashboard.
SupplierRecord = one row of info about a single supplier.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class KPIMetrics:
    """Key performance indicators for supply chain health."""

    risk_score: float
    lead_time: float
    inventory_coverage: float
    sentiment_score: float

    def as_dict(self) -> dict[str, float]:
        """Return KPI values keyed by metric name."""
        return {
            "risk_score": self.risk_score,
            "lead_time": self.lead_time,
            "inventory_coverage": self.inventory_coverage,
            "sentiment_score": self.sentiment_score,
        }


@dataclass(frozen=True)
class SupplierRecord:
    """A single supplier entity with risk and performance attributes."""

    supplier_id: str
    name: str
    region: str
    category: str
    risk_score: float
    lead_time_days: float
    on_time_delivery_pct: float
    sentiment_score: float
    last_disruption: date | None
