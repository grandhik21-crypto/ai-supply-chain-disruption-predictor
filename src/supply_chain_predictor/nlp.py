"""Lightweight, dependency-free NLP risk analysis for supply-chain news.

A transformer model would be overkill (and slow) for a demo, so this uses a
transparent lexicon of disruption-related terms plus mitigating terms. The
output is a normalised risk score in [0, 1] with an explainable breakdown of
which terms fired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Terms that indicate an elevated disruption risk, with relative weights.
RISK_LEXICON: dict[str, float] = {
    "strike": 3.0,
    "shortage": 3.0,
    "delay": 2.0,
    "delayed": 2.0,
    "backlog": 2.0,
    "congestion": 2.5,
    "port closure": 3.0,
    "shutdown": 3.0,
    "lockdown": 3.0,
    "bankruptcy": 3.5,
    "insolvency": 3.5,
    "recall": 2.5,
    "tariff": 2.0,
    "sanction": 3.0,
    "embargo": 3.0,
    "war": 3.5,
    "conflict": 2.5,
    "earthquake": 3.0,
    "hurricane": 3.0,
    "flood": 2.5,
    "fire": 2.5,
    "typhoon": 3.0,
    "drought": 2.0,
    "pandemic": 3.0,
    "outage": 2.5,
    "disruption": 3.0,
    "halt": 2.5,
    "cyberattack": 3.0,
}

# Terms that indicate recovery / reduced risk.
MITIGATION_LEXICON: dict[str, float] = {
    "resolved": 2.5,
    "recovered": 2.5,
    "resumed": 2.5,
    "restored": 2.5,
    "on schedule": 2.0,
    "on track": 2.0,
    "normal": 1.5,
    "stabilized": 2.0,
    "stabilised": 2.0,
    "ramp up": 1.5,
    "surplus": 1.5,
}


@dataclass
class NewsRiskResult:
    """Explainable output of a news-risk analysis."""

    score: float
    matched_risk_terms: dict[str, int] = field(default_factory=dict)
    matched_mitigation_terms: dict[str, int] = field(default_factory=dict)
    headline_count: int = 0


class NewsRiskAnalyzer:
    """Scores free-text supply-chain news for near-term disruption risk."""

    def __init__(
        self,
        risk_lexicon: dict[str, float] | None = None,
        mitigation_lexicon: dict[str, float] | None = None,
        saturation: float = 6.0,
    ) -> None:
        self.risk_lexicon = risk_lexicon or RISK_LEXICON
        self.mitigation_lexicon = mitigation_lexicon or MITIGATION_LEXICON
        # `saturation` controls how quickly the score approaches 1.0 as more
        # weighted risk terms appear.
        self.saturation = saturation

    @staticmethod
    def _count(term: str, text: str) -> int:
        pattern = r"\b" + re.escape(term) + r"\b"
        return len(re.findall(pattern, text))

    def analyze(self, headlines: list[str] | str) -> NewsRiskResult:
        """Return a risk score in [0, 1] for one or more news headlines."""
        if isinstance(headlines, str):
            headlines = [headlines]
        headlines = [h for h in headlines if h and h.strip()]

        text = " ".join(headlines).lower()

        risk_hits: dict[str, int] = {}
        risk_weight = 0.0
        for term, weight in self.risk_lexicon.items():
            count = self._count(term, text)
            if count:
                risk_hits[term] = count
                risk_weight += weight * count

        mitigation_hits: dict[str, int] = {}
        mitigation_weight = 0.0
        for term, weight in self.mitigation_lexicon.items():
            count = self._count(term, text)
            if count:
                mitigation_hits[term] = count
                mitigation_weight += weight * count

        net_weight = max(0.0, risk_weight - 0.5 * mitigation_weight)
        # Bounded, monotonically increasing mapping from weight to [0, 1].
        score = 1.0 - (self.saturation / (self.saturation + net_weight))

        return NewsRiskResult(
            score=round(float(score), 4),
            matched_risk_terms=risk_hits,
            matched_mitigation_terms=mitigation_hits,
            headline_count=len(headlines),
        )
