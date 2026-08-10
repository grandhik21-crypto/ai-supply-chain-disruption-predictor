"""
Sentiment-related data types for the web application.

NewsArticle = one news story about a supplier.
ArticleSentimentResult = FinBERT output for a single article.
"""

from dataclasses import dataclass  # Simple classes to hold data
from datetime import date, datetime  # For article publish dates


@dataclass(frozen=True)
class NewsArticle:
    """A news article linked to a supplier."""

    text: str  # Full article body (FinBERT reads this)
    supplier_id: str  # Which supplier this news is about (e.g. "SUP-1000")
    published_date: date | datetime  # When the article was published
    headline: str = ""  # Optional short title
    source: str = ""  # Optional news source name


@dataclass(frozen=True)
class ArticleSentimentResult:
    """Sentiment analysis output for one article."""

    supplier_id: str  # Supplier the article belongs to
    published_date: date  # Date of the article (date only, no time)
    headline: str  # Article headline (for reference in output)
    text: str  # Original article text
    sentiment_label: str  # FinBERT label: positive, negative, or neutral
    confidence: float  # How sure the model is (0.0 to 1.0)
    sentiment_score: float  # Single number from 0 (bad) to 1 (good)
