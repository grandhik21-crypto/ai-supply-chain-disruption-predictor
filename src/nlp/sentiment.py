"""
FinBERT sentiment analysis for supply chain news.

Reads a list of news articles, scores each one with Hugging Face FinBERT,
groups scores by supplier and day, and saves the results to a CSV file.

Typical usage:
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_articles(articles)
    daily = analyzer.aggregate_daily_by_supplier(results)
    analyzer.save_to_csv(daily)
"""

from __future__ import annotations  # Modern type hint support

from dataclasses import dataclass, field  # For settings class
from datetime import date, datetime  # For grouping articles by day
from pathlib import Path  # For file paths

import pandas as pd  # For tables and CSV export

from src.models.sentiment import ArticleSentimentResult, NewsArticle
from src.utils.logging_config import get_logger  # Logging helper

logger = get_logger(__name__)  # Logger for this module

# Hugging Face model name — FinBERT is trained on financial news text
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# Default folder for saving processed sentiment CSV files
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "processed"
)
DEFAULT_DAILY_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "daily_supplier_sentiment.csv"
DEFAULT_ARTICLE_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "article_sentiment.csv"

# Map FinBERT text labels to a single number between 0 and 1
# Higher = more positive / less risky sentiment
LABEL_BASE_SCORES: dict[str, float] = {
    "positive": 1.0,
    "neutral": 0.5,
    "negative": 0.0,
}


def _to_date(value: date | datetime) -> date:
    """Convert a datetime or date to a plain date object."""
    if isinstance(value, datetime):
        return value.date()
    return value


def compute_sentiment_score(label: str, confidence: float) -> float:
    """
    Turn a FinBERT label + confidence into one number from 0 to 1.

    Step 1: Start from a base score (positive=1, neutral=0.5, negative=0)
    Step 2: Blend toward 0.5 when confidence is low (model is unsure)
    """
    normalized_label = label.lower()  # FinBERT may return "Positive" or "positive"
    base = LABEL_BASE_SCORES.get(normalized_label, 0.5)
    # Example: positive with 0.9 confidence -> 0.5 + (1.0 - 0.5) * 0.9 = 0.95
    return round(0.5 + (base - 0.5) * confidence, 4)


@dataclass
class SentimentAnalyzer:
    """
    Runs FinBERT sentiment analysis on supplier news articles.

    Steps:
        1. Load FinBERT from Hugging Face (lazy — only when first needed)
        2. Score each article -> label, confidence, numerical score
        3. Aggregate mean score per supplier per day
        4. Save results to CSV
    """

    model_name: str = FINBERT_MODEL_NAME
    daily_output_path: Path = field(default_factory=lambda: DEFAULT_DAILY_OUTPUT_PATH)
    article_output_path: Path = field(
        default_factory=lambda: DEFAULT_ARTICLE_OUTPUT_PATH
    )
    _pipeline: object | None = field(default=None, init=False, repr=False)

    def _get_pipeline(self):
        """
        Step 1 — Load the FinBERT model (only once, then reuse it).

        Uses Hugging Face 'transformers' library sentiment-analysis pipeline.
        First run downloads the model (~400 MB); later runs use the cache.
        """
        if self._pipeline is None:
            logger.info("Loading FinBERT model '%s' from Hugging Face...", self.model_name)
            # Import here so the app starts fast even if sentiment is not used yet
            from transformers import pipeline

            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
            )
            logger.info("FinBERT model loaded successfully")
        return self._pipeline

    def analyze_article(self, article: NewsArticle) -> ArticleSentimentResult:
        """
        Step 2a — Score a single news article with FinBERT.

        Returns label (positive/negative/neutral), confidence, and numerical score.
        """
        # FinBERT expects a string; use headline + body for better context
        text_input = f"{article.headline}. {article.text}".strip(". ").strip()
        if not text_input:
            raise ValueError("Article text cannot be empty")

        # Run FinBERT — returns list with one dict: {"label": "...", "score": 0.XX}
        prediction = self._get_pipeline()(text_input[:512])[0]
        label = str(prediction["label"])
        confidence = round(float(prediction["score"]), 4)
        score = compute_sentiment_score(label, confidence)

        logger.debug(
            "Scored article for %s on %s: %s (conf=%.2f, score=%.2f)",
            article.supplier_id,
            _to_date(article.published_date),
            label,
            confidence,
            score,
        )

        return ArticleSentimentResult(
            supplier_id=article.supplier_id,
            published_date=_to_date(article.published_date),
            headline=article.headline,
            text=article.text,
            sentiment_label=label,
            confidence=confidence,
            sentiment_score=score,
        )

    def analyze_articles(self, articles: list[NewsArticle]) -> list[ArticleSentimentResult]:
        """
        Step 2b — Score a list of news articles.

        Loops through each article and collects all FinBERT results.
        """
        if not articles:
            logger.warning("No articles provided for sentiment analysis")
            return []

        logger.info("Analyzing sentiment for %d articles...", len(articles))
        results = [self.analyze_article(article) for article in articles]
        logger.info("Finished scoring %d articles", len(results))
        return results

    def results_to_dataframe(
        self, results: list[ArticleSentimentResult]
    ) -> pd.DataFrame:
        """Convert article-level sentiment results to a pandas table."""
        if not results:
            return pd.DataFrame(
                columns=[
                    "supplier_id",
                    "published_date",
                    "headline",
                    "sentiment_label",
                    "confidence",
                    "sentiment_score",
                ]
            )

        return pd.DataFrame(
            [
                {
                    "supplier_id": r.supplier_id,
                    "published_date": r.published_date,
                    "headline": r.headline,
                    "sentiment_label": r.sentiment_label,
                    "confidence": r.confidence,
                    "sentiment_score": r.sentiment_score,
                }
                for r in results
            ]
        )

    def aggregate_daily_by_supplier(
        self, results: list[ArticleSentimentResult]
    ) -> pd.DataFrame:
        """
        Step 3 — Combine article scores into one row per supplier per day.

        For each supplier + date group we compute:
          - dominant sentiment label (most common)
          - average confidence
          - average sentiment score
          - number of articles that day
        """
        if not results:
            logger.warning("No results to aggregate")
            return pd.DataFrame(
                columns=[
                    "supplier_id",
                    "date",
                    "sentiment_label",
                    "avg_confidence",
                    "sentiment_score",
                    "article_count",
                ]
            )

        df = self.results_to_dataframe(results)
        logger.info("Aggregating daily sentiment for %d article scores...", len(df))

        def _dominant_label(labels: pd.Series) -> str:
            """Pick the most frequent sentiment label in the group."""
            return labels.mode().iloc[0]

        daily = (
            df.groupby(["supplier_id", "published_date"], as_index=False)
            .agg(
                sentiment_label=("sentiment_label", _dominant_label),
                avg_confidence=("confidence", "mean"),
                sentiment_score=("sentiment_score", "mean"),
                article_count=("headline", "count"),
            )
            .rename(columns={"published_date": "date"})
        )

        # Round numbers for clean CSV output
        daily["avg_confidence"] = daily["avg_confidence"].round(4)
        daily["sentiment_score"] = daily["sentiment_score"].round(4)
        daily["article_count"] = daily["article_count"].astype(int)

        logger.info(
            "Daily aggregation complete: %d supplier-day rows", len(daily)
        )
        return daily

    def save_to_csv(
        self,
        df: pd.DataFrame,
        output_path: Path | str | None = None,
    ) -> Path:
        """
        Step 4 — Save a DataFrame to a CSV file.

        Creates the output folder if it does not exist yet.
        """
        path = Path(output_path) if output_path is not None else self.daily_output_path
        path.parent.mkdir(parents=True, exist_ok=True)  # Create data/processed/ if needed
        df.to_csv(path, index=False)
        logger.info("Saved %d rows to '%s'", len(df), path)
        return path

    def run_pipeline(
        self,
        articles: list[NewsArticle],
        *,
        save_daily: bool = True,
        save_articles: bool = True,
    ) -> pd.DataFrame:
        """
        Run the full sentiment pipeline end-to-end.

        1. Score all articles with FinBERT
        2. Aggregate by supplier and day
        3. Save article-level and daily CSV files
        4. Return the daily aggregated DataFrame
        """
        logger.info("=== Starting sentiment pipeline ===")

        # Step 2: Score each article
        article_results = self.analyze_articles(articles)

        # Step 3: Group by supplier + date
        daily_df = self.aggregate_daily_by_supplier(article_results)

        # Step 4: Write CSV files
        if save_articles and article_results:
            self.save_to_csv(
                self.results_to_dataframe(article_results),
                self.article_output_path,
            )
        if save_daily and not daily_df.empty:
            self.save_to_csv(daily_df, self.daily_output_path)

        logger.info("=== Sentiment pipeline complete ===")
        return daily_df


def load_articles_from_csv(csv_path: Path | str) -> list[NewsArticle]:
    """
    Helper — load news articles from a CSV file.

    Expected columns: supplier_id, published_date, headline, text, source (optional)
    """
    path = Path(csv_path)
    logger.info("Loading news articles from '%s'", path)
    df = pd.read_csv(path, parse_dates=["published_date"])

    articles = [
        NewsArticle(
            text=str(row["text"]),
            supplier_id=str(row["supplier_id"]),
            published_date=row["published_date"],
            headline=str(row.get("headline", "")),
            source=str(row.get("source", "")),
        )
        for _, row in df.iterrows()
    ]
    logger.info("Loaded %d news articles", len(articles))
    return articles
