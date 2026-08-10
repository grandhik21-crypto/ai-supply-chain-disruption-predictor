#!/usr/bin/env python3
"""
Run the FinBERT sentiment pipeline from the command line.

Steps performed:
  1. Load sample news articles from data/raw/news_articles.csv
  2. Score each article with Hugging Face FinBERT
  3. Aggregate daily sentiment by supplier
  4. Save results to data/processed/*.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path so imports work when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.sentiment import (  # noqa: E402
    DEFAULT_ARTICLE_OUTPUT_PATH,
    DEFAULT_DAILY_OUTPUT_PATH,
    SentimentAnalyzer,
    load_articles_from_csv,
)

NEWS_CSV = PROJECT_ROOT / "data" / "raw" / "news_articles.csv"


def main() -> None:
    print("Step 1: Loading news articles...")
    articles = load_articles_from_csv(NEWS_CSV)
    print(f"  Loaded {len(articles)} articles")

    print("Step 2: Running FinBERT sentiment analysis...")
    analyzer = SentimentAnalyzer()
    daily_df = analyzer.run_pipeline(articles)

    print("Step 3: Results summary")
    print(daily_df.to_string(index=False))

    print("\nStep 4: Output files saved:")
    print(f"  Daily:    {DEFAULT_DAILY_OUTPUT_PATH}")
    print(f"  Articles: {DEFAULT_ARTICLE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
