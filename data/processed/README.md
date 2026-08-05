# Processed output files (created by the sentiment pipeline)

After running the FinBERT sentiment module, CSV files are saved here:

| File | Description |
|------|-------------|
| `daily_supplier_sentiment.csv` | One row per supplier per day (aggregated scores) |
| `article_sentiment.csv` | One row per news article (individual FinBERT scores) |
| `ml_features.csv` | ML-ready dataset with engineered features (from feature pipeline) |

Run the pipelines:

```bash
python3 scripts/run_sentiment.py
python3 scripts/run_feature_engineering.py
```
