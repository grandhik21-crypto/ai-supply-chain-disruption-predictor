# Processed output files (created by the sentiment pipeline)

After running the FinBERT sentiment module, CSV files are saved here:

| File | Description |
|------|-------------|
| `daily_supplier_sentiment.csv` | One row per supplier per day (aggregated scores) |
| `article_sentiment.csv` | One row per news article (individual FinBERT scores) |

Run the pipeline:

```bash
python3 scripts/run_sentiment.py
```
