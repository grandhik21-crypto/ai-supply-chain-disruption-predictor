# Processed output files

Demo CSVs in this folder are **committed** so deployments can train or serve
predictions without re-running FinBERT (which needs heavy optional deps).

| File | Description |
|------|-------------|
| `daily_supplier_sentiment.csv` | One row per supplier per day (aggregated scores) |
| `article_sentiment.csv` | One row per news article (individual FinBERT scores) |
| `ml_features.csv` | ML-ready dataset with engineered features |

Rebuild:

```bash
python3 scripts/run_sentiment.py              # optional; needs requirements-nlp.txt
python3 scripts/run_feature_engineering.py
# or simply:
python3 scripts/bootstrap.py
```
