# Saved machine learning models

The demo model is **committed** so Streamlit Cloud and fresh clones work without
a manual train step. Retrain anytime with the commands below.

| File | Description |
|------|-------------|
| `disruption_xgb.joblib` | XGBoost shipment disruption classifier (joblib bundle) |

The joblib file includes:
- the trained model
- feature column names
- evaluation metrics from the last training run

Train / rebuild:

```bash
python3 scripts/bootstrap.py          # build only what's missing
python3 scripts/bootstrap.py --force  # rebuild features + retrain
```
