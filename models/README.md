# Saved machine learning models

Trained models are written here by `scripts/train_model.py`.

| File | Description |
|------|-------------|
| `disruption_xgb.joblib` | XGBoost shipment disruption classifier (joblib bundle) |

The joblib file includes:
- the trained model
- feature column names
- evaluation metrics from the last training run

Train:

```bash
python3 scripts/run_feature_engineering.py   # if features CSV is missing
python3 scripts/train_model.py
```
