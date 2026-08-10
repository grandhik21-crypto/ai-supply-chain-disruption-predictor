# Deployment guide

## Why the live site said “No trained model yet”

The dashboard needs these files:

- `models/disruption_xgb.joblib`
- `data/processed/ml_features.csv`

They used to be gitignored, so Streamlit Cloud (and similar hosts) cloned the
repo **without** a trained model. That is fixed: the demo model and feature
CSVs are now committed. After you pull / redeploy, the dashboard should load
predictions immediately.

If the model is still missing on a host, open the Dashboard and click
**Build features and train model**, or set `SC_ALLOW_TRAINING_IN_APP=true`
and run:

```bash
python3 scripts/bootstrap.py
```

---

## Streamlit Community Cloud

1. Push this branch (or merge to `main`).
2. In [share.streamlit.io](https://share.streamlit.io): **New app**.
3. Set:
   - **Repository**: your fork / this repo  
   - **Branch**: `main` (or this feature branch)  
   - **Main file path**: `app/main.py`
4. Keep the default Python environment using root `requirements.txt`
   (it intentionally **excludes** PyTorch / Transformers so the cloud install fits).
5. Root `packages.txt` may list Debian apt packages (one name per line, **no comments** —
   Streamlit passes lines to `apt-get`, and a `/` in a comment breaks the install).
6. Deploy, then **reboot** the app after the model files land in git.

Optional secrets (App settings → Secrets), TOML format:

```toml
SC_ALLOW_TRAINING_IN_APP = "true"
SC_LOG_LEVEL = "INFO"
```

See `.streamlit/secrets.toml.example` for a full template.

---

## Local / Docker-style run

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

Optional NLP (FinBERT only):

```bash
python3 -m pip install -r requirements-nlp.txt
python3 scripts/run_sentiment.py
python3 scripts/bootstrap.py --force
```

---

## Checklist if deploy still shows the empty state

1. Confirm `models/disruption_xgb.joblib` exists in the deployed git revision.
2. Confirm Main file path is `app/main.py`.
3. Confirm install uses slim `requirements.txt` (not a custom file that drops `xgboost` / `joblib`).
4. Reboot the Streamlit Cloud app after pushing model files.
5. Use the in-app **Build features and train model** button as a fallback.
