# AGENTS.md

## Cursor Cloud specific instructions

### Project overview
This repo is a Python **AI Supply Chain Disruption Predictor**. It blends
industrial-engineering supplier metrics (a scikit-learn logistic-regression
model) with lexicon-based NLP risk analysis of supply-chain news, exposed
through a Streamlit UI.

- Package source: `src/supply_chain_predictor/` (`model.py`, `nlp.py`,
  `data.py`, `predictor.py`). `SupplyChainPredictor` in `predictor.py` is the
  main API.
- UI entry point: `app.py` (Streamlit).
- Tests: `tests/` (pytest).
- Dependency manifests: `requirements.txt` (runtime), `requirements-dev.txt`
  (adds pytest + ruff), and `pyproject.toml` (package metadata, ruff + pytest
  config).

### Environment
- The startup update script creates/uses a virtualenv at `.venv`, installs
  `requirements-dev.txt`, and does an editable install (`pip install -e .`).
  Activate it before running anything: `source .venv/bin/activate`.
- `python3-venv` (system apt package) is required to create the venv; it is a
  system dependency, not part of the update script.

### Non-obvious gotchas
- The package lives under `src/`, so `streamlit run app.py` only resolves
  `import supply_chain_predictor` because the package is installed **editable**
  (`pip install -e .`). If imports fail with `ModuleNotFoundError`, re-run
  `pip install -e .` rather than adding `sys.path` hacks. (pytest resolves it
  independently via `pythonpath = ["src"]` in `pyproject.toml`.)
- On first request, the app trains the model once and caches it via
  `@st.cache_resource`, so the UI briefly shows "Training disruption model...".
  The model is trained on synthetic data with a fixed seed, so outputs are
  deterministic.

### Common commands (run inside the activated venv)
- Lint: `ruff check .`
- Tests: `pytest -q`
- Run app (dev): `streamlit run app.py` (add
  `--server.headless true --server.port 8501` when running without a browser).
