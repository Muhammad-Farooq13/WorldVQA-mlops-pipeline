# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] - 2025-07-16

### Added
- `streamlit_app.py` — 5-tab dashboard (Overview, Model Results, Analytics, Pipeline & API, 🔮 Classify)
- `train_demo.py` — builds `models/demo_bundle.pkl` from 600 synthetic VQA questions across 5 categories (no HuggingFace download required)
- `.streamlit/config.toml` — dark theme (primaryColor `#4FC3F7`)
- `requirements-ci.txt` — lean CI dependencies (no datasets/streamlit)
- `runtime.txt`, `packages.txt` — deployment metadata

### Changed
- `.github/workflows/ci.yml` — upgraded to Python 3.11/3.12 matrix, added ruff linting, `pytest-cov`, `codecov-action@v5`, Docker buildx job
- `requirements.txt` — added `streamlit>=1.36.0`, `plotly>=5.16.0`; removed `pytest` (moved to CI file)
- `src/models/train_model.py` — graceful fallback for stratified split on very small datasets
- `.gitignore` — added `!models/demo_bundle.pkl`
- `README.md` — complete rewrite with badges, Quick Start, API reference, project structure

---

## [0.1.0] - initial

### Added
- Initial WorldVQA MLOps pipeline: `src/data/load_data.py`, `src/features/build_features.py`, `src/models/train_model.py`
- Flask REST API (`flask_app.py`) with `/health` and `/predict`
- CLI orchestrator (`mlops_pipeline.py`)
- pytest suite (`tests/`) — 3 tests, 1 skipped (requires network)
- GitHub Actions CI (Python 3.10, single job)