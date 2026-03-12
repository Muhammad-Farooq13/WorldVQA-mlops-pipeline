# WorldVQA MLOps Pipeline

[![CI](https://github.com/Muhammad-Farooq13/WorldVQA-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Farooq13/WorldVQA-mlops-pipeline/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20LR-orange)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-lightgrey)](https://flask.palletsprojects.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready MLOps pipeline for **Visual Question Answering (VQA)** over world-knowledge data.
Classifies natural-language questions into five semantic categories using a TF-IDF + Logistic Regression baseline.

---

## Features

- **TF-IDF + Logistic Regression** baseline (extensible to transformer-based multimodal models)
- **Flask REST API** with `/health` and `/predict` endpoints
- **Streamlit dashboard** — 5 tabs including live 🔮 Classify
- **Synthetic demo bundle** — no network download required for the dashboard
- **CI/CD** — GitHub Actions matrix (Python 3.11 & 3.12), ruff linting, pytest + coverage, Docker build

---

## Quick Start

```bash
git clone https://github.com/Muhammad-Farooq13/WorldVQA-mlops-pipeline.git
cd WorldVQA-mlops-pipeline
pip install -r requirements.txt

# Build the demo bundle (synthetic data, no HuggingFace download needed)
python train_demo.py

# Launch the Streamlit dashboard
streamlit run streamlit_app.py
```

---

## Flask API

```bash
# Start the API (models/vectorizer.joblib + models/model.joblib required)
python -c "from flask_app import create_app; create_app().run(port=5000)"

# Health check
curl http://localhost:5000/health
# {"status": "ok"}

# Predict
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["What is the capital of France?"]}'
# {"predictions": ["geography"]}
```

---

## MLOps Pipeline (full — requires HuggingFace network access)

```bash
python mlops_pipeline.py run-all   # load data + train on real WorldVQA dataset
python mlops_pipeline.py load-data
python mlops_pipeline.py train-model
```

---

## Tests

```bash
pip install -r requirements-ci.txt
pytest tests/ -v --cov=src --cov-report=term-missing
# 3 passed, 1 skipped
```

---

## Project Structure

```
worldvqa/
├── src/
│   ├── data/load_data.py          # HuggingFace dataset loader
│   ├── features/build_features.py # TF-IDF feature builder
│   ├── models/train_model.py      # Train & save model artifacts
│   └── utils/config.py            # Directory helpers
├── tests/                         # pytest suite
├── models/
│   ├── vectorizer.joblib          # Flask API artifact
│   ├── model.joblib               # Flask API artifact
│   └── demo_bundle.pkl            # Streamlit dashboard bundle
├── flask_app.py                   # REST API factory
├── mlops_pipeline.py              # CLI orchestrator
├── train_demo.py                  # Synthetic demo bundle builder
├── streamlit_app.py               # Streamlit dashboard (5 tabs)
├── requirements.txt               # Runtime dependencies
├── requirements-ci.txt            # CI-only dependencies
└── .github/workflows/ci.yml       # GitHub Actions (3.11 & 3.12)
```

---

## License

MIT