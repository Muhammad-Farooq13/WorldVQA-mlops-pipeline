# WorldVQA MLOps Project

## Objective

The goal of this project is to build a reproducible, production-ready
machine learning pipeline around the `moonshotai/WorldVQA` dataset
from Hugging Face. It demonstrates best practices in:

- **Data science workflow** (exploration, feature engineering, modeling)
- **MLOps** (version control, automated tests, reproducible pipelines)
- **Deployment** via **Flask** and **Docker**

The final deliverable is a trained model that can be served via a
Flask API and containerized for deployment.

## Project Structure

```text
.
├── data
│   ├── raw/          # Raw datasets (downloaded / original)
│   └── processed/    # Cleaned and processed datasets
├── notebooks/        # Jupyter notebooks for EDA and experiments
├── src
│   ├── data/         # Data loading and preprocessing scripts
│   ├── features/     # Feature engineering code
│   ├── models/       # Model training, evaluation, and persistence
│   ├── visualization/# Visualization/EDA helpers
│   └── utils/        # Config, paths, and other utilities
├── tests/            # Unit tests (pytest)
├── models/           # Saved model artifacts (vectorizer, model, etc.)
├── flask_app.py      # Flask application to serve predictions
├── mlops_pipeline.py # Orchestrated end-to-end pipeline
├── requirements.txt  # Python dependencies
├── Dockerfile        # Containerization for deployment
└── README.md         # Project documentation (this file)
```

## Dataset Overview

This project uses the **WorldVQA** dataset hosted on Hugging Face:

- **Source**: `moonshotai/WorldVQA` (loaded via `datasets.load_dataset`)
- **Modality**: Visual Question Answering (VQA) focused on world knowledge
- **Typical fields** (exact schema should be confirmed by inspecting the dataset):
  - Question text (e.g., `question`)
  - Answer/label (e.g., `answer`)
  - Optional additional fields (e.g., image paths, metadata, etc.)

You can inspect the dataset schema by running:

```bash
python -m src.data.load_data
```

This will print the available splits and a sample example. Make sure to
update any column names in `src/features/build_features.py` and
`mlops_pipeline.py` to match the actual dataset schema.

### Preprocessing

Preprocessing is intentionally kept minimal in the initial version:

- Loading performed via `src.data.load_data.load_worldvqa_dataset`
- Conversion to pandas via `src.features.build_features.dataset_to_dataframe`
- Basic text/label extraction via `src.features.build_features.build_basic_features`

You should extend this with:

- Text normalization (lowercasing, punctuation removal, etc.)
- Advanced NLP or multimodal feature extraction tailored to VQA
- Additional derived features or metadata usage

## Model Development

### Model Selection

The initial baseline model is:

- **Vectorizer**: `TfidfVectorizer` (scikit-learn)
- **Classifier**: `LogisticRegression`

The training logic is implemented in `src.models.train_model.train_baseline_model`.
It:

- Splits the data into train/validation sets
- Fits TF-IDF on the training text
- Trains the classifier
- Prints a **classification report** on the validation set
- Persists the vectorizer and model under `models/`

You should experiment with alternative models (e.g., gradient boosting,
transformer-based encoders, or specialized VQA architectures) and compare:

- Accuracy / F1-score or other relevant metrics
- Inference latency and resource usage

Document model comparisons and decisions in your notebooks and optionally
summarize them here.

### Hyperparameter Tuning

For hyperparameter tuning you can:

- Use dedicated notebooks under `notebooks/` (e.g., `notebooks/hyperparameter_search.ipynb`)
- Or create additional scripts under `src/models/`

Examples of techniques:

- Grid search / Random search (`sklearn.model_selection`)
- Bayesian optimization (e.g., Optuna)

Be sure to:

- Log your experiments (e.g., using MLflow, Weights & Biases, or manual logs)
- Record chosen hyperparameters and rationale in the notebook and/or README

## MLOps and Pipeline

### MLOps Concepts Used

- **Version control**: All code and configuration are intended to be tracked in Git
- **Automated testing**: Unit tests live under `tests/` and are executed with `pytest`
- **Reproducible pipelines**: `mlops_pipeline.py` defines a simple, scriptable pipeline
  that can be invoked from CI/CD tools (e.g., GitHub Actions)
- **Model artifact management**: Trained artifacts are saved in the `models/` directory

You can further enhance this by:

- Adding a GitHub Actions workflow to run tests and the pipeline on each push
- Integrating MLflow for experiment tracking and model registry
- Adding monitoring hooks around the Flask app (e.g., logging, metrics)

### Pipeline Usage

The main entry point is `mlops_pipeline.py`, which supports multiple subcommands:

```bash
# Run the full pipeline: load data and train model
python mlops_pipeline.py run-all

# Just load and inspect the dataset
python mlops_pipeline.py load-data

# Train (or retrain) the baseline model
python mlops_pipeline.py train-model
```

This is a convenient hook for CI/CD: your CI job can run
`python mlops_pipeline.py run-all` after tests pass to retrain and
refresh model artifacts.

## Running the Project Locally

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Explore the dataset

Use the notebooks under `notebooks/` (e.g. `notebooks/eda.ipynb`) and
the utilities in `src/` to perform EDA and experiments.

### 4. Train a model

```bash
python mlops_pipeline.py run-all
```

This will:

- Load the WorldVQA dataset
- Build basic features
- Train the baseline model
- Save artifacts into `models/`

## Deployment

### Running the Flask App Locally

After training a model (ensuring `models/model.joblib` and
`models/vectorizer.joblib` exist), you can start the Flask server:

```bash
export FLASK_APP=flask_app.py           # On Windows: set FLASK_APP=flask_app.py
flask run --host 0.0.0.0 --port 5000
```

Or using the module directly:

```bash
python flask_app.py
```

Once running, you can test the API:

```bash
curl -X POST "http://localhost:5000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"texts\": [\"Example question here\"]}"
```

You should receive a JSON response with model predictions.

### Docker Deployment

To build and run the Docker image:

```bash
# Build the image
docker build -t worldvqa-mlops .

# Run the container
docker run -p 8000:8000 worldvqa-mlops
```

The application will be served via Gunicorn on port `8000` inside the
container, mapped to `localhost:8000` on your machine. You can then call:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"texts\": [\"Example question here\"]}"
```

### MLOps Integration Notes

For CI/CD, a typical setup might include:

- On each push or pull request:
  - Install dependencies
  - Run `pytest` for unit tests
  - Optionally run `python mlops_pipeline.py run-all` for training
- On main-branch merges:
  - Build and push the Docker image to a registry
  - Deploy the updated container to your environment (e.g., Kubernetes, ECS, etc.)

These steps can be implemented via GitHub Actions, GitLab CI, or any
other CI/CD service of your choice.

## Testing

Unit tests live in the `tests/` directory and use **pytest**.

To run the test suite:

```bash
pytest
```

Add more tests as you extend the codebase (e.g., for new preprocessing,
feature engineering, modeling, or inference logic).

## Notes and Next Steps

- Confirm actual WorldVQA schema and update column names accordingly
- Enhance feature engineering and modeling for VQA-specific tasks
- Integrate experiment tracking (e.g., MLflow)
- Add CI/CD configuration (e.g., GitHub Actions) to automate tests and pipeline runs
## Project Maintainer

- **Name**: Muhammad Farooq
- **Email**: mfarooqshafee333@gmail.com
- **GitHub**: [Muhammad-Farooq-13](https://github.com/Muhammad-Farooq-13)

