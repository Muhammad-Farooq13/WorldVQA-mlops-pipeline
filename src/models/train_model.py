"""
Model training and evaluation for the WorldVQA project.

This module currently implements a simple baseline model using
scikit-learn. It is structured so that more advanced models
can be plugged in later while keeping the training loop and
evaluation pipeline stable.
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def train_baseline_model(
    X_text: pd.Series,
    y: pd.Series,
    model_dir: Path = Path("models"),
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """
    Train a simple TF-IDF + Logistic Regression baseline model.

    Parameters
    ----------
    X_text : pd.Series
        Text features (e.g., questions).
    y : pd.Series
        Target labels.
    model_dir : Path
        Directory where the model artifacts will be stored.

    Returns
    -------
    vectorizer, model
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_text.values, y.values, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_text.values, y.values, test_size=0.2, random_state=42
        )

    vectorizer = TfidfVectorizer(max_features=20000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_val_vec)
    print("Validation classification report:")
    print(classification_report(y_val, y_pred))

    # Persist artifacts
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    joblib.dump(model, model_dir / "model.joblib")

    return vectorizer, model


def load_model_artifacts(model_dir: Path = Path("models")) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """
    Load a previously trained vectorizer and model from disk.
    """
    vectorizer = joblib.load(model_dir / "vectorizer.joblib")
    model = joblib.load(model_dir / "model.joblib")
    return vectorizer, model


def predict(
    texts: list[str],
    model_dir: Path = Path("models"),
) -> np.ndarray:
    """
    Run predictions for a list of input texts.
    """
    vectorizer, model = load_model_artifacts(model_dir)
    X_vec = vectorizer.transform(texts)
    return model.predict(X_vec)


if __name__ == "__main__":
    # Minimal example stub (would be wired to real data in pipelines/notebooks).
    example_X = pd.Series(["example question 1", "example question 2"])
    example_y = pd.Series([0, 1])
    train_baseline_model(example_X, example_y)

