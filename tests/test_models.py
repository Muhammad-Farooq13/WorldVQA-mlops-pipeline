from pathlib import Path

import numpy as np
import pandas as pd

from src.models.train_model import predict, train_baseline_model


def test_train_and_predict_baseline_model(tmp_path: Path):
    # Minimal synthetic data just to validate the training loop
    X = pd.Series(["question one", "question two", "question three", "question four"])
    y = pd.Series([0, 1, 0, 1])

    model_dir = tmp_path / "models"
    vectorizer, model = train_baseline_model(X, y, model_dir=model_dir)

    assert vectorizer is not None
    assert model is not None

    preds = predict(["new question"], model_dir=model_dir)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (1,)

