"""
Flask application for serving the WorldVQA model.

This app exposes a simple `/predict` endpoint that accepts JSON input
and returns model predictions. It is designed to be run either directly
with Flask or via Gunicorn inside a Docker container.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request

from src.models.train_model import predict


def create_app(model_dir: Path | None = None) -> Flask:
    """
    Application factory for the Flask app.

    Parameters
    ----------
    model_dir : Path, optional
        Path to the directory containing model artifacts.
    """
    app = Flask(__name__)

    if model_dir is None:
        model_dir = Path("models")

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify({"status": "ok"}), 200

    @app.route("/predict", methods=["POST"])
    def predict_endpoint() -> Any:
        payload: Dict[str, Any] = request.get_json(force=True) or {}
        texts: List[str] = payload.get("texts", [])

        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            return jsonify({"error": "`texts` must be a list of strings"}), 400

        try:
            preds = predict(texts, model_dir=model_dir)
            return jsonify({"predictions": preds.tolist()}), 200
        except FileNotFoundError:
            return (
                jsonify(
                    {
                        "error": "Model artifacts not found. "
                        "Train and save a model before calling /predict."
                    }
                ),
                500,
            )

    return app


if __name__ == "__main__":
    # For local development only; prefer Gunicorn in production.
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)

