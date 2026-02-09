"""
Simple MLOps pipeline orchestration script for the WorldVQA project.

This script is intentionally lightweight and focuses on:
- reproducible data loading
- feature engineering
- model training and evaluation
- saving artifacts for deployment

It is structured so that it can be wired into CI/CD (e.g., GitHub Actions)
by invoking `python mlops_pipeline.py run-all` as part of an automated job.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.load_data import load_worldvqa_dataset
from src.features.build_features import build_basic_features, dataset_to_dataframe
from src.models.train_model import train_baseline_model
from src.utils.config import ensure_directories


def step_load_data() -> None:
    print("Loading WorldVQA dataset...")
    ds = load_worldvqa_dataset()
    print(ds)


def step_train_model(model_dir: Path = Path("models")) -> None:
    print("Training baseline model...")
    ensure_directories()

    ds = load_worldvqa_dataset(split="train")
    df = dataset_to_dataframe(ds)

    # NOTE: Replace `question` and `answer` with the actual column names
    # from the WorldVQA dataset schema.
    text_column = "question"
    label_column = "answer"

    X, y = build_basic_features(df, text_column=text_column, label_column=label_column)
    train_baseline_model(X, y, model_dir=model_dir)


def run_all() -> None:
    """
    Run the end-to-end pipeline: load data, train model.
    """
    step_load_data()
    step_train_model()
    print("Pipeline completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldVQA MLOps pipeline")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("load-data", help="Load and inspect the WorldVQA dataset")
    subparsers.add_parser("train-model", help="Train the baseline model")
    subparsers.add_parser("run-all", help="Run the full pipeline")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "load-data":
        step_load_data()
    elif args.command == "train-model":
        step_train_model()
    else:
        # Default to running the full pipeline
        run_all()


if __name__ == "__main__":
    main()

