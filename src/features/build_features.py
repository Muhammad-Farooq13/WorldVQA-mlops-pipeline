"""
Feature engineering for the WorldVQA project.

This module defines functions that transform the raw dataset
into model-ready feature matrices (e.g., pandas DataFrames,
NumPy arrays, or sparse matrices).
"""

from typing import Tuple

import pandas as pd
from datasets import Dataset


def dataset_to_dataframe(ds_split: Dataset) -> pd.DataFrame:
    """
    Convert a Hugging Face Dataset split into a pandas DataFrame.

    Parameters
    ----------
    ds_split : Dataset
        A split of the WorldVQA dataset (e.g., train/validation).

    Returns
    -------
    pd.DataFrame
        DataFrame with all columns from the dataset.
    """
    return ds_split.to_pandas()


def build_basic_features(df: pd.DataFrame, text_column: str, label_column: str) -> Tuple[pd.Series, pd.Series]:
    """
    Build a simple baseline of features and labels from a DataFrame.

    For a real project, you would replace this with more sophisticated
    NLP or multimodal feature extraction tailored to WorldVQA.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    text_column : str
        Name of the text/question column.
    label_column : str
        Name of the target/label column.

    Returns
    -------
    X, y : Tuple[pd.Series, pd.Series]
        Features (currently the raw text) and labels.
    """
    X = df[text_column]
    y = df[label_column]
    return X, y

