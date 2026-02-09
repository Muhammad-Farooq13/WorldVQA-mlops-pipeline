"""
Data loading utilities for the WorldVQA project.

This module wraps Hugging Face `datasets` to load the
`moonshotai/WorldVQA` dataset and provides helpers to convert it
into pandas-friendly formats for downstream processing.
"""

from typing import Optional

from datasets import DatasetDict, load_dataset


def load_worldvqa_dataset(split: Optional[str] = None) -> DatasetDict:
    """
    Load the WorldVQA dataset from Hugging Face Datasets.

    Parameters
    ----------
    split : Optional[str]
        If provided, return only the specified split
        (e.g., "train", "validation", "test"). If None,
        returns the full DatasetDict.

    Returns
    -------
    DatasetDict or Dataset
        The loaded dataset (or a single split).
    """
    ds = load_dataset("moonshotai/WorldVQA")
    if split is not None:
        return ds[split]
    return ds


def main() -> None:
    """Small CLI entrypoint for manual inspection."""
    ds = load_worldvqa_dataset()
    print(ds)
    if "train" in ds:
        print("First training example:")
        print(ds["train"][0])


if __name__ == "__main__":
    main()

