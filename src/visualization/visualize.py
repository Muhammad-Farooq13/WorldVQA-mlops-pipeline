"""
Visualization helpers for the WorldVQA project.

These functions are meant to be used from notebooks or scripts
to quickly inspect label distributions, question lengths, etc.
"""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_label_distribution(df: pd.DataFrame, label_column: str, top_k: Optional[int] = 20) -> None:
    """
    Plot the distribution of labels.
    """
    counts = df[label_column].value_counts()
    if top_k is not None:
        counts = counts.head(top_k)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Label distribution")
    plt.tight_layout()
    plt.show()


def plot_question_length_distribution(df: pd.DataFrame, text_column: str) -> None:
    """
    Plot the distribution of question lengths.
    """
    lengths = df[text_column].astype(str).str.len()
    plt.figure(figsize=(8, 4))
    sns.histplot(lengths, bins=50, kde=True)
    plt.title("Question length distribution")
    plt.xlabel("Length (characters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

