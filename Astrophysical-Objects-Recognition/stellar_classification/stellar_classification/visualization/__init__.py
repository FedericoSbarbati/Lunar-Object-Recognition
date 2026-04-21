"""Visualization helpers for stellar classification."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_class_distribution(y, title: str = 'Class Distribution') -> None:
    """Bar chart of class frequencies.

    Parameters
    ----------
    y : array-like
        Target labels (raw strings or encoded integers).
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    unique, counts = np.unique(y, return_counts=True)
    ax.bar([str(u) for u in unique], counts, color=sns.color_palette('Set3', len(unique)))
    ax.set_title(title)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names=None, title: str = 'Confusion Matrix') -> None:
    """Heatmap of a confusion matrix.

    Parameters
    ----------
    cm : array-like of shape (n_classes, n_classes)
        Confusion matrix from ``sklearn.metrics.confusion_matrix``.
    class_names : list[str], optional
        Tick labels for axes.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    heatmap = sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names or 'auto',
        yticklabels=class_names or 'auto',
        ax=ax,
    )
    heatmap.set_title(title)
    heatmap.set_xlabel('Predicted')
    heatmap.set_ylabel('Actual')
    plt.tight_layout()
    plt.show()


def plot_permutation_importance(
    imp: pd.Series,
    top_n: int = 10,
    title: str = 'Permutation Feature Importance',
) -> None:
    """Horizontal bar chart of permutation importances.

    Parameters
    ----------
    imp : pd.Series
        Sorted importances (index = feature names) from
        :func:`~stellar_classification.inference.predictor.compute_permutation_importance`.
    top_n : int
        How many features to display.
    title : str
        Plot title.
    """
    top = imp.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index, top.values)
    ax.set_xlabel('Mean Importance')
    ax.set_title(f'{title} (Top {top_n})')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
