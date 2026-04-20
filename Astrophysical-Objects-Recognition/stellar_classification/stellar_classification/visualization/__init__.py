"""Visualization helpers for stellar classification."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_class_distribution(y, title: str = 'Class Distribution'):
    sb = sns.countplot(x=y, palette='Set3')
    sb.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, title: str = 'Confusion Matrix'):
    sb = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    sb.set_title(title)
    sb.set_xlabel('Predicted')
    sb.set_ylabel('Actual')
    plt.tight_layout()
    plt.show()


def plot_permutation_importance(imp: pd.Series, top_n: int = 10, title: str = 'Feature Importance'):
    top = imp.head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(top.index, top.values)
    plt.xlabel('Permutation Importance')
    plt.title(f'{title} (Top {top_n})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
