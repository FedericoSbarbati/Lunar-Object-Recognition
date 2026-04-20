"""Print-friendly metrics formatting."""

import pandas as pd


def print_metrics(metrics: dict):
    """Pretty-print a dict of classification metrics."""
    print(f"\n{metrics['model']}:")
    for key in ['accuracy', 'precision', 'recall', 'f1']:
        if key in metrics:
            print(f"  {key.capitalize()}: {metrics[key]:.2f}%")
    if 'confusion_matrix' in metrics:
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
