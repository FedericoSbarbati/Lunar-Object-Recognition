"""Print-friendly metrics formatting."""


def print_metrics(metrics: dict) -> None:
    """Pretty-print a classification metrics dict.

    Expects the dict produced by :func:`~stellar_classification.trainer.compute_metrics`
    or :func:`~stellar_classification.inference.predictor.evaluate_test_set`.

    ``accuracy`` is assumed to be stored as a percentage (0-100).
    """
    dataset = f" on {metrics['dataset']}" if 'dataset' in metrics else ''
    print(f"\n{metrics['model']}{dataset}:")
    print(f"  Accuracy  : {metrics['accuracy']:.2f}%")
    for key in ('precision', 'recall', 'f1'):
        if key in metrics:
            print(f"  {key.capitalize():<9}: {metrics[key]:.4f}")
    if 'confusion_matrix' in metrics:
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
