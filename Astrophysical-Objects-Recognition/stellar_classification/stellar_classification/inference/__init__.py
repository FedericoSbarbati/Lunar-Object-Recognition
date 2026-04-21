"""Inference sub-package: evaluation and feature importance."""

from .predictor import (  # noqa: F401
    evaluate_test_set,
    evaluate_neural,
    compute_permutation_importance,
    compute_shap,
)

__all__ = [
    'evaluate_test_set',
    'evaluate_neural',
    'compute_permutation_importance',
    'compute_shap',
]
