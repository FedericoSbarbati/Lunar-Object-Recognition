"""stellar_classification — top-level package for SDSS stellar-object classification.

Quick-start
-----------
>>> from stellar_classification.data.preprocessing import prepare_splits, to_dataloaders
>>> from stellar_classification.trainer import train_traditional, train_voting, train_neural
>>> from stellar_classification.inference.predictor import (
...     evaluate_test_set, evaluate_neural, compute_permutation_importance
... )
>>> from stellar_classification.visualization import (
...     plot_class_distribution, plot_confusion_matrix, plot_permutation_importance
... )
>>> from stellar_classification.utils.metrics import print_metrics
"""

from .data.preprocessing import (  # noqa: F401
    remove_outliers,
    prepare_splits,
    to_dataloaders,
)
from .trainer import (  # noqa: F401
    compute_metrics,
    train_traditional,
    train_voting,
    train_neural,
)
from .inference.predictor import (  # noqa: F401
    evaluate_test_set,
    evaluate_neural,
    compute_permutation_importance,
)
from .visualization import (  # noqa: F401
    plot_class_distribution,
    plot_confusion_matrix,
    plot_permutation_importance,
)
from .utils.metrics import print_metrics  # noqa: F401
from .models.network import SimpleNN     # noqa: F401

__all__ = [
    # data
    'remove_outliers', 'prepare_splits', 'to_dataloaders',
    # training
    'compute_metrics', 'train_traditional', 'train_voting', 'train_neural',
    # inference
    'evaluate_test_set', 'evaluate_neural', 'compute_permutation_importance',
    # visualization
    'plot_class_distribution', 'plot_confusion_matrix', 'plot_permutation_importance',
    # utils
    'print_metrics',
    # models
    'SimpleNN',
]
