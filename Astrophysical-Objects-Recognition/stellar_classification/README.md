# stellar_classification

Stellar object classification pipeline for SDSS data using traditional ML, ensemble voting, and PyTorch neural networks with SHAP interpretability.

## Feature Classes

- **0**: Dwarf Star (main-sequence)
- **1**: Giant Star
- **2**: White Dwarf

## Package Structure

```
stellar_classification/
├── data/preprocessing.py   — Outlier removal, StandardScaler, SMOTE, train/val/test splits, DataLoader creation
├── models/network.py       — SimpleNN (128-hidden linear network)
├── trainer.py              — train_traditional(), train_voting(), train_neural(), compute_metrics()
├── inference/predictor.py  — Test evaluation, permutation importance, SHAP KernelExplainer
├── visualization/          — Confusion matrix, feature importance, class distribution plots
└── utils/metrics.py        — Print-friendly metric formatting
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage from Notebook

```python
import sys
sys.path.insert(0, 'stellar_classification')

from stellar_classification.data.preprocessing import prepare_splits, to_dataloaders
from stellar_classification.models.network import SimpleNN
from stellar_classification.trainer import train_traditional, train_voting, train_neural
from stellar_classification.inference.predictor import evaluate_test_set, evaluate_neural, compute_permutation_importance, compute_shap
from stellar_classification.visualization import plot_confusion_matrix, plot_permutation_importance
from stellar_classification.utils.metrics import print_metrics

# Data pipeline
X_train, X_val, X_test, y_train, y_val, y_test, le = prepare_splits(df)
train_loader, val_loader, test_loader = to_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test)

# Traditional ML
models = train_traditional(X_train, y_train, X_val, y_val, X_test, y_test)

# Voting ensemble
voting_clf = train_voting(X_train, y_train, X_val, y_val)

# Neural network
nn_model = train_neural(train_loader, val_loader, input_size=X_train.shape[1], num_classes=len(le.classes_))
```

## Training Config

See `configs/training_config.yaml` for hyperparameters.
