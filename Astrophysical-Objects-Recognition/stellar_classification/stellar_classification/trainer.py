"""Traditional ML trainer: fits ensemble models, evaluates, and produces metrics."""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import gc

from .models.network import SimpleNN


MODEL_MAP = {
    'LinearSVC': lambda: LinearSVC(),
    'DecisionTree': lambda: DecisionTreeClassifier(),
    'RandomForest': lambda: RandomForestClassifier(),
    'CatBoost': lambda: CatBoostClassifier(task_type='GPU' if torch.cuda.is_available() else 'CPU', verbose=0),
    'LightGBM': lambda: LGBMClassifier(device='gpu' if torch.cuda.is_available() else 'cpu'),
}


def compute_metrics(y_true, y_pred, dataset_name: str, model_name: str) -> dict:
    """Compute and return metrics as a dict."""
    return {
        'dataset': dataset_name,
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }


def train_traditional(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> dict[str, object]:
    """Train all traditional ML models and return fitted estimators."""
    models = {name: fn() for name, fn in MODEL_MAP.items()}
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def train_voting(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> VotingClassifier:
    """Train a hard VotingClassifier over all traditional models."""
    models = train_traditional(X_train, y_train, X_val, y_val, X_val, y_val)
    estimators = [
        ('svc', models['LinearSVC']),
        ('dt', models['DecisionTree']),
        ('rf', models['RandomForest']),
        ('catboost', models['CatBoost']),
        ('lgbm', models['LightGBM']),
    ]

    class _Voting(VotingClassifier):
        def _predict(self, X):
            return np.asarray([e.predict(X).ravel() for e in self.estimators_]).T

    return _Voting(estimators=estimators, voting='hard')


def train_neural(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    input_size: int,
    num_classes: int,
    num_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.001,
) -> SimpleNN:
    """Train the PyTorch neural network and return the fitted model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNN(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        # Validation accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                _, pred = model(Xb).max(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        print(f'Epoch {epoch+1}/{num_epochs}, Val Acc: {100*correct/total:.2f}%')

    gc.collect()
    return model
