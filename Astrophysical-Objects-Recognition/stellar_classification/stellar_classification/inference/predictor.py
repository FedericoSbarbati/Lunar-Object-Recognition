"""Inference helpers: test evaluation and feature importance."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import shap
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_test_set(
    y_test,
    y_pred,
    model_name: str = 'Model',
) -> dict:
    """Compute test metrics and return as a dict."""
    return {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred) * 100,
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }


def evaluate_neural(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    model_name: str = 'Neural Network',
) -> dict:
    """Run inference on test loader and return metrics."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            _, pred = model(Xb).max(1)
            preds.extend(pred.cpu().numpy())
            trues.extend(yb.numpy())
    preds = np.asarray(preds)
    trues = np.asarray(trues)
    return {
        'model': model_name,
        'accuracy': accuracy_score(trues, preds) * 100,
        'precision': precision_score(trues, preds, average='macro'),
        'recall': recall_score(trues, preds, average='macro'),
        'f1': f1_score(trues, preds, average='macro'),
        'confusion_matrix': confusion_matrix(trues, preds),
    }


def compute_permutation_importance(
    model,
    X_test,
    y_test,
    feature_names,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance and return sorted DataFrame."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    imp = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)
    return imp


def compute_shap(
    model,
    X_train_background,
    X_test_subset,
    feature_names,
    background_size: int = 100,
) -> shap.KernelExplainer:
    """Return a SHAP KernelExplainer fitted on the model."""
    def predict_fn(X):
        return model.predict(X)

    explainer = shap.KernelExplainer(predict_fn, X_train_background[:background_size])
    shap_values = explainer.shap_values(X_test_subset)
    return explainer, shap_values
