"""Data preprocessing: outlier removal, scaling, splitting, SMOTE, and tensor conversion."""

import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import DataLoader, TensorDataset


# Metadata columns that carry no predictive signal
METADATA_COLUMNS = ['run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'fiber_ID', 'obj_ID']


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers using the IQR method on all numeric columns.

    Returns a **copy** of the DataFrame with outlier rows dropped.
    """
    df = df.copy()
    for col in df.select_dtypes(include='number').columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def prepare_splits(
    df: pd.DataFrame,
    target_col: str = 'class',
    test_size: float = 0.2,
    val_ratio: float = 0.25,
    random_state: int = 42,
    apply_outlier_removal: bool = True,
) -> tuple:
    """Encode labels, (optionally) remove outliers, split, scale and apply SMOTE.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (loaded from CSV).
    target_col : str
        Name of the target column.
    test_size : float
        Fraction of data reserved for the test set.
    val_ratio : float
        Fraction of the remaining training data reserved for validation
        (0.25 × 0.80 = 0.20 of total).
    random_state : int
        Seed for reproducibility.
    apply_outlier_removal : bool
        Whether to call :func:`remove_outliers` before splitting.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
        Feature arrays (scaled; X_train additionally over-sampled via SMOTE).
    y_train, y_val, y_test : np.ndarray
        Label arrays.
    label_encoder : LabelEncoder
        Fitted encoder (use `.inverse_transform` to recover class names).
    scaler : StandardScaler
        Fitted scaler (re-use for unseen inference data).
    feature_names : list[str]
        Ordered list of feature column names (useful for permutation importance).
    """
    df = df.copy()

    # ── 1. Label-encode the target ────────────────────────────────────────────
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])

    # ── 2. Drop metadata columns ──────────────────────────────────────────────
    df.drop(columns=METADATA_COLUMNS, inplace=True, errors='ignore')

    # ── 3. (Optional) remove outliers ─────────────────────────────────────────
    if apply_outlier_removal:
        n_before = len(df)
        df = remove_outliers(df)
        print(f"Outliers removed: {n_before - len(df):,} rows  ({len(df):,} remain)")

    # ── 4. Feature / target split ─────────────────────────────────────────────
    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].values
    y = df[target_col].values

    # ── 5. Train / val / test split ───────────────────────────────────────────
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_ratio, random_state=random_state, stratify=y_tv
    )

    # ── 6. Standardize ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ── 7. SMOTE on training set ──────────────────────────────────────────────
    smote = SMOTE(random_state=1)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    gc.collect()
    return X_train, X_val, X_test, y_train, y_val, y_test, le, scaler, feature_names


def to_dataloaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    batch_size: int = 64,
) -> tuple:
    """Convert numpy arrays to PyTorch ``DataLoader`` objects.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    def _make_loader(X, y, shuffle):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (
        _make_loader(X_train, y_train, shuffle=True),
        _make_loader(X_val,   y_val,   shuffle=False),
        _make_loader(X_test,  y_test,  shuffle=False),
    )
