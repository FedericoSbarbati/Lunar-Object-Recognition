"""Data preprocessing: outlier removal, scaling, splitting, SMOTE, and tensor conversion."""

import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch


OUTLIER_COLUMNS = ['run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'fiber_ID', 'obj_ID']


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers using the IQR method on all numeric columns."""
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, LabelEncoder]:
    """Load CSV, encode labels, remove outliers, split, scale, and apply SMOTE.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test, label_encoder).
    """
    # Label encode
    le = LabelEncoder()
    df = df.copy()
    df[target_col] = le.fit_transform(df[target_col])

    # Drop metadata columns
    df.drop(columns=OUTLIER_COLUMNS, inplace=True, errors='ignore')

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train / test split, then train / validation split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
    )

    # Standardize
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)

    # SMOTE on training set
    smote = SMOTE(random_state=1)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    gc.collect()
    return X_train, X_val, X_test, y_train, y_val, y_test, le


def to_dataloaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    batch_size: int = 64,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Convert numpy arrays to PyTorch DataLoaders (long for classification)."""
    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
