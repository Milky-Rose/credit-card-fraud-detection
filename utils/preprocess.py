"""
utils/preprocess.py
--------------------
Shared preprocessing helpers used by both train_model.py and app.py.
"""

import io
import numpy as np
import pandas as pd


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Fill numeric NaNs with column median.
    - Drop any fully-empty columns.
    """
    # Drop fully-empty columns
    df.dropna(axis=1, how="all", inplace=True)

    # Fill remaining numeric NaNs with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    return df


def prepare_upload(file_storage, feature_names: list, scaler) -> np.ndarray:
    """
    Read an uploaded CSV (Flask FileStorage), align columns to training
    feature set, scale, and return a numpy array ready for prediction.

    Parameters
    ----------
    file_storage  : werkzeug.datastructures.FileStorage  (request.files['file'])
    feature_names : list of str  –  columns the model was trained on
    scaler        : fitted StandardScaler

    Returns
    -------
    X_scaled : np.ndarray  shape (n_rows, n_features)
    n_rows   : int
    """
    content = file_storage.read()
    df = pd.read_csv(io.BytesIO(content))

    # Drop label column if present (user may upload labelled data)
    if "Class" in df.columns:
        df.drop(columns=["Class"], inplace=True)

    df = clean_dataframe(df)

    # Align to training features: add missing cols as 0, drop extras
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_names]   # reorder to match training order

    X_scaled = scaler.transform(df.values.astype(float))
    return X_scaled, len(df)


def iforest_predict(model, X: np.ndarray) -> np.ndarray:
    """
    IsolationForest returns +1 (normal) / -1 (anomaly).
    Map to 0 / 1 to match the fraud label convention.
    """
    raw = model.predict(X)
    return np.where(raw == -1, 1, 0)