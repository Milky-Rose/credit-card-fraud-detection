"""
train_model.py
--------------
Script to train fraud detection models on the credit card dataset.
Run this ONCE to generate saved_models.pkl before launching the Flask app.

Expected dataset: CSV with a 'Class' column (0 = legit, 1 = fraud)
and numeric feature columns (e.g., V1–V28, Amount, Time).

Usage:
    python model/train_model.py --data path/to/creditcard.csv
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(MODEL_DIR, "saved_models.pkl")


# ── helpers ────────────────────────────────────────────────────────────────────
def load_and_preprocess(csv_path: str):
    """Load CSV, clean data, scale features, return X_train/test, y_train/test + scaler."""
    print(f"[INFO] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # ── basic validation ──────────────────────────────────────────────────────
    if "Class" not in df.columns:
        sys.exit("[ERROR] Dataset must contain a 'Class' column (0=legit, 1=fraud).")

    print(f"[INFO] Dataset shape : {df.shape}")
    print(f"[INFO] Class distribution:\n{df['Class'].value_counts()}\n")

    # ── handle missing values ─────────────────────────────────────────────────
    missing = df.isnull().sum().sum()
    if missing:
        print(f"[WARN] {missing} missing values found – filling with column median.")
        df.fillna(df.median(numeric_only=True), inplace=True)

    # ── features / label ──────────────────────────────────────────────────────
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # ── scale (fit only on train split to avoid leakage) ─────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"[INFO] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}\n")
    return X_train_sc, X_test_sc, y_train.values, y_test.values, scaler, list(X.columns)


def evaluate(name: str, model, X_test, y_test, is_iforest=False):
    """Print and return accuracy, error-rate, confusion matrix for a model."""
    if is_iforest:
        # IsolationForest returns -1 (anomaly) / +1 (normal) → map to 1/0
        raw  = model.predict(X_test)
        preds = np.where(raw == -1, 1, 0)
    else:
        preds = model.predict(X_test)

    acc   = accuracy_score(y_test, preds)
    err   = 1 - acc
    cm    = confusion_matrix(y_test, preds)

    print(f"── {name} ──")
    print(f"   Accuracy  : {acc:.4f}  |  Error rate: {err:.4f}")
    print(f"   Confusion matrix:\n{cm}")
    print(f"   Report:\n{classification_report(y_test, preds, zero_division=0)}")
    return {"accuracy": round(acc, 4), "error_rate": round(err, 4), "confusion_matrix": cm.tolist()}


# ── training ───────────────────────────────────────────────────────────────────
def train_all(csv_path: str):
    X_tr, X_te, y_tr, y_te, scaler, feature_names = load_and_preprocess(csv_path)

    models_meta = {}

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("[TRAIN] Logistic Regression …")
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_tr, y_tr)
    models_meta["Logistic Regression"] = {
        "model": lr,
        **evaluate("Logistic Regression", lr, X_te, y_te)
    }

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("[TRAIN] Random Forest …")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    models_meta["Random Forest"] = {
        "model": rf,
        **evaluate("Random Forest", rf, X_te, y_te)
    }

    # ── SVM (use linear kernel + probability; subsample if large dataset) ─────
    print("[TRAIN] SVM …")
    # SVM is O(n²) – cap training rows at 50 000 for speed
    MAX_SVM = 50_000
    if len(X_tr) > MAX_SVM:
        print(f"   [WARN] Dataset large – sampling {MAX_SVM:,} rows for SVM training.")
        idx = np.random.default_rng(42).choice(len(X_tr), MAX_SVM, replace=False)
        X_svm, y_svm = X_tr[idx], y_tr[idx]
    else:
        X_svm, y_svm = X_tr, y_tr

    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_svm, y_svm)
    models_meta["SVM"] = {
        "model": svm,
        **evaluate("SVM", svm, X_te, y_te)
    }

    # ── Isolation Forest ──────────────────────────────────────────────────────
    print("[TRAIN] Isolation Forest …")
    contamination = float(y_tr.mean()) or 0.01   # use actual fraud ratio
    iforest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iforest.fit(X_tr)   # unsupervised – no labels needed
    models_meta["Isolation Forest"] = {
        "model": iforest,
        **evaluate("Isolation Forest", iforest, X_te, y_te, is_iforest=True)
    }

    # ── persist everything ────────────────────────────────────────────────────
    bundle = {
        "scaler":        scaler,
        "feature_names": feature_names,
        "models":        models_meta,
    }
    with open(MODELS_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n[SUCCESS] Models saved → {MODELS_PATH}")


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud-detection models.")
    parser.add_argument("--data", required=True, help="Path to creditcard CSV dataset.")
    args = parser.parse_args()
    train_all(args.data)