"""
app.py
------
Flask backend for the Credit Card Fraud Detection application.

Routes
------
GET  /          → renders the upload homepage (index.html)
POST /predict   → accepts CSV upload, runs predictions, renders result.html

Pre-trained models are loaded ONCE at startup from model/saved_models.pkl.
DO NOT retrain on every request.
"""

import os
import sys
import pickle
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for, flash
)
from werkzeug.utils import secure_filename

# ── make sure utils/ is importable regardless of working directory ─────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.preprocess import prepare_upload, iforest_predict

# ── app config ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
import pandas as pd

# df = pd.read_csv("dataset/creditcard.csv")
# df.sample(2000).to_csv("sample.csv", index=False)
app.secret_key = "fraud-detection-secret-2024"

ALLOWED_EXTENSIONS = {"csv"}
MAX_CONTENT_LENGTH  = 200 * 1024 * 1024   # 200 MB upload limit
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "saved_models.pkl")


# ── load models at startup ─────────────────────────────────────────────────────
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        print(
            f"[WARN] Model file not found at {MODEL_PATH}.\n"
            "       Run  python model/train_model.py --data <csv>  first."
        )
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


BUNDLE = load_bundle()   # { scaler, feature_names, models: {name: {model, accuracy, ...}} }


# ── helpers ────────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_predictions(file_storage) -> dict:
    """
    Core prediction logic.

    Returns a dict ready to be passed to result.html:
    {
        total        : int,
        fraud        : int,
        legit        : int,
        fraud_pct    : float,
        legit_pct    : float,
        model_results: [ {name, accuracy, error_rate, fraud, legit}, … ]
    }
    """
    if BUNDLE is None:
        raise RuntimeError("Models not loaded. Train models first.")

    scaler        = BUNDLE["scaler"]
    feature_names = BUNDLE["feature_names"]
    models_meta   = BUNDLE["models"]

    X_scaled, n_rows = prepare_upload(file_storage, feature_names, scaler)

    model_results = []
    # Use Random Forest as the "primary" model for the summary card
    primary_fraud = primary_legit = 0

    for idx, (name, meta) in enumerate(models_meta.items()):
        model = meta["model"]

        # Predict
        if name == "Isolation Forest":
            preds = iforest_predict(model, X_scaled)
        else:
            preds = model.predict(X_scaled)

        fraud = int(np.sum(preds == 1))
        legit = int(np.sum(preds == 0))

        if name == "Random Forest" or idx == 0:
            primary_fraud = fraud
            primary_legit = legit

        model_results.append({
            "name":       name,
            "accuracy":   f"{meta['accuracy'] * 100:.2f}",
            "error_rate": f"{meta['error_rate'] * 100:.2f}",
            "fraud":      fraud,
            "legit":      legit,
        })

    fraud_pct = round(primary_fraud / n_rows * 100, 2) if n_rows else 0
    legit_pct = round(100 - fraud_pct, 2)

    return {
        "total":         n_rows,
        "fraud":         primary_fraud,
        "legit":         primary_legit,
        "fraud_pct":     fraud_pct,
        "legit_pct":     legit_pct,
        "model_results": model_results,
    }


# ── routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Homepage – render file-upload form."""
    models_ready = BUNDLE is not None
    model_names  = list(BUNDLE["models"].keys()) if models_ready else []
    return render_template("index.html", models_ready=models_ready, model_names=model_names)


@app.route("/predict", methods=["POST"])
def predict():
    """Accept CSV upload, run models, return results page."""

    # ── validate upload ───────────────────────────────────────────────────────
    if "file" not in request.files:
        flash("No file part in the request.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected. Please choose a CSV file.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Only CSV files are accepted.", "error")
        return redirect(url_for("index"))

    if BUNDLE is None:
        flash(
            "Models are not trained yet. "
            "Run  python model/train_model.py --data <your_csv>  and restart.",
            "error"
        )
        return redirect(url_for("index"))

    # ── run predictions ───────────────────────────────────────────────────────
    try:
        results = run_predictions(file)
    except Exception as exc:
        flash(f"Prediction failed: {exc}", "error")
        return redirect(url_for("index"))

    return render_template(
        "result.html",
        filename=secure_filename(file.filename),
        **results
    )


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)