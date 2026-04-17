# FraudShield · Credit Card Fraud Detection

ML-powered Flask web app that runs uploaded transaction CSVs through four trained models
and returns fraud counts + accuracy metrics — without retraining on every request.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models (one-time)
Download the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(`creditcard.csv`), then run:

```bash
python model/train_model.py --data path/to/creditcard.csv
```

This creates `model/saved_models.pkl` containing all four trained models + scaler.

### 3. Start the Flask server
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 4. Upload a CSV and get results
Upload any CSV that matches the dataset schema (columns V1–V28, Amount, Time).
The `Class` column is optional — it will be dropped automatically.

---

## Project Structure

```
credit-card-fraud-detection/
├── app.py                  ← Flask app (routes: / and /predict)
├── requirements.txt
├── model/
│   ├── train_model.py      ← One-time training script
│   └── saved_models.pkl    ← Generated after training
├── utils/
│   └── preprocess.py       ← Shared cleaning + scaling helpers
├── templates/
│   ├── index.html          ← Upload homepage
│   └── result.html         ← Results dashboard
└── static/
    └── style.css           ← Full custom stylesheet
```

## Models

| Model               | Notes                                              |
|---------------------|----------------------------------------------------|
| Logistic Regression | Fast baseline, good on linearly separable data     |
| Random Forest       | Strong ensemble, handles class imbalance well      |
| SVM (RBF kernel)    | Subsampled to 50k rows if dataset is large         |
| Isolation Forest    | Unsupervised anomaly detector, no labels at train  |

## Dataset Format

Expected: Kaggle `creditcard.csv`
- Columns: `Time`, `V1`–`V28`, `Amount`, `Class`
- `Class`: 0 = legitimate, 1 = fraud
- The `Class` column is dropped during prediction (upload labelled or unlabelled data)
