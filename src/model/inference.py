# src/model/inference.py
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "model" / "xgboost_model.pkl"

FEATURE_COLS = [
    "payer",
    "patient_age",
    "ICD_10_code",
    "CPT_code",
    "modifier",
    "documentation_quality_score",
    "prior_auth_obtained",
    "billed_amount",
    "submission_days_delay",
    "past_denial_count",
]

_model = joblib.load(MODEL_PATH)

def predict_claim(features: dict):
    df = pd.DataFrame([features], columns=FEATURE_COLS)
    proba = _model.predict_proba(df)[0][1]
    label = "Denied" if proba >= 0.5 else "Approved"
    return {"denial_probability": float(proba), "predicted_status": label}
