# src/model/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # project root
DATA_PATH = BASE_DIR / "data" / "claimshield_training_data_refined.csv"
MODEL_PATH = BASE_DIR / "model" / "xgboost_model.pkl"

def main():
    df = pd.read_csv(DATA_PATH)

    feature_cols = [
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
    X = df[feature_cols]
    y = (df["claim_status"] == "Denied").astype(int)

    cat_cols = ["payer", "ICD_10_code", "CPT_code", "modifier"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)

    print("Train accuracy:", clf.score(X_train, y_train))
    print("Test accuracy:", clf.score(X_test, y_test))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
