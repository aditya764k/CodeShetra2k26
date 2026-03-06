# backend/main.py
import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from src.model.inference import predict_claim
from src.rules.remediate import get_remediation
from src.vision.extract_codes import extract_from_note

app = FastAPI()

class ClaimFeatures(BaseModel):
    payer: str
    patient_age: int
    ICD_10_code: str
    CPT_code: str
    modifier: str
    documentation_quality_score: float
    prior_auth_obtained: int
    billed_amount: float
    submission_days_delay: int
    past_denial_count: int

def simple_reason_from_features(features: dict, predicted_status: str) -> str:
    if predicted_status == "Approved":
        return "Approved"

    if features["prior_auth_obtained"] == 0 and features["CPT_code"] == "99215":
        return "Missing Prior Auth"
    if features["documentation_quality_score"] < 0.65 and features["CPT_code"] in ["99214", "99215"]:
        return "CO-97"
    if features["ICD_10_code"] == "J01.90" and features["CPT_code"] == "93000":
        return "CO-50"
    if features["modifier"] == "None":
        return "CO-4"
    return "CO-45"

@app.post("/predict")
def predict_endpoint(claim: ClaimFeatures):
    features = claim.model_dump() # Using model_dump() for newer Pydantic versions
    model_out = predict_claim(features)
    status = model_out["predicted_status"]
    prob = model_out["denial_probability"]

    reason_code = simple_reason_from_features(features, status)
    remediation = get_remediation(reason_code, status)

    return {
        "denial_probability": prob,
        "predicted_status": status,
        "denial_reason_code": reason_code,
        "remediation": remediation,
    }

@app.post("/full_pipeline")
async def full_pipeline(file: UploadFile = File(...), payer: str = "Medicare"):
    # 1) Save uploaded image temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # 2) OCR: extract ICD & CPT candidates
    ocr_result = extract_from_note(temp_path)
    
    # Ensure we always have at least one code to test, even if OCR fails
    icd_list = ocr_result.get("icd10", []) or ["J01.90"]
    cpt_list = ocr_result.get("cpt", []) or ["99213"]

    # Variables to track the highest-risk code combination
    worst_prob = -1.0
    worst_status = "Approved"
    worst_reason = "Approved"
    worst_remediation = ""
    worst_features = {}

    # 3) Check EVERY combination of extracted codes
    for icd in icd_list:
        for cpt in cpt_list:
            
            # Build features for this specific pair
            features = {
                "payer": payer,
                "patient_age": 45,
                "ICD_10_code": icd,
                "CPT_code": cpt,
                "modifier": "None",
                "documentation_quality_score": 0.8,
                "prior_auth_obtained": 0,  # conservative: No by default
                "billed_amount": 150.0,
                "submission_days_delay": 3,
                "past_denial_count": 0,
            }

            # ==========================================
            # 4) THE RULES ENGINE GUARDRAIL (Runs FIRST)
            # ==========================================
            rule_failed = False
            
            # Rule A: Sinusitis + ECG (Medical Necessity)
            if icd == "J01.90" and cpt == "93000":
                prob = 0.99
                status = "Denied"
                reason_code = "CO-50 (Medical Necessity)"
                remediation = "Remove CPT 93000. An ECG is not medically necessary for Acute Sinusitis."
                rule_failed = True
            
            # Rule B: Level 5 Visit Without Prior Auth
            elif cpt == "99215" and features["prior_auth_obtained"] == 0:
                prob = 0.85
                status = "Denied"
                reason_code = "Missing Prior Auth"
                remediation = "Level 5 complex visits require prior authorization. Please attach auth number."
                rule_failed = True

            # Rule C: Upcoding Fraud
            elif icd == "J01.90" and cpt == "99215":
                prob = 0.95
                status = "Denied"
                reason_code = "CO-11 (Inconsistent with Diagnosis)"
                remediation = "Downcode to 99213. A routine sinus infection does not meet Level 5 criteria."
                rule_failed = True

            # Rule D: Back Pain + Blood Test
            elif icd == "M54.50" and cpt == "85025":
                prob = 0.98
                status = "Denied"
                reason_code = "CO-50 (Medical Necessity)"
                remediation = "Remove CPT 85025. Routine blood work (CBC) is not indicated for mechanical back pain."
                rule_failed = True

            # ==========================================
            # 5) THE ML PREDICTION (Runs only if rules pass)
            # ==========================================
            if not rule_failed:
                model_out = predict_claim(features)
                status = model_out["predicted_status"]
                prob = model_out["denial_probability"]
                reason_code = simple_reason_from_features(features, status)
                remediation = get_remediation(reason_code, status)

            # 6) If this is the highest risk pairing found so far, overwrite the variables
            if prob > worst_prob:
                worst_prob = prob
                worst_status = status
                worst_reason = reason_code
                worst_remediation = remediation
                worst_features = features.copy() 

    # 7) Clean up the temporary image so your hard drive doesn't fill up!
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Return the worst-case scenario to the frontend
    return {
        "ocr": ocr_result,
        "default_features": worst_features, # Passes back the exact codes that caused the denial
        "denial_probability": worst_prob,
        "predicted_status": worst_status,
        "denial_reason_code": worst_reason,
        "remediation": worst_remediation,
    }