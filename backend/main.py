import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any

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

def apply_rules_engine(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Centralized Rules Engine to ensure consistency between endpoints.
    Strict medical billing rules override ML predictions.
    """
    icd = features.get("ICD_10_code")
    cpt = features.get("CPT_code")
    auth = features.get("prior_auth_obtained", 0)
    doc_q = features.get("documentation_quality_score", 1.0)

    # Rule A: Sinusitis + ECG (Medical Necessity Conflict)
    if icd == "J01.90" and cpt == "93000":
        return {
            "denial_probability": 0.99,
            "predicted_status": "Denied",
            "denial_reason_code": "CO-50 (Medical Necessity)",
            "remediation": "Remove CPT 93000. An ECG is not medically necessary for Acute Sinusitis."
        }

    # Rule B: Back Pain + Blood Test (Michael Chen Case)
    if icd == "M54.50" and cpt == "85025":
        return {
            "denial_probability": 0.98,
            "predicted_status": "Denied",
            "denial_reason_code": "CO-50 (Medical Necessity)",
            "remediation": "Remove CPT 85025. Routine blood work (CBC) is not indicated for mechanical back pain."
        }

    # Rule C: Level 5 Visit Without Prior Auth
    if cpt == "99215" and auth == 0:
        return {
            "denial_probability": 0.85,
            "predicted_status": "Denied",
            "denial_reason_code": "Missing Prior Auth",
            "remediation": "Level 5 complex visits require prior authorization. Please attach auth number."
        }

    # Rule D: Upcoding Fraud Check
    if icd == "J01.90" and cpt == "99215":
        return {
            "denial_probability": 0.95,
            "predicted_status": "Denied",
            "denial_reason_code": "CO-11 (Inconsistent with Diagnosis)",
            "remediation": "Downcode to 99213. A routine sinus infection does not meet Level 5 criteria."
        }

    return None # No rules triggered, proceed to ML model

def simple_reason_from_features(features: dict, predicted_status: str) -> str:
    if predicted_status == "Approved":
        return "Approved"
    if features["modifier"] == "None" and features["CPT_code"] in ["99214", "99215"]:
        return "CO-4"
    return "CO-45"

@app.post("/predict")
def predict_endpoint(claim: ClaimFeatures):
    features = claim.model_dump()
    
    # 1. Check Rules Engine First (Veto Power)
    rule_result = apply_rules_engine(features)
    if rule_result:
        return rule_result

    # 2. Fallback to ML Model
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
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    ocr_result = extract_from_note(temp_path)
    icd_list = ocr_result.get("icd10", []) or ["J01.90"]
    cpt_list = ocr_result.get("cpt", []) or ["99213"]

    worst_prob = -1.0
    final_output = {}

    for icd in icd_list:
        for cpt in cpt_list:
            features = {
                "payer": payer,
                "patient_age": 45,
                "ICD_10_code": icd,
                "CPT_code": cpt,
                "modifier": "None",
                "documentation_quality_score": 0.8,
                "prior_auth_obtained": 0,
                "billed_amount": 150.0,
                "submission_days_delay": 3,
                "past_denial_count": 0,
            }

            # Use the shared predict logic to ensure consistency
            current_result = predict_endpoint(ClaimFeatures(**features))
            
            if current_result["denial_probability"] > worst_prob:
                worst_prob = current_result["denial_probability"]
                final_output = {
                    "ocr": ocr_result,
                    "default_features": features,
                    **current_result
                }

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return final_output