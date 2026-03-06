# src/rules/remediate.py

REMEDIATION_MAP = {
    "CO-45": "Downgrade CPT code or align with payer fee schedule (charges exceed allowed amount).",
    "CO-97": "Add modifier 59 or check bundling rules – service is bundled into another procedure.",
    "CO-4": "Missing or invalid modifier – Medicare requires specific modifier for this CPT.",
    "Missing Prior Auth": "Obtain prior authorization before resubmission.",
    "Non-Billable": "Improve documentation: add exam findings, laterality, and medical necessity statement.",
}

DEFAULT_REMEDIATION = "Review payer policy, documentation, and coding for this claim before resubmission."

def get_remediation(denial_reason_code: str, predicted_status: str) -> str:
    """
    Returns a human-readable remediation suggestion.
    - If claim predicted Approved -> give a reassurance / best-practice tip.
    - If Denied -> use REMEDIATION_MAP or fallback.
    """
    if predicted_status == "Approved":
        return "Low predicted denial risk. Ensure documentation and prior auth are in place before submission."

    # For Denied predictions
    if denial_reason_code in REMEDIATION_MAP:
        return REMEDIATION_MAP[denial_reason_code]

    # Fallback when reason is unknown or generic
    return DEFAULT_REMEDIATION
