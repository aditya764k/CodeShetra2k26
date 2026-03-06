import pandas as pd
import numpy as np

np.random.seed(42)
n = 5000

# 1. Base setup: payers, diagnoses, procedures
payers = ['Medicare', 'BlueCross', 'Aetna']

icd10_map = {
    'J01.90': 'Acute sinusitis, unspecified',
    'M54.50': 'Low back pain, unspecified',
    'E11.9':  'Type 2 diabetes mellitus without complications',
    'I10':    'Essential (primary) hypertension',
    'J44.9':  'Chronic obstructive pulmonary disease, unspecified'
}

# ICD → typical CPT + billed amount (correct mapping)
icd_to_cpt = {
    'J01.90': ('99213', 120),  # Sinusitis -> Level 3 office visit
    'M54.50': ('99214', 160),  # Back pain -> Level 4 visit
    'E11.9':  ('85025', 45),   # Diabetes -> CBC/lab
    'I10':    ('93000', 85),   # Hypertension -> ECG
    'J44.9':  ('99215', 250)   # COPD -> Complex visit
}

# 2. Initialize base data
df = pd.DataFrame({
    'payer': np.random.choice(payers, n, p=[0.4, 0.3, 0.3]),
    'ICD_10_code': np.random.choice(
        list(icd10_map.keys()),
        n,
        p=[0.25, 0.25, 0.2, 0.2, 0.1]  # sinus/back more common
    ),
    'documentation_quality_score': np.random.uniform(0.4, 1.0, n),
    'prior_auth_obtained': np.random.choice([0, 1], n, p=[0.2, 0.8])  # 80% correctly obtained
})

# 3. Age assignment: older for chronic conditions
def assign_age(icd):
    if icd == 'J44.9':  # COPD
        return np.random.randint(55, 85)
    elif icd in ['I10', 'E11.9']:  # HTN / Diabetes
        return np.random.randint(40, 85)
    else:  # sinusitis / back pain
        return np.random.randint(18, 65)

df['patient_age'] = df['ICD_10_code'].apply(assign_age)

# 4. Procedure + billed amount with 85% correct mapping, 15% medical necessity errors
def assign_procedure(icd):
    if np.random.rand() < 0.85:
        # Correct ICD → CPT mapping
        return icd_to_cpt[icd]
    else:
        # Introduce a medically questionable choice:
        # either over-code complexity or use ECG in wrong context
        if icd == 'J01.90':
            return ('93000', 85)   # ECG for sinusitis -> bad necessity
        elif icd == 'M54.50':
            return ('99215', 250)  # Over-code complex visit for back pain
        elif icd == 'E11.9':
            return ('99215', 250)  # Over-code complex visit for diabetes check
        elif icd == 'I10':
            return ('99215', 250)  # Over-code complex visit for HTN
        elif icd == 'J44.9':
            # Already complex, minor random alternative
            return ('99214', 160)

df[['CPT_code', 'billed_amount']] = df['ICD_10_code'].apply(
    lambda x: pd.Series(assign_procedure(x))
)

# 5. Modifiers & submission delay & past_denial_count
df['modifier'] = np.random.choice(
    ['25', '59', 'None'],
    n,
    p=[0.4, 0.3, 0.3]
)
df['submission_days_delay'] = np.random.randint(0, 30, n)
df['past_denial_count'] = np.random.poisson(1.2, n)

# 6. Rule-based claim status + denial reason
def determine_denial(row):
    # Rule A: Missing Prior Auth for complex visits (99215) with expensive care
    if row['CPT_code'] == '99215' and row['prior_auth_obtained'] == 0:
        return 'Denied', 'Missing Prior Auth'

    # Rule B: Poor documentation for high-level visits (downcoding scenario)
    if row['CPT_code'] in ['99214', '99215'] and row['documentation_quality_score'] < 0.65:
        return 'Denied', 'CO-97 (Documentation Lacking)'

    # Rule C: Medical necessity mismatch (e.g., ECG for sinusitis)
    if row['ICD_10_code'] == 'J01.90' and row['CPT_code'] == '93000':
        return 'Denied', 'CO-50 (Medical Necessity)'

    # Rule D: Payer-specific strictness (Aetna randomly denies some claims)
    if row['payer'] == 'Aetna' and np.random.rand() < 0.05:
        return 'Denied', 'CO-45 (Fee Schedule Limit)'

    # Otherwise approved
    return 'Approved', 'None'

df[['claim_status', 'denial_reason_code']] = df.apply(
    lambda r: pd.Series(determine_denial(r)),
    axis=1
)

# 7. Denial probability (for regression models / risk scoring)
def compute_denial_probability(row):
    base = 0.05  # baseline low denial risk

    # Payer personality
    if row['payer'] == 'Medicare':
        base += 0.05
    elif row['payer'] == 'BlueCross':
        base += 0.02
    elif row['payer'] == 'Aetna':
        base += 0.10

    # Clinical/operational factors
    if row['CPT_code'] == '99215':
        base += 0.15
    if row['CPT_code'] in ['99214', '99215'] and row['documentation_quality_score'] < 0.65:
        base += 0.25
    if row['ICD_10_code'] == 'J01.90' and row['CPT_code'] == '93000':
        base += 0.30
    if row['prior_auth_obtained'] == 0 and row['CPT_code'] == '99215':
        base += 0.25
    if row['modifier'] == 'None':
        base += 0.05
    if row['submission_days_delay'] > 20:
        base += 0.05
    if row['past_denial_count'] >= 3:
        base += 0.07

    # Approved claims should still have low—but non-zero—predicted risk
    if row['claim_status'] == 'Approved':
        base = min(base, 0.25)

    # Denied claims should have high predicted risk
    if row['claim_status'] == 'Denied':
        base = max(base, 0.65)

    # Add small noise and clip
    base += np.random.normal(0, 0.03)
    return float(np.clip(base, 0.0, 1.0))

df['denial_probability'] = df.apply(compute_denial_probability, axis=1)

# 8. Human-readable diagnosis/procedure names (for UI)
df['ICD_10_description'] = df['ICD_10_code'].map(icd10_map)

cpt_name_map = {
    '99213': 'Standard office visit (15 min)',
    '99214': 'Extended office visit (25 min)',
    '99215': 'Complex office visit (40 min)',
    '93000': 'Electrocardiogram (ECG)',
    '85025': 'Complete blood count (CBC)'
}
df['CPT_description'] = df['CPT_code'].map(cpt_name_map)

# 9. Export
df.to_csv('data/claimshield_training_data_refined.csv', index=False)
print(df['claim_status'].value_counts())
print(df['denial_probability'].describe())
