# app/app.py
import streamlit as st
import requests

API_PREDICT = "http://127.0.0.1:8000/predict"
API_FULL = "http://127.0.0.1:8000/full_pipeline"

# --- UI CONFIGURATION ---
st.set_page_config(page_title="ClaimShield AI", page_icon="🛡️", layout="wide")

# Custom CSS for a professional "MedTech" look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .status-approved { color: #238636; font-weight: bold; border: 1px solid #238636; padding: 5px 10px; border-radius: 5px; }
    .status-denied { color: #da3633; font-weight: bold; border: 1px solid #da3633; padding: 5px 10px; border-radius: 5px; }
    .shield-header { font-size: 42px; font-weight: 800; color: #58a6ff; margin-bottom: 0px; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<p class="shield-header">🛡️ ClaimShield AI</p>', unsafe_allow_html=True)
st.markdown("#### *Real-time Clinical Documentation & Denial Prevention Workflow*")
st.divider()

# Initialize Session State to lock results on screen
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# --- MAIN WORKFLOW ---
col_upload, col_process = st.columns([1, 2])

with col_upload:
    st.subheader("1. Clinical Input")
    payer_ocr = st.selectbox("Select Payer", ["Medicare", "BlueCross", "Aetna"])
    uploaded_file = st.file_uploader("Upload Doctor's Note (Image)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Source Document", use_container_width=True)
        if st.button("✨ Analyze with ClaimShield", use_container_width=True, type="primary"):
            with st.spinner("AI analyzing clinical context and inferring codes..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {"payer": payer_ocr}
                try:
                    resp = requests.post(API_FULL, files=files, params=params)
                    st.session_state.analysis_results = resp.json()
                except Exception as e:
                    st.error(f"Shield Analysis Interrupted: {e}")

    if st.button("🔄 Clear & Restart"):
        st.session_state.analysis_results = None
        st.rerun()

with col_process:
    if st.session_state.analysis_results:
        data = st.session_state.analysis_results
        ocr = data["ocr"]
        defaults = data["default_features"]

        st.success("AI Extraction Complete")
        
        # --- Step 2: Verification Panel ---
        st.subheader("2. AI Extraction & Human Verification")
        st.info("Review inferred data and adjust fields if necessary.")
        
        v_col1, v_col2 = st.columns(2)
        
        # Define valid options for selectboxes
        icd_options = ["J01.90", "M54.50", "E11.9", "I10", "J44.9"]
        cpt_options = ["99213", "99214", "99215", "93000", "85025"]
        
        with v_col1:
            icd_val = st.selectbox(
                "Primary ICD-10 code (Diagnosis)", 
                icd_options,
                index=icd_options.index(defaults["ICD_10_code"]) if defaults["ICD_10_code"] in icd_options else 0
            )
            cpt_val = st.selectbox(
                "Primary CPT code (Procedure)", 
                cpt_options,
                index=cpt_options.index(defaults["CPT_code"]) if defaults["CPT_code"] in cpt_options else 0
            )
            mod_val = st.selectbox("Modifier", ["None", "25", "59"], index=0)

        with v_col2:
            # AGE FIELD ADDED HERE
            age_val = st.slider("Patient Age", 18, 100, int(defaults.get("patient_age", 45)))
            
            doc_q = st.slider(
                "Documentation Quality Score", 
                0.0, 1.0, 
                float(defaults.get("documentation_quality_score", 0.8)),
                step=0.01
            )
            auth_val = st.radio(
                "Prior Authorization obtained?", 
                ["Yes", "No"], 
                index=0 if defaults["prior_auth_obtained"] == 1 else 1, 
                horizontal=True
            )

        # --- Step 3: Final Shield Result ---
        st.divider()
        st.subheader("3. Final Shield Result")
        
        # Automatically re-predict whenever any of the above widgets are changed
        payload = {
            "payer": payer_ocr,
            "patient_age": age_val,
            "ICD_10_code": icd_val,
            "CPT_code": cpt_val,
            "modifier": mod_val,
            "documentation_quality_score": doc_q,
            "prior_auth_obtained": 1 if auth_val == "Yes" else 0,
            "billed_amount": float(defaults.get("billed_amount", 150.0)),
            "submission_days_delay": int(defaults.get("submission_days_delay", 3)),
            "past_denial_count": int(defaults.get("past_denial_count", 0)),
        }
        
        try:
            final_resp = requests.post(API_PREDICT, json=payload)
            res = final_resp.json()
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Denial Probability", f"{res['denial_probability']:.2%}")
            
            status_html = f'<span class="status-approved">APPROVED</span>' if res['predicted_status'] == "Approved" else f'<span class="status-denied">DENIED / RISK</span>'
            m2.markdown(f"**Predicted Status**<br>{status_html}", unsafe_allow_html=True)
            
            m3.metric("Reason Code", res['denial_reason_code'])
            
            st.warning(f"**🛡️ Remediation Strategy:** {res['remediation']}")
            
            if res['predicted_status'] == "Approved":
                st.balloons()
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            
    else:
        st.info("👈 Upload a clinical note and click 'Analyze with ClaimShield' to start.")