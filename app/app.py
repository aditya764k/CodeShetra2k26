import streamlit as st
import requests

# API Endpoints
API_PREDICT = "http://127.0.0.1:8000/predict"
API_FULL = "http://127.0.0.1:8000/full_pipeline"

# --- UI CONFIGURATION ---
st.set_page_config(page_title="ClaimShield AI", page_icon="🛡️", layout="wide")

# Custom CSS for "MedTech" Professional Branding
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    .status-approved { color: #238636; font-weight: bold; font-size: 24px; border: 2px solid #238636; padding: 10px; border-radius: 8px; text-align: center; display: block; }
    .status-denied { color: #da3633; font-weight: bold; font-size: 24px; border: 2px solid #da3633; padding: 10px; border-radius: 8px; text-align: center; display: block; }
    .shield-header { font-size: 42px; font-weight: 800; color: #58a6ff; margin-bottom: 0px; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<p class="shield-header">🛡️ ClaimShield AI</p>', unsafe_allow_html=True)
st.markdown("#### *Real-time Clinical Documentation & Denial Prevention Pipeline*")
st.divider()

# Initialize Session State
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
        if st.button("✨ Run Shield Pipeline", use_container_width=True, type="primary"):
            with st.spinner("Gemini 3 Flash extracting clinical context..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {"payer": payer_ocr}
                try:
                    # Execute Full Pipeline (OCR -> Logic -> Prediction)
                    resp = requests.post(API_FULL, files=files, params=params)
                    st.session_state.analysis_results = resp.json()
                except Exception as e:
                    st.error(f"Pipeline Interrupted: {e}")

    if st.button("🔄 Reset System"):
        st.session_state.analysis_results = None
        st.rerun()

with col_process:
    if st.session_state.analysis_results:
        # Step 2: Display what the AI Found
        res = st.session_state.analysis_results
        defaults = res["default_features"]
        ocr_data = res["ocr"]

        st.success("✅ Multimodal Extraction Complete")
        
        st.subheader("2. AI Clinical Extraction")
        st.info(f"**AI Clinical Insight:** {ocr_data.get('notes', 'Confirmed clinical encounter.')}")
        
        # Display the extracted codes from the Backend
        c1, c2, c3 = st.columns(3)
        icd_val = c1.selectbox("Extracted ICD-10", ["J01.90", "M54.50", "E11.9", "I10", "J44.9"], 
                               index=["J01.90", "M54.50", "E11.9", "I10", "J44.9"].index(defaults["ICD_10_code"]))
        cpt_val = c2.selectbox("Extracted CPT", ["99213", "99214", "99215", "93000", "85025"], 
                               index=["99213", "99214", "99215", "93000", "85025"].index(defaults["CPT_code"]))
        auth_val = c3.radio("Prior Auth?", ["Yes", "No"], index=0 if defaults["prior_auth_obtained"] == 1 else 1)

        # Step 3: Final Risk Verdict
        st.divider()
        st.subheader("3. Risk Verdict & Remediation")
        
        # We re-verify with /predict to ensure widgets and backend stay in sync
        payload = {
            **defaults,
            "ICD_10_code": icd_val,
            "CPT_code": cpt_val,
            "prior_auth_obtained": 1 if auth_val == "Yes" else 0
        }
        
        try:
            live_resp = requests.post(API_PREDICT, json=payload)
            live_data = live_resp.json()
            
            m1, m2, m3 = st.columns(3)
            
            # Probability Gauge
            m1.metric("Denial Risk", f"{live_data['denial_probability']:.2%}")
            
            # Status Indicator
            if live_data['predicted_status'] == "Approved":
                m2.markdown('<div class="status-approved">✅ LOW RISK / APPROVED</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                m2.markdown('<div class="status-denied">❌ HIGH RISK / DENIED</div>', unsafe_allow_html=True)
            
            # Industry Reason Code
            m3.metric("Payer Reason Code", live_data['denial_reason_code'])
            
            # The "Shield" Remediation Box
            if live_data['predicted_status'] != "Approved":
                st.error(f"**🛡️ Shield Remediation:** {live_data['remediation']}")
            else:
                st.success(f"**🛡️ Shield Remediation:** Claim meets all medical necessity rules.")

            # Extra: Technical Log for Judges
            with st.expander("🛠️ View Pipeline Metadata"):
                st.json(live_data)
                
        except Exception as e:
            st.error(f"Final Prediction Error: {e}")
            
    else:
        st.info("👈 Upload a medical document to trigger the ClaimShield AI pipeline.")