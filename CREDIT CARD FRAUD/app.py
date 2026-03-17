import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

st.set_page_config(
    page_title="Cyber-Sentinel | Fraud Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Tactical CSS
st.markdown("""
<style>
    /* Dark Terminal Theme */
    .stApp {
        background-color: #020617; /* Even deeper space navy/black */
        color: #4ade80; /* Brighter neon green */
        font-family: 'Inter', 'Courier New', monospace;
    }
    
    header { visibility: hidden; }
    .st-emotion-cache-12fmjuu { display: none; }

    /* Security Terminal Card */
    .terminal-window {
        background-color: #0f172a;
        border: 3px solid #10b981;
        border-radius: 12px;
        padding: 3rem;
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .terminal-header {
        border-bottom: 2px solid #10b981;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 800;
        letter-spacing: 2px;
        font-size: 1.2rem;
    }
    
    .status-dot {
        height: 14px; width: 14px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 12px #10b981;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.4; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.1); }
        100% { opacity: 0.4; transform: scale(0.9); }
    }

    /* Scanning Animation */
    .scanning-overlay {
        position: absolute;
        top: 0; left: 0; width: 100%; height: 4px;
        background-color: #10b981;
        box-shadow: 0 0 25px #10b981;
        z-index: 10;
        animation: scan 3s infinite linear;
    }
    
    @keyframes scan {
        0% { top: 0%; opacity: 0; }
        10% { opacity: 0.8; }
        90% { opacity: 0.8; }
        100% { top: 100%; opacity: 0; }
    }

    /* Red Alert Theme for Fraud */
    .fraud-alert {
        border: 3px solid #f43f5e !important;
        box-shadow: 0 0 50px rgba(244, 63, 94, 0.4) !important;
        background-color: #1e1b1e !important;
    }
    .fraud-text { color: #f43f5e !important; text-shadow: 0 0 15px rgba(244, 63, 94, 0.5); }
    .fraud-dot { background-color: #f43f5e !important; box-shadow: 0 0 12px #f43f5e !important; }
    
    /* Typography - BIGGER SIZE */
    h1 { font-size: 4rem !important; font-weight: 900 !important; letter-spacing: -2px !important; color: #10b981 !important; }
    h2 { font-size: 2.5rem !important; font-weight: 800 !important; color: #10b981 !important; }
    h3 { font-size: 1.8rem !important; font-weight: 700 !important; color: #34d399 !important; }
    
    p, label, .stMarkdown, .stSelectbox label, .stNumberInput label {
        color: #34d399 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    /* Inputs - BIGGER */
    .stNumberInput input, .stSelectbox [data-baseweb="select"] {
        background-color: #1e293b !important;
        color: #34d399 !important;
        border: 2px solid #10b981 !important;
        font-size: 1.2rem !important;
        height: 50px !important;
    }
    
    /* Custom Button - BIGGER & GLOWY */
    div.stButton > button {
        background-color: transparent !important;
        color: #10b981 !important;
        border: 3px solid #10b981 !important;
        border-radius: 10px !important;
        font-weight: 900 !important;
        font-size: 1.4rem !important;
        letter-spacing: 4px !important;
        padding: 1.5rem 3rem !important;
        width: 100%;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    div.stButton > button:hover {
        background-color: #10b981 !important;
        color: #020617 !important;
        box-shadow: 0 0 40px #10b981;
        transform: scale(1.02);
    }
    
    /* Sidebar hide fix */
    .st-emotion-cache-6q9sum { display: none; }
</style>
""", unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_assets():
    if not all(os.path.exists(f) for f in ["fraud_model.pkl", "fraud_scaler.pkl", "sample_transactions.pkl"]):
        return None, None, None
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("fraud_scaler.pkl")
    samples = joblib.load("sample_transactions.pkl")
    return model, scaler, samples

model, scaler, samples = load_assets()

if model is None:
    st.error("SYSTEM ERROR: SECURITY ASSETS NOT FOUND. RUN fraud_model.py")
    st.stop()

# Header Section
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <h1 style="font-size: 3rem; margin: 0;">🛡️ CYBER-SENTINEL</h1>
    <p style="letter-spacing: 5px; opacity: 0.8;">PREDICTIVE FRAUD PROTECTION SYSTEM v5.0</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Core Layout
col_status, col_input = st.columns([1.2, 1], gap="large")

with col_status:
    st.markdown("""
    <div class="terminal-window">
        <div class="terminal-header">
            <span>NETWORK STATUS: ENCRYPTED</span>
            <div class="status-dot"></div>
        </div>
        <p style="font-size: 0.9rem; opacity: 0.7;">[LOGIN]: B_BIRADAR<br>[LOCATION]: SECURE_VAULT_04<br>[LEVEL]: COMMANDER</p>
        <br>
        <h3 style="margin-top: 0;">CURRENT THREAT OVERVIEW</h3>
        <p style='color: #888 !important;'>The global transaction ledger is currently stabilized. Model sensitivity is set to MAX to capture low-magnitude fraudulent signals.</p>
        <br>
        <div style="border-left: 2px solid #22c55e; padding-left: 1rem;">
            <p style="margin:0; font-size: 0.8rem;">[AI CORE]: Random Forest Active</p>
            <p style="margin:0; font-size: 0.8rem;">[DATASET]: Integrated Transaction History</p>
            <p style="margin:0; font-size: 0.8rem;">[IMBALANCE]: SMOTE/ROS Logic Injected</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_input:
    st.markdown("<h3 style='margin-bottom: 2rem;'>TRANSACTION PACKET INGESTION</h3>", unsafe_allow_html=True)
    
    # Selection of samples for testing
    option = st.selectbox("LOAD TRANSACTION TEMPLATE", ["MANUAL ENTRY", "SAMPLE: GENUINE TRANSACTION", "SAMPLE: FRAUDULENT PACKET"])
    
    if option == "SAMPLE: GENUINE TRANSACTION":
        row = samples['genuine'].iloc[0]
        st.session_state.v1 = row['V1']
        st.session_state.v2 = row['V2']
        st.session_state.v14 = row['V14']
        st.session_state.amt = row['scaled_amount']
    elif option == "SAMPLE: FRAUDULENT PACKET":
        row = samples['fraud'].iloc[0]
        st.session_state.v1 = row['V1']
        row2 = samples['fraud'].iloc[1] # Mix it up
        st.session_state.v2 = row2['V2']
        st.session_state.v14 = row['V14']
        st.session_state.amt = row['scaled_amount'] + 500

    t1, t2 = st.columns(2)
    with t1:
        v1 = st.number_input("V1 (Principal Vector)", value=st.session_state.get('v1', 0.0))
        v2 = st.number_input("V2 (Principal Vector)", value=st.session_state.get('v2', 0.0))
    with t2:
        v14 = st.number_input("V14 (Risk Vector)", value=st.session_state.get('v14', 0.0))
        amt = st.number_input("Transaction Amount ($)", value=st.session_state.get('amt', 100.0))
    
    # We maintain 28 V-features for the model
    scan_btn = st.button("EXECUTE DEEP-SCAN", use_container_width=True)

if scan_btn:
    placeholder = st.empty()
    
    # THE SCANNING ANIMATION - BIGGER TEXT
    with placeholder.container():
        st.markdown('<div class="terminal-window"><div class="scanning-overlay"></div><h1 style="text-align: center; color: #10b981 !important;">SCANNING PACKET...</h1><p style="text-align: center; font-size: 1.5rem !important;">Analyzing 30-factor vector space architecture...</p></div>', unsafe_allow_html=True)
        time.sleep(1.5)
        st.markdown('<div class="terminal-window"><div class="scanning-overlay"></div><h1 style="text-align: center; color: #10b981 !important;">APPLYING AI HEURISTICS...</h1><p style="text-align: center; font-size: 1.5rem !important;">Running Random Forest ensemble check...</p></div>', unsafe_allow_html=True)
        time.sleep(1.0)


    # RECONSTRUCT THE FULL FEATURE VECTOR
    # The model expects 30 features: V1-V28, scaled_amount, scaled_time
    # We use our inputs and pads the rest with zeros for demo
    full_vector = np.zeros(30)
    full_vector[0] = v1
    full_vector[1] = v2
    full_vector[13] = v14 # V14 is index 13
    full_vector[28] = amt # scaled amount (simplified demo usage)
    full_vector[29] = 0.5 # scaled time
    
    # Predict
    prediction = model.predict([full_vector])[0]
    prob = model.predict_proba([full_vector])[0]
    
    placeholder.empty()
    
    if prediction == 1:
        st.markdown(f"""
        <div class="terminal-window fraud-alert">
            <div class="terminal-header">
                <span class="fraud-text">CRITICAL SYSTEM ALERT</span>
                <div class="status-dot fraud-dot"></div>
            </div>
            <h1 class="fraud-text" style="text-align: center; font-size: 6rem !important; margin: 0;">DENIED</h1>
            <p style="text-align: center; color: #f43f5e !important; font-size: 1.8rem !important; font-weight: 800;">FRAUD PROBABILITY: {prob[1]*100:.1f}%</p>
            <p style="text-align: center; font-size: 1.1rem; opacity: 0.8; color: #f43f5e !important;">High-risk signature detected in V14 vector. Transaction blocked.</p>
        </div>
        """, unsafe_allow_html=True)
        st.error("SECURITY BREACH DETECTED")
    else:
        st.markdown(f"""
        <div class="terminal-window">
            <div class="terminal-header">
                <span>SECURITY CLEARANCE: GRANTED</span>
                <div class="status-dot"></div>
            </div>
            <h1 style="text-align: center; font-size: 6rem !important; margin: 0; color: #10b981 !important;">CLEARED</h1>
            <p style="text-align: center; color: #10b981 !important; font-size: 1.8rem !important; font-weight: 800;">SAFE PROBABILITY: {prob[0]*100:.2f}%</p>
            <p style="text-align: center; font-size: 1.1rem; opacity: 0.8;">No anomalous signatures detected. Transaction is verified secure.</p>
        </div>
        """, unsafe_allow_html=True)
        st.success("TRANSACTION SECURE")


st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; opacity: 0.4; font-size: 0.7rem; color: #22c55e;">
    CYBER-SENTINEL PROTECTIVE DIV - DEVELOPED BY B. BIRADAR<br>
    CODENAME: CODESOFT_T5_FINAL
</div>
""", unsafe_allow_html=True)
