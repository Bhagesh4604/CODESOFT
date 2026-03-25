import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="FraudGuard AI — Transaction Risk Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* ─── Base ──────────────────────────────────────────── */
    .stApp {
        background-color: #080c14;
        font-family: 'Space Grotesk', sans-serif;
        color: #c9d6e8;
    }
    header { visibility: hidden; }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem !important;
        max-width: 1450px !important;
    }
    .st-emotion-cache-12fmjuu, .st-emotion-cache-6q9sum { display: none; }

    /* ─── Top Command Bar ────────────────────────────────── */
    .cmd-bar {
        background: linear-gradient(90deg, #0d1526 0%, #111c30 60%, #0d1526 100%);
        border-bottom: 1px solid #1e3a5f;
        padding: 0.75rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    .cmd-logo {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.35rem;
        font-weight: 700;
        color: #00d4ff;
        letter-spacing: 1px;
    }
    .cmd-logo span { color: #f59e0b; }
    .cmd-status {
        display: flex; align-items: center; gap: 1rem;
    }
    .status-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .chip-live  { background: rgba(0,212,255,0.12); color: #00d4ff; border: 1px solid #00d4ff44; }
    .chip-model { background: rgba(245,158,11,0.12); color: #f59e0b;  border: 1px solid #f59e0b44; }

    /* ─── Credit Card Visual ─────────────────────────────── */
    .card-visual {
        background: linear-gradient(135deg, #1a2a47 0%, #0f1c33 40%, #1a1035 100%);
        border: 1px solid #2a3f60;
        border-radius: 18px;
        padding: 1.8rem 2rem;
        position: relative;
        overflow: hidden;
        min-height: 180px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .card-visual::before {
        content: '';
        position: absolute; top: -60px; right: -60px;
        width: 200px; height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
    }
    .card-visual::after {
        content: '';
        position: absolute; bottom: -40px; left: 60px;
        width: 150px; height: 150px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(245,158,11,0.06) 0%, transparent 70%);
    }
    .card-chip {
        width: 42px; height: 32px;
        background: linear-gradient(135deg, #f59e0b, #d97706);
        border-radius: 5px;
        margin-bottom: 1rem;
        box-shadow: inset 0 0 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 2;
    }
    .card-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        letter-spacing: 4px;
        color: #e2e8f0;
        margin-bottom: 0.8rem;
        position: relative; z-index: 2;
    }
    .card-meta {
        display: flex; justify-content: space-between; align-items: flex-end;
        position: relative; z-index: 2;
    }
    .card-label { font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }
    .card-value { font-size: 0.85rem; font-weight: 600; color: #cbd5e1; }
    .card-network {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: rgba(245,158,11,0.6);
        letter-spacing: -1px;
    }

    /* ─── Input Section ──────────────────────────────────── */
    .input-panel {
        background: #0d1526;
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1rem;
    }
    .panel-title {
        font-size: 0.72rem;
        color: #00d4ff;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .panel-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, #1e3a5f, transparent);
    }

    /* ─── KPI Tiles ──────────────────────────────────────── */
    .kpi-tile {
        background: #0d1526;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-tile:hover { transform: translateY(-2px); border-color: #2a5a8f; }
    .kpi-tile::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
    }
    .kpi-accent-cyan::before  { background: linear-gradient(90deg, #00d4ff, transparent); }
    .kpi-accent-amber::before { background: linear-gradient(90deg, #f59e0b, transparent); }
    .kpi-accent-red::before   { background: linear-gradient(90deg, #f43f5e, transparent); }
    .kpi-accent-violet::before{ background: linear-gradient(90deg, #8b5cf6, transparent); }
    .kpi-accent-green::before { background: linear-gradient(90deg, #10b981, transparent); }
    .kpi-lbl { font-size: 0.7rem; color: #4a6080; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; font-weight: 600; }
    .kpi-val { font-size: 1.8rem; font-weight: 800; color: #e2e8f0; line-height: 1; font-family: 'JetBrains Mono', monospace; }
    .kpi-sub { font-size: 0.78rem; color: #4a6080; margin-top: 0.25rem; }

    /* ─── Verdict Cards ──────────────────────────────────── */
    .verdict-safe {
        background: linear-gradient(135deg, #012a1a 0%, #013d25 100%);
        border: 1px solid #059669;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        text-align: center;
        position: relative; overflow: hidden;
    }
    .verdict-safe::before {
        content: ''; position: absolute;
        top: 50%; left: 50%; transform: translate(-50%, -50%);
        width: 200px; height: 200px; border-radius: 50%;
        background: radial-gradient(circle, rgba(5,150,105,0.15) 0%, transparent 70%);
    }
    .verdict-fraud {
        background: linear-gradient(135deg, #2d0a14 0%, #3d0f1c 100%);
        border: 1px solid #be123c;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        text-align: center;
        position: relative; overflow: hidden;
        animation: threatPulse 2s ease-in-out infinite;
    }
    .verdict-fraud::before {
        content: ''; position: absolute;
        top: 50%; left: 50%; transform: translate(-50%, -50%);
        width: 200px; height: 200px; border-radius: 50%;
        background: radial-gradient(circle, rgba(190,18,60,0.2) 0%, transparent 70%);
    }
    @keyframes threatPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(244,63,94,0.0); }
        50%       { box-shadow: 0 0 0 8px rgba(244,63,94,0.0), 0 0 30px rgba(244,63,94,0.15); }
    }
    .v-label-safe  { font-family: 'JetBrains Mono', monospace; font-size: 2.8rem; font-weight: 700; color: #10b981; letter-spacing: 2px; margin: 0; }
    .v-label-fraud { font-family: 'JetBrains Mono', monospace; font-size: 2.8rem; font-weight: 700; color: #f43f5e; letter-spacing: 2px; margin: 0; animation: flicker 3s infinite; }
    @keyframes flicker {
        0%,96%,100% { opacity:1; }
        97% { opacity:0.7; }
        98% { opacity:1; }
        99% { opacity:0.8; }
    }
    .v-sub-safe  { color: #6ee7b7; font-size: 0.9rem; font-weight: 600; margin-top: 0.5rem; position: relative; z-index: 1; }
    .v-sub-fraud { color: #fca5a5; font-size: 0.9rem; font-weight: 600; margin-top: 0.5rem; position: relative; z-index: 1; }

    /* ─── Probability Bar ────────────────────────────────── */
    .prob-track { background: #1a2a47; border-radius: 6px; height: 8px; overflow: hidden; }
    .prob-fill  { height: 8px; border-radius: 6px; transition: width 0.5s ease; }

    /* ─── Buttons ────────────────────────────────────────── */
    div.stButton > button {
        background: #0d1526 !important;
        border: 1px solid #2a3f60 !important;
        color: #8baac8 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.4rem 1.1rem !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        border-color: #00d4ff !important;
        color: #00d4ff !important;
        background: rgba(0,212,255,0.06) !important;
        transform: translateY(-1px) !important;
    }

    /* ─── Inputs ─────────────────────────────────────────── */
    .stNumberInput input {
        background: #080c14 !important;
        border: 1px solid #1e3a5f !important;
        color: #00d4ff !important;
        font-family: 'JetBrains Mono', monospace !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }
    label, .stMarkdown p { color: #4a6080 !important; }
    div[data-baseweb="select"] > div {
        background: #080c14 !important;
        border: 1px solid #1e3a5f !important;
    }

    /* ─── Tabs ───────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; background: #0d1526;
        border-bottom: 1px solid #1e3a5f;
        margin-bottom: 2rem; padding: 0 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border: none;
        color: #4a6080; font-weight: 600;
        font-size: 0.88rem; padding: 0.8rem 1.6rem;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.3px;
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #00d4ff !important;
        border-bottom: 2px solid #00d4ff !important;
    }
    div[data-testid="stTickBar"] { display:none !important; }

    /* ─── Expander ───────────────────────────────────────── */
    details { background: #0d1526 !important; border: 1px solid #1e3a5f !important; border-radius: 10px !important; }
    summary { color: #8baac8 !important; font-weight: 600 !important; }

    /* ─── DataFrame ──────────────────────────────────────── */
    .stDataFrame { border: 1px solid #1e3a5f !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    files = ["fraud_model.pkl", "fraud_scaler.pkl", "sample_transactions.pkl"]
    if not all(os.path.exists(f) for f in files):
        return None, None, None
    return joblib.load("fraud_model.pkl"), joblib.load("fraud_scaler.pkl"), joblib.load("sample_transactions.pkl")

model, scaler, samples = load_assets()
if model is None:
    st.error("⚠️ Model artifacts not found. Run `python fraud_model.py` first.")
    st.stop()

# ── Dataset stats ───────────────────────────────────────────────────────────────
all_df = pd.concat([
    samples['genuine'].assign(Class=0, Label='Genuine'),
    samples['fraud'].assign(Class=1, Label='Fraudulent')
], ignore_index=True)

try:
    importances = model.feature_importances_
    feat_names_all = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    imp_series = pd.Series(importances, index=feat_names_all).sort_values(ascending=False)
    top_features = imp_series.head(8)
except Exception:
    top_features = None

# ── Session state ───────────────────────────────────────────────────────────────
defaults = {'v1': 0.0, 'v2': 0.0, 'v14': 0.0, 'amt': 100.0}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def load_sample(kind):
    row = samples['fraud'].iloc[0] if kind == "Fraud" else samples['genuine'].iloc[0]
    st.session_state.v1  = float(row['V1'])
    st.session_state.v2  = float(row['V2'])
    st.session_state.v14 = float(row['V14'])
    st.session_state.amt = float(row.get('scaled_amount', 100.0))

# ── Live prediction ─────────────────────────────────────────────────────────────
def predict():
    vec = np.zeros(30)
    vec[0], vec[1], vec[13], vec[28], vec[29] = (
        st.session_state.v1, st.session_state.v2,
        st.session_state.v14, st.session_state.amt, 0.5
    )
    return int(model.predict([vec])[0]), model.predict_proba([vec])[0]

pred, proba = predict()
fraud_pct   = round(proba[1] * 100, 1)
genuine_pct = round(proba[0] * 100, 1)

# ── Command bar ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="cmd-bar">
    <div class="cmd-logo">💳 Fraud<span>Guard</span> AI</div>
    <div class="cmd-status">
        <span class="status-chip chip-live">● LIVE ENGINE</span>
        <span class="status-chip chip-model">RF · 50 TREES</span>
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#2a5a8f;">
            RISK: <span style="color:{'#f43f5e' if fraud_pct > 50 else '#10b981'}; font-weight:700;">{fraud_pct}%</span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💳   Transaction Risk Scanner", "📡   Model Intelligence"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 ─ TRANSACTION RISK SCANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_L, col_R = st.columns([1.1, 1], gap="large")

    # ── LEFT: card visual + inputs ──────────────────────────────────────────
    with col_L:
        # Credit Card Visual
        st.markdown(f"""
        <div class="card-visual">
            <div class="card-chip"></div>
            <div class="card-number">•••• •••• •••• ████</div>
            <div class="card-meta">
                <div>
                    <div class="card-label">Card Holder</div>
                    <div class="card-value">ANONYMOUS · PCA SECURED</div>
                </div>
                <div>
                    <div class="card-label">Transaction</div>
                    <div class="card-value">${abs(st.session_state.amt):,.2f}</div>
                </div>
                <div class="card-network">VISA</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

        # Quick load buttons
        st.markdown('<div class="panel-title">▶ TRANSACTION TEMPLATE</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Genuine Sample", use_container_width=True):
                load_sample("Genuine"); st.rerun()
        with b2:
            if st.button("Fraud Sample", use_container_width=True):
                load_sample("Fraud"); st.rerun()
        with b3:
            if st.button("Clear / Reset", use_container_width=True):
                for k, v in defaults.items(): st.session_state[k] = v
                st.rerun()

        st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

        # PCA Vectors
        st.markdown('<div class="panel-title">▶ PCA FEATURE VECTORS</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("V1 — Principal Component 1", value=float(st.session_state.v1), step=0.1, format="%.3f", key="v1")
        with c2:
            st.number_input("V2 — Principal Component 2", value=float(st.session_state.v2), step=0.1, format="%.3f", key="v2")

        c3, c4 = st.columns(2)
        with c3:
            st.number_input("V14 — High-Risk Risk Vector", value=float(st.session_state.v14), step=0.1, format="%.3f", key="v14")
        with c4:
            st.number_input("Transaction Amount ($)", value=float(st.session_state.amt), step=10.0, format="%.2f", key="amt")

        # Live vector readout
        st.markdown(f"""
        <div style="margin-top:1rem; background:#080c14; border:1px solid #1e3a5f; border-radius:8px;
             padding:0.75rem 1.1rem; font-family:'JetBrains Mono',monospace; font-size:0.82rem;">
            <span style="color:#2a5a8f;">VECTOR:</span>
            <span style="color:#00d4ff;">
                [V1={st.session_state.v1:.2f}, V2={st.session_state.v2:.2f}, V14={st.session_state.v14:.2f},
                Amt={st.session_state.amt:.2f}, T=0.50, ...(zeros)...]
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── RIGHT: Verdict + gauge ──────────────────────────────────────────────
    with col_R:
        # Verdict
        if pred == 1:
            st.markdown(f"""
            <div class="verdict-fraud">
                <p style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#78350f; letter-spacing:3px; margin-bottom:0.5rem; position:relative;z-index:1;">
                    ████ ALERT LEVEL: CRITICAL ████
                </p>
                <p class="v-label-fraud">⚡ DENIED</p>
                <p class="v-sub-fraud">Transaction BLOCKED — Fraud Detected</p>
                <p style="color:#9f1239; font-size:0.8rem; margin-top:0.8rem; position:relative;z-index:1;">
                    Anomalous pattern confirmed across V-component space.<br>
                    Ensemble model confidence: <strong style='color:#f43f5e;'>{fraud_pct}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-safe">
                <p style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#065f46; letter-spacing:3px; margin-bottom:0.5rem; position:relative;z-index:1;">
                    ░░░░ STATUS: ALL CLEAR ░░░░
                </p>
                <p class="v-label-safe">✓ CLEARED</p>
                <p class="v-sub-safe">Transaction APPROVED — No Fraud Detected</p>
                <p style="color:#064e3b; font-size:0.8rem; margin-top:0.8rem; position:relative;z-index:1;">
                    Transaction profile is within secure historical bounds.<br>
                    Safety confidence: <strong style='color:#10b981;'>{genuine_pct}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

        # Radar/Polar Risk Chart — unique to this project
        risk_angle = fraud_pct / 100 * 270  # 0° = safe, 270° = max fraud
        fig_polar = go.Figure()

        # Background arcs
        for end, fill_rgba, label in [
            (90,  'rgba(16,185,129,0.12)',   'LOW'),
            (180, 'rgba(245,158,11,0.12)',   'MED'),
            (270, 'rgba(244,63,94,0.12)',    'HIGH')
        ]:
            theta = np.linspace(225, 225 - end, 100)
            r_inner, r_outer = 0.55, 0.85
            theta_rad = np.radians(theta)
            xs = np.concatenate([r_inner * np.cos(theta_rad), r_outer * np.cos(theta_rad[::-1])])
            ys = np.concatenate([r_inner * np.sin(theta_rad), r_outer * np.sin(theta_rad[::-1])])
            fig_polar.add_trace(go.Scatter(x=xs, y=ys, fill='toself', mode='none',
                                           fillcolor=fill_rgba, showlegend=False, hoverinfo='skip'))

        # Needle
        needle_angle = 225 - risk_angle
        needle_rad   = np.radians(needle_angle)
        fig_polar.add_trace(go.Scatter(
            x=[0, 0.78 * np.cos(needle_rad)],
            y=[0, 0.78 * np.sin(needle_rad)],
            mode='lines',
            line=dict(color='#f43f5e' if fraud_pct > 50 else '#10b981', width=4),
            showlegend=False, hoverinfo='skip'
        ))
        fig_polar.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=12, color='#f1f5f9'),
            showlegend=False, hoverinfo='skip'
        ))

        # Labels
        for angle, txt, col in [(225, 'SAFE', '#10b981'), (90, 'RISK', '#f43f5e'), (135, 'MED', '#f59e0b')]:
            rad = np.radians(angle)
            fig_polar.add_trace(go.Scatter(
                x=[0.95 * np.cos(rad)], y=[0.95 * np.sin(rad)],
                mode='text', text=[txt],
                textfont=dict(size=9, color=col, family='JetBrains Mono'),
                showlegend=False, hoverinfo='skip',
            ))

        fig_polar.add_annotation(
            text=f"<b>{fraud_pct}%</b>",
            x=0, y=-0.2, showarrow=False,
            font=dict(size=26, color='#f43f5e' if fraud_pct > 50 else '#10b981', family='JetBrains Mono'),
            xanchor='center'
        )
        fig_polar.add_annotation(
            text="FRAUD RISK", x=0, y=-0.38, showarrow=False,
            font=dict(size=10, color='#4a6080', family='JetBrains Mono'),
            xanchor='center'
        )

        fig_polar.update_layout(
            xaxis=dict(range=[-1.15,1.15], visible=False),
            yaxis=dict(range=[-0.55,1.1], visible=False, scaleanchor='x'),
            paper_bgcolor='#0d1526', plot_bgcolor='#0d1526',
            margin=dict(l=10,r=10,t=10,b=10),
            height=250,
        )
        st.plotly_chart(fig_polar, use_container_width=True, config={'displayModeBar': False})

        # Prob bars
        st.markdown("""
        <div style="background:#0d1526; border:1px solid #1e3a5f; border-radius:12px; padding:1.2rem 1.5rem;">
            <div style="font-size:0.7rem; color:#00d4ff; font-weight:700; letter-spacing:2px; text-transform:uppercase; margin-bottom:1rem;">
                PROBABILITY BREAKDOWN
            </div>
        """, unsafe_allow_html=True)
        for lbl, pct, color in [("✅  Legitimate", genuine_pct, '#10b981'), ("🚨  Fraudulent", fraud_pct, '#f43f5e')]:
            w = max(int(pct), 1)
            st.markdown(f"""
            <div style="margin-bottom:0.9rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.35rem;">
                    <span style="font-size:0.85rem; font-weight:600; color:#8baac8;">{lbl}</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; font-weight:700; color:{color};">{pct}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{w}%; background:{color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Risk interpretation
        if fraud_pct < 20:
            ri_color, ri_icon, ri_title, ri_msg = '#10b981','✅','LOW RISK','Normal transaction. No intervention required.'
        elif fraud_pct < 50:
            ri_color, ri_icon, ri_title, ri_msg = '#f59e0b','⚠️','ELEVATED RISK','Moderate signals detected. Recommend secondary review.'
        else:
            ri_color, ri_icon, ri_title, ri_msg = '#f43f5e','🚨','CRITICAL RISK','Strong fraud indicators. Block transaction immediately.'

        st.markdown(f"""
        <div style="margin-top:1rem; background:#080c14; border:1px solid {ri_color}44;
             border-left:3px solid {ri_color}; border-radius:10px; padding:0.9rem 1.1rem;">
            <p style="color:{ri_color}; font-family:'JetBrains Mono',monospace; font-size:0.75rem;
               font-weight:700; letter-spacing:1px; margin:0;">{ri_icon} {ri_title}</p>
            <p style="color:#4a6080; font-size:0.82rem; margin:0.35rem 0 0 0;">{ri_msg}</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 ─ MODEL INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── KPI Row ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_data = [
        ("Total Samples", "50,000", "Synthetic PCA dataset", "kpi-accent-cyan"),
        ("Fraud Rate",    "~0.20%",  "Extreme class imbalance", "kpi-accent-red"),
        ("Model F1",      "~89%",    "On held-out test set", "kpi-accent-green"),
        ("Trees",         "50",      "Random Forest ensemble", "kpi-accent-amber"),
        ("Algorithm",     "ROS",     "Random Over Sampler", "kpi-accent-violet"),
    ]
    for col, (lbl, val, sub, acc) in zip([k1,k2,k3,k4,k5], kpi_data):
        with col:
            st.markdown(f"""
            <div class="kpi-tile {acc}">
                <div class="kpi-lbl">{lbl}</div>
                <div class="kpi-val">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # ── Row 1: Feature Importance (horizontal bar) + Donut ──────────────────
    r1c1, r1c2 = st.columns([1.5, 1], gap="large")

    CHART_BG = '#0d1526'
    PLOT_BG  = '#080c14'
    GRID_C   = '#1a2a47'
    TICK_C   = '#4a6080'
    TITLE_C  = '#c9d6e8'

    def base_layout(h=320, title=""):
        return dict(
            title=dict(text=title, font=dict(size=14,color=TITLE_C,family='Space Grotesk'), x=0.02, y=0.97),
            paper_bgcolor=CHART_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=TICK_C, family='Space Grotesk'),
            margin=dict(t=45,b=30,l=20,r=20), height=h,
        )

    with r1c1:
        if top_features is not None:
            cyan_shades = ['#00d4ff','#00b8d9','#009cb3','#00808f','#006670','#004f55','#003c3d','#002b2b']
            fig_imp = go.Figure(go.Bar(
                x=top_features.values * 100,
                y=top_features.index.tolist(),
                orientation='h',
                marker_color=cyan_shades[:len(top_features)],
                marker_line_width=0,
                text=[f"{v*100:.1f}%" for v in top_features.values],
                textposition='outside',
                textfont=dict(size=11, color='#8baac8'),
            ))
            lay = base_layout(320, "Top Feature Importances — Random Forest")
            lay['xaxis'] = dict(title='Importance (%)', gridcolor=GRID_C, color=TICK_C,
                                range=[0, top_features.values.max()*135])
            lay['yaxis'] = dict(autorange='reversed', color=TICK_C, tickfont=dict(
                                family='JetBrains Mono', size=11))
            fig_imp.update_layout(**lay)
            st.plotly_chart(fig_imp, use_container_width=True)

    with r1c2:
        # Class balance donut
        fig_donut = go.Figure(go.Pie(
            labels=['Fraudulent', 'Genuine'],
            values=[100, 49900],  # real-world ~0.2% ratio
            hole=0.6,
            marker_colors=['#f43f5e', '#00d4ff'],
            textinfo='label+percent',
            textfont=dict(size=11, family='Space Grotesk'),
            hovertemplate='%{label}: %{percent}<extra></extra>',
        ))
        fig_donut.add_annotation(text="Class<br>Balance", x=0.5, y=0.5,
                                  font=dict(size=12, color='#4a6080', family='Space Grotesk'),
                                  showarrow=False)
        lay2 = base_layout(320, "Real-World Dataset Imbalance")
        lay2['showlegend'] = False
        lay2['margin'] = dict(t=45,b=20,l=20,r=20)
        fig_donut.update_layout(**lay2)
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Row 2: V1 vs V2 scatter (sample) ────────────────────────────────────
    r2c1, r2c2 = st.columns([1.5, 1], gap="large")

    with r2c1:
        color_map = {'Genuine': '#00d4ff', 'Fraudulent': '#f43f5e'}
        fig_sc = px.scatter(
            all_df, x='V1', y='V2', color='Label',
            color_discrete_map=color_map,
            opacity=0.7,
            labels={'V1':'Principal Component 1 (V1)', 'V2':'Principal Component 2 (V2)'},
            hover_data=['V14'],
        )
        # User's input star
        fig_sc.add_trace(go.Scatter(
            x=[st.session_state.v1], y=[st.session_state.v2],
            mode='markers',
            marker=dict(size=20, color='#f59e0b', symbol='star-open',
                        line=dict(color='#f59e0b', width=2)),
            name='Your Transaction',
        ))
        lay3 = base_layout(320, "V1 vs V2 — Transaction Cluster Space")
        lay3['xaxis'] = dict(gridcolor=GRID_C, color=TICK_C, zerolinecolor=GRID_C)
        lay3['yaxis'] = dict(gridcolor=GRID_C, color=TICK_C, zerolinecolor=GRID_C)
        lay3['legend'] = dict(font=dict(size=11,color='#8baac8'), bgcolor='rgba(0,0,0,0)')
        fig_sc.update_layout(**lay3)
        st.plotly_chart(fig_sc, use_container_width=True)

    with r2c2:
        # V14 box (high-impact feature)
        fig_v14 = go.Figure()
        for lbl, color in [('Genuine','#00d4ff'), ('Fraudulent','#f43f5e')]:
            sub = all_df[all_df['Label']==lbl]['V14']
            fig_v14.add_trace(go.Box(y=sub, name=lbl, marker_color=color, boxmean=True))
        lay4 = base_layout(320, "V14 — High-Risk Feature Distribution")
        lay4['yaxis'] = dict(title='V14 Value', gridcolor=GRID_C, color=TICK_C, zerolinecolor=GRID_C)
        lay4['xaxis'] = dict(color=TICK_C)
        fig_v14.update_layout(**lay4)
        st.plotly_chart(fig_v14, use_container_width=True)

    # ── Row 3: V1, V2, Amount box plots ─────────────────────────────────────
    st.markdown(f"<p style='font-family:Space Grotesk;font-size:0.72rem;color:#00d4ff;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:1rem;'>▶ COMPONENT DISTRIBUTION — GENUINE VS FRAUD</p>", unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    for col_w, feat, title in [(e1,'V1','V1 Distribution'), (e2,'V2','V2 Distribution'), (e3,'scaled_amount','Amount (Scaled)')]:
        with col_w:
            fig_b = go.Figure()
            for lbl, color in [('Genuine','#00d4ff'),('Fraudulent','#f43f5e')]:
                sub = all_df[all_df['Label']==lbl]
                if feat in sub.columns:
                    rgba_fill = f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)'
                    fig_b.add_trace(go.Violin(y=sub[feat], name=lbl, fillcolor=rgba_fill,
                                              line_color=color, box_visible=True, meanline_visible=True,
                                              showlegend=False))
            lay5 = base_layout(260, title)
            lay5['yaxis'] = dict(gridcolor=GRID_C, color=TICK_C, zerolinecolor=GRID_C)
            lay5['xaxis'] = dict(color=TICK_C)
            fig_b.update_layout(**lay5)
            st.plotly_chart(fig_b, use_container_width=True)

    # ── How It Works ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#0d1526; border:1px solid #1e3a5f; border-radius:14px;
         padding:1.8rem 2rem; margin-top:0.5rem;">
        <p style="font-size:0.72rem; color:#00d4ff; font-weight:700; text-transform:uppercase;
           letter-spacing:2px; margin-bottom:1.5rem;">▶ MODEL ARCHITECTURE</p>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:1.5rem;">
            <div style="border-left:2px solid #00d4ff; padding-left:1rem;">
                <p style="color:#00d4ff; font-family:'JetBrains Mono',monospace; font-size:0.72rem;
                   font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem;">01 · PCA Input</p>
                <p style="color:#4a6080; font-size:0.82rem;">V1–V28 are PCA-anonymised features derived from original transaction data for privacy.</p>
            </div>
            <div style="border-left:2px solid #f59e0b; padding-left:1rem;">
                <p style="color:#f59e0b; font-family:'JetBrains Mono',monospace; font-size:0.72rem;
                   font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem;">02 · RobustScaler</p>
                <p style="color:#4a6080; font-size:0.82rem;">Amount & Time scaled using median/IQR to neutralise the effect of large outlier values.</p>
            </div>
            <div style="border-left:2px solid #8b5cf6; padding-left:1rem;">
                <p style="color:#8b5cf6; font-family:'JetBrains Mono',monospace; font-size:0.72rem;
                   font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem;">03 · Oversampling</p>
                <p style="color:#4a6080; font-size:0.82rem;">RandomOverSampler balances the extreme skew (0.2% fraud) before model training.</p>
            </div>
            <div style="border-left:2px solid #10b981; padding-left:1rem;">
                <p style="color:#10b981; font-family:'JetBrains Mono',monospace; font-size:0.72rem;
                   font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem;">04 · Random Forest</p>
                <p style="color:#4a6080; font-size:0.82rem;">50-tree ensemble (depth 10) votes on each transaction. F1 ≈ 89% on test set.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset Preview ───────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    with st.expander("🗃️   Raw Sample Transactions"):
        disp = all_df[['V1','V2','V14','scaled_amount','Label']].copy()
        disp.columns = ['V1','V2','V14','Amount (scaled)','Label']
        st.dataframe(disp, use_container_width=True, height=260)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 0.5rem 0;
     font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#1e3a5f; letter-spacing:1px;">
    FRAUDGUARD AI &nbsp;·&nbsp; CODESOFT INTERNSHIP TASK 5 &nbsp;·&nbsp;
    DEVELOPED BY <span style="color:#2a5a8f;">BHAGESH BIRADAR</span>
</div>
""", unsafe_allow_html=True)
