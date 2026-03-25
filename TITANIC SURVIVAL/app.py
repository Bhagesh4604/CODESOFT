import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RMS Titanic | Survival Predictor",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    p, span, div, label {
        font-size: 1.05rem !important;
        color: #1e293b !important;
    }
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    /* Hide Streamlit's default header bar */
    header[data-testid="stHeader"] { visibility: hidden; height: 0; }

    /* ---- FIXED NAVY HEADER ---- */
    /* Streamlit wraps its content in .stApp > .appview-container > .main > .block-container
       We target the topmost element to inject a fixed bar. */
    .fixed-nav {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        width: 100vw;
        z-index: 99999;
        background-color: #001f3f;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.85rem 2.5rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.25);
    }
    .fixed-nav *, .fixed-nav span, .fixed-nav div, .fixed-nav p, .fixed-nav label {
        color: white !important;
        font-size: 1rem !important;
    }
    .fixed-nav .nav-brand {
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .fixed-nav .nav-badge {
        background: white;
        padding: 4px 16px;
        border-radius: 6px;
    }
    .fixed-nav .nav-badge span {
        color: #001f3f !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
    }
    .fixed-nav .nav-meta {
        display: flex;
        align-items: center;
        gap: 18px;
        font-size: 0.9rem !important;
        opacity: 0.9;
    }
    .fixed-nav .nav-circle {
        background: rgba(255,255,255,0.2);
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
    }

    /* Push content down from behind the fixed header */
    .block-container {
        padding-top: 4.2rem !important;
        padding-bottom: 1rem !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 2px solid #e2e8f0;
        padding-top: 64px;
    }
    .sidebar-brand {
        font-size: 1.4rem !important;
        font-weight: 800;
        color: #001f3f !important;
        border-bottom: 2px solid #cbd5e1;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .nav-item {
        padding: 0.9rem 0;
        color: #475569 !important;
        font-size: 1.1rem !important;
        font-weight: 600;
    }
    .nav-item.active {
        color: #001f3f !important;
        font-weight: 800;
        border-left: 4px solid #001f3f;
        padding-left: 10px;
    }
    .nav-divider { height: 2px; background: #cbd5e1; margin: 1.5rem 0; }

    /* Cards */
    .bi-card {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04);
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.9rem !important;
        color: #64748b !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }

    /* Predict Button */
    .predict-btn-wrap .stButton > button {
        background-color: #cc5500 !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        width: 100% !important;
        padding: 0.85rem 2rem !important;
        margin-top: 1.5rem;
        border-radius: 50px !important;
        box-shadow: 0 4px 10px rgba(204,85,0,0.35);
        transition: all 0.2s ease;
    }
    .predict-btn-wrap .stButton > button:hover {
        background-color: #a34400 !important;
        transform: translateY(-2px);
    }

    /* Table */
    [data-testid="stTable"] { background-color: transparent !important; }
    [data-testid="stTable"] td, [data-testid="stTable"] th {
        color: #1e293b !important;
        font-weight: 500 !important;
        border-bottom: 1px solid #f1f5f9 !important;
        font-size: 1rem !important;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 8px 8px 0 0 !important;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: #001f3f !important;
        background: white !important;
        border-bottom: 3px solid #001f3f !important;
    }
</style>
""", unsafe_allow_html=True)

# ---- FIXED HEADER ----
st.markdown("""
<div class="fixed-nav">
    <div class="nav-brand">&#9881; Survivor Compass</div>
    <div class="nav-badge"><span>RMS Titanic &middot; 1912 Voyage Analysis</span></div>
    <div class="nav-meta">
        <span>Developed by <b>Bhagesh Biradar</b></span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- MODELS & DATA ---
@st.cache_resource
def load_assets():
    if not os.path.exists('titanic_survival_model.pkl'): return None, None
    try: model = joblib.load('titanic_survival_model.pkl')
    except: model = None
    try:
        import kagglehub, glob
        path = kagglehub.dataset_download("yasserh/titanic-dataset")
        df = pd.read_csv(glob.glob(os.path.join(path, "*.csv"))[0])
        df.columns = [c.lower() for c in df.columns]
        df['sex'] = df['sex'].str.title()
        df['embarked'] = df['embarked'].fillna('C').map({'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'})
    except: df = None
    return model, df

model, df = load_assets()

import base64
def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
    return ""

img_b64 = get_image_base64("public/Image.jpeg")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">🚢 Titanic Predictor</div>
    <div class="nav-item">📊 Titanic EDA</div>
    <div class="nav-item">🔍 Global Filters</div>
    <div class="nav-item active">👤 Predict Fate</div>
    <div class="nav-item">📈 Key Metrics</div>
    <div class="nav-divider"></div>
    """, unsafe_allow_html=True)

# ---- TABS ----
tab1, tab2 = st.tabs(["⚓  Survival Predictor", "📊  Analysis Dashboard"])

# =====================================================================
# TAB 1 - PREDICTOR
# =====================================================================
with tab1:
    col_in, col_out = st.columns([1, 1.4], gap="large")

    with col_in:
        st.markdown("### Passenger Demographics")
        st.markdown('<div style="font-size:1rem;color:#64748b;margin-bottom:1rem;">Essential personal attributes</div>', unsafe_allow_html=True)
        gender_in = st.radio("Sex", ["Male", "Female"], horizontal=True)
        age_in = st.slider("Age (Years)", 0, 80, 30)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### Travel Details")
        st.markdown('<div style="font-size:1rem;color:#64748b;margin-bottom:1rem;">Ticket and boarding information</div>', unsafe_allow_html=True)
        pclass_in = st.selectbox("Ticket Class", [1, 2, 3], format_func=lambda x: f"{x}st Class — Upper" if x==1 else (f"{x}nd Class — Middle" if x==2 else f"{x}rd Class — Lower"))
        fare_in = st.number_input("Fare Paid in GBP (£)", 0.0, 512.0, 32.0)
        embarked_in = st.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"])

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### Family Composition")
        st.markdown('<div style="font-size:1rem;color:#64748b;margin-bottom:1rem;">Companions aboard</div>', unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        with f1: sib_in = st.number_input("Siblings / Spouses", 0, 8, 0)
        with f2: par_in = st.number_input("Parents / Children", 0, 9, 0)

        st.markdown('<div class="predict-btn-wrap">', unsafe_allow_html=True)
        predict_btn = st.button("Predict Fate")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_out:
        if "res_prob" not in st.session_state: st.session_state.res_prob = None

        if predict_btn and model:
            input_row = pd.DataFrame([{
                'age': float(age_in), 'sex': gender_in.lower(), 'pclass': pclass_in,
                'fare': float(fare_in), 'cabin': 1 if fare_in > 50 else 0,
                'FamilySize': sib_in + par_in + 1, 'Title': 'Miss' if gender_in == 'Female' else 'Mr'
            }])
            st.session_state.res_prob = model.predict_proba(input_row)[0][1]

        if st.session_state.res_prob is not None:
            p = st.session_state.res_prob
            survived = p > 0.5
            verdict_text = "Survived" if survived else "Did Not Survive"

            influences = []
            if pclass_in == 3: influences.append("🎫 3rd Class Ticket (strong)")
            elif pclass_in == 1: influences.append("🎫 1st Class Ticket (strong)")
            if fare_in < 15: influences.append(f"💰 Low Fare £{fare_in:.2f} (moderate)")
            elif fare_in > 50: influences.append(f"💰 High Fare £{fare_in:.2f} (moderate)")
            influences.append(f"⚓ Embarked at {embarked_in.split()[0]} (minor)")
            if sib_in + par_in > 3: influences.append("👨‍👩‍👧‍👦 Large family aboard (impact)")
            elif sib_in + par_in == 0: influences.append("👤 Traveling alone (minor)")

            influences_html = "<br>".join(influences)
            rationale_html = ("The passenger's survival is highly likely due to prioritizing women and first-class passengers on lifeboats."
                              if survived else
                              "Low fare and 3rd class boarding are the most influential factors decreasing survival probability.")

            html_str = f"""
            <div class="bi-card" style="display:flex;gap:2rem;align-items:stretch;margin-bottom:1.5rem;">
                <div style="flex:0 0 160px;display:flex;justify-content:center;align-items:center;">
                    <img src="{img_b64}" style="width:100%;object-fit:contain;">
                </div>
                <div style="flex:1;display:flex;flex-direction:column;">
                    <div style="font-size:1rem;color:#64748b;margin-bottom:0.5rem;text-transform:uppercase;">Prediction Result</div>
                    <div style="font-size:2.8rem;margin-bottom:0.5rem;color:{'#118a3d' if survived else '#cb3b3b'};font-weight:800;">
                        {verdict_text} <span style="color:#ea580c;font-size:2.2rem;margin-left:10px;">{p*100:.0f}%</span>
                    </div>
                    <div style="font-size:1rem;color:#475569;margin-bottom:1.5rem;line-height:1.5;">
                        <b>Model Rationale:</b> {rationale_html}
                    </div>
                    <div style="display:flex;gap:1rem;flex:1;">
                        <div style="flex:1;background:#f8fafc;padding:1rem;border-radius:8px;border:1px solid #e2e8f0;">
                            <div style="font-size:0.85rem;font-weight:700;color:#64748b;margin-bottom:0.5rem;text-transform:uppercase;">Confidence</div>
                            <div style="font-size:1.4rem;font-weight:800;color:#0f172a;line-height:1.2;">
                                {p*100:.0f}% chance of<br>survival
                            </div>
                        </div>
                        <div style="flex:1.2;background:#f8fafc;padding:1rem;border-radius:8px;border:1px solid #e2e8f0;">
                            <div style="font-size:0.85rem;font-weight:700;color:#64748b;margin-bottom:0.5rem;text-transform:uppercase;">Top Influences</div>
                            <div style="font-size:0.95rem;color:#0f172a;line-height:1.4;font-weight:500;">{influences_html}</div>
                        </div>
                    </div>
                </div>
            </div>
            """
            html_str = "\n".join([line.strip() for line in html_str.split("\n")])
            st.markdown(html_str, unsafe_allow_html=True)

            col_out_1, col_out_2 = st.columns(2)

            with col_out_1:
                st.markdown("### Historic Rates by Passenger Class")
                if df is not None:
                    rates = df.groupby('pclass')['survived'].mean().reset_index()
                    rates['label'] = (rates['survived'] * 100).round(1).astype(str) + '%'
                    fig = px.bar(rates, x='pclass', y='survived', text='label', color_discrete_sequence=['#6b9bd2'])
                    fig.update_traces(textposition='outside', textfont=dict(size=14, color='#0f172a'))
                    fig.update_layout(height=270, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    fig.update_xaxes(title="", tickmode='array', tickvals=[1,2,3], ticktext=['1st','2nd','3rd'])
                    fig.update_yaxes(title="", showgrid=False, showticklabels=False, range=[0,1.2])
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            with col_out_2:
                st.markdown("### Passenger Snapshot Vector")
                vec_html = f"""
                <div class="bi-card">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:1rem;margin-bottom:1rem;">
                        <div>
                            <div style="color:#64748b;font-size:0.85rem;font-weight:600;">Sex</div>
                            <div style="font-weight:700;margin-bottom:0.5rem;color:#0f172a;">{gender_in}</div>
                            <div style="color:#64748b;font-size:0.85rem;font-weight:600;">Age</div>
                            <div style="font-weight:700;margin-bottom:0.5rem;color:#0f172a;">{age_in}</div>
                            <div style="color:#64748b;font-size:0.85rem;font-weight:600;">Pclass</div>
                            <div style="font-weight:700;margin-bottom:0.5rem;color:#0f172a;">{pclass_in}</div>
                        </div>
                        <div>
                            <div style="color:#64748b;font-size:0.85rem;font-weight:600;">Fare</div>
                            <div style="font-weight:700;margin-bottom:0.5rem;color:#0f172a;">£{fare_in:.2f}</div>
                            <div style="color:#64748b;font-size:0.85rem;font-weight:600;">Embarked</div>
                            <div style="font-weight:700;margin-bottom:0.5rem;color:#0f172a;">{embarked_in.split()[0]}</div>
                            <div style="color:#64748b;font-size:0.85rem;font-weight:600;">SibSp / Parch</div>
                            <div style="font-weight:700;margin-bottom:0.5rem;color:#0f172a;">{sib_in} / {par_in}</div>
                        </div>
                    </div>
                    <div style="padding-top:1rem;border-top:2px solid #f1f5f9;font-size:0.9rem;color:#475569;">
                        <div style="margin-bottom:0.5rem;font-weight:600;">Feature Vector (for model)</div>
                        <div style="font-family:monospace;font-size:0.85rem;background:#f8fafc;padding:0.8rem;border-radius:6px;word-break:break-all;color:#0f172a;">
                            [Sex:{0 if gender_in=='Male' else 1}, Age:{age_in}, Pclass:{pclass_in}, Fare:{fare_in}, Embarked:{embarked_in[0]}, SibSp:{sib_in}, Parch:{par_in}]
                        </div>
                    </div>
                </div>
                """
                vec_html = "\n".join([line.strip() for line in vec_html.split("\n")])
                st.markdown(vec_html, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="bi-card" style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;opacity:0.45;min-height:400px;">
                <img src="{img_b64}" width="200" style="margin-top:2rem;">
                <p style="margin-top:2rem;font-weight:700;font-size:1.3rem;color:#0f172a;">Configure inputs and press "Predict Fate"</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div style="text-align:center;padding:0.5rem 1rem;color:#64748b;font-size:1rem;">Developed by <b style="color:#001f3f;">Bhagesh Biradar</b></div>', unsafe_allow_html=True)


# =====================================================================
# TAB 2 - ANALYSIS DASHBOARD
# =====================================================================
with tab2:
    st.markdown('<div class="page-title" style="font-size:2rem;font-weight:800;color:#0f172a;margin-bottom:0.3rem;">Survival Analysis Overview</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.1rem;color:#475569;margin-bottom:1.5rem;">Interactive dashboard visualizing Titanic survival patterns.</div>', unsafe_allow_html=True)
    st.divider()

    # Filters
    if 'f_sex' not in st.session_state: st.session_state.f_sex = "All"
    if 'f_pclass' not in st.session_state: st.session_state.f_pclass = "All"
    if 'f_embarked' not in st.session_state: st.session_state.f_embarked = "All"

    fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns([1, 1, 1, 0.7, 1])
    with fcol1: sex_filter = st.selectbox("Sex", ["All", "Male", "Female"], key="da_sex")
    with fcol2: pclass_filter = st.selectbox("Class", ["All", "1st", "2nd", "3rd"], key="da_pclass")
    with fcol3: embarked_filter = st.selectbox("Port", ["All", "Cherbourg", "Queenstown", "Southampton"], key="da_port")
    with fcol4:
        if st.button("Reset"):
            st.rerun()

    # Apply filters
    df_filtered = df.copy() if df is not None else pd.DataFrame()
    if not df_filtered.empty:
        if sex_filter != "All": df_filtered = df_filtered[df_filtered['sex'] == sex_filter]
        if pclass_filter != "All": df_filtered = df_filtered[df_filtered['pclass'] == int(pclass_filter[0])]
        if embarked_filter != "All": df_filtered = df_filtered[df_filtered['embarked'] == embarked_filter]

    # KPI Row
    st.markdown("<br>", unsafe_allow_html=True)
    kc1, kc2, kc3 = st.columns(3)
    total_p = len(df_filtered) if not df_filtered.empty else 0
    surv_rate = (df_filtered['survived'].mean() * 100) if not df_filtered.empty and len(df_filtered) > 0 else 0
    avg_fare = df_filtered['fare'].mean() if not df_filtered.empty and len(df_filtered) > 0 else 0

    with kc1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Passengers</div><div class="metric-value">{total_p} 👥</div></div>', unsafe_allow_html=True)
    with kc2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Survival Rate</div><div class="metric-value" style="color:#118a3d !important;">{surv_rate:.1f}%</div></div>', unsafe_allow_html=True)
    with kc3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Average Fare</div><div class="metric-value">${avg_fare:.2f}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    ch1, ch2, ch3 = st.columns([1.2, 1, 1], gap="large")

    with ch1:
        st.markdown("### Survival Count by Sex & Class")
        if not df_filtered.empty:
            agg = df_filtered.groupby(['pclass', 'sex', 'survived']).size().reset_index(name='count')
            agg['survived_str'] = agg['survived'].map({1: 'Survived', 0: 'Perished'})
            agg['x_label'] = agg['pclass'].astype(str) + "cls " + agg['sex']
            fig1 = px.bar(agg, x="x_label", y="count", color='survived_str',
                         color_discrete_map={'Survived': '#118a3d', 'Perished': '#64748b'}, barmode='stack',
                         text='count')
            fig1.update_traces(textposition='inside', textfont=dict(size=12, color='white'))
            fig1.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0), plot_bgcolor='rgba(0,0,0,0)',
                               paper_bgcolor='rgba(0,0,0,0)', showlegend=True,
                               legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
            fig1.update_xaxes(showgrid=False, title="")
            fig1.update_yaxes(showgrid=False, showticklabels=False, title="")
            st.plotly_chart(fig1, use_container_width=True)

        st.markdown("### Age Distribution by Survival")
        st.markdown('<div style="font-size:0.9rem;color:#64748b;margin-bottom:0.5rem;">Passenger count by age group — green = survived, gray = perished</div>', unsafe_allow_html=True)
        if not df_filtered.empty and 'age' in df_filtered.columns:
            age_df2 = df_filtered.dropna(subset=['age']).copy()
            age_df2['Survival Status'] = age_df2['survived'].map({1: 'Survived', 0: 'Perished'})
            fig2 = px.histogram(
                age_df2, x="age",
                color="Survival Status",
                color_discrete_map={'Survived': '#118a3d', 'Perished': '#94a3b8'},
                barmode="overlay",
                nbins=30,
                opacity=0.78,
                labels={"age": "Age (years)", "count": "Passengers"}
            )
            fig2.update_traces(marker_line_width=0.4, marker_line_color='white')
            fig2.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=20, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(
                    title="",
                    orientation="h",
                    yanchor="top", y=0.99,
                    xanchor="right", x=0.99,
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#e2e8f0', borderwidth=1,
                    font=dict(size=11)
                )
            )
            fig2.update_xaxes(title="Age (years)", showgrid=True, gridcolor='#f1f5f9', tickfont=dict(size=11))
            fig2.update_yaxes(title="Passenger Count", showgrid=True, gridcolor='#f1f5f9',
                              showticklabels=True, tickfont=dict(size=11))
            st.plotly_chart(fig2, use_container_width=True)

    with ch2:
        st.markdown("### Age vs Fare (Survival Colored)")
        if not df_filtered.empty:
            sc_df = df_filtered.dropna(subset=['age', 'fare']).copy()
            sc_df['norm_age'] = sc_df['age'] - sc_df['age'].mean()
            sc_df['norm_fare'] = sc_df['fare'] - sc_df['fare'].mean()
            fig3 = px.scatter(sc_df, x="norm_age", y="norm_fare", color="survived",
                              color_discrete_map={1: '#118a3d', 0: '#334155'}, opacity=0.65)
            fig3.add_shape(type="line", x0=0, y0=-200, x1=0, y1=500, line=dict(color="#cbd5e1", width=2))
            fig3.add_shape(type="line", x0=-50, y0=0, x1=50, y1=0, line=dict(color="#cbd5e1", width=2))
            fig3.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0), plot_bgcolor='rgba(0,0,0,0)',
                               paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
            fig3.update_xaxes(showgrid=False, showticklabels=False, title="Age (centered)")
            fig3.update_yaxes(showgrid=False, showticklabels=False, title="Fare (centered)")
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### Legend & Quick Stats")
        if not df_filtered.empty:
            s_c = len(df_filtered[df_filtered['survived']==1])
            p_c = len(df_filtered[df_filtered['survived']==0])
            s_age = df_filtered[df_filtered['survived']==1]['age'].median()
            p_age = df_filtered[df_filtered['survived']==0]['age'].median()
            s_fare = df_filtered[df_filtered['survived']==1]['fare'].median()
            p_fare = df_filtered[df_filtered['survived']==0]['fare'].median()
            st.markdown(f"""
            <div class="bi-card">
                <div style="margin-bottom:1rem;">
                    <span style="color:#118a3d;font-weight:800;font-size:1.1rem;">● Survived:</span>
                    <span style="font-weight:600;color:#0f172a;"> {s_c} passengers</span>
                    <div style="color:#475569;font-size:0.95rem;margin-top:0.3rem;">Median Age: <b>{s_age:.0f}</b> | Median Fare: <b>${s_fare:.1f}</b></div>
                </div>
                <div>
                    <span style="color:#64748b;font-weight:800;font-size:1.1rem;">● Perished:</span>
                    <span style="font-weight:600;color:#0f172a;"> {p_c} passengers</span>
                    <div style="color:#475569;font-size:0.95rem;margin-top:0.3rem;">Median Age: <b>{p_age:.0f}</b> | Median Fare: <b>${p_fare:.1f}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with ch3:
        st.markdown("### Passenger Records")
        if not df_filtered.empty:
            disp = df_filtered[['passengerid', 'name', 'age', 'sex', 'pclass', 'fare']].copy()
            disp['fare'] = disp['fare'].apply(lambda x: f"${x:.2f}")
            disp['age'] = disp['age'].fillna("").apply(lambda x: f"{float(x):.0f}" if x != "" else "—")
            disp.columns = ['ID', 'Name', 'Age', 'Sex', 'Pclass', 'Fare']
            st.dataframe(disp, hide_index=True, use_container_width=True, height=640)

    st.divider()
    st.markdown('<div style="text-align:center;padding:0.5rem 1rem;color:#64748b;font-size:1rem;">Developed by <b style="color:#001f3f;">Bhagesh Biradar</b></div>', unsafe_allow_html=True)
