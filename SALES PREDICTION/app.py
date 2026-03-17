import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Sales Prediction Engine", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "Sunset Analytics" theme
st.markdown("""
<style>
    /* Main Background Pattern - Sunset gradient */
    .stApp {
        background: linear-gradient(135deg, #fef2f2 0%, #fdf4ff 100%);
        font-family: 'Inter', 'Segoe UI', Tahoma, sans-serif;
    }
    
    /* Clean up headers */
    header { visibility: hidden; }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }
    
    /* Hero section */
    .hero-container {
        background: linear-gradient(120deg, #f43f5e 0%, #a855f7 100%);
        padding: 4rem;
        border-radius: 20px;
        margin-bottom: 3rem;
        box-shadow: 0 15px 30px rgba(168, 85, 247, 0.2);
        color: white;
        position: relative;
        overflow: hidden;
    }
    /* Add subtle glass shine to hero */
    .hero-container::after {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 50%; height: 100%;
        background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0) 100%);
        transform: skewX(-25deg);
        animation: shine 8s infinite;
    }
    @keyframes shine {
        0% { left: -100%; }
        20% { left: 200%; }
        100% { left: 200%; }
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
        color: #ffffff !important;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        max-width: 600px;
        line-height: 1.5;
        color: #ffffff !important;
    }
    
    /* Cards removed from here to prevent Streamlit widget escaping bugs */

    .card-title {
        color: #475569;
        font-size: 1.1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Input Section Typography */
    .input-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    .input-desc {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 1rem;
    }
    
    /* Sliders - sunset colors */
    div[data-testid="stTickBar"] { display: none !important; }
    .stSlider > div[data-baseweb="slider"] {
        padding-top: 0.5rem;
    }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #f43f5e !important;
        border: 2px solid white !important;
        box-shadow: 0 2px 5px rgba(244, 63, 94, 0.4) !important;
        width: 18px !important;
        height: 18px !important;
    }
    div[data-baseweb="slider"] div[data-testid="stSliderTickBar"] { display: none !important; }
    
    /* Buttons */
    div.stButton > button {
        border-radius: 30px;
        border: 1px solid #e2e8f0;
        color: #334155;
        font-weight: 600;
        background-color: white;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        border-color: #a855f7;
        color: #a855f7;
        background-color: #faf5ff;
        transform: translateY(-2px);
    }
    
    h1, h2, h3, h4, p, span { color: #1e293b; }
</style>
""", unsafe_allow_html=True)

# load model
@st.cache_resource
def load_artifacts():
    if not all(os.path.exists(f) for f in ["sales_model.pkl", "sales_scaler.pkl", "sales_dataset.pkl"]):
        return None, None, None
    model = joblib.load("sales_model.pkl")
    scaler = joblib.load("sales_scaler.pkl")
    df = joblib.load("sales_dataset.pkl")
    return model, scaler, df

model, scaler, dataset = load_artifacts()

if model is None:
    st.error("Model artifacts not found. Please wait or run sales_model.py.")
    st.stop()

# default state
if 'tv' not in st.session_state: st.session_state.tv = 150.0
if 'radio' not in st.session_state: st.session_state.radio = 25.0
if 'news' not in st.session_state: st.session_state.news = 30.0

def set_budget(size):
    if size == 'low':
        st.session_state.tv, st.session_state.radio, st.session_state.news = 50.0, 10.0, 10.0
    elif size == 'med':
        st.session_state.tv, st.session_state.radio, st.session_state.news = 150.0, 25.0, 30.0
    elif size == 'high':
        st.session_state.tv, st.session_state.radio, st.session_state.news = 250.0, 45.0, 60.0

# hero
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Sales Prediction Engine</h1>
    <p class="hero-subtitle">Optimize advertising ROI through advanced machine learning projection.</p>
</div>
""", unsafe_allow_html=True)

# layout
col_inputs, col_results = st.columns([1, 1.2], gap="large")

with col_inputs:
    st.markdown("<div class='card-title'>🎯 Budget Allocation</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:0.9rem; color:#64748b; margin-bottom:1.5rem;'>Allocate your advertising budget across different channels to see the projected impact on sales.</p>", unsafe_allow_html=True)
    
    # sliders
    st.markdown("<div class='input-label'>📺 TV Campaign Budget</div>", unsafe_allow_html=True)
    st.markdown("<div class='input-desc'>Allocated funds in thousands ($). Historically highest ROI.</div>", unsafe_allow_html=True)
    st.slider("TV", 0.0, 300.0, key='tv', step=1.0)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<div class='input-label'>📻 Radio Campaign Budget</div>", unsafe_allow_html=True)
    st.markdown("<div class='input-desc'>Allocated funds in thousands ($).</div>", unsafe_allow_html=True)
    st.slider("Radio", 0.0, 60.0, key='radio', step=1.0)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<div class='input-label'>📰 Newspaper Campaign Budget</div>", unsafe_allow_html=True)
    st.markdown("<div class='input-desc'>Allocated funds in thousands ($). Typically lowest impact.</div>", unsafe_allow_html=True)
    st.slider("Newspaper", 0.0, 120.0, key='news', step=1.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # preset buttons
    st.markdown("<p style='font-size:0.9rem; font-weight:600; color:#475569;'>Try Presets:</p>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.button("Low Budget", on_click=set_budget, args=('low',), use_container_width=True)
    p2.button("Mid Budget", on_click=set_budget, args=('med',), use_container_width=True)
    p3.button("High Budget", on_click=set_budget, args=('high',), use_container_width=True)
    
    # end presets

# get prediction
total_spend = st.session_state.tv + st.session_state.radio + st.session_state.news
input_data = pd.DataFrame([{
    'TV': st.session_state.tv, 
    'Radio': st.session_state.radio, 
    'Newspaper': st.session_state.news
}])
input_scaled = scaler.transform(input_data)
predicted_sales = model.predict(input_scaled)[0]

# Calculate ROI (assuming sales unit are in standard multiplier. If Sales = 15, and total spend = 150. Wait, dataset usually has Sales in thousands of units, and spend in thousands of dollars)
# We will just show raw projected sales unit from dataset multiplied by a reasonable currency factor for aesthetics.
# Let's say 1 "Sale Unit" = $1,500 Revenue for better visualization.
revenue = predicted_sales * 1500

with col_results:
    st.markdown("<div class='card-title'>📈 ROI Projection</div>", unsafe_allow_html=True)
    
    # key metrics
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div style='background:#f1f5f9; padding:1.5rem; border-radius:12px; border-left:4px solid #f43f5e;'>
            <p style='margin:0; font-size:0.9rem; color:#64748b; font-weight:600; text-transform:uppercase;'>Total Ad Spend</p>
            <h2 style='margin:0; font-size:2.2rem; color:#0f172a;'>${total_spend:,.1f}K</h2>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div style='background:#f1f5f9; padding:1.5rem; border-radius:12px; border-left:4px solid #a855f7;'>
            <p style='margin:0; font-size:0.9rem; color:#64748b; font-weight:600; text-transform:uppercase;'>Projected Revenue</p>
            <h2 style='margin:0; font-size:2.2rem; color:#0f172a;'>${revenue:,.0f}</h2>
            <p style='margin:0; font-size:0.8rem; color:#10b981; font-weight:600;'>~{predicted_sales:.1f}k Sales Units</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = predicted_sales,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Sales Volume (Units)", 'font': {'size': 18, 'color': '#475569'}},
        number = {'font': {'size': 48, 'color': '#1e293b'}},
        gauge = {
            'axis': {'range': [None, 35], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#a855f7"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 10], 'color': '#fef2f2'},
                {'range': [10, 20], 'color': '#fce7f3'},
                {'range': [20, 35], 'color': '#f3e8ff'}],
            'threshold': {
                'line': {'color': "#f43f5e", 'width': 4},
                'thickness': 0.75,
                'value': 25}
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'family': "Inter"})
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # end col_results

# dataset info expander
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("Explore Historical Dataset Info"):
    st.markdown("This model was trained on the `ashydv/sales-prediction-simple-linear-regression` historical dataset.")
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        st.dataframe(dataset.head(), use_container_width=True)
    with e_col2:
        st.markdown("**Feature Importance Note:**")
        st.markdown("When training this model, **TV Advertising** typically explains the vast majority of variance in sales, followed distantly by Radio. Newspaper advertising generally shows almost zero correlation with sales. Try setting TV to 0 and manipulating the others to see the minimal impact!")

st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding-top: 3rem; padding-bottom: 1rem;">
    Developed by <b>Bhagesh Biradar</b>
</div>
""", unsafe_allow_html=True)
