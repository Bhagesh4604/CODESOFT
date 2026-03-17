import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Iris Species Predictor", 
    page_icon="🌿", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# my css
st.markdown("""
<style>
    /* background */
    .stApp {
        background-color: #fafafa;
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
    }
    
    /* clean container */
    header { visibility: hidden; }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }
    
    .st-emotion-cache-12fmjuu { display: none; }
    
    /* top part */
    .hero-container {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        padding: 3rem 4rem;
        border-radius: 16px;
        margin-bottom: 3rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: nowrap;
        gap: 2rem;
    }
    .hero-text-content {
        flex: 1;
        min-width: 0;
    }
    .hero-title-main {
        font-size: 3.2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        line-height: 1.1;
        letter-spacing: -1px;
    }
    .hero-subtitle-main {
        font-size: 1.25rem;
        color: #475569;
        margin-top: 0.75rem;
        font-weight: 400;
    }
    .hero-graphic {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        object-fit: cover;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        border: 4px solid white;
        flex-shrink: 0;
    }
    
    h2, h3, h4, p, span {
        color: #1e293b;
    }
    
    /* button style */
    div.stButton > button {
        border-radius: 20px;
        border: 1px solid #cbd5e1;
        color: #0f172a;
        font-weight: 600;
        background-color: #ffffff;
        padding: 0.25rem 1rem;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        border-color: #94a3b8;
        background-color: #f8fafc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #000000;
    }
    
    /* specific buttons */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1) div.stButton > button {
        color: #2563eb !important;
        background-color: #eff6ff !important;
        border-color: #eff6ff !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1) div.stButton > button:hover { background-color: #dbeafe !important; }
    
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div.stButton > button {
        color: #16a34a !important;
        background-color: #f0fdf4 !important;
        border-color: #f0fdf4 !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) div.stButton > button:hover { background-color: #dcfce7 !important; }
    
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div.stButton > button {
        color: #9333ea !important;
        background-color: #faf5ff !important;
        border-color: #faf5ff !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(3) div.stButton > button:hover { background-color: #f3e8ff !important; }
    
    /* hide slider parts */
    div[data-testid="stTickBar"] { display: none !important; }
    .stSlider > div[data-baseweb="slider"] {
        padding-top: 0.5rem;
    }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #0f172a !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    div[data-baseweb="slider"] div[data-testid="stSliderTickBar"] { display: none !important; }
    
    .stSlider [data-testid="stMarkdownContainer"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# load model
@st.cache_resource
def load_artifacts():
    if not all(os.path.exists(f) for f in ["iris_model.pkl", "iris_scaler.pkl"]):
        return None, None
    model = joblib.load("iris_model.pkl")
    scaler = joblib.load("iris_scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

if model is None:
    st.error("Model artifacts not found. Please run the training script.")
    st.stop()

# default values
if 'sl' not in st.session_state: st.session_state.sl = 5.8
if 'sw' not in st.session_state: st.session_state.sw = 3.0
if 'pl' not in st.session_state: st.session_state.pl = 4.4
if 'pw' not in st.session_state: st.session_state.pw = 1.3

def set_sample(type_str):
    if type_str == 'setosa':
        st.session_state.sl, st.session_state.sw, st.session_state.pl, st.session_state.pw = 5.0, 3.4, 1.5, 0.2
    elif type_str == 'versicolor':
        st.session_state.sl, st.session_state.sw, st.session_state.pl, st.session_state.pw = 5.8, 2.7, 4.1, 1.0
    elif type_str == 'virginica':
        st.session_state.sl, st.session_state.sw, st.session_state.pl, st.session_state.pw = 6.8, 3.2, 5.9, 2.3
    elif type_str == 'median':
        st.session_state.sl, st.session_state.sw, st.session_state.pl, st.session_state.pw = 5.8, 3.0, 4.4, 1.3

# header
st.markdown("""
<div class="hero-container">
    <div class="hero-text-content">
        <h1 class="hero-title-main">Iris Analytical Engine</h1>
        <p class="hero-subtitle-main">High-precision species classification leveraging classical machine learning.</p>
    </div>
    <img src="https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg" class="hero-graphic" alt="Iris Flower">
</div>
""", unsafe_allow_html=True)


col_left, col_right = st.columns([1.1, 1.0], gap="large")

with col_left:
    
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem;">
        <h2 style="margin:0; padding:0; font-size: 1.8rem; font-weight:700; color: #0f172a;">Iris Species Predictor</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # buttons
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.button("Setosa example", use_container_width=True, on_click=set_sample, args=('setosa',))
    with ex2:
        st.button("Versicolor example", use_container_width=True, on_click=set_sample, args=('versicolor',))
    with ex3:
        st.button("Virginica example", use_container_width=True, on_click=set_sample, args=('virginica',))
        
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
    
    # inputs
    # Sepal Length
    st.markdown("<p style='font-size:1.1rem; font-weight:700; color:#1e293b; margin-bottom:0;'>Sepal Length</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.85rem; color:#94a3b8; margin-top:0; margin-bottom:0;'>Range 4.0 - 8.0 cm &bull; step 0.1</p>", unsafe_allow_html=True)
    st.slider("Sepal Length", 4.0, 8.0, key='sl', step=0.1, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sepal Width
    st.markdown("<p style='font-size:1.1rem; font-weight:700; color:#1e293b; margin-bottom:0;'>Sepal Width</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.85rem; color:#94a3b8; margin-top:0; margin-bottom:0;'>Range 2.0 - 5.0 cm &bull; step 0.1</p>", unsafe_allow_html=True)
    st.slider("Sepal Width", 2.0, 5.0, key='sw', step=0.1, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Petal Length
    st.markdown("<p style='font-size:1.1rem; font-weight:700; color:#1e293b; margin-bottom:0;'>Petal Length</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.85rem; color:#94a3b8; margin-top:0; margin-bottom:0;'>Range 1.0 - 7.0 cm &bull; step 0.1</p>", unsafe_allow_html=True)
    st.slider("Petal Length", 1.0, 7.0, key='pl', step=0.1, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Petal Width
    st.markdown("<p style='font-size:1.1rem; font-weight:700; color:#1e293b; margin-bottom:0;'>Petal Width</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.85rem; color:#94a3b8; margin-top:0; margin-bottom:0;'>Range 0.1 - 2.5 cm &bull; step 0.1</p>", unsafe_allow_html=True)
    st.slider("Petal Width", 0.1, 2.5, key='pw', step=0.1, label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns([1, 1, 1])
    with r1:
        st.markdown("<p style='color:#94a3b8; font-size:0.85rem; margin-top:10px;'>Adjust any control — prediction updates live</p>", unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <style>
            div[data-testid="stHorizontalBlock"]:last-of-type div.stButton > button {
                color: #0f172a !important; background-color: #ffffff !important; border-color: #cbd5e1 !important; margin-top: 5px;
            }
            div[data-testid="stHorizontalBlock"]:last-of-type div.stButton > button:hover {
                background-color: #f8fafc !important;
            }
        </style>
        """, unsafe_allow_html=True)
        st.button("Reset to medians", use_container_width=True, on_click=set_sample, args=('median',))


# get prediction
input_data = pd.DataFrame([{
    'sepal_length': st.session_state.sl, 
    'sepal_width': st.session_state.sw, 
    'petal_length': st.session_state.pl, 
    'petal_width': st.session_state.pw
}])
input_scaled = scaler.transform(input_data)
pred_class = model.predict(input_scaled)[0]
probs = model.predict_proba(input_scaled)[0]

clean_class = pred_class.replace("Iris-", "").capitalize()

color_map = {
    "Setosa": "#2563eb",      # exact blue from mockup
    "Versicolor": "#22c55e",  # exact green from mockup
    "Virginica": "#8b5cf6"    # exact purple from mockup
}
bg_color = color_map.get(clean_class, "#22c55e")

with col_right:
    # Top padding to perfectly align with the left side headers
    st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)
        
    st.markdown(f"""
    <div style="background-color: {bg_color}; color: white; padding: 10px 24px; border-radius: 6px; font-size: 1.8rem; font-weight: 700; width: fit-content; margin-bottom: 0.2rem; display: inline-block; letter-spacing: -0.5px;">
        {clean_class}
    </div>
    <p style="color:#94a3b8; font-size:0.95rem; margin-top:0.5rem; margin-bottom:2.5rem;">Dominant prediction (updates live)</p>
    """, unsafe_allow_html=True)
    
    # progress bars
    classes_clean = [c.replace("Iris-", "").capitalize() for c in model.classes_]
    
    html_bars = ""
    for idx, cls in enumerate(classes_clean):
        pct = int(probs[idx] * 100)
        c_color = color_map.get(cls, "#cbd5e1")
        # Ensure at least 1% width so the rounded corner is visible, if pct is 0 make it 0
        w = max(pct, 1) if pct > 0 else 0 
        
        html_bars += f"""
        <div style="margin-bottom: 2.2rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
                <span style="font-weight:700; color:#1e293b; font-size:1.05rem;">{cls}</span>
                <span style="font-weight:700; color:#1e293b; font-size:1.05rem;">{pct}%</span>
            </div>
            <div style="background-color:#f1f5f9; border-radius:10px; height:12px; width:100%; overflow:hidden;">
                <div style="background-color:{c_color}; height:100%; width:{w}%; border-radius:10px; transition: width 0.4s ease;"></div>
            </div>
        </div>
        """
    st.markdown(html_bars, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top:4rem;'></div>", unsafe_allow_html=True)
    
    # raw data
    st.markdown("<p style='color:#94a3b8; font-size:0.9rem; margin-bottom:0.2rem;'>Feature vector</p>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background-color:#f8fafc; border:1px solid #f1f5f9; border-radius:8px; padding:14px 16px; font-family: 'Courier New', monospace; color:#334155; font-size:1.15rem; margin-bottom:2.5rem; letter-spacing: 1px;">
        [{st.session_state.sl:.1f}, {st.session_state.sw:.1f}, {st.session_state.pl:.1f}, {st.session_state.pw:.1f}]
    </div>
    """, unsafe_allow_html=True)
    


