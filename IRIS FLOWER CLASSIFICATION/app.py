import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background-color: #fafafa;
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
    }
    header { visibility: hidden; }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1300px !important;
    }
    .st-emotion-cache-12fmjuu { display: none; }

    /* hero */
    .hero-container {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        padding: 3rem 4rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: nowrap;
        gap: 2rem;
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
        font-size: 1.2rem;
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

    /* kpi cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .kpi-label { font-size: 0.8rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
    .kpi-value { font-size: 2rem; font-weight: 800; color: #0f172a; line-height: 1; margin-bottom: 0.2rem; }
    .kpi-sub { font-size: 0.85rem; color: #64748b; }

    /* comparison table */
    .cmp-table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
    .cmp-table th {
        padding: 0.6rem 1rem;
        background: #f8fafc;
        color: #475569;
        font-weight: 600;
        text-align: left;
        border-bottom: 2px solid #e2e8f0;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .cmp-table td { padding: 0.6rem 1rem; border-bottom: 1px solid #f1f5f9; color: #1e293b; }
    .cmp-table tr:last-child td { border-bottom: none; }
    .cmp-table tr.highlight td { background: #f0fdf4; font-weight: 700; }
    .cmp-highlight { color: #16a34a; font-weight: 700; }
    .cmp-above { color: #dc2626; }
    .cmp-below { color: #2563eb; }

    /* contribution bars */
    .contrib-wrap { margin-bottom: 1.6rem; }
    .contrib-label { display: flex; justify-content: space-between; margin-bottom: 0.35rem; }
    .contrib-track { background: #f1f5f9; border-radius: 6px; height: 10px; width: 100%; }
    .contrib-fill { height: 10px; border-radius: 6px; }

    /* section card */
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .section-title { font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 1rem; }

    h2, h3, h4, p, span { color: #1e293b; }

    /* buttons */
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

    /* sliders */
    div[data-testid="stTickBar"] { display: none !important; }
    .stSlider > div[data-baseweb="slider"] { padding-top: 0.5rem; }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #0f172a !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    .stSlider [data-testid="stMarkdownContainer"] { display: none !important; }

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px 8px 0 0;
        color: #64748b;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.6rem 1.4rem;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #0f172a !important;
        border-bottom: 2px solid #0f172a !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    files = ["iris_model.pkl", "iris_scaler.pkl", "iris_dataset.pkl"]
    if not all(os.path.exists(f) for f in files):
        return None, None, None
    model = joblib.load("iris_model.pkl")
    scaler = joblib.load("iris_scaler.pkl")
    df = joblib.load("iris_dataset.pkl")
    return model, scaler, df

model, scaler, df = load_artifacts()

if model is None:
    st.error("Model artifacts not found. Please run `python iris_model.py` first.")
    st.stop()

# Normalise species names
df['species'] = df['species'].str.replace("Iris-", "").str.capitalize()
FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
FEATURE_LABELS = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
COLORS = {'Setosa': '#2563eb', 'Versicolor': '#22c55e', 'Virginica': '#8b5cf6'}
COLOR_LIST = [COLORS[s] for s in SPECIES]

species_stats = df.groupby('species')[FEATURES].mean()

# ── Hero ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div>
        <h1 class="hero-title-main">Iris Analytical Engine</h1>
        <p class="hero-subtitle-main">High-precision species classification with full data analytics.</p>
    </div>
    <img src="https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg" class="hero-graphic" alt="Iris Flower">
</div>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────────
defaults = {'sl': 5.8, 'sw': 3.0, 'pl': 4.4, 'pw': 1.3}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def set_sample(t):
    presets = {
        'setosa':     (5.0, 3.4, 1.5, 0.2),
        'versicolor': (5.8, 2.7, 4.1, 1.0),
        'virginica':  (6.8, 3.2, 5.9, 2.3),
        'median':     (5.8, 3.0, 4.4, 1.3),
    }
    st.session_state.sl, st.session_state.sw, st.session_state.pl, st.session_state.pw = presets[t]

# ── Live prediction ─────────────────────────────────────────────────────────────
inp = pd.DataFrame([{
    'sepal_length': st.session_state.sl,
    'sepal_width':  st.session_state.sw,
    'petal_length': st.session_state.pl,
    'petal_width':  st.session_state.pw,
}])
inp_scaled = scaler.transform(inp)
pred_class = model.predict(inp_scaled)[0].replace("Iris-", "").capitalize()
probs = model.predict_proba(inp_scaled)[0]
classes_clean = [c.replace("Iris-", "").capitalize() for c in model.classes_]
prob_dict = dict(zip(classes_clean, probs))

# Feature importance (from trained model)
importances = model.feature_importances_

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🌸  Predictor", "📊  Analytics Dashboard"])

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.15, 1.0], gap="large")

    with col_left:
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem;">
            <h2 style="margin:0; font-size:1.8rem; font-weight:700; color:#0f172a;">Iris Species Predictor</h2>
        </div>""", unsafe_allow_html=True)

        # Sample preset buttons
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            if st.button("Setosa", use_container_width=True):
                set_sample('setosa')
                st.rerun()
        with ex2:
            if st.button("Versicolor", use_container_width=True):
                set_sample('versicolor')
                st.rerun()
        with ex3:
            if st.button("Virginica", use_container_width=True):
                set_sample('virginica')
                st.rerun()

        st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

        # Sliders
        for key, label, lo, hi in [
            ('sl', 'Sepal Length', 4.0, 8.0),
            ('sw', 'Sepal Width',  2.0, 5.0),
            ('pl', 'Petal Length', 1.0, 7.0),
            ('pw', 'Petal Width',  0.1, 2.5),
        ]:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0;">
                <p style='font-size:1.05rem; font-weight:700; color:#1e293b; margin:0;'>{label}</p>
                <span style='font-size:1.2rem; font-weight:800; color:#0f172a; background:#f1f5f9;
                      padding:2px 10px; border-radius:6px; font-variant-numeric:tabular-nums;'>
                    {st.session_state[key]:.1f} cm
                </span>
            </div>
            <p style='font-size:0.82rem; color:#94a3b8; margin-top:2px; margin-bottom:0;'>Range {lo} – {hi} cm &bull; step 0.1</p>""", unsafe_allow_html=True)
            st.slider(label, lo, hi, key=key, step=0.1, label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)

        r1, _, r3 = st.columns([1.4, 0.2, 1])
        with r1:
            st.markdown("<p style='color:#94a3b8; font-size:0.82rem; margin-top:12px;'>Adjust any slider — prediction updates live</p>", unsafe_allow_html=True)
        with r3:
            if st.button("Reset to medians", use_container_width=True):
                set_sample('median')
                st.rerun()

        # ── Species Comparison Table ────────────────────────────────────────────
        st.markdown("<div style='margin-top:2.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-title'>📏 Your Input vs. Species Averages</p>", unsafe_allow_html=True)

        inp_vals = [st.session_state.sl, st.session_state.sw, st.session_state.pl, st.session_state.pw]
        rows = ""
        for feat, label, val in zip(FEATURES, FEATURE_LABELS, inp_vals):
            avgs = species_stats[feat]
            cells = ""
            for sp in SPECIES:
                avg = avgs[sp]
                diff = val - avg
                badge = f'<span class="cmp-above">▲ +{diff:.1f}</span>' if diff > 0.1 else (f'<span class="cmp-below">▼ {diff:.1f}</span>' if diff < -0.1 else '<span style="color:#64748b">≈ avg</span>')
                cells += f"<td>{avg:.1f} {badge}</td>"
            rows += f"<tr><td><strong>{label}</strong></td><td><strong style='color:#0f172a;'>{val:.1f} cm</strong></td>{cells}</tr>"

        st.markdown(f"""
        <table class="cmp-table">
            <thead>
                <tr>
                    <th>Feature</th><th>Your Input</th>
                    <th style="color:{COLORS['Setosa']}">Setosa avg</th>
                    <th style="color:{COLORS['Versicolor']}">Versicolor avg</th>
                    <th style="color:{COLORS['Virginica']}">Virginica avg</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT COLUMN ─────────────────────────────────────────────────────────────
    with col_right:
        st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)
        bg = COLORS.get(pred_class, '#22c55e')

        # Prediction badge
        st.markdown(f"""
        <div style="background:{bg}; color:white; padding:10px 24px; border-radius:8px;
             font-size:1.8rem; font-weight:700; width:fit-content; margin-bottom:0.2rem; letter-spacing:-0.5px;">
            {pred_class}
        </div>
        <p style="color:#94a3b8; font-size:0.92rem; margin-top:0.5rem; margin-bottom:2rem;">Dominant prediction · updates live</p>
        """, unsafe_allow_html=True)

        # Probability bars
        for cls in classes_clean:
            pct = int(prob_dict[cls] * 100)
            c_color = COLORS.get(cls, '#cbd5e1')
            w = max(pct, 1) if pct > 0 else 0
            st.markdown(f"""
            <div style="margin-bottom:2rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                    <span style="font-weight:700; color:#1e293b;">{cls}</span>
                    <span style="font-weight:700; color:#1e293b;">{pct}%</span>
                </div>
                <div style="background:#f1f5f9; border-radius:10px; height:12px; overflow:hidden;">
                    <div style="background:{c_color}; height:100%; width:{w}%; border-radius:10px; transition:width 0.4s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Feature Contribution ──────────────────────────────────────────────────
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-title'>🎯 Feature Contribution to Prediction</p>", unsafe_allow_html=True)

        max_imp = max(importances)
        contrib_colors = ['#2563eb', '#16a34a', '#f59e0b', '#8b5cf6']
        for i, (label, imp) in enumerate(zip(FEATURE_LABELS, importances)):
            pct_bar = int((imp / max_imp) * 100)
            st.markdown(f"""
            <div class="contrib-wrap">
                <div class="contrib-label">
                    <span style="font-size:0.88rem; font-weight:600; color:#334155;">{label}</span>
                    <span style="font-size:0.88rem; font-weight:700; color:#0f172a;">{imp*100:.1f}%</span>
                </div>
                <div class="contrib-track">
                    <div class="contrib-fill" style="width:{pct_bar}%; background:{contrib_colors[i]};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Feature vector
        st.markdown(f"""
        <p style='color:#94a3b8; font-size:0.88rem; margin-bottom:0.3rem; margin-top:1rem;'>Feature vector</p>
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 16px;
             font-family:'Courier New',monospace; color:#334155; font-size:1.1rem; letter-spacing:1px;">
            [{st.session_state.sl:.1f}, {st.session_state.sw:.1f}, {st.session_state.pl:.1f}, {st.session_state.pw:.1f}]
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── KPI Row ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    total = len(df)
    counts = df['species'].value_counts()

    kpis = [
        ("Total Samples", str(total), "Fisher's Iris dataset"),
        ("Setosa", str(counts.get("Setosa", 0)), f"{counts.get('Setosa',0)/total*100:.0f}% of dataset"),
        ("Versicolor", str(counts.get("Versicolor", 0)), f"{counts.get('Versicolor',0)/total*100:.0f}% of dataset"),
        ("Virginica", str(counts.get("Virginica", 0)), f"{counts.get('Virginica',0)/total*100:.0f}% of dataset"),
        ("Model Accuracy", "96–100%", "Random Forest, 100 trees"),
    ]
    for col, (label, val, sub) in zip([k1, k2, k3, k4, k5], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # ── Row 1: Distribution + Scatter ────────────────────────────────────────────
    c1, c2 = st.columns([1, 1.6], gap="large")

    with c1:
        # Donut chart
        fig_pie = go.Figure(go.Pie(
            labels=SPECIES,
            values=[counts.get(s, 0) for s in SPECIES],
            hole=0.55,
            marker_colors=COLOR_LIST,
            textinfo='label+percent',
            textfont_size=13,
        ))
        fig_pie.update_layout(
            title=dict(text="Species Distribution", font=dict(size=15, color='#0f172a'), x=0),
            showlegend=False,
            margin=dict(t=50, b=20, l=20, r=20),
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        # Scatter — Petal dimensions, coloured by species
        fig_scatter = px.scatter(
            df, x='petal_length', y='petal_width', color='species',
            color_discrete_map=COLORS,
            labels={'petal_length': 'Petal Length (cm)', 'petal_width': 'Petal Width (cm)', 'species': 'Species'},
            title='Petal Dimensions by Species',
            hover_data=['sepal_length', 'sepal_width'],
        )
        # Mark user's input
        fig_scatter.add_scatter(
            x=[st.session_state.pl], y=[st.session_state.pw],
            mode='markers',
            marker=dict(size=16, color='#f59e0b', symbol='star', line=dict(color='white', width=1.5)),
            name='Your Input',
        )
        fig_scatter.update_layout(
            title=dict(font=dict(size=15, color='#0f172a'), x=0),
            margin=dict(t=50, b=20, l=20, r=20),
            paper_bgcolor='white', plot_bgcolor='#fafafa',
            legend=dict(font=dict(size=12)),
            height=300,
            xaxis=dict(gridcolor='#f1f5f9'), yaxis=dict(gridcolor='#f1f5f9'),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Row 2: Box Plots ──────────────────────────────────────────────────────────
    st.markdown("#### Feature Distributions by Species")
    b1, b2, b3, b4 = st.columns(4)
    box_cols = [b1, b2, b3, b4]

    for col_widget, feat, label in zip(box_cols, FEATURES, FEATURE_LABELS):
        with col_widget:
            fig_box = go.Figure()
            for sp in SPECIES:
                fig_box.add_trace(go.Box(
                    y=df[df['species'] == sp][feat],
                    name=sp,
                    marker_color=COLORS[sp],
                    boxmean=True,
                    showlegend=False,
                ))
            fig_box.update_layout(
                title=dict(text=label, font=dict(size=13, color='#0f172a'), x=0.05),
                margin=dict(t=40, b=20, l=10, r=10),
                paper_bgcolor='white', plot_bgcolor='#fafafa',
                yaxis=dict(title='cm', gridcolor='#f1f5f9', tickfont=dict(size=11)),
                height=280,
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # ── Row 3: Sepal Scatter + Feature Importance ─────────────────────────────────
    d1, d2 = st.columns([1.6, 1], gap="large")

    with d1:
        fig_sep = px.scatter(
            df, x='sepal_length', y='sepal_width', color='species',
            color_discrete_map=COLORS,
            labels={'sepal_length': 'Sepal Length (cm)', 'sepal_width': 'Sepal Width (cm)', 'species': 'Species'},
            title='Sepal Dimensions by Species',
        )
        fig_sep.add_scatter(
            x=[st.session_state.sl], y=[st.session_state.sw],
            mode='markers',
            marker=dict(size=16, color='#f59e0b', symbol='star', line=dict(color='white', width=1.5)),
            name='Your Input',
        )
        fig_sep.update_layout(
            title=dict(font=dict(size=15, color='#0f172a'), x=0),
            margin=dict(t=50, b=20, l=20, r=20),
            paper_bgcolor='white', plot_bgcolor='#fafafa',
            legend=dict(font=dict(size=12)),
            height=320,
            xaxis=dict(gridcolor='#f1f5f9'), yaxis=dict(gridcolor='#f1f5f9'),
        )
        st.plotly_chart(fig_sep, use_container_width=True)

    with d2:
        # Feature importance bar
        fig_imp = go.Figure(go.Bar(
            x=importances * 100,
            y=FEATURE_LABELS,
            orientation='h',
            marker_color=['#2563eb', '#16a34a', '#f59e0b', '#8b5cf6'],
            text=[f"{v*100:.1f}%" for v in importances],
            textposition='outside',
        ))
        fig_imp.update_layout(
            title=dict(text="Feature Importance (RF)", font=dict(size=15, color='#0f172a'), x=0),
            margin=dict(t=50, b=20, l=20, r=60),
            paper_bgcolor='white', plot_bgcolor='#fafafa',
            xaxis=dict(title='Importance (%)', gridcolor='#f1f5f9'),
            yaxis=dict(autorange='reversed'),
            height=320,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Row 4: Species Averages Radar ─────────────────────────────────────────────
    st.markdown("#### Species Profile — Radar Chart")
    fig_radar = go.Figure()

    for sp in SPECIES:
        vals = [species_stats.loc[sp, f] for f in FEATURES]
        vals_norm = [v / max(df[f]) for v, f in zip(vals, FEATURES)]  # normalise 0–1
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_norm + [vals_norm[0]],
            theta=FEATURE_LABELS + [FEATURE_LABELS[0]],
            fill='toself',
            name=sp,
            line_color=COLORS[sp],
            fillcolor=COLORS[sp],
            opacity=0.25,
        ))

    # User input on radar
    user_norm = [v / max(df[f]) for v, f in zip(inp_vals, FEATURES)]
    fig_radar.add_trace(go.Scatterpolar(
        r=user_norm + [user_norm[0]],
        theta=FEATURE_LABELS + [FEATURE_LABELS[0]],
        fill='toself',
        name='Your Input',
        line_color='#f59e0b',
        fillcolor='rgba(245,158,11,0.15)',
        line_dash='dash',
        line_width=2,
    ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10))),
        showlegend=True,
        legend=dict(font=dict(size=12)),
        paper_bgcolor='white',
        margin=dict(t=40, b=40, l=40, r=40),
        height=420,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Dataset Preview ───────────────────────────────────────────────────────────
    with st.expander("🗃️  View Raw Dataset"):
        st.dataframe(
            df.style.applymap(
                lambda v: f"color: {COLORS.get(v, '#1e293b')}; font-weight:600;" if isinstance(v, str) else "",
                subset=['species']
            ),
            use_container_width=True,
            height=320,
        )
