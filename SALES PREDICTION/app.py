import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Sales Prediction Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #fef2f2 0%, #fdf4ff 100%);
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    header { visibility: hidden; }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* Hero */
    .hero-container {
        background: linear-gradient(120deg, #f43f5e 0%, #a855f7 100%);
        padding: 3.5rem 4rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 30px rgba(168,85,247,0.2);
        position: relative;
        overflow: hidden;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 50%; height: 100%;
        background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0) 100%);
        transform: skewX(-25deg);
        animation: shine 8s infinite;
    }
    @keyframes shine { 0%{left:-100%} 20%{left:200%} 100%{left:200%} }
    .hero-title  { font-size:3.2rem; font-weight:800; margin:0; letter-spacing:-1px; color:#fff !important; }
    .hero-sub    { font-size:1.15rem; opacity:0.9; margin-top:0.5rem; color:#fff !important; }

    /* KPI cards */
    .kpi-card {
        background:#fff;
        border:1px solid #e2e8f0;
        border-radius:12px;
        padding:1.3rem 1.5rem;
        box-shadow:0 2px 4px rgba(0,0,0,0.04);
    }
    .kpi-label { font-size:0.75rem; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.3rem; }
    .kpi-value { font-size:1.8rem; font-weight:800; color:#0f172a; line-height:1; }
    .kpi-sub   { font-size:0.82rem; color:#64748b; margin-top:.2rem; }

    /* Insight cards */
    .insight-card {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: .8rem;
        border-left: 4px solid #f43f5e;
    }

    /* Slider value badge */
    .val-badge {
        display: inline-block;
        background: #f1f5f9;
        color: #0f172a;
        font-weight: 800;
        font-size: 1.1rem;
        padding: 2px 12px;
        border-radius: 6px;
        font-variant-numeric: tabular-nums;
    }

    /* Channel breakdown bars */
    .ch-bar-track { background:#f1f5f9; border-radius:8px; height:14px; margin-top:.4rem; }
    .ch-bar-fill  { height:14px; border-radius:8px; }

    /* Buttons */
    div.stButton > button {
        border-radius:30px; border:1px solid #e2e8f0;
        color:#334155; font-weight:600;
        background-color:#fff; padding:.4rem 1.4rem;
        transition:all .2s ease;
    }
    div.stButton > button:hover {
        border-color:#a855f7; color:#a855f7;
        background:#faf5ff; transform:translateY(-2px);
    }

    div[data-testid="stTickBar"] { display:none !important; }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color:#f43f5e !important;
        border:2px solid white !important;
        box-shadow:0 2px 5px rgba(244,63,94,.4) !important;
    }

    h1,h2,h3,h4,p,span { color:#1e293b; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap:.5rem; background:transparent;
        border-bottom:2px solid #e2e8f0; margin-bottom:2rem;
    }
    .stTabs [data-baseweb="tab"] {
        background:transparent; border-radius:8px 8px 0 0;
        color:#64748b; font-weight:600; font-size:.95rem;
        padding:.6rem 1.4rem; border:none;
    }
    .stTabs [aria-selected="true"] {
        background:#fff !important; color:#0f172a !important;
        border-bottom:2px solid #f43f5e !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    files = ["sales_model.pkl", "sales_scaler.pkl", "sales_dataset.pkl"]
    if not all(os.path.exists(f) for f in files):
        return None, None, None
    return joblib.load("sales_model.pkl"), joblib.load("sales_scaler.pkl"), joblib.load("sales_dataset.pkl")

model, scaler, dataset = load_artifacts()
if model is None:
    st.error("Model artifacts not found. Run `python sales_model.py` first.")
    st.stop()

# ── Session state ───────────────────────────────────────────────────────────────
if 'tv'   not in st.session_state: st.session_state.tv   = 150.0
if 'radio' not in st.session_state: st.session_state.radio = 25.0
if 'news' not in st.session_state: st.session_state.news  = 30.0

def set_budget(size):
    p = {'low':(50,10,10), 'med':(150,25,30), 'high':(250,45,60)}[size]
    st.session_state.tv, st.session_state.radio, st.session_state.news = p

# ── Prediction ──────────────────────────────────────────────────────────────────
total_spend = st.session_state.tv + st.session_state.radio + st.session_state.news
inp = pd.DataFrame([{'TV': st.session_state.tv, 'Radio': st.session_state.radio, 'Newspaper': st.session_state.news}])
predicted_sales = model.predict(scaler.transform(inp))[0]

# Revenue: dataset Sales = thousands of units. Avg price ~$1 → Sales × $1000 = revenue in dollars
# We show the raw model unit as "Sales Units (thousands)" and explain this clearly.
avg_price_k = 1.0   # $1,000 per unit → revenue in $thousands
revenue_k = predicted_sales * avg_price_k   # $thousands
roi_pct = ((revenue_k - total_spend) / total_spend * 100) if total_spend > 0 else 0

# Feature importances from RF
importances = model.feature_importances_
feat_names  = ['TV', 'Radio', 'Newspaper']
feat_colors = ['#f43f5e', '#a855f7', '#f59e0b']
imp_dict    = dict(zip(feat_names, importances))

# ── Hero ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Sales Prediction Engine</h1>
    <p class="hero-sub">Allocate your ad budget · get instant sales & ROI projections · understand what actually drives revenue.</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯  Budget Planner", "📊  Analytics Dashboard"])

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 1 — BUDGET PLANNER
# ══════════════════════════════════════════════════════════════════════════════════
with tab1:
    col_inputs, col_results = st.columns([1, 1.2], gap="large")

    with col_inputs:
        st.markdown("<p style='font-size:1rem; color:#64748b; margin-bottom:1.5rem;'>Allocate your advertising budget across channels to project sales impact.</p>", unsafe_allow_html=True)

        # ── Sliders with live $ labels ──────────────────────────────────────────
        for key, icon, label, lo, hi, note in [
            ('tv',    '📺', 'TV Campaign Budget',        0.0, 300.0, 'Historically highest ROI channel'),
            ('radio', '📻', 'Radio Campaign Budget',     0.0,  60.0, 'Good secondary channel'),
            ('news',  '📰', 'Newspaper Campaign Budget', 0.0, 120.0, 'Typically lowest impact on sales'),
        ]:
            val = st.session_state[key]
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:2px;">
                <span style="font-size:1.05rem; font-weight:700; color:#1e293b;">{icon} {label}</span>
                <span class="val-badge">${val:,.0f}K</span>
            </div>
            <p style="font-size:0.82rem; color:#94a3b8; margin:0 0 4px 0;">{note} &bull; Range ${lo:.0f}K – ${hi:.0f}K</p>
            """, unsafe_allow_html=True)
            st.slider(label, lo, hi, key=key, step=1.0, label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)

        # Presets
        st.markdown("<p style='font-size:.9rem; font-weight:600; color:#475569; margin-bottom:.5rem;'>Quick Presets:</p>", unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        p1.button("💼 Low Budget",  on_click=set_budget, args=('low',),  use_container_width=True)
        p2.button("📦 Mid Budget",  on_click=set_budget, args=('med',),  use_container_width=True)
        p3.button("🚀 High Budget", on_click=set_budget, args=('high',), use_container_width=True)

        # ── Budget split breakdown ──────────────────────────────────────────────
        st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1rem; font-weight:700; color:#0f172a; margin-bottom:.8rem;'>Budget Split</p>", unsafe_allow_html=True)
        if total_spend > 0:
            for key, label, color in [('tv','TV','#f43f5e'), ('radio','Radio','#a855f7'), ('news','Newspaper','#f59e0b')]:
                val  = st.session_state[key]
                pct  = val / total_spend * 100
                w    = max(int(pct), 1) if pct > 0 else 0
                st.markdown(f"""
                <div style="margin-bottom:.9rem;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:.3rem;">
                        <span style="font-size:.88rem; font-weight:600; color:#334155;">{label}</span>
                        <span style="font-size:.88rem; font-weight:700; color:#0f172a;">${val:,.0f}K &nbsp;({pct:.0f}%)</span>
                    </div>
                    <div class="ch-bar-track">
                        <div class="ch-bar-fill" style="width:{w}%; background:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Results column ──────────────────────────────────────────────────────────
    with col_results:
        # KPI row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="kpi-card" style="border-left:4px solid #f43f5e;">
                <div class="kpi-label">Total Ad Spend</div>
                <div class="kpi-value">${total_spend:,.0f}K</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="kpi-card" style="border-left:4px solid #a855f7;">
                <div class="kpi-label">Predicted Sales</div>
                <div class="kpi-value">{predicted_sales:.1f}K</div>
                <div class="kpi-sub">units sold (thousands)</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            roi_color = '#10b981' if roi_pct >= 0 else '#ef4444'
            roi_sign  = '+' if roi_pct >= 0 else ''
            st.markdown(f"""
            <div class="kpi-card" style="border-left:4px solid {roi_color};">
                <div class="kpi-label">Est. ROI</div>
                <div class="kpi-value" style="color:{roi_color};">{roi_sign}{roi_pct:.0f}%</div>
                <div class="kpi-sub">revenue vs. spend</div>
            </div>""", unsafe_allow_html=True)

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_sales,
            domain={'x':[0,1],'y':[0,1]},
            title={'text':"Predicted Sales (thousands of units)", 'font':{'size':15,'color':'#475569'}},
            number={'font':{'size':52,'color':'#1e293b'}, 'suffix':'K'},
            gauge={
                'axis':{'range':[0,35],'tickwidth':1},
                'bar':{'color':'#a855f7'},
                'bgcolor':'white',
                'borderwidth':2, 'bordercolor':'#e2e8f0',
                'steps':[
                    {'range':[0,10],'color':'#fef2f2'},
                    {'range':[10,20],'color':'#fce7f3'},
                    {'range':[20,35],'color':'#f3e8ff'},
                ],
                'threshold':{'line':{'color':'#f43f5e','width':4},'thickness':0.75,'value':25}
            }
        ))
        fig_gauge.update_layout(height=320, margin=dict(l=20,r=20,t=50,b=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar':False})

        # ── Key Insights ────────────────────────────────────────────────────────
        tv_share = imp_dict['TV'] * 100
        news_share = imp_dict['Newspaper'] * 100
        tv_val = st.session_state.tv
        news_val = st.session_state.news

        st.markdown("<p style='font-size:.95rem; font-weight:700; color:#0f172a; margin-bottom:.6rem;'>💡 Key Insights</p>", unsafe_allow_html=True)

        if news_val > tv_val:
            st.markdown(f"""
            <div class="insight-card" style="border-left-color:#ef4444;">
                ⚠️ <strong>Budget mismatch detected</strong> — You're spending more on Newspaper (${news_val:.0f}K)
                than TV (${tv_val:.0f}K), but Newspaper accounts for only <strong>{news_share:.1f}%</strong> of
                the model's predictive power vs <strong>{tv_share:.1f}%</strong> for TV.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insight-card" style="border-left-color:#10b981;">
                ✅ <strong>TV dominates</strong> with <strong>{tv_share:.1f}%</strong> of the model's predictive
                power. Your TV budget of ${tv_val:.0f}K is driving most of the projected sales.
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-card" style="border-left-color:#f59e0b; margin-top:.5rem;">
            📰 <strong>Newspaper ROI warning</strong> — This channel contributes only
            <strong>{news_share:.1f}%</strong> to predictions. Reallocating newspaper budget to TV
            would likely increase projected sales significantly.
        </div>""", unsafe_allow_html=True)

        # Revenue note
        st.markdown("""
        <p style="font-size:0.78rem; color:#94a3b8; margin-top:1rem;">
            <em>ℹ️ Sales units are from the ISLR Advertising dataset (Hundreds of units sold).
            ROI estimate assumes revenue = Sales × $1K per hundred units vs. ad spend in $K.</em>
        </p>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── KPI cards ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("Total Records",    str(len(dataset)),              "Training samples"),
        ("Avg TV Spend",     f"${dataset['TV'].mean():.0f}K", "per campaign"),
        ("Avg Radio Spend",  f"${dataset['Radio'].mean():.0f}K", "per campaign"),
        ("Avg Sales",        f"{dataset['Sales'].mean():.1f}K", "units (thousands)"),
        ("Max Sales",        f"{dataset['Sales'].max():.1f}K",  "units in dataset"),
    ]
    for col, (label, val, sub) in zip([k1,k2,k3,k4,k5], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

    # ── Feature Importance + Budget Donut ──────────────────────────────────────
    c1, c2 = st.columns([1.3, 1], gap="large")

    with c1:
        fig_imp = go.Figure(go.Bar(
            x=importances * 100,
            y=feat_names,
            orientation='h',
            marker_color=feat_colors,
            text=[f"{v*100:.1f}%" for v in importances],
            textposition='outside',
        ))
        fig_imp.update_layout(
            title=dict(text="Feature Importance — What Actually Drives Sales?", font=dict(size=15, color='#0f172a'), x=0),
            xaxis=dict(title='Importance (%)', range=[0, max(importances)*130], gridcolor='#f1f5f9'),
            yaxis=dict(autorange='reversed'),
            paper_bgcolor='white', plot_bgcolor='#fafafa',
            margin=dict(t=50, b=20, l=20, r=60),
            height=280,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with c2:
        current_spend = [st.session_state.tv, st.session_state.radio, st.session_state.news]
        fig_donut = go.Figure(go.Pie(
            labels=feat_names,
            values=current_spend,
            hole=0.55,
            marker_colors=feat_colors,
            textinfo='label+percent',
            textfont_size=13,
        ))
        fig_donut.update_layout(
            title=dict(text="Your Current Budget Split", font=dict(size=15, color='#0f172a'), x=0),
            showlegend=False,
            margin=dict(t=50, b=20, l=20, r=20),
            paper_bgcolor='white',
            height=280,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Scatter plots: each channel vs Sales ───────────────────────────────────
    st.markdown("#### Ad Spend vs. Sales — Historical Relationship")
    s1, s2, s3 = st.columns(3)

    scatter_cfg = [
        (s1, 'TV',        'tv',    '#f43f5e', 'TV Spend ($K)'),
        (s2, 'Radio',     'radio', '#a855f7', 'Radio Spend ($K)'),
        (s3, 'Newspaper', 'news',  '#f59e0b', 'Newspaper Spend ($K)'),
    ]
    for col_w, feat, sess_key, color, xlabel in scatter_cfg:
        with col_w:
            # Trendline via numpy
            z = np.polyfit(dataset[feat], dataset['Sales'], 1)
            trend_x = np.linspace(dataset[feat].min(), dataset[feat].max(), 100)
            trend_y = np.polyval(z, trend_x)

            fig_sc = go.Figure()
            fig_sc.add_scatter(x=dataset[feat], y=dataset['Sales'],
                               mode='markers',
                               marker=dict(color=color, size=6, opacity=0.6),
                               name='Data points')
            fig_sc.add_scatter(x=trend_x, y=trend_y,
                               mode='lines',
                               line=dict(color=color, width=2, dash='dash'),
                               name='Trend')
            # Mark user's input
            fig_sc.add_scatter(
                x=[st.session_state[sess_key]],
                y=[predicted_sales],
                mode='markers',
                marker=dict(color='#0f172a', size=14, symbol='star'),
                name='Your input',
            )
            fig_sc.update_layout(
                title=dict(text=feat, font=dict(size=14, color='#0f172a')),
                xaxis=dict(title=xlabel, gridcolor='#f1f5f9'),
                yaxis=dict(title='Sales (K units)', gridcolor='#f1f5f9'),
                paper_bgcolor='white', plot_bgcolor='#fafafa',
                margin=dict(t=40, b=40, l=40, r=20),
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    # ── Correlation heatmap ────────────────────────────────────────────────────
    st.markdown("#### Correlation Heatmap")
    corr = dataset[['TV', 'Radio', 'Newspaper', 'Sales']].corr()
    fig_heat = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0,'#fdf4ff'],[0.5,'#a855f7'],[1,'#581c87']],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(size=14, color='white'),
        showscale=True,
    ))
    fig_heat.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor='white',
        height=320,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Dataset preview ────────────────────────────────────────────────────────
    with st.expander("🗃️  View Raw Dataset"):
        st.dataframe(dataset, use_container_width=True, height=320)

st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:.82rem; padding-top:2rem;">
    Developed by <b>Bhagesh Biradar</b>
</div>
""", unsafe_allow_html=True)
