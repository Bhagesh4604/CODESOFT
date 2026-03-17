import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Titanic EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PIXEL-PERFECT NAUTICAL CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global Text and Background Adjustments */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Make all default text bigger and darker for visibility */
    p, span, div, label {
        font-size: 1.05rem !important;
        color: #1e293b !important;
    }
    
    h1, h2, h3 { 
        color: #0f172a !important; 
        font-weight: 800 !important; 
    }
    
    /* Hide default header */
    header { visibility: hidden; }

    /* Modify sidebar appearance */
    [data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 2px solid #e2e8f0;
    }
    
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 1.5rem 0rem;
        font-size: 1.5rem !important;
        font-weight: 800;
        color: #001f3f !important;
        border-bottom: 2px solid #cbd5e1;
        margin-bottom: 1.5rem;
    }
    
    .nav-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 1rem 0rem;
        color: #475569 !important;
        font-size: 1.2rem !important;
        font-weight: 600;
    }
    .nav-item.active {
        color: #001f3f !important;
        font-weight: 800;
        border-left: 4px solid #001f3f;
        padding-left: 10px;
    }
    
    .nav-divider {
        height: 2px;
        background-color: #cbd5e1;
        margin: 2rem 0;
    }

    /* --- PAGE TITLE --- */
    .page-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        margin-bottom: 0.5rem;
    }
    .page-subtitle {
        font-size: 1.2rem !important;
        color: #475569 !important;
        margin-bottom: 0.5rem;
    }

    /* --- METRIC CARDS --- */
    .metric-card {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-title {
        font-size: 1.1rem !important;
        color: #64748b !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        margin-top: auto;
    }

    /* --- SPECIFIC FIXES --- */
    /* Remove white boxes by targeting Streamlit containers instead of injecting raw HTML */
    div[data-testid="stVerticalBlock"] > div.element-container {
        margin-bottom: 0px !important;
    }
    
    /* Bigger Buttons */
    .stButton > button {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border: 2px solid #cbd5e1 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        width: 100% !important;
        padding: 0.6rem !important;
        margin-top: 1.8rem;
        border-radius: 8px !important;
    }
    
    /* Make selectbox labels bigger and dark */
    div.row-widget.stSelectbox > div > label > div > p {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        import kagglehub
        import glob
        path = kagglehub.dataset_download("yasserh/titanic-dataset")
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        df = pd.read_csv(csv_files[0])
        df.columns = [c.lower() for c in df.columns]
        df['sex'] = df['sex'].str.title()
        df['embarked'] = df['embarked'].fillna('C').map({'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'})
        return df
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return pd.DataFrame()

df = load_data()

# --- SIDEBAR COMPONENT ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">🚢 Titanic Analysis</div>
    <div class="nav-item">📊 Titanic EDA</div>
    <div class="nav-item">🔍 Global Filters</div>
    <div class="nav-item">👥 Passengers</div>
    <div class="nav-item active">📈 Key Metrics</div>
    <div class="nav-divider"></div>
    """, unsafe_allow_html=True)

# --- MAIN LAYOUT ---
st.markdown('<div class="page-title">Survival Analysis Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Interactive dashboard visualizing Titanic survival patterns.</div>', unsafe_allow_html=True)
st.divider()

# Grid Layout 1: Filters & KPIs
col1, col2 = st.columns([1.5, 1], gap="large")

# Filter State
if 'f_sex' not in st.session_state: st.session_state.f_sex = "All"
if 'f_pclass' not in st.session_state: st.session_state.f_pclass = "All"
if 'f_embarked' not in st.session_state: st.session_state.f_embarked = "All"

with col1:
    st.markdown("### Global Filters")
    fcol1, fcol2, fcol3, fcol4 = st.columns([1, 1, 1, 0.8])
    with fcol1:
        sex_filter = st.selectbox("Sex", ["All", "Male", "Female"])
    with fcol2:
        pclass_filter = st.selectbox("Class", ["All", "1st", "2nd", "3rd"])
    with fcol3:
        embarked_filter = st.selectbox("Port", ["All", "Cherbourg", "Queenstown", "Southampton"])
    with fcol4:
        if st.button("Reset Filters"):
            st.session_state.f_sex = "All"
            st.session_state.f_pclass = "All"
            st.session_state.f_embarked = "All"
            st.rerun()

# Apply Filters
df_filtered = df.copy() if df is not None else pd.DataFrame()
if not df_filtered.empty:
    if sex_filter != "All":
        df_filtered = df_filtered[df_filtered['sex'] == sex_filter]
    if pclass_filter != "All":
        df_filtered = df_filtered[df_filtered['pclass'] == int(pclass_filter[0])]
    if embarked_filter != "All":
        df_filtered = df_filtered[df_filtered['embarked'] == embarked_filter]

with col2:
    mcol1, mcol2, mcol3 = st.columns(3)
    
    total_passengers = len(df_filtered) if not df_filtered.empty else 0
    survival_rate = (df_filtered['survived'].mean() * 100) if not df_filtered.empty and len(df_filtered) > 0 else 0
    avg_fare = df_filtered['fare'].mean() if not df_filtered.empty and len(df_filtered) > 0 else 0
    
    with mcol1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total<br>Passengers</div>
            <div class="metric-value">{total_passengers}</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Survival<br>Rate</div>
            <div class="metric-value" style="color:#118a3d !important;">{survival_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average<br>Fare</div>
            <div class="metric-value">${avg_fare:.1f}</div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("<br><br>", unsafe_allow_html=True)

# Grid Layout 2: Charts Area
col_left, col_mid, col_right = st.columns([1.2, 1, 1], gap="large")

with col_left:
    st.markdown("### Survival Count by Sex & Passenger Class")
    if not df_filtered.empty:
        agg_df = df_filtered.groupby(['pclass', 'sex', 'survived']).size().reset_index(name='count')
        agg_df['survived_str'] = agg_df['survived'].map({1: 'Survived', 0: 'Perished'})
        agg_df['x_label'] = agg_df['pclass'].astype(str) + " Class " + agg_df['sex']
        agg_df = agg_df.sort_values(by=['pclass', 'sex', 'survived'], ascending=[True, False, False])

        fig1 = px.bar(agg_df, x="x_label", y="count", color='survived_str', 
                     color_discrete_map={'Survived': '#118a3d', 'Perished': '#64748b'},
                     barmode='stack')
        fig1.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            height=320,
            xaxis_title="",
            yaxis_title="Passenger Count",
            legend_title="",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Age Distribution by Survival")
    if not df_filtered.empty and 'age' in df_filtered.columns:
        fig2 = px.histogram(df_filtered.dropna(subset=['age']), x="age", color="survived",
                            color_discrete_map={1: '#118a3d', 0: '#64748b'},
                            barmode="overlay")
        fig2.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            height=280,
            xaxis_title="Age",
            yaxis_title="Density",
            bargap=0.1,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig2, use_container_width=True)

with col_mid:
    st.markdown("### Age vs Fare (Survival Colored)")
    if not df_filtered.empty:
        scatter_df = df_filtered.dropna(subset=['age', 'fare']).copy()
        scatter_df['norm_age'] = scatter_df['age'] - scatter_df['age'].mean()
        scatter_df['norm_fare'] = scatter_df['fare'] - scatter_df['fare'].mean()
        
        fig3 = px.scatter(scatter_df, x="norm_age", y="norm_fare", color="survived",
                          color_discrete_map={1: '#118a3d', 0: '#0f172a'}, opacity=0.7)
        fig3.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=320,
            xaxis_title="Age (Centered)",
            yaxis_title="Fare (Centered)",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        # Add quadrant lines
        fig3.add_shape(type="line", x0=0, y0=-100, x1=0, y1=300, line=dict(color="#cbd5e1", width=2))
        fig3.add_shape(type="line", x0=-40, y0=0, x1=50, y1=0, line=dict(color="#cbd5e1", width=2))
        
        st.plotly_chart(fig3, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Legend & Quick Stats")
    if not df_filtered.empty:
        s_count = len(df_filtered[df_filtered['survived'] == 1])
        p_count = len(df_filtered[df_filtered['survived'] == 0])
        s_age = df_filtered[df_filtered['survived'] == 1]['age'].median()
        p_age = df_filtered[df_filtered['survived'] == 0]['age'].median()
        s_fare = df_filtered[df_filtered['survived'] == 1]['fare'].median()
        p_fare = df_filtered[df_filtered['survived'] == 0]['fare'].median()
        
        st.markdown(f"""
        <div class="metric-card" style="padding:1.5rem;">
            <div style="margin-bottom: 1rem;">
                <span style="color:#118a3d; font-weight:800; font-size:1.2rem;">● Survived:</span> 
                <span style="font-weight:600; font-size:1.1rem; color:#0f172a;">{s_count} passengers</span>
                <div style="color:#475569; font-size:1rem; margin-top:0.3rem;">Median Age: <b>{s_age:.0f}</b> | Median Fare: <b>${s_fare:.1f}</b></div>
            </div>
            <div>
                <span style="color:#64748b; font-weight:800; font-size:1.2rem;">● Perished:</span> 
                <span style="font-weight:600; font-size:1.1rem; color:#0f172a;">{p_count} passengers</span>
                <div style="color:#475569; font-size:1rem; margin-top:0.3rem;">Median Age: <b>{p_age:.0f}</b> | Median Fare: <b>${p_fare:.1f}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown("### Passenger Records")
    if not df_filtered.empty:
        display_df = df_filtered[['passengerid', 'name', 'age', 'sex', 'pclass', 'fare']]
        display_df.columns = ['ID', 'Name', 'Age', 'Sex', 'Pclass', 'Fare']
        # Native Streamlit dataframe with pagination handles the styling purely
        st.dataframe(display_df, hide_index=True, use_container_width=True, height=660)
