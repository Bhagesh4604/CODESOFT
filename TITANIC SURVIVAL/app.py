import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up the Streamlit app
st.set_page_config(page_title="Titanic Predictor", layout="wide")

st.markdown("""
    <style>
    /* Dark Ocean Background */
    .stApp {
        background: #0b1120;
        background-image: radial-gradient(circle at top right, #1a2744, #0b1120);
        color: #e2e8f0;
    }
    
    /* Solid Navy Header */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    header[data-testid="stHeader"] * { color: #e2e8f0 !important; }

    /* Main container */
    .main .block-container {
        padding: 0rem 2rem 2rem 2rem;
        background: transparent;
        margin-top: 1rem;
    }
    
    /* Tabs as Floating Pills */
    div.stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: transparent;
    }
    div.stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #1e293b;
        border-radius: 25px !important;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        color: #94a3b8 !important;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div.stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0ea5e9;
        color: #ffffff !important;
        border: none;
    }
    
    /* Typography Overrides */
    h1, h2, h3 { color: #ffffff !important; font-family: 'Inter', sans-serif; }
    h4 { color: #ffffff !important; font-size: 1.3rem !important; margin-top: 0.5rem; margin-bottom: 0.2rem; }
    p, span, div, label { color: #ffffff !important; }
    
    /* Inputs */
    .stNumberInput div[data-baseweb="input"] > div,
    .stSelectbox div[data-baseweb="select"] > div,
    .stSlider div[data-baseweb="slider"] {
        background: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 6px;
        color: white !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }
    .stSelectbox div[data-baseweb="select"] span {
        color: #f8fafc !important;
        font-weight: 500;
    }
    input[type="number"], input[type="text"] {
        color: #f8fafc !important;
        font-weight: 500;
    }
    /* Number input buttons */
    .stNumberInput button {
        background: #334155 !important;
        color: #f8fafc !important;
    }
    
    /* Button */
    div.stButton > button:first-child {
        background: #0ea5e9 !important; /* Ocean Teal */
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        transition: transform 0.1s ease;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
        width: 100%;
        margin-top: 1rem;
    }
    div.stButton > button:first-child:hover {
        background: #0284c7 !important;
        transform: translateY(-2px);
    }

    /* Hero Section CSS */
    .hero-container {
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        color: #f8fafc;
        line-height: 1.1;
        letter-spacing: -1px;
    }
    .hero-title span { color: #0ea5e9; }
    .hero-subtitle {
        font-size: 1.5rem;
        font-weight: 500;
        color: #0ea5e9; /* Teal */
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .hero-text {
        font-size: 1.1rem;
        color: #94a3b8;
        line-height: 1.6;
        max-width: 90%;
    }
    
    /* Oceanic Right Side Graphic */
    .ocean-graphic-container {
        position: relative;
        width: 100%;
        padding-bottom: 80%; /* rectangle */
        border-radius: 20px;
        /* Dramatic dark ocean texture */
        background-image: url("https://images.unsplash.com/photo-1518837695005-2083093ee35b?q=80&w=1000&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
        overflow: hidden;
        border: 2px solid #1e293b;
    }
    .ocean-graphic-container::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        /* Diagonal teal gradient overlay */
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(14, 165, 233, 0.3) 100%);
        z-index: 1;
    }
    .ocean-text {
        position: absolute;
        bottom: 20px;
        right: 20px;
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        text-align: right;
        z-index: 2;
        font-style: italic;
        text-shadow: 0 2px 4px rgba(0,0,0,0.8);
    }
    
    /* Ticket Style Prediction Box */
    .ticket-box {
        background: #1e293b;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px dashed #334155;
        border-left: 8px solid #0ea5e9;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    }
    .ticket-value {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .ticket-survived { color: #10b981 !important; }
    .ticket-perished { color: #f43f5e !important; }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1a2744;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #1e293b;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        border-top: 3px solid #0ea5e9; /* Teal accent */
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.8rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('titanic_survival_model.pkl')

@st.cache_data
def load_dataset():
    try:
        import kagglehub
        import glob
        import os
        path = kagglehub.dataset_download("yasserh/titanic-dataset")
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        df = pd.read_csv(csv_files[0])
        df.columns = [c.lower() for c in df.columns]
        return df
    except:
        return None

model = load_model()
df = load_dataset()

# --- HERO SECTION ---
st.markdown("<div class='hero-container'>", unsafe_allow_html=True)
col_h_left, col_h_right = st.columns([1.2, 1])

with col_h_left:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Welcome to<br><span>Titanic Survivor</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Navigate data, Predict destinies</div>", unsafe_allow_html=True)
    st.markdown("""
<div class='hero-text'>
    Discover the definitive tool for evaluating historical passenger survival probabilities on the RMS Titanic.
    <br><br>
    Uncover insights hidden in the depths of historical data. Here, your analytical curiosity meets powerful modeling to reveal the true factors behind nautical survival.
</div>
    """, unsafe_allow_html=True)

with col_h_right:
    st.markdown("""
<div class="ocean-graphic-container">
    <div class="ocean-text">
        "Some survived by chance,<br>others by data."
    </div>
</div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2 = st.tabs(["🚢 Survival Engine", "🌊 Data Deep-Dive"])

with tab1:
    st.subheader("Passenger details")
    
    col_demo, col_journey = st.columns(2)
    
    with col_demo:
        st.markdown("#### Passenger Demographics")
        age = st.slider(label="Age (Years)", min_value=0.0, max_value=100.0, value=25.0, step=0.5)
        sex = st.selectbox(label="Gender", options=['male', 'female'], index=0)
        title = st.selectbox(label="Title", options=["Mr", "Miss", "Mrs", "Master", "Rare"], index=0)
        
    with col_journey:
        st.markdown("#### Journey Details")
        pclass = st.selectbox(label="Ticket Class", options=[1, 2, 3], index=2, format_func=lambda x: f"{x} Class")
        fare = st.number_input(label="Fare Paid (£)", min_value=0.0, max_value=600.0, value=15.0, step=1.0)
        embarked = st.selectbox(label="Embarked Port", options=["Southampton", "Cherbourg", "Queenstown"], index=0)
        sibsp = st.number_input(label="Siblings / Spouses Aboard", min_value=0, max_value=10, value=0)
        parch = st.number_input(label="Parents / Children Aboard", min_value=0, max_value=10, value=0)
        has_cabin = st.selectbox(label="Assigned a Cabin?", options=["No", "Yes"], index=0)
        
    family_size = sibsp + parch + 1

    predict_btn = st.button(label="Calculate Survival Probability", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'pclass': pclass,
            'fare': fare,
            'cabin': 1 if has_cabin == "Yes" else 0,
            'FamilySize': family_size,
            'Title': title
        }])
        
        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        survive_pct = probs[1] * 100
        
        st.divider()
        st.subheader("Model Assessment")
        
        r1, r2 = st.columns([1, 1])
        
        with r1:
            outcome_text = "Survived" if prediction == 1 else "Perished"
            outcome_class = "ticket-survived" if prediction == 1 else "ticket-perished"
            
            st.markdown(f"""
            <div class="ticket-box">
                <h3 style='margin:0; color:#e2e8f0; font-size:1.1rem;'>Survival Assessment</h3>
                <div class="ticket-value {outcome_class}">{survive_pct:.1f}<span style='font-size: 1.5rem; color:#94a3b8;'>%</span></div>
                <p style='margin:0; font-size:1rem; color:#cbd5e1; font-weight:600;'>{outcome_text}</p>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown("#### Feature Importance")
            st.caption("Key factors influencing this prediction")
            
            classifier = model.named_steps['classifier']
            importances = classifier.feature_importances_
            preprocessor = model.named_steps['preprocessor']
            
            try:
                feature_names = preprocessor.get_feature_names_out()
                clean_names = [name.split('__')[-1].title() for name in feature_names]
                
                imp_df = pd.DataFrame({'Factor': clean_names, 'Weight': importances}).sort_values('Weight', ascending=True)
                
                fig_bar = px.bar(imp_df, x='Weight', y='Factor', orientation='h', color_discrete_sequence=['#0ea5e9'])
                fig_bar.update_traces(opacity=0.9, marker_line_width=0)
                fig_bar.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=280,
                    xaxis=dict(showgrid=True, title=""),
                    yaxis=dict(title="", showgrid=False)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            except:
                st.warning("Feature importance unavailable.")

with tab2:
    if df is not None:
        st.subheader("Dataset Analytics")
        # --- ROW 1: KPI METRICS ---
        st.markdown("#### High-Level Survivor Metrics")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.metric(label="Total Passengers", value=f"{len(df):,}")
        with kpi2:
            overall_survival = df['survived'].mean() * 100
            st.metric(label="Overall Survival Rate", value=f"{overall_survival:.1f}%")
        with kpi3:
            avg_fare = df['fare'].mean()
            st.metric(label="Average Fare Paid", value=f"£{avg_fare:.2f}")
        with kpi4:
            avg_age = df['age'].mean()
            st.metric(label="Average Age", value=f"{avg_age:.1f} yrs")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- ROW 2: MAIN COMPARATIVES ---
        col_main_left, col_main_right = st.columns([2, 1.2]) # 2/3 and 1/3 split
        
        with col_main_left:
            # Checking out how Class and Gender impact survival rates together
            st.markdown("#### Survival by Ticket Class & Gender")
            surv_multi = df.groupby(['pclass', 'sex'])['survived'].mean().reset_index()
            surv_multi['survived_pct'] = surv_multi['survived'] * 100
            surv_multi['pclass_label'] = surv_multi['pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
            surv_multi['sex_label'] = surv_multi['sex'].str.capitalize()
            
            fig_multi = px.bar(
                surv_multi, x='pclass_label', y='survived_pct', color='sex_label', barmode='group',
                color_discrete_map={'Female': '#0ea5e9', 'Male': '#818cf8'},
                labels={'pclass_label': 'Ticket Class', 'survived_pct': 'Survival Rate %', 'sex_label': 'Gender'}
            )
            fig_multi.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0),
                height=350,
                xaxis=dict(showgrid=False, title=""),
                yaxis=dict(showgrid=False, title="Survival Probability (%)"),
                legend=dict(title="")
            )
            st.plotly_chart(fig_multi, use_container_width=True)
            
        with col_main_right:
            # Looking for correlations to see which numeric features drive survival the most
            st.markdown("#### Feature Correlation")
            corr_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
            df_corr = df[corr_cols].dropna().corr()
            
            fig_corr = px.imshow(
                df_corr, text_auto=".1f", aspect="auto",
                color_continuous_scale="Teal_r", zmin=-1, zmax=1
            )
            fig_corr.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                height=350,
                coloraxis_showscale=False,
                xaxis=dict(tickangle=-45)
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- ROW 3: DISTRIBUTION DRILL-DOWNS ---
        col_dist_left, col_dist_right = st.columns(2)
        
        with col_dist_left:
            # Building a violin plot because I want to see the age distribution shapes between those who lived and died
            st.markdown("#### Age Distribution by Outcome")
            df_age = df.copy().dropna(subset=['age'])
            df_age['Outcome'] = df_age['survived'].map({0: 'Perished', 1: 'Survived'})
            
            fig_violin = px.violin(
                df_age, x='Outcome', y='age', color='Outcome', box=True, points="all",
                color_discrete_map={'Perished': '#334155', 'Survived': '#0ea5e9'},
                labels={'age': 'Age', 'Outcome': 'Status'}
            )
            fig_violin.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                height=300,
                xaxis=dict(title="", showgrid=False),
                yaxis=dict(showgrid=False),
                showlegend=False
            )
            st.plotly_chart(fig_violin, use_container_width=True)
            
        with col_dist_right:
            # Does having a bigger family help or hurt? Let's plot the average survival rate by family size
            st.markdown("#### Survival by Family Size")
            df_fam = df.copy()
            df_fam['FamilySize'] = df_fam['sibsp'] + df_fam['parch'] + 1
            
            fam_agg = df_fam.groupby('FamilySize')['survived'].agg(['mean', 'sem']).reset_index()
            fam_agg['mean_pct'] = fam_agg['mean'] * 100
            
            fig_fam = px.line(
                fam_agg, x='FamilySize', y='mean_pct', markers=True,
                labels={'FamilySize': 'Total Family Size', 'mean_pct': 'Survival Rate %'}
            )
            fig_fam.update_traces(line_color='#0ea5e9', marker=dict(size=8, color='#0ea5e9'))
            fig_fam.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                height=300,
                xaxis=dict(showgrid=False, dtick=1),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_fam, use_container_width=True)
            
    else:
        st.warning("Historical dataset could not be loaded for graphics.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #718096; font-size: 0.85rem; padding: 10px;">
    Developed by <strong>Bhagesh Biradar</strong>
</div>
""", unsafe_allow_html=True)
