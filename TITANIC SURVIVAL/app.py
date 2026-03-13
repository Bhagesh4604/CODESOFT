import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up the Streamlit app
st.set_page_config(page_title="Titanic Predictor", layout="centered")

st.markdown("""
    <style>
    /* Adjust Top Empty Space */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 0rem !important;
    }
    
    /* Main Title Styling */
    .main-title {
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        color: #0f172a !important;
        padding-bottom: 0.5rem !important;
        margin-bottom: 0rem !important;
        line-height: 1.2 !important;
    }
    
    h3 {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        padding-bottom: 0.5rem;
    }
    h4 {
        font-size: 1.4rem !important;
        font-weight: 500 !important;
        color: #334155 !important;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    p, span, label, .stMarkdown p, div[data-testid="stMarkdownContainer"] p, .stText, div[data-testid="stText"] {
        font-size: 1.15rem !important;
    }
    .stCaption, div[data-testid="stCaptionContainer"] p {
        font-size: 1.05rem !important;
        color: #64748b !important;
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

# Header
st.markdown("<div class='main-title'>🚢 Titanic Survival Predictor</div>", unsafe_allow_html=True)
st.markdown("Predict the survival probability of passengers based on historical data.")

tab1, tab2 = st.tabs(["Prediction", "Analytics"])

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
            if prediction == 1:
                st.success(f"**Survived**\n\nProbability: {survive_pct:.1f}%")
            else:
                st.error(f"**Did not survive**\n\nProbability: {survive_pct:.1f}%")
                
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
                
                fig_bar = px.bar(imp_df, x='Weight', y='Factor', orientation='h')
                fig_bar.update_traces(marker_color='#10b981', opacity=0.9, marker_line_width=0)
                fig_bar.update_layout(
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
        
        # 1. Multivariate Categorical Analysis
        st.markdown("#### Survival by Ticket Class & Gender")
        
        surv_multi = df.groupby(['pclass', 'sex'])['survived'].mean().reset_index()
        surv_multi['survived_pct'] = surv_multi['survived'] * 100
        surv_multi['pclass_label'] = surv_multi['pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
        surv_multi['sex_label'] = surv_multi['sex'].str.capitalize()
        
        fig_multi = px.bar(
            surv_multi, x='pclass_label', y='survived_pct', color='sex_label', barmode='group',
            color_discrete_map={'Female': '#10b981', 'Male': '#3b82f6'},
            labels={'pclass_label': 'Ticket Class', 'survived_pct': 'Survival Rate %', 'sex_label': 'Gender'}
        )
        fig_multi.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            height=400,
            xaxis=dict(showgrid=True, title=""),
            yaxis=dict(showgrid=True, title="Survival Probability (%)"),
            legend=dict(title="")
        )
        st.plotly_chart(fig_multi, use_container_width=True)
        
        st.divider()
        
        # 2. Feature Correlation Matrix
        st.markdown("#### Feature Correlation Matrix")
        
        corr_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
        df_corr = df[corr_cols].dropna().corr()
        
        fig_corr = px.imshow(
            df_corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1
        )
        fig_corr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            height=450,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.divider()
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            # 3. Density Distributions (Violin Plot)
            st.markdown("#### Age Distribution by Outcome")
            
            df_age = df.copy().dropna(subset=['age'])
            df_age['Outcome'] = df_age['survived'].map({0: 'Perished', 1: 'Survived'})
            
            fig_violin = px.violin(
                df_age, x='Outcome', y='age', color='Outcome', box=True, points="all",
                color_discrete_map={'Perished': '#f43f5e', 'Survived': '#10b981'},
                labels={'age': 'Age', 'Outcome': 'Status'}
            )
            fig_violin.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                height=400,
                xaxis=dict(title=""),
                showlegend=False
            )
            st.plotly_chart(fig_violin, use_container_width=True)
            
        with col_d:
            # 4. Family Size Impact (Line Plot)
            st.markdown("#### Survival by Family Size")
            
            df_fam = df.copy()
            df_fam['FamilySize'] = df_fam['sibsp'] + df_fam['parch'] + 1
            
            fam_agg = df_fam.groupby('FamilySize')['survived'].agg(['mean', 'sem']).reset_index()
            fam_agg['mean_pct'] = fam_agg['mean'] * 100
            fam_agg['sem_pct'] = fam_agg['sem'] * 100
            
            fig_fam = px.line(
                fam_agg, x='FamilySize', y='mean_pct', error_y='sem_pct', markers=True,
                labels={'FamilySize': 'Total Family Size', 'mean_pct': 'Survival Rate %'}
            )
            fig_fam.update_traces(line_color='#8b5cf6', marker=dict(size=8, color='#8b5cf6'))
            fig_fam.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                height=400,
                xaxis=dict(showgrid=True, dtick=1),
                yaxis=dict(showgrid=True)
            )
            st.plotly_chart(fig_fam, use_container_width=True)
            
    else:
        st.warning("Historical dataset could not be loaded for graphics.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 1.15rem; padding-top: 2rem; border-top: 1px solid #e2e8f0;'>
    Developed by <strong>Bhagesh Biradar</strong> • CodeSoft Data Science Internship
</div>
""", unsafe_allow_html=True)
