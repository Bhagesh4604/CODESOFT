import streamlit as st
import pandas as pd
import joblib
import os
import category_encoders as ce

st.set_page_config(
    page_title="Movie Rating Predictor", 
    page_icon="🎥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# My custom CSS to make it look like the beautiful magenta reference image
st.markdown("""
<style>
    /* Stable, bright background */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        background-color: #fdfbfb;
    }
    
    /* Override Streamlit's top header to be solid magenta */
    header[data-testid="stHeader"] {
        background-color: #9D174D !important;
        background: #9D174D !important;
    }
    
    /* Make the burger menu icon and text white so it shows up on the magenta background */
    header[data-testid="stHeader"] * {
        color: #FFFFFF !important;
    }

    /* Main container card styling */
    .main .block-container {
        padding: 0rem 2rem 2rem 2rem; /* Reduce top padding since we have the solid header now */
        background: transparent;
        margin-top: 2rem;
    }
    
    /* Create white cards for Tabs */
    div.stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    div.stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 8px;
        padding-left: 1rem;
        padding-right: 1rem;
        color: #718096 !important;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    div.stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FDF2F8;
        color: #9D174D !important; /* Magenta tab text */
        border-bottom: 3px solid #9D174D;
    }
    
    /* Modify sidebar */
    [data-testid="stSidebar"] {
        background: #9D174D !important; /* Rich Magenta/Burgundy */
        border-right: none;
    }
    
    [data-testid="stSidebar"] * {
        color: #FDF2F8 !important; /* Soft pink-white text */
    }
    
    /* Input backgrounds in the sidebar */
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput div[data-baseweb="input"] > div {
        background: #831843 !important; /* Darker Magenta for inputs */
        border-radius: 6px;
        border: 1px solid #BE185D;
        color: white !important;
    }
    /* Ensure the selected text inside the sidebar inputs is white */
    .stSelectbox div[data-baseweb="select"] span {
        color: #FFFFFF !important;
    }
    
    /* Fix the expanded dropdown menu list items so text is visible */
    div[data-baseweb="popover"] {
        background-color: #9D174D !important;
    }
    div[data-baseweb="popover"] ul li,
    div[data-baseweb="popover"] ul li * {
        color: #FDF2F8 !important;
        background-color: #9D174D !important;
    }
    div[data-baseweb="popover"] ul li:hover,
    div[data-baseweb="popover"] ul li:hover * {
        background-color: #BE185D !important; /* Lighter Magenta hover */
        color: #FFFFFF !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2D3748 !important;
        font-family: 'Inter', sans-serif;
    }
    p, span, div {
        color: #4A5568;
    }
    
    /* Primary button styling */
    div.stButton > button:first-child {
        background: #9D174D !important; /* Magenta button */
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        transition: transform 0.1s ease;
        box-shadow: 0 4px 15px rgba(157, 23, 77, 0.3);
        width: 100%;
        margin-top: 1rem;
    }
    div.stButton > button:first-child:hover {
        background: #BE185D !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(157, 23, 77, 0.5);
    }
    
    /* Prediction Box - Bright Card */
    .prediction-box {
        background: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #FBCFE8;
        border-left: 6px solid #9D174D; /* Magenta accent */
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #9D174D !important; /* Magenta text */
        margin: 0.5rem 0;
    }
    .prediction-box h3 {
        color: #2D3748 !important;
        font-weight: 600;
        margin: 0;
        font-size: 1.25rem;
    }
    .prediction-box p {
        color: #718096 !important;
        margin: 0;
        font-size: 0.9rem;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #2D3748 !important;
        font-size: 2rem !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 1.1rem !important;
    }
    [data-testid="stMetricDelta"] svg {
        fill: #9D174D !important; /* Change up-arrow color to match theme */
    }
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #FBCFE8; /* Light pink border */
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-top: 4px solid #9D174D; /* Magenta top accent */
    }
</style>
""", unsafe_allow_html=True)

import plotly.express as px

# Caching this so it doesn't reload the 8MB model every time I click a button
@st.cache_resource
def load_artifacts():
    artifacts_exist = all(os.path.exists(f) for f in ["rf_model.pkl", "encoder.pkl", "scaler.pkl", "dropdown_options.pkl", "movie_dataset.pkl"])
    if not artifacts_exist:
        return None, None, None, None, None
        
    model = joblib.load("rf_model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    options = joblib.load("dropdown_options.pkl")
    dataset = joblib.load("movie_dataset.pkl")
    return model, encoder, scaler, options, dataset

# --- HERO SECTION ---
st.markdown("""
<style>
    .hero-container {
        padding: 1rem 0 3rem 0;
    }
    .hero-title {
        font-size: 4.5rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.1;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .hero-subtitle {
        font-size: 1.8rem;
        font-weight: 600;
        color: #9D174D;
        margin-bottom: 1.5rem;
    }
    .hero-text {
        font-size: 1.15rem;
        color: #4B5563;
        line-height: 1.6;
    }
    .aesthetic-circle {
        position: relative;
        width: 100%;
        padding-bottom: 100%; /* perfect square aspect ratio */
        border-radius: 50%;
        /* Use a high-quality dandelion image to match the reference */
        background-image: url("https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?q=80&w=1000&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        box-shadow: 0 20px 40px rgba(157, 23, 77, 0.2);
        overflow: hidden;
    }
    /* Add a subtle magenta tint overlay to the image so text remains readable */
    .aesthetic-circle::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(to bottom, transparent 40%, rgba(253, 242, 248, 0.9) 100%);
        z-index: 1;
    }
    /* Pseudo-elements for the overlapping background circles shown in the reference image */
    .aesthetic-container {
        position: relative;
        padding: 2rem;
    }
    .bg-circle-top {
        position: absolute;
        top: 0;
        left: 0;
        width: 40%;
        height: 40%;
        border-radius: 50%;
        background-color: #9D174D;
        z-index: 0;
    }
    .bg-circle-bottom {
        position: absolute;
        bottom: 0px;
        right: 15%;
        width: 30%;
        height: 30%;
        border-radius: 50%;
        background-color: #BE185D;
        z-index: 0;
    }
    .circle-content {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: 2;
    }
    .circle-text-box {
        position: absolute;
        bottom: 15%;
        left: 50%;
        transform: translateX(-50%);
        width: 75%;
        text-align: center;
        color: #111827;
        font-size: 1.25rem;
        font-weight: 600;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='hero-container'>", unsafe_allow_html=True)
col_h_left, col_h_right = st.columns([1.1, 1])

with col_h_left:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Welcome to<br>Movie Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Believe in data, Invest in insights</div>", unsafe_allow_html=True)
    st.markdown("""
<div class='hero-text'>
    Welcome to the ultimate tool for evaluating and predicting the success of Indian cinema.
    <br><br>
    Analyzing movies can be daunting, but at the Movie Predictor, 
    we provide a robust, data-driven environment where your cinematic curiosity 
    meets powerful analytics to deliver actionable insights.
</div>
    """, unsafe_allow_html=True)

with col_h_right:
    st.markdown("""
<div class="aesthetic-container">
    <div class="bg-circle-top"></div>
    <div class="bg-circle-bottom"></div>
    <div class="aesthetic-circle">
        <div class="circle-content">
            <div class="circle-text-box">
                Here, you can borrow my<br>analytical insights, until you<br>can find your own again.
            </div>
        </div>
    </div>
</div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

model, encoder, scaler, options, dataset = load_artifacts()

if model is None:
    st.error("Model artifacts not found. Please ensure the model training script `movie_rating_model.py` has been executed.")
    st.stop()

st.sidebar.header("Auto-Fill (Optional)")
st.sidebar.markdown("<p style='font-size: 0.85rem; color: #A0AEC0;'>Select a known movie to automatically fill the features below.</p>", unsafe_allow_html=True)

# Getting all unique movie names for the search dropdown
movie_list = ["-- Manual Entry --"] + sorted(dataset['Name'].dropna().unique().tolist())
selected_movie = st.sidebar.selectbox("Find Historical Movie", movie_list)

# Default values
def_year, def_duration, def_votes = 2020, 120, 1000
def_genre, def_director, def_actor1, def_actor2 = 'Unknown', 'Unknown', 'Unknown', 'Unknown'

# Auto-fill the sliders if a user picks a specific movie from the list
if selected_movie != "-- Manual Entry --":
    movie_info = dataset[dataset['Name'] == selected_movie].iloc[0]
    
    # Pulling values safely just in case there are NaNs
    def_year = int(movie_info['Year']) if pd.notna(movie_info['Year']) else 2020
    def_duration = int(movie_info['Duration']) if pd.notna(movie_info['Duration']) else 120
    def_votes = int(movie_info['Votes']) if pd.notna(movie_info['Votes']) else 1000
    
    # For categorical, ensure it exists in the options list, otherwise fallback to index 0
    def_genre = movie_info['Primary_Genre'] if pd.notna(movie_info['Primary_Genre']) else 'Unknown'
    def_director = movie_info['Director'] if pd.notna(movie_info['Director']) else 'Unknown'
    def_actor1 = movie_info['Actor 1'] if pd.notna(movie_info['Actor 1']) else 'Unknown'
    def_actor2 = movie_info['Actor 2'] if pd.notna(movie_info['Actor 2']) else 'Unknown'
    
    st.sidebar.success(f"Loaded features for: **{selected_movie}**")

st.sidebar.markdown("---")
st.sidebar.header("Input Features")

# Helper to find index safely
def get_index(options_list, val):
    return options_list.index(val) if val in options_list else 0

# Extract options lists
genre_opts = options.get('Primary_Genre', ['Unknown'])
dir_opts = options.get('Director', ['Unknown'])
a1_opts = options.get('Actor 1', ['Unknown'])
a2_opts = options.get('Actor 2', ['Unknown'])

# Inputs using the collected unique values
year = st.sidebar.slider("Release Year", 1930, 2024, def_year)
duration = st.sidebar.slider("Duration (minutes)", 45, 300, def_duration)
votes = st.sidebar.number_input("Number of Votes", 0, 10000000, def_votes)

primary_genre = st.sidebar.selectbox("Primary Genre", genre_opts, index=get_index(genre_opts, def_genre))
director = st.sidebar.selectbox("Director", dir_opts, index=get_index(dir_opts, def_director))
actor_1 = st.sidebar.selectbox("Actor 1", a1_opts, index=get_index(a1_opts, def_actor1))
actor_2 = st.sidebar.selectbox("Actor 2", a2_opts, index=get_index(a2_opts, def_actor2))

# Create Application Tabs
tab_predict, tab_analytics, tab_similar = st.tabs(["🔮 Rating Predictor", "📈 Analytics Dashboard", "🍿 Similar Movies"])

with tab_predict:
    if st.sidebar.button("Predict Rating!", type="primary"):
        with st.spinner("Calculating rating..."):
            # Prepare input DataFrame with the same column names
            input_data = pd.DataFrame({
                'Year': [year],
                'Duration': [duration],
                'Votes': [votes],
                'Primary_Genre': [primary_genre],
                'Director': [director],
                'Actor 1': [actor_1],
                'Actor 2': [actor_2]
            })
            
            # Predict Logic
            input_encoded = encoder.transform(input_data)
            input_scaled = scaler.transform(input_encoded)
            prediction = model.predict(input_scaled)[0]
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style='margin:0; color:#2D3748;'>Predicted IMDb Rating</h3>
                <div class="prediction-value">⭐ {prediction:.1f} <span style='font-size: 1.2rem; color:#A0AEC0;'>/ 10</span></div>
                <p style='margin:0; font-size:0.9rem; color:#718096;'>Powered by Random Forest Regression</p>
            </div>
            """, unsafe_allow_html=True)
            
            # What-If Analysis (Feature Sensitivity)
            st.markdown("### 🤔 What-If Analysis")
            st.markdown("See how tweaking certain features would impact the rating:")
            
            # +50,000 Votes
            whatif_data1 = input_data.copy()
            whatif_data1['Votes'] += 50000
            pred_whatif1 = model.predict(scaler.transform(encoder.transform(whatif_data1)))[0]
            delta1 = pred_whatif1 - prediction
            
            # Duration tweaking (Dynamic)
            # If duration > 100, checking shorter by 20. If <= 100, checking longer by 20.
            whatif_data2 = input_data.copy()
            if duration > 100:
                whatif_data2['Duration'] -= 20
                label_dur = f"If Duration was {duration - 20} mins"
            else:
                whatif_data2['Duration'] += 20
                label_dur = f"If Duration was {duration + 20} mins"
                
            pred_whatif2 = model.predict(scaler.transform(encoder.transform(whatif_data2)))[0]
            delta2 = pred_whatif2 - prediction
            
            col1, col2 = st.columns(2)
            col1.metric(label=f"If it had {votes + 50000:,} Votes", value=f"⭐ {pred_whatif1:.1f}", delta=f"{delta1:.2f}")
            col2.metric(label=label_dur, value=f"⭐ {pred_whatif2:.1f}", delta=f"{delta2:.2f}")

            # Display Feature Importance Visualization if it exists
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📊 What drives the rating?")
            
            # Pulling out the feature importances from the Random Forest
            importances = model.feature_importances_
            feature_names = encoder.get_feature_names_out()
            
            # Cleaning up the names since TargetEncoder alters them a bit
            clean_names = [name.split('_')[0] if '_' in name and not name.startswith('Target') else name for name in feature_names]
            
            # Grouping them back together and grabbing the top 6 factors
            feat_imp_df = pd.DataFrame({'Feature': clean_names, 'Importance': importances})
            # Combine importances of same-named features (e.g. multiple Director columns after encoding)
            feat_imp_df = feat_imp_df.groupby('Feature')['Importance'].sum().reset_index()
            feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(6)
            
            fig_imp = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h', 
                             title="Feature Impact on Predicted Rating",
                             color_discrete_sequence=['#9D174D'], text='Importance')
            fig_imp.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_imp.update_layout(
                yaxis={'categoryorder':'total ascending'},
                template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='#2D3748', size=12), height=300, margin=dict(l=10, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_imp, use_container_width=True, theme=None)
    else:
        st.info("👈 Set the movie features in the sidebar and click **Predict Rating!** to see the results.")

with tab_analytics:
    st.header("📈 Dataset Insights")
    st.markdown("Explore trends from the historical IMDb Indian Movies dataset in a dense, multi-metric view.")
    
    # ROW 1: KPI Metrics
    total_movies = len(dataset)
    avg_rating = dataset['Rating'].mean()
    total_votes = dataset['Votes'].sum()
    avg_duration = dataset['Duration'].dropna().mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", f"{total_movies:,}")
    col2.metric("Average Rating", f"⭐ {avg_rating:.2f}")
    col3.metric("Total Votes", f"{total_votes:,.0f}")
    col4.metric("Avg Duration", f"{avg_duration:.0f} mins")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ROW 2: Line Chart (Wide) & Horizontal Bar Chart (Narrow)
    col_l2, col_r2 = st.columns([2, 1])
    with col_l2:
        # Chart 1: Rating by Release Year
        yearly_avg = dataset.groupby('Year')['Rating'].mean().reset_index()
        fig_year = px.line(yearly_avg, x='Year', y='Rating', title='Rating Trend Over Time', 
                           color_discrete_sequence=['#BE185D'], text='Rating')
        fig_year.update_traces(texttemplate='%{text:.1f}', textposition='top center')
        fig_year.update_layout(
            template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='#2D3748', size=12), height=350, margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_year, use_container_width=True, theme=None)
        
    with col_r2:
        # Chart 2: Top Directors
        dir_counts = dataset['Director'].value_counts()
        valid_dirs = dir_counts[dir_counts >= 5].index
        top_dirs = dataset[dataset['Director'].isin(valid_dirs)].groupby('Director')['Rating'].mean().sort_values(ascending=False).head(5).reset_index()
        fig_dir = px.bar(top_dirs, x='Rating', y='Director', orientation='h', title='Top Directors',
                         color_discrete_sequence=['#9D174D'], text='Rating')
        fig_dir.update_traces(texttemplate='%{text:.1f}', textposition='inside')
        fig_dir.update_layout(
            yaxis={'categoryorder':'total ascending'}, template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='#2D3748', size=11), height=350, margin=dict(l=10, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_dir, use_container_width=True, theme=None)

    # ROW 3: Donut Chart, Scatter Plot, and Data Table
    col_l3, col_m3, col_r3 = st.columns([1, 1, 1.5])
    with col_l3:
        # Chart 3: Top Genres (Donut)
        top_genres = dataset['Primary_Genre'].value_counts().head(5).reset_index()
        top_genres.columns = ['Genre', 'Count']
        fig_genre = px.pie(top_genres, values='Count', names='Genre', title='Top Genres', hole=0.5,
                           color_discrete_sequence=['#9D174D', '#BE185D', '#DB2777', '#F472B6', '#FBCFE8'])
        fig_genre.update_traces(textposition='inside', textinfo='percent+label')
        fig_genre.update_layout(
            template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='#2D3748', size=11), height=300, margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_genre, use_container_width=True, theme=None)
        
    with col_m3:
        # Chart 4: Duration vs Rating Scatter
        sample_df = dataset.dropna(subset=['Duration', 'Rating']).sample(min(2000, len(dataset)), random_state=42)
        fig_scatter = px.scatter(sample_df, x='Duration', y='Rating', color='Rating', title='Duration vs Rating',
                                 color_continuous_scale='RdPu', opacity=0.6)
        fig_scatter.update_layout(
            template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='#2D3748', size=11), height=300, margin=dict(l=20, r=20, t=40, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_scatter, use_container_width=True, theme=None)
        
    with col_r3:
        st.markdown("<div style='color:#2D3748; font-weight:600; font-size:16px; margin-bottom:15px;'>Top Rated Movies</div>", unsafe_allow_html=True)
        top_movies_df = dataset.dropna(subset=['Name']).sort_values(by=['Rating', 'Votes'], ascending=[False, False])
        top_movies_df = top_movies_df[['Name', 'Year', 'Primary_Genre', 'Rating']].head(7)
        # Clean up year for display
        top_movies_df['Year'] = top_movies_df['Year'].fillna(0).astype(int).astype(str)
        top_movies_df['Year'] = top_movies_df['Year'].replace('0', 'Unknown')
        
        st.dataframe(top_movies_df, use_container_width=True, hide_index=True, height=270)

with tab_similar:
    st.header("🍿 Find Similar Movies")
    st.markdown("Based on the **Genre** and **Director** selected in the sidebar, here are highly-rated movies from the dataset you might like:")
    
    # Simple recommendation logic: just filtering the dataset by the same director or genre
    similar_movies = dataset[
        (dataset['Primary_Genre'] == primary_genre) | 
        (dataset['Director'] == director)
    ]
    
    # Filter out missing Names if any, sort by Rating and Votes
    similar_movies = similar_movies.dropna(subset=['Name']).sort_values(by=['Rating', 'Votes'], ascending=[False, False])
    
    if len(similar_movies) > 0:
        # Show top 5
        st.dataframe(
            similar_movies[['Name', 'Year', 'Primary_Genre', 'Director', 'Rating']].head(5),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No similar historical movies found for the selected criteria.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #A0AEC0; font-size: 0.85rem; padding: 10px;">
    Developed by <b>Bhagesh Biradar</b> • CodeSoft Data Science Internship
</div>
""", unsafe_allow_html=True)
