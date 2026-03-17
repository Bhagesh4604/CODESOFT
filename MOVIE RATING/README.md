# Movie Rating Predictor

An analytical tool designed to predict the IMDb ratings of Indian movies and recommend similar historical cinema based on production attributes.

## How it Works
The application leverages a **Random Forest Regressor** to find patterns between movie directors, actors, genres, and their ratings. It also features a **Weighted Similarity Engine** to suggest related movies from the dataset based on your inputs.

## Technologies Used
- **Python**: Machine Learning and Data Processing.
- **Streamlit**: Premium "Magenta" themed dashboard.
- **Scikit-Learn**: Regression modeling.
- **Category Encoders**: Handling large categorical feature sets.

## How to Run
1. Install dependencies:
   ```bash
   pip install streamlit pandas scikit-learn joblib plotly category_encoders
   ```
2. Train the model:
   ```bash
   python movie_rating_model.py
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

Developed by **Bhagesh Biradar**
