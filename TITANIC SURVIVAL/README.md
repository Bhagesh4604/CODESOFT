# Titanic Survival Predictor

A machine learning application that predicts the survival probability of passengers on the RMS Titanic based on their age, gender, ticket class, and other factors.

## How it Works
This project uses a **Random Forest Classifier** trained on historical passenger data. It captures key survival trends (like the "women and children first" policy) to estimate probabilities for new passenger profiles.

## Technologies Used
- **Python**: Core logic and machine learning.
- **Streamlit**: Elegant web interface with a "Dark Ocean" theme.
- **Scikit-Learn**: Model training and evaluation.
- **Plotly**: Dynamic data visualizations.

## How to Run
1. Install dependencies:
   ```bash
   pip install streamlit pandas scikit-learn joblib plotly kagglehub
   ```
2. Prepare the model (only needed once):
   ```bash
   python titanic_model.py
   ```
3. Launch the application:
   ```bash
   streamlit run app.py
   ```

Developed by **Bhagesh Biradar**
