# Titanic Survival Prediction

A machine learning pipeline and interactive dashboard for predicting passenger survival on the Titanic dataset.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. To run the interactive web application:
   ```bash
   streamlit run app.py
   ```

3. To retrain the Random Forest model from scratch:
   ```bash
   python titanic_prediction.py
   ```

## Stack
- **Dashboard:** Streamlit & Plotly
- **Model:** Scikit-Learn (RandomForestClassifier, Pipeline, GridSearchCV)
- **Data:** Kaggle `yasserh/titanic-dataset`
