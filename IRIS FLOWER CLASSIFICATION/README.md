# Iris Flower Classification

A scientific classification engine that identifies iris flower species (Setosa, Versicolor, Virginica) based on petal and sepal measurements.

## How it Works
This project uses a **Random Forest Classifier** to distinguish between the three species. The app uses **Streamlit Session State** to provide an instant, "live-updating" prediction experience as you manipulate the sliders.

## Technologies Used
- **Python**: Classification logic.
- **Streamlit**: "Botanical Glassmorphic" user interface.
- **Scikit-Learn**: Feature scaling and classification.

## How to Run
1. Install dependencies:
   ```bash
   pip install streamlit pandas scikit-learn joblib
   ```
2. Train the model:
   ```bash
   python iris_model.py
   ```
3. Launch the app:
   ```bash
   streamlit run app.py
   ```

Developed by **Bhagesh Biradar**
