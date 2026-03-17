# Sales Prediction Engine

A business intelligence dashboard that forecasts sales revenue based on advertising budgets across TV, Radio, and Newspaper channels.

## How it Works
The engine uses a **Random Forest Regressor** to determine the ROI of various advertising combinations. It emphasizes the high correlation between TV spending and sales while showing the negligible impact of Newspaper ads.

## Technologies Used
- **Python**: Regression analysis.
- **Streamlit**: "Sunset Analytics" themed UI with revenue gauges.
- **Plotly**: Interactive indicator charts.
- **Scikit-Learn**: High-precision modeling.

## How to Run
1. Install dependencies:
   ```bash
   pip install streamlit pandas scikit-learn joblib plotly
   ```
2. Train the model:
   ```bash
   python sales_model.py
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

Developed by **Bhagesh Biradar**
