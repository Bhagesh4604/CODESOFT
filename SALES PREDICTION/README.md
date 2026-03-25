# Sales Prediction

Task 4 — predicting product sales based on advertising spend across TV, Radio and Newspaper channels. I used the ISLR Advertising dataset which is small (200 rows) but has a clear signal.

## What I built

An interactive budget planner where you adjust three sliders (TV, Radio, Newspaper budgets) and see the projected sales units update in real time, along with estimated ROI. There's also an analytics tab with scatter plots showing the relationship between each channel and sales, a correlation heatmap, and feature importance bars.

## Findings

The most interesting thing this model reveals is how little Newspaper advertising actually matters. TV has by far the strongest correlation with sales (~0.78) while Newspaper is nearly irrelevant (~0.23). Radio is a decent secondary channel. The model picks this up clearly from feature importances.

I tried Linear Regression first which gave decent results (R² ~0.9) but Random Forest pushed it higher and handles any non-linear patterns better. I did a basic grid search to tune the number of estimators and max depth.

## Stack

- Python + Scikit-learn + Pandas
- Streamlit
- Plotly (gauge chart, scatter plots, heatmap)

## Running it

```
pip install streamlit pandas scikit-learn joblib plotly
```

Train:
```
python sales_model.py
```

Run:
```
streamlit run app.py
```

Budget sliders go live as soon as the app loads — no button needed.

---
Bhagesh Biradar | CODESOFT Internship Task 4
