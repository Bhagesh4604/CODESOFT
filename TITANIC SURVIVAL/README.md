# Titanic Survival Predictor

This was my first task during the CODESOFT internship and honestly one of the most interesting datasets I've worked with. The Titanic dataset is a classic in machine learning but there's still a lot to explore in terms of feature engineering.

## What I built

A web app where you can enter passenger details (age, gender, ticket class, fare etc.) and it tells you whether that person would have survived. I also added an analytics dashboard with charts showing survival patterns across different groups.

## Approach

I used a Random Forest Classifier for the prediction. Before training I spent some time on preprocessing — handling missing ages, encoding the gender and embarkation columns, and extracting titles from names (Mr, Mrs, Miss etc.) which turned out to be a surprisingly useful feature.

The model gets around 82-85% accuracy on the test split which I'm happy with given the dataset size.

## Stack

- Python + Scikit-learn (model training)
- Streamlit (the web interface)
- Plotly (charts in the dashboard)
- Joblib (saving and loading the model)

## Running it locally

First install the dependencies:
```
pip install streamlit pandas scikit-learn joblib plotly
```

Then train the model (only needs to be done once):
```
python titanic_prediction.py
```

Then start the app:
```
streamlit run app.py
```

---
Made by Bhagesh Biradar | CODESOFT Data Science Internship
