# Iris Flower Classification

Task 3 — the classic Iris dataset. I know it's a basic one but I wanted to make the interface as interactive as possible rather than just running a notebook and showing results.

## What I built

A live classification app where sliders for sepal/petal measurements instantly update the predicted species without needing to click a "predict" button. I also added a radar chart showing how the input measurements compare to the average profile of each species, and scatter plots of the dataset colored by species.

## Model

Random Forest Classifier trained on Fisher's Iris dataset (150 samples, 4 features, 3 classes). With enough trees and a proper train/test split the accuracy is basically 96-100% on this dataset — it's not a hard classification problem since the species are fairly well separated in feature space (except Versicolor and Virginica which have some overlap in sepal dimensions).

I used StandardScaler before training since the feature ranges are slightly different.

## Live prediction

The way Streamlit session state works is that slider changes trigger a rerun of the whole script, so I just compute the prediction at the top level every time the script runs rather than inside a button click. This gives the "live" feel.

## Stack

- Python + Scikit-learn
- Streamlit (with session state for live sliders)
- Plotly (scatter plots, radar chart, box plots)

## Running it

Install deps:
```
pip install streamlit pandas scikit-learn joblib plotly
```

Train (fast, takes ~5 seconds):
```
python iris_model.py
```

Run:
```
streamlit run app.py
```

---
Bhagesh Biradar | CODESOFT Internship Task 3
