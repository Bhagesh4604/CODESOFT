# Credit Card Fraud Detection

Task 5 and probably the most challenging one. The actual Kaggle dataset (mlg-ulb/creditcardfraud) is 144MB so I couldn't include it, but I generated a synthetic dataset that matches its structure — 50,000 transactions with ~0.2% fraud rate — using the same PCA feature format (V1-V28).

## The problem

Credit card fraud detection is tricky because the dataset is *extremely* imbalanced. Less than 0.2% of transactions are fraud. If you just train a model on raw data it'll predict "not fraud" for everything and still get 99.8% accuracy — which is totally useless.

## How I solved it

I used RandomOverSampler from the imbalanced-learn library to oversample the fraud cases in the training set. This gives the model enough examples of fraud to actually learn what distinguishes them.

The features V1–V28 are already PCA-transformed (the original dataset is anonymised for privacy), so I just needed to scale the Amount and Time columns using RobustScaler (better than StandardScaler for financial data with outliers).

Model: Random Forest Classifier with 50 trees, max depth 10. Gets around F1 ~0.89 on the test split.

## What the app does

- You enter or load sample PCA feature vectors for a transaction
- The model gives an instant verdict (FRAUD or CLEARED) with a probability score
- There's a custom polar risk dial that visually shows threat level
- Analytics tab has feature importance chart, scatter plots, violin plots for distribution comparison

## Stack

- Python + Scikit-learn + imbalanced-learn
- Streamlit
- Plotly
- Numpy + Pandas

## Running it

```
pip install streamlit pandas scikit-learn imbalanced-learn joblib plotly
```

Generate the synthetic data and train the model:
```
python fraud_model.py
```

Run the app:
```
streamlit run app.py
```

The model takes about a minute to train because of the oversampling step.

---
Bhagesh Biradar | CODESOFT Internship Task 5
