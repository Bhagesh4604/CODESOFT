# Movie Rating Predictor

Task 2 of the CODESOFT internship. I went with an Indian movies dataset from Kaggle which has ratings, directors, actors, genres etc. I thought it would be more interesting than a generic movies dataset since I'm familiar with Bollywood films.

## What I built

A Streamlit app where you pick a director, genre, cast members and year, and it predicts what IMDb rating that movie would get. There's also a "similar movies" section that finds movies from the dataset with similar attributes.

## How the model works

The trickiest part here was handling the categorical features. Directors and actors are high-cardinality columns with hundreds of unique values, so I used `category_encoders` target encoding instead of one-hot encoding (which would have made the feature matrix huge).

I went with Random Forest Regression in the end. I also tried Linear Regression first but the RF gave much better results once I tuned it a bit.

For the similar movies feature I wrote a simple weighted scoring function — movies get points for sharing the same genre, director, actors etc. It's not a fancy recommendation system but it works reasonably well.

## Stack

- Python + Scikit-learn + Category Encoders
- Streamlit (dashboard)
- Plotly (charts)
- Pandas + Numpy

## To run

```
pip install streamlit pandas scikit-learn joblib plotly category_encoders
```

Train the model first (this downloads and processes the dataset):
```
python movie_rating_model.py
```

Then:
```
streamlit run app.py
```

Note: The model file is around 8MB so it might take a moment to load.

---
Bhagesh Biradar | CODESOFT Internship Task 2
