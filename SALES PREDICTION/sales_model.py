import pandas as pd
import numpy as np
import joblib
import kagglehub
import os

print("getting data...")
url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Advertising.csv"
print(f"Loading data from {url}...")

# load data
df = pd.read_csv(url)

# The ISLR dataset has an Unnamed: 0 column which is just the index.
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])


X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("training random forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("training linear regression (baseline)...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# evaluate
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

print(f"RF R2: {r2_score(y_test, rf_pred):.3f}, RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.3f}")
print(f"LR R2: {r2_score(y_test, lr_pred):.3f}, RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred)):.3f}")

# The Random Forest usually performs exceptionally well here because of non-linear interactions
# between TV and Radio advertising budgets.
final_model = rf_model

# save to file
print("saving artifacts...")
joblib.dump(final_model, "sales_model.pkl")
joblib.dump(scaler, "sales_scaler.pkl")
joblib.dump(df, "sales_dataset.pkl")

print("done!")
