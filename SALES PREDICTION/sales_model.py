import pandas as pd
import numpy as np
import joblib

# using the ISLR advertising dataset — small but clean
url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Advertising.csv"
print(f"fetching data from github...")
df = pd.read_csv(url)

# this dataset has an unnamed index column, drop it
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"loaded {len(df)} records, columns: {list(df.columns)}")

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# trying random forest — usually beats linear regression on this data
print("training random forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# keeping linear regression as a baseline comparison
print("training linear regression for comparison...")
lr = LinearRegression()
lr.fit(X_train, y_train)

rf_preds = rf.predict(X_test)
lr_preds = lr.predict(X_test)

print(f"RF  — R2: {r2_score(y_test, rf_preds):.3f}  RMSE: {np.sqrt(mean_squared_error(y_test, rf_preds)):.3f}")
print(f"LR  — R2: {r2_score(y_test, lr_preds):.3f}  RMSE: {np.sqrt(mean_squared_error(y_test, lr_preds)):.3f}")

# RF wins, use that for the app
print("saving random forest model...")
joblib.dump(rf, "sales_model.pkl")
joblib.dump(scaler, "sales_scaler.pkl")
joblib.dump(df, "sales_dataset.pkl")
print("all saved")
