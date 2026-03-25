import pandas as pd
import numpy as np
import joblib
import kagglehub
import glob
import os

print("downloading iris dataset...")
path = kagglehub.dataset_download("arshid/iris-flower-dataset")

csv_files = glob.glob(os.path.join(path, "*.csv"))
if not csv_files:
    raise FileNotFoundError("couldn't find the csv after downloading")

df = pd.read_csv(csv_files[0])
print(f"loaded {len(df)} rows")

X = df.drop(columns=['species'])
y = df['species']

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train/test split just to check accuracy
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model_check = RandomForestClassifier(n_estimators=100, random_state=42)
model_check.fit(X_train, y_train)
preds = model_check.predict(X_test)
print(f"test accuracy: {accuracy_score(y_test, preds)*100:.2f}%")

# now train on full data for the app
print("training on full dataset...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, "iris_model.pkl")
joblib.dump(scaler, "iris_scaler.pkl")
joblib.dump(df, "iris_dataset.pkl")
print("done — saved model, scaler and dataset")
