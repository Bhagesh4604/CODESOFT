import pandas as pd
import numpy as np
import joblib
import kagglehub
import os

print("Downloading dataset...")
# Download latest version
path = kagglehub.dataset_download("arshid/iris-flower-dataset")
print("Path to dataset files:", path)

# get the csv file
import glob
csv_files = glob.glob(os.path.join(path, "*.csv"))
if not csv_files:
    raise FileNotFoundError("Could not find the downloaded CSV file.")

csv_path = csv_files[0]
print(f"Loading data from {csv_path}...")

# load data
df = pd.read_csv(csv_path)
X = df.drop(columns=['species'])
y = df['species']

# scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# random forest
from sklearn.ensemble import RandomForestClassifier
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# test locally
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
test_model = RandomForestClassifier(n_estimators=100, random_state=42)
test_model.fit(X_train, y_train)
y_pred = test_model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# save to file
joblib.dump(model, "iris_model.pkl")
joblib.dump(scaler, "iris_scaler.pkl")
joblib.dump(df, "iris_dataset.pkl")
