import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import RandomOverSampler

# the actual kaggle dataset (mlg-ulb/creditcardfraud) is 144MB so I can't
# include it in the repo. generating a synthetic version with the same structure
# 50k transactions, ~0.2% fraud rate — same as the real dataset

print("generating synthetic transaction data...")
np.random.seed(42)
n = 50000
n_fraud = 100  # keeping it realistic (0.2%)

# V1-V28 are PCA features in the real dataset, simulating them here
v_features = np.random.randn(n, 28)
# fraud transactions look different on a few key components (V1, V2, V14)
v_features[:n_fraud, [0, 1, 13]] += 3

time_col = np.random.uniform(0, 172792, n)
amount_col = np.random.exponential(scale=88, size=n)

labels = np.zeros(n)
labels[:n_fraud] = 1

df = pd.DataFrame(v_features, columns=[f'V{i}' for i in range(1, 29)])
df['Time'] = time_col
df['Amount'] = amount_col
df['Class'] = labels
df = df.sample(frac=1).reset_index(drop=True)

print(f"dataset: {df.shape}, fraud cases: {int(df['Class'].sum())}")

# scale Amount and Time — using RobustScaler because financial data has big outliers
# regular StandardScaler would let a few huge transactions distort everything
scaler = RobustScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# the whole point of this project — handling extreme class imbalance
# without oversampling the model just predicts "not fraud" for everything
# and gets 99.8% accuracy which is completely useless
print("oversampling fraud cases in training data...")
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)
print(f"after resampling: {X_res.shape}")

print("training model...")
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_res, y_res)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"F1 score: {f1_score(y_test, y_pred):.3f}")

print("saving files...")
joblib.dump(model, "fraud_model.pkl")
joblib.dump(scaler, "fraud_scaler.pkl")

# save a few genuine and fraud examples so the app can load them as demo inputs
fraud_samples = df[df['Class'] == 1].sample(5, random_state=42)
genuine_samples = df[df['Class'] == 0].sample(5, random_state=42)
joblib.dump({'fraud': fraud_samples, 'genuine': genuine_samples}, "sample_transactions.pkl")

print("done")
