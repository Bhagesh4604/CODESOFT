import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import category_encoders as ce
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

def main():
    print("Fetching dataset from Kaggle...")
    # Download latest version
    path = kagglehub.dataset_download("adrianmcmahon/imdb-india-movies")
    print(f"Path to dataset files: {path}")
    
    # Trying to find the right csv file in the downloaded folder
    # Assuming it's the IMDb one
    csv_file = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_file = os.path.join(root, file)
                break
        if csv_file:
            break
            
    if not csv_file:
        raise FileNotFoundError("Could not find the CSV file in the downloaded Kaggle dataset.")
        
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file, encoding='latin1')
    
    print("\n--- Initial Data Overview ---")
    print(df.head())
    print("\nData Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    
    print("\n--- Data Preprocessing ---")
    # Step 1: Drop empty ratings since that's what I'm trying to predict
    df.dropna(subset=['Rating'], inplace=True)
    
    # 2. Clean Year column: Remove parentheses and convert to integer
    df['Year'] = df['Year'].str.replace(r'[()]', '', regex=True)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # 3. Clean Duration column: Remove ' min' and convert to integer
    df['Duration'] = df['Duration'].str.replace(' min', '')
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    
    # Step 4: Votes have commas (e.g., 5,000), need to remove them so I can convert to numbers
    df['Votes'] = df['Votes'].str.replace(',', '')
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
    
    # Step 5: Filling missing numbers with median instead of mean to avoid outlier issues
    df['Duration'].fillna(df['Duration'].median(), inplace=True)
    df['Year'].fillna(df['Year'].median(), inplace=True)
    df['Votes'].fillna(df['Votes'].median(), inplace=True)

    # Step 6: Text columns are tricky, I'll just fill NaNs with 'Unknown' for now
    cat_columns = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    for col in cat_columns:
        df[col].fillna('Unknown', inplace=True)
        
    print(f"Data Shape after cleaning target/numerical: {df.shape}")

    # The Genre column has multiple genres separated by commas (Action, Drama, etc)
    # I'll just extract the first one as the 'Primary' genre to make modeling simpler
    df['Primary_Genre'] = df['Genre'].apply(lambda x: x.split(',')[0] if pd.notnull(x) else 'Unknown')
    
    print("\n--- Feature Engineering & Encoding ---")
    # I decided to use Target Encoding for Director and Actors because One-Hot Encoding 
    # created thousands of columns and crashed my laptop memory.
    # Target Encoding replaces a text category with the average rating for that category!
    
    features = ['Year', 'Duration', 'Votes', 'Primary_Genre', 'Director', 'Actor 1', 'Actor 2']
    X = df[features]
    y = df['Rating']
    
    # Have to split data BEFORE encoding to avoid leaking test info into the training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Setting up TargetEncoder with smoothing
    # (smoothing=10 makes sure rare directors/actors don't skew the results too much)
    encoder = ce.TargetEncoder(cols=['Primary_Genre', 'Director', 'Actor 1', 'Actor 2'], smoothing=10)
    
    # Fit on training data, transform both
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    X_test_encoded = encoder.transform(X_test)
    
    # Scale numerical features (Year, Duration, Votes, and the newly target-encoded features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)
    
    print("Features ready for modeling.")
    
    print("\n--- Model Training ---")
    # 1. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_r2 = r2_score(y_test, lr_pred)
    
    print("Linear Regression Results:")
    print(f"RMSE: {lr_rmse:.4f}")
    print(f"R2 Score: {lr_r2:.4f}")
    
    # 2. Random Forest Regressor
    print("\nTraining Random Forest (this might take a moment)...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    
    print("Random Forest Results:")
    print(f"RMSE: {rf_rmse:.4f}")
    print(f"R2 Score: {rf_r2:.4f}")
    
    # 3. Simple comparison DataFrame
    results_df = pd.DataFrame({
        'Actual Rating': y_test.values[:10],
        'LR Predicted': lr_pred[:10],
        'RF Predicted': rf_pred[:10]
    })
    
    print("\n--- Sample Predictions ---")
    print(results_df)

    # Output feature importances for Random Forest
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n--- Feature Importances (Random Forest) ---")
    print("\n--- Feature Importances (Random Forest) ---")
    for i in range(len(features)):
        print(f"{features[indices[i]]}: {importances[indices[i]]:.4f}")

    print("\n--- Saving Artifacts ---")
    joblib.dump(rf_model, "rf_model.pkl")
    joblib.dump(encoder, "encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    dropdown_options = {
        'Primary_Genre': sorted(df['Primary_Genre'].unique().tolist()),
        'Director': sorted(df['Director'].unique().tolist()),
        'Actor 1': sorted(df['Actor 1'].unique().tolist()),
        'Actor 2': sorted(df['Actor 2'].unique().tolist()),
    }
    joblib.dump(dropdown_options, "dropdown_options.pkl")
    
    # Saving the cleaned dataset as a pickle so I can load it directly in the Streamlit app
    joblib.dump(df, "movie_dataset.pkl")
    
    print("Artifacts saved successfully.")

if __name__ == "__main__":
    main()
