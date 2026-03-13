import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import subprocess
import sys
import glob
import os

def main():
    print("Loading dataset from kaggle.com/yasserh/titanic-dataset...")
    import kagglehub
    
    try:
        path = kagglehub.dataset_download("yasserh/titanic-dataset")
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
        else:
            print("Error: No CSV found in downloaded dataset.")
            return
    except Exception as e:
        print(f"Error loading Kaggle dataset: {e}")
        return

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    print("Data loaded successfully.")

    # Ensure target is integer
    df['survived'] = df['survived'].astype(int)

    # Feature Engineering
    # 1. Family Size (SibSp + Parch + 1 for self)
    df['FamilySize'] = df['sibsp'] + df['parch'] + 1
    
    # 2. Extract Titles from Names
    # Note: the kaggle dataset 'name' column contains formats like "Braund, Mr. Owen Harris"
    df['Title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Consolidate rare titles
    rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Convert cabin to a boolean value indicating if they had a cabin or not
    df['cabin'] = df['cabin'].notna().astype(int)

    # Required features
    features = ['age', 'sex', 'pclass', 'fare', 'cabin', 'FamilySize', 'Title']
    target = 'survived'
    
    X = df[features].copy()
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing
    numerical_features = ['age', 'fare', 'pclass', 'FamilySize']
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['sex', 'Title']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('passthrough', 'passthrough', ['cabin']) 
        ])

    # Model Pipeline
    from sklearn.model_selection import GridSearchCV
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    print("Executing GridSearchCV for Hyperparameter Tuning...")
    # Define the grid of parameters to search
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 5, 10, 15],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    # Run the grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Extract the absolute best model
    model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")

    print("Evaluating optimal model...")
    y_pred = model.predict(X_test)
    
    print("\n--- Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save best model
    joblib.dump(model, 'titanic_survival_model.pkl')
    print("Optimal model successfully trained and saved to 'titanic_survival_model.pkl'")

if __name__ == "__main__":
    main()
