import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import glob
import os

def main():
    print("loading titanic dataset from kaggle...")
    import kagglehub

    try:
        path = kagglehub.dataset_download("yasserh/titanic-dataset")
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
        else:
            print("no csv found in the download")
            return
    except Exception as e:
        print(f"kaggle download failed: {e}")
        return

    df.columns = [c.lower() for c in df.columns]
    df['survived'] = df['survived'].astype(int)

    # adding family size — combines siblings/spouses and parents/children 
    df['FamilySize'] = df['sibsp'] + df['parch'] + 1

    # pulling the title out of the name column (e.g. "Mr", "Mrs", "Miss")
    # the name format is like "Braund, Mr. Owen Harris" so this regex works well
    df['Title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # grouping rare titles together so they don't get treated as separate categories
    rare = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare, 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # convert cabin to 1/0 — just whether someone had a cabin, not which one
    df['cabin'] = df['cabin'].notna().astype(int)

    features = ['age', 'sex', 'pclass', 'fare', 'cabin', 'FamilySize', 'Title']
    X = df[features].copy()
    y = df['survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # preprocessing pipeline — median impute for numerics, one-hot for categoricals
    num_cols = ['age', 'fare', 'pclass', 'FamilySize']
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_cols = ['sex', 'Title']
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
        ('passthrough', 'passthrough', ['cabin'])
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # running a grid search to find better hyperparameters
    # takes a few minutes but worth it
    print("running grid search, this might take a bit...")
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 5, 10, 15],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    print(f"best params: {grid_search.best_params_}")

    y_pred = model.predict(X_test)
    print(f"\naccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'titanic_survival_model.pkl')
    print("saved model to titanic_survival_model.pkl")

if __name__ == "__main__":
    main()
