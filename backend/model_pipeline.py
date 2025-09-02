# backend/model_pipeline.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def build_and_train(
    csv_path="/Users/topfuture/Downloads/Crop_recommendation.csv",
    save_path="model_pipeline.joblib",
):
    df = pd.read_csv(csv_path)

    # simple feature list — adapt if your CSV differs
    target_col = "label"
    feature_cols = [c for c in df.columns if c != target_col]

    # separate types
    numeric_features = (
        df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    )
    categorical_features = (
        df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    )

    # transformers
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # full pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # train/test split
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    # save
    joblib.dump({"pipeline": pipeline, "feature_columns": feature_cols}, save_path)
    print("Saved pipeline to", save_path)

    return pipeline, (X_test, y_test)


if __name__ == "__main__":
    pipeline, (X_test, y_test) = build_and_train()
    print("Done training")
    
#Creates a small helper that accepts raw inputs(dictionary) and returns the predicted crop
def predict_from_dict(input_dict, model_path='model_pipeline.joblib'):
    obj=joblib.load(model_path)
    pipeline=obj['pipeline']
    #order doesb't strictly matter because ColumnTransformer picks by column name
    X = pd.DataFrame([input_dict])
    pred = pipeline.predict(X)[0]
    proba = None
    if hasattr(pipeline, 'predict_proba'):
        proba = pipeline.predict_proba(X).max()
    return {'prediction': pred, 'Confidence': float(proba) if proba is not None else None}
#Example usage:
#input_data = {'N': 90, 'P': 42, 'K': 43, 'temperature': 20.879743, 'humidity': 82.002744, 'ph':6.502985, 'rainfall':202.935536}    
#print(predict_from_dict(input_data))