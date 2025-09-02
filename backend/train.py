import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/Users/topfuture/Downloads/Crop_recommendation.csv")
# quick checks
print("rows, columns:", df.shape)
print(df.head())
print("\ninfo:")
print(df.info())
print("\ndescribe:")
print(df.describe())
print("\nmissing values:", df.isnull().sum().values.sum())
# explain target distribution
print("\nlabel counts:")
print(df["label"].value_counts())

## Train and save RandomForest pipeline
X = df[[c for c in df.columns if c != "label"]]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline(
    [("scaler", StandardScaler()), ("clf", RandomForestClassifier(random_state=42))]
)
pipeline.fit(X_train, y_train)

# Save the pipeline
joblib.dump({"pipeline": pipeline}, "model_pipeline.joblib")

# Evaluate
preds = pipeline.predict(X_test)
print("accuracy:", accuracy_score(y_test, preds))
print("\nclassification report:\n", classification_report(y_test, preds))
print("\nconfusion matrix:\n", confusion_matrix(y_test, preds))
