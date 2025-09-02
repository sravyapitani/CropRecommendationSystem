from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the saved pipeline
model_obj = joblib.load("model_pipeline.joblib")
pipeline = model_obj["pipeline"]


# Home route (GET)
@app.route("/", methods=["GET"])
def home():
    return (
        "Crop Recommender System API is running. Use POST /predict to get predictions."
    )


# Prediction route (POST)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not isinstance(data, dict):
        return jsonify({"error": "Input data should be a dictionary"}), 400

    X = pd.DataFrame([data])

    try:
        pred = pipeline.predict(X)[0]
        proba = None
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X).max()

        return jsonify(
            {
                "prediction": str(pred),
                "confidence": float(proba) if proba is not None else None,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
