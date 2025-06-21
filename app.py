from flask import Flask, request, jsonify, render_template
import pickle
from utils.preprocess import preprocess_input
import os

app = Flask(__name__)

# Load model
# Load model from relative path
model_path = os.path.join(os.path.dirname(__file__), "model", "heart_disease_model.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found at {model_path}. Please run model_training.py first.")



@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

import pandas as pd

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json(force=True)
        features = pd.DataFrame([input_data])  # <-- Now send raw input as DataFrame
        prediction = model.predict(features)[0]
        return jsonify({"prediction": int(prediction)})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)






