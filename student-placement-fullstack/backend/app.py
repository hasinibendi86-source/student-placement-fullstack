"""
app.py
-------
Flask REST API backend for the Student Placement Prediction System.

Exposes:
    POST /predict   -> takes student details as JSON, returns prediction + probability
    GET  /health     -> simple health check endpoint

Run with:
    python app.py

The server starts at: http://127.0.0.1:5000
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------- App setup ----------
app = Flask(__name__)
CORS(app)  # Allows the separate frontend (opened as a static file / different port) to call this API

MODEL_PATH = os.path.join(os.path.dirname(__file__), "placement_model.pkl")

# ---------- Load the trained model once, when the server starts ----------
model_bundle = None
if os.path.exists(MODEL_PATH):
    model_bundle = joblib.load(MODEL_PATH)
    print("Model loaded successfully from placement_model.pkl")
else:
    print("WARNING: placement_model.pkl not found. Run 'python train_model.py' first.")


# ---------- Validation helper ----------
def validate_input(data):
    """
    Checks that all required fields are present and within valid ranges.
    Returns (is_valid, error_message).
    """
    required_fields = [
        "cgpa", "tenth", "twelfth",
        "technical", "aptitude", "communication", "internship"
    ]

    # Check all fields are present
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == "":
            return False, f"Missing required field: '{field}'"

    # Check numeric types and ranges
    try:
        cgpa = float(data["cgpa"])
        tenth = float(data["tenth"])
        twelfth = float(data["twelfth"])
        technical = float(data["technical"])
        aptitude = float(data["aptitude"])
        communication = float(data["communication"])
        internship = int(data["internship"])
    except (ValueError, TypeError):
        return False, "All score fields must be valid numbers."

    if not (0 <= cgpa <= 10):
        return False, "CGPA must be between 0 and 10."
    if not (0 <= tenth <= 100):
        return False, "10th Percentage must be between 0 and 100."
    if not (0 <= twelfth <= 100):
        return False, "12th Percentage must be between 0 and 100."
    if not (0 <= technical <= 100):
        return False, "Technical Score must be between 0 and 100."
    if not (0 <= aptitude <= 100):
        return False, "Aptitude Score must be between 0 and 100."
    if not (0 <= communication <= 100):
        return False, "Communication Score must be between 0 and 100."
    if internship not in (0, 1):
        return False, "Internship must be 0 (No) or 1 (Yes)."

    return True, None


# ---------- Routes ----------

@app.route("/health", methods=["GET"])
def health():
    """Simple endpoint to check if the server and model are ready."""
    return jsonify({
        "status": "ok",
        "model_loaded": model_bundle is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
      "cgpa": 8.2,
      "tenth": 85,
      "twelfth": 88,
      "technical": 80,
      "aptitude": 75,
      "communication": 82,
      "internship": 1
    }

    Returns JSON:
    {
      "prediction": "Placed",
      "probability": 87.5
    }
    """
    # 1. Make sure the model is available
    if model_bundle is None:
        return jsonify({"error": "Model not found on server. Please run train_model.py first."}), 500

    # 2. Parse JSON body safely
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body."}), 400

    # 3. Validate input values
    is_valid, error_message = validate_input(data)
    if not is_valid:
        return jsonify({"error": error_message}), 400

    # 4. Build a DataFrame in the exact column order used during training
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feature_columns = model_bundle["feature_columns"]

    input_row = pd.DataFrame([[
        float(data["cgpa"]),
        float(data["tenth"]),
        float(data["twelfth"]),
        float(data["technical"]),
        float(data["aptitude"]),
        float(data["communication"]),
        int(data["internship"]),
    ]], columns=feature_columns)

    # 5. Scale input using the SAME scaler fit during training
    try:
        input_scaled = scaler.transform(input_row)
    except Exception as e:
        return jsonify({"error": f"Failed to process input: {str(e)}"}), 500

    # 6. Predict class and probability
    try:
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100  # probability of "Placed"
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    result_label = "Placed" if prediction == 1 else "Not Placed"

    return jsonify({
        "prediction": result_label,
        "probability": round(float(probability), 2)
    })


# ---------- Error handlers for cleaner messages ----------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ---------- Run the server ----------
if __name__ == "__main__":
    # debug=True auto-reloads the server when you edit code (fine for local development)
    app.run(debug=True, host="127.0.0.1", port=5000)
