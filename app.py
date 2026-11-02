
# Q1: Create a Flask REST API for Student Result Prediction

from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "student_result_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Trained model not found. Run train_model.py first.")

model = joblib.load(MODEL_PATH)

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "service": "student-result-prediction",
        "status": "ok"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "JSON object required"}), 400

        missing = [feature for feature in FEATURES if feature not in data]

        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing
            }), 400

        values = [float(data[feature]) for feature in FEATURES]

        if any(value < 0 or value > 100 for value in values):
            return jsonify({
                "error": "All input values must be between 0 and 100"
            }), 400

        prediction = int(model.predict([values])[0])

        return jsonify({
            "prediction": "PASS" if prediction == 1 else "FAIL",
            "prediction_code": prediction
        })

    except (ValueError, TypeError):
        return jsonify({"error": "Input values must be numeric"}), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
