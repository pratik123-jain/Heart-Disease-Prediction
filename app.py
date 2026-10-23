
import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


# --------------------------------------------------
# 1. App configuration
# --------------------------------------------------

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BACKEND_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

DATABASE_PATH = os.path.join(BACKEND_DIR, "predictions.db")


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

model = None
scaler = None
feature_columns = None
model_load_error = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    if isinstance(feature_columns, pd.DataFrame):
        feature_columns = feature_columns.columns.tolist()
    elif hasattr(feature_columns, "tolist"):
        feature_columns = feature_columns.tolist()
    else:
        feature_columns = list(feature_columns)

    print("ML model loaded successfully.")

except Exception as error:
    model_load_error = str(error)
    print("Model loading error:", model_load_error)
    print("Check backend/model/ and run train_model.py if needed.")


# --------------------------------------------------
# 3. Create database and table
# --------------------------------------------------

def init_database():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT
                    (datetime('now', 'localtime')),
                result TEXT NOT NULL,
                probability REAL,
                patient_name TEXT DEFAULT 'Unknown',
                age INTEGER
            )
        """)

        # Add columns if an older version of the table exists.
        existing_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(predictions)"
            ).fetchall()
        }

        if "patient_name" not in existing_columns:
            connection.execute("""
                ALTER TABLE predictions
                ADD COLUMN patient_name TEXT DEFAULT 'Unknown'
            """)

        if "age" not in existing_columns:
            connection.execute("""
                ALTER TABLE predictions
                ADD COLUMN age INTEGER
            """)

        connection.commit()


init_database()


# --------------------------------------------------
# 4. Home page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# 5. Predict heart disease
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None or feature_columns is None:
        return jsonify({
            "success": False,
            "error": (
                "ML model files are missing or could not be loaded. "
                "Please check backend/model/ and train the model again."
            )
        }), 503

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Invalid request. Please submit the form again."
        }), 400

    patient_name = str(data.get("patient_name", "")).strip()

    if not patient_name:
        return jsonify({
            "success": False,
            "error": "Patient name is required."
        }), 400

    required_fields = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal"
    ]

    missing_fields = [
        field for field in required_fields
        if field not in data or data[field] is None
        or str(data[field]).strip() == ""
    ]

    if missing_fields:
        return jsonify({
            "success": False,
            "error": "Missing fields: " + ", ".join(missing_fields)
        }), 400

    try:
        input_values = {}

        for field in required_fields:
            value = float(data[field])

            if not np.isfinite(value):
                raise ValueError(f"{field} must be a valid number.")

            input_values[field] = value

        age = int(input_values["age"])

        if not 1 <= age <= 120:
            return jsonify({
                "success": False,
                "error": "Please enter a valid age."
            }), 400

        # Keep the input columns in the same order as training.
        input_df = pd.DataFrame([input_values])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        scaled_input = scaler.transform(input_df)

        prediction = int(model.predict(scaled_input)[0])

        # Score for the predicted class, not a confirmed medical risk.
        score = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_input)[0]
            predicted_index = list(model.classes_).index(prediction)
            score = round(float(probabilities[predicted_index]) * 100, 2)

        elif hasattr(model, "decision_function"):
            # A decision score is not a probability, so do not label it
            # as a percentage.
            score = None

        if prediction == 1:
            result_label = "Higher Risk"
            message = (
                "The model indicates a higher-risk result. "
                "Please consult a qualified healthcare professional."
            )
        else:
            result_label = "Lower Risk"
            message = (
                "The model indicates a lower-risk result. "
                "This does not rule out heart disease or replace medical advice."
            )

        # Save prediction to SQLite.
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute("""
                INSERT INTO predictions
                    (patient_name, age, result, probability)
                VALUES (?, ?, ?, ?)
            """, (
                patient_name,
                age,
                result_label,
                score
            ))

            connection.commit()

        return jsonify({
            "success": True,
            "patient_name": patient_name,
            "age": age,
            "prediction": prediction,
            "result": result_label,
            "probability": score,
            "message": message,
            "disclaimer": (
                "This is an educational ML prediction, "
                "not a medical diagnosis."
            )
        })

    except (TypeError, ValueError) as error:
        return jsonify({
            "success": False,
            "error": f"Invalid input: {error}"
        }), 400

    except Exception as error:
        app.logger.exception("Prediction failed")

        return jsonify({
            "success": False,
            "error": (
                "Prediction failed. Check that the trained model, "
                "scaler, and feature columns match the dataset."
            )
        }), 500


# --------------------------------------------------
# 6. Dashboard statistics and prediction history
# --------------------------------------------------

@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    try:
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.row_factory = sqlite3.Row

            total = connection.execute("""
                SELECT COUNT(*) AS count
                FROM predictions
            """).fetchone()["count"]

            higher_risk = connection.execute("""
                SELECT COUNT(*) AS count
                FROM predictions
                WHERE result = 'Higher Risk'
            """).fetchone()["count"]

            lower_risk = connection.execute("""
                SELECT COUNT(*) AS count
                FROM predictions
                WHERE result = 'Lower Risk'
            """).fetchone()["count"]

            rows = connection.execute("""
                SELECT patient_name, age, result, probability, created_at
                FROM predictions
                ORDER BY id DESC
                LIMIT 100
            """).fetchall()

            history = []

            for row in rows:
                history.append({
                    "patient_name": row["patient_name"],
                    "age": row["age"],
                    "result": row["result"],
                    "probability": (
                        round(row["probability"], 2)
                        if row["probability"] is not None
                        else None
                    ),
                    "created_at": row["created_at"]
                })

        return jsonify({
            "success": True,
            "total": total,
            "higher_risk": higher_risk,
            "lower_risk": lower_risk,
            "history": history
        })

    except Exception:
        app.logger.exception("Dashboard loading failed")

        return jsonify({
            "success": False,
            "error": "Could not load dashboard data."
        }), 500


# --------------------------------------------------
# 7. Run Flask application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
