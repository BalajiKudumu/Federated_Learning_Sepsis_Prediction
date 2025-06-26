from flask import Flask, render_template, request
import numpy as np
import json
import os
import joblib

app = Flask(__name__)

def load_scaler_params():
    try:
        with open("webapp/scaler_params.json", "r") as f:
            params = json.load(f)
            return np.array(params["mean"]), np.array(params["scale"])
    except FileNotFoundError:
        return None, None

def load_model():
    try:
        model = joblib.load("webapp/rf_model.pkl")
        return model
    except FileNotFoundError:
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Read form inputs
            f1 = float(request.form["feature1"])
            f2 = float(request.form["feature2"])
            f3 = float(request.form["feature3"])

            # Create feature array
            features = np.array([[f1, f2, f3]])

            # Load scaler
            mean, scale = load_scaler_params()
            if mean is None or scale is None:
                prediction = "Scaler parameters not found. Please train the model."
                return render_template("index.html", prediction=prediction)

            # Scale features
            features_scaled = (features - mean) / scale

            # Load model
            model = load_model()
            if model is None:
                prediction = "Model not found. Please train the model."
                return render_template("index.html", prediction=prediction)

            # Make prediction
            pred_class = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0][1]

            # prediction = f"Sepsis Prediction: {pred_class} (Probability: {prob:.4f})"
            prediction = f"Sepsis Prediction: {pred_class}"
            print(f'Probability: {prob}')

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
