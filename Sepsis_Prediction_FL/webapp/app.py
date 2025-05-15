from flask import Flask, render_template, request
import numpy as np
import json
import os

app = Flask(__name__)

# Load model parameters
def load_model_params():
    try:
        with open("model_params.json", "r") as f:
            params = json.load(f)
            coef = np.array(params["coef"]).reshape(1, -1)
            intercept = np.array(params["intercept"])
            return coef, intercept
    except FileNotFoundError:
        return None, None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            f1 = float(request.form["feature1"])
            f2 = float(request.form["feature2"])
            f3 = float(request.form["feature3"])
            features = np.array([[f1, f2, f3]])

            coef, intercept = load_model_params()
            if coef is None or intercept is None:
                prediction = "Model not yet trained."
            else:
                logits = np.dot(features, coef.T) + intercept
                prob = 1 / (1 + np.exp(-logits))
                print(prob)
                predicted_class = int(prob[0][0] >= 0.5)
                prediction = f"Sepsis Prediction: {predicted_class} (Probability: {prob[0][0]:.4f})"
                
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
