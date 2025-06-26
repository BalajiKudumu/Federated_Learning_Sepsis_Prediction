import joblib
import numpy as np
import json

# Load model
model = joblib.load("webapp/rf_model.pkl")

# Load scaler params
with open("webapp/scaler_params.json") as f:
    scaler_params = json.load(f)
mean = np.array(scaler_params["mean"])
scale = np.array(scaler_params["scale"])

# Inputs to test
test_inputs = [
    [20, 0.0, 0.0],
    [25, 0.1, 0.2],
    [15, 0.0, 0.0]
]

for inp in test_inputs:
    X = np.array([inp])
    X_scaled = (X - mean) / scale
    prob = model.predict_proba(X_scaled)[0][1]
    print(f"Input: {inp} -> Prob: {prob:.4f}")
