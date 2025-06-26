import flwr as fl
import numpy as np
import json
import os
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Set client ID
client_id = 1

# Load dataset
dataset = fetch_ucirepo(id=827)  # Sepsis dataset
X = dataset.data.features
y = dataset.data.targets.values.ravel()

# You can select specific features if needed
# X = X[['age_years', 'some_other_feature', 'another_feature']]

# Split data between clients
split_index = len(X) // 2
if client_id == 1:
    X, y = X[:split_index], y[:split_index]
else:
    X, y = X[split_index:], y[split_index:]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler parameters
scaler_params = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist()
}
os.makedirs("webapp", exist_ok=True)
with open("webapp/scaler_params.json", "w") as f:
    json.dump(scaler_params, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f'BINCount: {np.bincount(y_train)}')

# Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100,class_weight="balanced", random_state=42)
        self.model_trained = False

    def get_parameters(self, config):
        # Not meaningful for RF; return dummy
        return [np.zeros(X_train.shape[1])]

    def set_parameters(self, parameters):
        # Not applicable for RF
        pass

    def fit(self, parameters, config):
        # Train RF model
        self.model.fit(X_train, y_train)
        self.model_trained = True

        # Save the model
        joblib.dump(self.model, "webapp/rf_model.pkl")
        print(f"[Client {client_id}] Model saved to webapp/rf_model.pkl")

        # Predict a sample for log
        sample = X_test[0].reshape(1, -1)
        pred = self.model.predict(sample)
        prob = self.model.predict_proba(sample)[0][1]

        print(f"[Client {client_id}] Sample Prediction: {pred[0]} (Prob: {prob:.4f})")

        return self.get_parameters(config), len(X_train), {
            "sample_prediction": int(pred[0]),
            "sample_prob": float(prob)
        }

    def evaluate(self, parameters, config):
        if not self.model_trained:
            print(f"[Client {client_id}] Skipping evaluation, model not trained.")
            return 0.0, len(X_test), {}

        acc = self.model.score(X_test, y_test)
        print(f"[Client {client_id}] Evaluation accuracy: {acc:.4f}")
        return float(acc), len(X_test), {}

# Start Flower client
fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())
