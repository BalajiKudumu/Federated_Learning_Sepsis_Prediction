import flwr as fl
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set client ID
client_id = 1

# Load dataset
dataset = fetch_ucirepo(id=827)  # Heart Disease dataset
X = dataset.data.features
y = dataset.data.targets.values.ravel()

# Split dataset between clients
split_index = len(X) // 2
if client_id == 1:
    X, y = X[:split_index], y[:split_index]
else:
    X, y = X[split_index:], y[split_index:]

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flower client class
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = LogisticRegression()
        self.model_trained = False

    def get_parameters(self, config):
        try:
            return [self.model.coef_.flatten(), self.model.intercept_]
        except AttributeError:
            # Return default zero parameters before training
            return [np.zeros(X.shape[1]), np.zeros(1)]

    def set_parameters(self, parameters):
        self.model.coef_ = np.array(parameters[0]).reshape(1, -1)
        self.model.intercept_ = np.array([parameters[1]]) if np.isscalar(parameters[1]) else np.array(parameters[1])
        self.model.classes_ = np.array([0, 1])  # Required by sklearn to avoid predict errors

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Train the model
        self.model.fit(X_train, y_train)
        self.model_trained = True

        # Predict on a single sample
        sample = X_test[0].reshape(1, -1)
        pred = self.model.predict(sample)

        # Logging
        print(f"[Client {client_id}] Trained Coefficients: {self.model.coef_}")
        print(f"[Client {client_id}] Trained Intercept: {self.model.intercept_}")
        print(f"[Client {client_id}] Sample Prediction: {pred}")

        return (
            self.get_parameters(config),
            len(X_train),
            {
                "intercept": float(self.model.intercept_[0]),
                "sample_prediction": int(pred[0]),
                "coef_sample": float(self.model.coef_.flatten()[0])  # Just log one value
            }
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        if not self.model_trained:
            print(f"[Client {client_id}] Skipping evaluation, model not trained.")
            return 0.0, len(X_test), {}

        accuracy = self.model.score(X_test, y_test)
        print(f"[Client {client_id}] Evaluation Accuracy: {accuracy}")
        return float(accuracy), len(X_test), {}

# Start Flower client
fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())
