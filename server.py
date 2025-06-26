import flwr as fl
import sys
from flwr.server.strategy import FedAvg

# --- Strategy that logs client metrics ---
class LoggingStrategy(FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        _, metrics_aggregated = super().aggregate_fit(rnd, results, failures)

        print(f"\n[Server] Round {rnd} Client Metrics:")
        for i, (client_proxy, fit_res) in enumerate(results):
            metrics = fit_res.metrics
            if metrics:
                print(f"  Client {i+1}:")
                print(f"    Sample Prediction: {metrics.get('sample_prediction')}")
                print(f"    Sample Prob: {metrics.get('sample_prob'):.4f}")
            else:
                print(f"  Client {i+1}: No metrics received.")

        return None, metrics_aggregated

# --- Main ---
if __name__ == "__main__":
    strategy = LoggingStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2
    )

    print("[Server] Starting Federated Server for RandomForestClassifier (no aggregation)")
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=7),
        strategy=strategy,
    )
