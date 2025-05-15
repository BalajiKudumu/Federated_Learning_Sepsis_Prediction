import flwr as fl
import numpy as np
import sys
from flwr.server.strategy import FedAvg, FedMedian, FedAdagrad, FedYogi, FedAdam
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import os
import json
# --- Strategy Selector ---
def get_strategy(strategy_name: str):
    strategy_map = {
        "fedavg": FedAvg,
        "fedmedian": FedMedian,
        "fedadagrad": FedAdagrad,
        "fedyogi": FedYogi,
        "fedadam": FedAdam,
    }

    strategy_cls = strategy_map.get(strategy_name.lower())
    if not strategy_cls:
        raise ValueError(f"Unsupported strategy: {strategy_name}")

    # Create initial parameters (dummy params to begin with)
    initial_params = [np.zeros(10), np.zeros(1)]  # Example: 10 features + intercept

    # Convert to Flower parameter format
    initial_parameters = ndarrays_to_parameters(initial_params)

    # Extend selected strategy with logging
    class LoggingStrategy(strategy_cls):
        def aggregate_fit(self, rnd, results, failures):
            aggregated_params, _ = super().aggregate_fit(rnd, results, failures)

            print(f"\n[Server] Round {rnd} Client Metrics:")
            for i, (client, fit_res) in enumerate(results):
                metrics = fit_res.metrics
                if metrics:
                    print(f"  Client {i+1} Metrics:")
                    print(f"    Coef: {metrics.get('coef')}")
                    print(f"    Intercept: {metrics.get('intercept')}")
                    print(f"    Prediction: {metrics.get('sample_prediction')}")
                else:
                    print(f"  Client {i+1}: No metrics received.")

            if aggregated_params is not None:
                ndarrays = parameters_to_ndarrays(aggregated_params)
                coef, intercept = ndarrays[0], ndarrays[1]
                print(f"\n[Server] Round {rnd} Aggregated Coefficients:\n{coef}")
                print(f"[Server] Round {rnd} Aggregated Intercept:\n{intercept}")

                # Save aggregated model parameters
                params_dict = {
                    "coef": coef.tolist(),
                    "intercept": intercept.tolist()
                }
                os.makedirs("webapp", exist_ok=True)
                with open("webapp/model_params.json", "w") as f:
                    json.dump(params_dict, f)
            else:
                print(f"\n[Server] Round {rnd}: No aggregated parameters received.")

            return aggregated_params, {}


    return LoggingStrategy(fraction_fit=1.0, min_fit_clients=2, min_available_clients=2, initial_parameters=initial_parameters)

# --- Main ---
if __name__ == "__main__":
    strategy_arg = sys.argv[1] if len(sys.argv) > 1 else "fedavg"
    print(f"[Server] Starting Federated Server with strategy: {strategy_arg}")

    strategy = get_strategy(strategy_arg)

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
