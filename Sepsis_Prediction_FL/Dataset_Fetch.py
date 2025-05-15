from ucimlrepo import fetch_ucirepo
import pandas as pd

# Load the dataset with ID 827 (Sepsis Dataset)
dataset = fetch_ucirepo(id=827)

# Features and targets
X = dataset.data.features
y = dataset.data.targets

# Show the first few rows
print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())

# Metadata (optional)
print("\nDataset Metadata:")
print(f"Name: {dataset.metadata.name}")
print(f"Number of instances: {dataset.metadata.num_instances}")
print(f"Number of features: {dataset.metadata.num_features}")
print(f"Target feature: {dataset.metadata.target}")
