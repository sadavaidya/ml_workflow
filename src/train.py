import os
import json
import yaml
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_model(X_train, y_train, model_params):
    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    return {
        "rmse": rmse,
        "mae": mean_absolute_error(y_test, preds),
    }



def main():
    config = load_config("configs/train_config.yaml")

    # Load processed data
    df = pd.read_csv(config["paths"]["processed_data"])

    X = df.drop(columns=["charges"])
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
    )

    # Train model
    model = train_model(X_train, y_train, config["model"]["params"])

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(config["paths"]["model_output"]), exist_ok=True)

    # Save model
    joblib.dump(model, config["paths"]["model_output"])

    # Save metrics
    with open(config["paths"]["metrics_output"], "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
