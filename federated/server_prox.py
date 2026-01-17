import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays

import torch
from torchvision import datasets, transforms

from src.model import SimpleLungCNN
from src.config import DEVICE, IMG_SIZE


# ---------------------------------------------------------------------
# Compute GLOBAL class weights (all clients merged)
# ---------------------------------------------------------------------
def compute_global_class_weights():
    train_root = "data"
    counts = {}

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    last_dataset = None

    for d in os.listdir(train_root):
        if d.startswith("client_"):
            train_path = os.path.join(train_root, d, "train")
            dataset = datasets.ImageFolder(train_path, transform=transform)
            last_dataset = dataset

            for _, label in dataset:
                counts[label] = counts.get(label, 0) + 1

    if last_dataset is None or len(counts) == 0:
        raise RuntimeError("No client_* train folders found under ./data")

    print("\n✅ GLOBAL TRAIN counts (all clients merged):")
    for k in sorted(counts.keys()):
        print(f"  idx={k} -> {last_dataset.classes[k]}: {counts[k]}")

    total = sum(counts.values())
    num_classes = len(last_dataset.classes)

    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]

    print("\n✅ GLOBAL class weights (same order as class_to_idx):")
    for i, w in enumerate(weights):
        print(f"  w[{i}] for {last_dataset.classes[i]} = {w:.4f}")

    return weights


# ---------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------
def aggregate_fit_metrics(metrics):
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    loss = sum(n * m.get("loss", 0.0) for n, m in metrics) / total
    ce = sum(n * m.get("ce_loss", 0.0) for n, m in metrics) / total
    return {"loss": float(loss), "ce_loss": float(ce)}


def aggregate_eval_metrics(metrics):
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    acc = sum(n * m.get("accuracy", 0.0) for n, m in metrics) / total
    return {"accuracy": float(acc)}


# ---------------------------------------------------------------------
# Custom FedAvg (store last parameters)
# ---------------------------------------------------------------------
class MyFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_parameters = None

    def aggregate_fit(self, rnd, results, failures):
        params, metrics = super().aggregate_fit(rnd, results, failures)
        self.last_parameters = params
        return params, metrics


# ---------------------------------------------------------------------
# Start server (FedProx via config + client proximal term)
# ---------------------------------------------------------------------
def start_server(num_rounds=50, proximal_mu=0.1):
    print(f"🚀 Starting Federated Server (FedProx-style) | mu={proximal_mu}")

    class_weights = compute_global_class_weights()
    class_weights_json = json.dumps(class_weights)

    strategy = MyFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        on_fit_config_fn=lambda rnd: {
            "local_epochs": 1,
            "class_weights_json": class_weights_json,
            "proximal_mu": float(proximal_mu),
        },
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )

    print("💾 Saving final global FedProx model...")

    if strategy.last_parameters is None:
        print("❌ ERROR: No final parameters captured (means FIT never succeeded).")
        return

    ndarrays = parameters_to_ndarrays(strategy.last_parameters)

    model = SimpleLungCNN().to(DEVICE)
    state_dict = {}
    for (k, _), v in zip(model.state_dict().items(), ndarrays):
        state_dict[k] = torch.tensor(v)

    model.load_state_dict(state_dict)

    os.makedirs("results", exist_ok=True)
    out_path = f"results/model_federated_fedprox_mu{proximal_mu}_rounds{num_rounds}.pth"
    torch.save(model.state_dict(), out_path)
    print(f"✅ Model saved: {out_path}")


if __name__ == "__main__":
    # Option A asked: mu ~ 0.01
    start_server(num_rounds=50, proximal_mu=0.1)
