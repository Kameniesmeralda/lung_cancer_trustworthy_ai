import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays

import torch

from src.model import SimpleLungCNN
from src.config import DEVICE


# ---------------------------------------------------------------------
# Agrégation pondérée des métriques
# ---------------------------------------------------------------------
def aggregate_fit_metrics(metrics):
    if len(metrics) == 0:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    avg_loss = sum(num_examples * m.get("loss", 0.0) for num_examples, m in metrics) / total_examples
    return {"loss": float(avg_loss)}


def aggregate_eval_metrics(metrics):
    if len(metrics) == 0:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    avg_acc = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics) / total_examples
    return {"accuracy": float(avg_acc)}


# ---------------------------------------------------------------------
# Custom FedAvg pour récupérer les derniers paramètres globaux
# ---------------------------------------------------------------------
class MyFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_parameters = None   # stockage des poids globaux

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(rnd, results, failures)

        # Sauvegarde des paramètres de ce round
        self.last_parameters = aggregated_parameters

        return aggregated_parameters, metrics


# ---------------------------------------------------------------------
# 3. Lancement du serveur FL
# ---------------------------------------------------------------------
def start_server(num_rounds: int = 5):

    print(" Lancement du serveur Federated Learning (FedAvg)")

    strategy = MyFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,

        on_fit_config_fn=lambda rnd: {"local_epochs": 1},
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
    )

    # Lancement serveur
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )

    # -----------------------------------------------------------------
    # Sauvegarde du modèle global après le dernier round
    # -----------------------------------------------------------------
    print(" Sauvegarde du modèle fédéré global...")

    final_parameters = strategy.last_parameters  # OK pour ta version

    if final_parameters is None:
        print(" ERREUR : Aucune information de paramètres récupérée.")
        return

    final_ndarrays = parameters_to_ndarrays(final_parameters)

    global_model = SimpleLungCNN().to(DEVICE)

    # chargement manuel
    state_dict = {}
    for (key, _), array in zip(global_model.state_dict().items(), final_ndarrays):
        state_dict[key] = torch.tensor(array)

    global_model.load_state_dict(state_dict)

    os.makedirs("results", exist_ok=True)
    torch.save(global_model.state_dict(), "results/model_federated_round5.pth")

    print(" Modèle fédéré sauvegardé dans results/model_federated_round5.pth")


if __name__ == "__main__":
    start_server(num_rounds=5)
