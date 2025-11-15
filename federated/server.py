import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
from flwr.server import ServerConfig
import torch
from src.model import SimpleLungCNN
from src.config import NUM_CLASSES, DEVICE


# ---------------------------------------------------------------------
# 1. Fonction pour récupérer les poids d'un modèle PyTorch
# ---------------------------------------------------------------------
def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


# ---------------------------------------------------------------------
# 2. Fonction pour charger des poids dans un modèle PyTorch
# ---------------------------------------------------------------------
def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------
# 3. Lancement du serveur FL
# ---------------------------------------------------------------------
def start_server(num_rounds: int = 3):
    """Lance un serveur FedAvg classique."""
    print("🚀 Lancement du serveur Federated Learning (FedAvg)")

    # Stratégie : FedAvg
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,        # Tous les clients participent à chaque round
        fraction_evaluate=1.0,
        min_fit_clients=3,       # Au moins 3 clients pour commencer
        min_available_clients=3, # On attend 3 clients
        min_evaluate_clients=3,
    )

    # Démarrer le serveur FL
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


if __name__ == "__main__":
    # Exemple : 3 rounds
    start_server(num_rounds=3)
