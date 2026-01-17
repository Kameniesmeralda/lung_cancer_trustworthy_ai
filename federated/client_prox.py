import sys
import os
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
from flwr.client import start_client

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.model import SimpleLungCNN
from src.config import LEARNING_RATE, NUM_CLASSES, DEVICE, IMG_SIZE, BATCH_SIZE


# -------------------------------------------------------------------------
# Load local data for a given client (client_1, client_2, ...)
# -------------------------------------------------------------------------
def load_client_data(client_id: int):
    train_path = f"data/client_{client_id}/train"
    val_path = f"data/client_{client_id}/val"

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    val_dataset = datasets.ImageFolder(val_path, transform=transform)

    print(f"✅ class_to_idx: {train_dataset.class_to_idx}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


# -------------------------------------------------------------------------
# Federated client (FedProx)
# -------------------------------------------------------------------------
class LungClientProx(fl.client.NumPyClient):
    def __init__(self, train_loader, val_loader):
        self.model = SimpleLungCNN(num_classes=NUM_CLASSES).to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=DEVICE) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def _update_loss_from_config(self, config):
        """Update CrossEntropyLoss(weight=.) if server provides weights."""
        cw_json = config.get("class_weights_json", None)
        if cw_json is None:
            self.criterion = nn.CrossEntropyLoss()
            return

        cw_list = json.loads(cw_json)
        weight_tensor = torch.tensor(cw_list, dtype=torch.float32, device=DEVICE)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    def fit(self, parameters, config):
        print("▶️ ClientProx: local training.")
        self.set_parameters(parameters)

        # weighted CE (optional)
        self._update_loss_from_config(config)

        # FedProx mu
        mu = float(config.get("proximal_mu", 0.0))

        # snapshot of global params w_global (after receiving them)
        global_params = [p.detach().clone() for p in self.model.parameters()]

        self.model.train()
        epochs = int(config.get("local_epochs", 1))

        running_ce = 0.0
        running_total = 0.0
        total = 0

        for _ in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(images)

                ce_loss = self.criterion(outputs, labels)

                # FedProx proximal penalty
                if mu > 0.0:
                    prox = 0.0
                    for p, g in zip(self.model.parameters(), global_params):
                        prox = prox + torch.sum((p - g) ** 2)
                    loss = ce_loss + (mu / 2.0) * prox
                else:
                    loss = ce_loss

                loss.backward()
                self.optimizer.step()

                bs = labels.size(0)
                running_ce += ce_loss.item() * bs
                running_total += loss.item() * bs
                total += bs

        avg_ce = running_ce / total if total > 0 else 0.0
        avg_total = running_total / total if total > 0 else 0.0

        print(f"✔️ Training done. CE={avg_ce:.4f} | Total(FedProx)={avg_total:.4f} | mu={mu}")
        return self.get_parameters(), total, {"ce_loss": float(avg_ce), "loss": float(avg_total), "mu": float(mu)}

    def evaluate(self, parameters, config):
        print("📊 ClientProx: local evaluation.")
        self.set_parameters(parameters)

        # same weighted CE for evaluation consistency
        self._update_loss_from_config(config)

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                bs = labels.size(0)
                running_loss += loss.item() * bs
                _, predicted = outputs.max(1)
                total += bs
                correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        print(f"   ➤ Val loss={avg_loss:.4f}, acc={acc*100:.2f}%")
        return float(avg_loss), total, {"accuracy": float(acc)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    args = parser.parse_args()

    print(f"🟦 ClientProx {args.client_id} started")
    train_loader, val_loader = load_client_data(args.client_id)

    start_client(
        server_address="127.0.0.1:8080",
        client=LungClientProx(train_loader, val_loader).to_client(),
    )
