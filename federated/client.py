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
def load_client_data(client_id):
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

    return train_loader, val_loader, train_dataset.class_to_idx


# -------------------------------------------------------------------------
# Federated client
# -------------------------------------------------------------------------
class LungClient(fl.client.NumPyClient):
    def __init__(self, train_loader, val_loader):
        self.model = SimpleLungCNN(num_classes=NUM_CLASSES).to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # default (will be overwritten if weights provided)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def _update_loss_from_config(self, config):
        """Update CrossEntropyLoss(weight=...) if server provides weights."""
        cw_json = config.get("class_weights_json", None)
        if cw_json is None:
            self.criterion = nn.CrossEntropyLoss()
            return

        # cw_json is a string -> parse to list[float]
        cw_list = json.loads(cw_json)
        weight_tensor = torch.tensor(cw_list, dtype=torch.float32, device=DEVICE)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    def fit(self, parameters, config):
        print("▶️ Client: local training...")
        self.set_parameters(parameters)

        # ✅ apply (global) class weights for this round
        self._update_loss_from_config(config)

        self.model.train()
        epochs = int(config.get("local_epochs", 1))

        running_loss = 0.0
        total = 0

        for _ in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                total += labels.size(0)

        avg_loss = running_loss / total if total > 0 else 0.0
        print("✔️ Training done.")
        return self.get_parameters(), total, {"loss": float(avg_loss)}

    def evaluate(self, parameters, config):
        print("📊 Client: local evaluation...")
        self.set_parameters(parameters)

        # use same weighted loss for evaluation (consistent)
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

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0

        print(f"   ➤ Val loss={avg_loss:.4f}, acc={acc*100:.2f}%")
        return float(avg_loss), total, {"accuracy": float(acc)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    args = parser.parse_args()

    print(f"🟦 Client {args.client_id} started")

    train_loader, val_loader, _ = load_client_data(args.client_id)

    start_client(
        server_address="127.0.0.1:8080",
        client=LungClient(train_loader, val_loader).to_client(),
    )
