import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flwr as fl
from flwr.client import start_client
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.model import SimpleLungCNN
from src.config import LEARNING_RATE, NUM_CLASSES, DEVICE, IMG_SIZE, BATCH_SIZE


# -------------------------------------------------------------------------
# Charger les données locales d'un client donné (client_1, client_2, etc.)
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


# -------------------------------------------------------------------------
# Définition du client Federated Learning
# -------------------------------------------------------------------------
class LungClient(fl.client.NumPyClient):

    def __init__(self, train_loader, val_loader):
        self.model = SimpleLungCNN(num_classes=NUM_CLASSES).to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config=None):
        print("▶️ Client : Entraînement local...")
        self.set_parameters(parameters)

        self.model.train()
        for images, labels in self.train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

        print("✔️ Entraînement terminé.")
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config=None):
        print("📊 Client : Évaluation locale...")
        self.set_parameters(parameters)
        self.model.eval()

        correct = 0
        total = 0
        loss_total = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_total += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = correct / total
        return float(loss_total / total), total, {"accuracy": accuracy}


# -------------------------------------------------------------------------
# Main : lire le client_id et démarrer le client
# -------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    args = parser.parse_args()

    print(f"🟦 Client {args.client_id} démarré")

    train_loader, val_loader = load_client_data(args.client_id)

    start_client(
        server_address="127.0.0.1:8080",
        client=LungClient(train_loader, val_loader).to_client(),
    )
