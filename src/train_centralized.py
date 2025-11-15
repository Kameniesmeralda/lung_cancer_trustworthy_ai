import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import torch
import torch.nn as nn
import torch.optim as optim

from src.model import SimpleLungCNN
from src.dataset import get_dataloaders
from src.config import EPOCHS, LEARNING_RATE, DEVICE


def train_one_epoch(model, loader, criterion, optimizer):
    """Entraîne le modèle sur **une seule époque**."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # prédiction = classe ayant la probabilité max
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100 * correct / total


def evaluate(model, loader, criterion):
    """Évalue le modèle sur la validation/test."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100 * correct / total


def main():
    print("📁 Chargement des données...")
    train_loader, val_loader, _ = get_dataloaders()

    print("🧠 Initialisation du modèle...")
    model = SimpleLungCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("🚀 Début de l'entraînement centralisé...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Époque {epoch+1}/{EPOCHS}")
        print(f"  🔹 Train : loss={train_loss:.4f}, acc={train_acc:.2f}%")
        print(f"  🔹 Val   : loss={val_loss:.4f}, acc={val_acc:.2f}%")

    # Sauvegarde du modèle
    torch.save(model.state_dict(), "results/model_centralized.pth")
    print("💾 Modèle sauvegardé dans results/model_centralized.pth")


if __name__ == "__main__":
    main()
