import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.dataset import get_dataloaders
from src.model import SimpleLungCNN
from src.config import DEVICE, NUM_CLASSES, LEARNING_RATE as CFG_LR


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_one_run(epochs: int, lr: float, run_name: str):
    print(f"📁 Chargement des données...")
    train_loader, val_loader, _ = get_dataloaders()

    print(f"🧠 Initialisation du modèle (lr={lr}, epochs={epochs})...")
    model = SimpleLungCNN(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print("🚀 Début de l'entraînement centralisé...")

    for epoch in range(1, epochs + 1):
        # ---------- TRAIN ----------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # ---------- VALIDATION ----------
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_total += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss_total / val_total
        val_acc = val_correct / val_total

        # ---------- LOG ----------
        print(f"Époque {epoch}/{epochs}")
        print(f"  🔹 Train : loss={train_loss:.4f}, acc={train_acc*100:.2f}%")
        print(f"  🔹 Val   : loss={val_loss:.4f}, acc={val_acc*100:.2f}%")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    # ---------- SAUVEGARDE DU MODÈLE ----------
    model_path = os.path.join(RESULTS_DIR, f"model_centralized_{run_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f" Modèle sauvegardé dans {model_path}")

    # ---------- SAUVEGARDE DE L'HISTORIQUE ----------
    history_path = os.path.join(RESULTS_DIR, f"history_centralized_{run_name}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f" Historique sauvegardé dans {history_path}")

    # ---------- GRAPHIQUES ----------
    df = pd.DataFrame(history)

    # Courbe des pertes
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.lineplot(x="epoch", y="train_loss", data=df, label="Train loss")
    sns.lineplot(x="epoch", y="val_loss", data=df, label="Val loss")
    plt.title(f"Courbe de loss (run: {run_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    loss_fig_path = os.path.join(RESULTS_DIR, f"loss_curve_{run_name}.png")
    plt.savefig(loss_fig_path, bbox_inches="tight")
    plt.close()
    print(f" Courbe de loss sauvegardée dans {loss_fig_path}")

    # Courbe des accuracies
    plt.figure(figsize=(8, 5))
    sns.lineplot(x="epoch", y="train_acc", data=df, label="Train acc")
    sns.lineplot(x="epoch", y="val_acc", data=df, label="Val acc")
    plt.title(f"Courbe d'accuracy (run: {run_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    acc_fig_path = os.path.join(RESULTS_DIR, f"accuracy_curve_{run_name}.png")
    plt.savefig(acc_fig_path, bbox_inches="tight")
    plt.close()
    print(f" Courbe d'accuracy sauvegardée dans {acc_fig_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Nombre d'époques pour l'entraînement centralisé",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=CFG_LR,
        help="Learning rate (par défaut: valeur du config.py)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Nom du run (pour distinguer les expériences)",
    )

    args = parser.parse_args()

    # Si aucun run_name n'est donné, on en génère un avec la date/heure
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"run_{timestamp}"

    return args


if __name__ == "__main__":
    args = parse_args()
    train_one_run(epochs=args.epochs, lr=args.lr, run_name=args.run_name)
