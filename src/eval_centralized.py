import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

from src.model import SimpleLungCNN
from src.config import DEVICE, NUM_CLASSES, DATA_DIR

# -------------------------------------------------------------------------
# 1. Chargement des données de test
# -------------------------------------------------------------------------
def load_test_data():
    test_dir = os.path.join(DATA_DIR, "test")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return test_loader, test_dataset.classes


# -------------------------------------------------------------------------
# 2. Évaluation du modèle centralisé
# -------------------------------------------------------------------------
def evaluate(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    loss_total = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_total += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = loss_total / total
    accuracy = correct / total

    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


# -------------------------------------------------------------------------
# 3. Matrice de confusion
# -------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Centralized Model - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_centralized.png")
    plt.close()


# -------------------------------------------------------------------------
# 4. Programme principal
# -------------------------------------------------------------------------
def main():

    print("\n Chargement du modèle centralisé...")
    model_path = "results/model_centralized_baseline_5ep_lr1e-3.pth"

    model = SimpleLungCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    test_loader, class_names = load_test_data()

    print(" Évaluation du modèle centralisé...")
    loss, acc, labels, preds = evaluate(model, test_loader)

    print(f"\n Test Loss = {loss:.4f}")
    print(f" Test Accuracy = {acc*100:.2f}%")

    print("\n Génération de la matrice de confusion...")
    plot_confusion_matrix(labels, preds, class_names)

    print("\n Évaluation terminée. Fichier généré : results/confusion_matrix_centralized.png")


if __name__ == "__main__":
    main()
