import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np

from src.model import SimpleLungCNN
from src.config import DEVICE, DATA_DIR, IMG_SIZE


# -------------------------------------------------------------------------
# 1. Chargement des données de test
# -------------------------------------------------------------------------
def load_test_data():
    test_dir = os.path.join(DATA_DIR, "test")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),                   # (224,224)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),   # IMPORTANT: same as training
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print("✅ TEST class_to_idx:", test_dataset.class_to_idx)
    print("✅ TEST classes order:", test_dataset.classes)

    return test_loader, test_dataset.classes


# -------------------------------------------------------------------------
# 2. Évaluation du modèle
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

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = loss_total / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


# -------------------------------------------------------------------------
# 3. Matrice de confusion
# -------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Federated (Weighted Clients) - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------------------------------------------------------
# 4. Chargement + Évaluation
# -------------------------------------------------------------------------
def main():
    print("\n📁 Loading federated weighted model...")

    # ✅ CHANGE THIS to your actual file name produced by the server
    model_path = "results/model_federated_fedprox_mu0.001_rounds50.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Check your results/ folder and update model_path."
        )

    model = SimpleLungCNN(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    test_loader, class_names = load_test_data()

    print("\n📊 Evaluating on TEST split...")
    loss, acc, labels, preds = evaluate(model, test_loader)

    print(f"\n🎯 Test Loss = {loss:.4f}")
    print(f"🎯 Test Accuracy = {acc*100:.2f}%")

    print("\n🧾 Classification report:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    print("\n📈 Generating confusion matrix...")
    os.makedirs("results", exist_ok=True)
    out_path = "results/confusion_matrix_federated_fedprox_mu0.001_rounds50.png"
    plot_confusion_matrix(labels, preds, class_names, out_path)

    print(f"\n✔️ Done. Confusion matrix saved to: {out_path}")


if __name__ == "__main__":
    main()
