import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders
from src.model import SimpleLungCNN
from src.config import DEVICE, NUM_CLASSES


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc, all_labels, all_preds


def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Centralized Weighted)")
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # numbers
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    print("\n📁 Loading weighted centralized model...")
    model = SimpleLungCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load("results/model_centralized_weighted.pth", map_location=DEVICE))

    # Load test loader + also get class names order from ImageFolder
    train_loader, val_loader, test_loader = get_dataloaders()

    if test_loader is None:
        raise ValueError("TEST_DIR not found or empty. Make sure data/test exists.")

    # IMPORTANT: same order as ImageFolder
    class_names = test_loader.dataset.classes
    print("✅ class order (from test ImageFolder):", class_names)

    # For evaluation we can use normal CE (weights not required for test measurement)
    criterion = nn.CrossEntropyLoss()

    print("\n📊 Evaluating on TEST split...")
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion)

    print(f"\n🎯 Test Loss = {test_loss:.4f}")
    print(f"🎯 Test Accuracy = {test_acc*100:.2f}%")

    # Confusion matrix + report
    cm = confusion_matrix(y_true, y_pred)
    os.makedirs("results", exist_ok=True)
    out_cm = "results/confusion_matrix_centralized_weighted.png"
    plot_confusion_matrix(cm, class_names, out_cm)

    print("\n🧾 Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print(f"\n✔️ Done. Confusion matrix saved to: {out_cm}")
