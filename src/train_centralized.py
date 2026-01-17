import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.model import SimpleLungCNN
from src.config import TRAIN_DIR, VAL_DIR, IMG_SIZE, DEVICE, BATCH_SIZE, NUM_WORKERS, NUM_CLASSES

def get_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

def compute_class_weights(train_dataset: datasets.ImageFolder) -> torch.Tensor:
    targets = torch.tensor(train_dataset.targets)
    counts = torch.bincount(targets, minlength=len(train_dataset.classes)).float()
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.mean()  # normalize
    return weights

def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

def main():
    os.makedirs("results", exist_ok=True)

    transform = get_transforms()
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

    print("\n✅ class_to_idx (IMPORTANT):")
    print(train_dataset.class_to_idx)

    # Quick sanity check: label range
    print("\n✅ num_classes:", len(train_dataset.classes))
    assert len(train_dataset.classes) == NUM_CLASSES, \
        f"NUM_CLASSES={NUM_CLASSES} but ImageFolder found {len(train_dataset.classes)}"

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ----- Weighted loss (AUTO, correct order) -----
    class_weights = compute_class_weights(train_dataset).to(DEVICE)
    print("\n✅ Using Weighted CrossEntropyLoss with weights (same order as class_to_idx):")
    print(class_weights.detach().cpu().tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model = SimpleLungCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        # ---- TRAIN ----
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        n_train = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            train_acc_sum += accuracy_from_logits(logits, y) * bs
            n_train += bs

        train_loss = train_loss_sum / n_train
        train_acc = train_acc_sum / n_train

        # ---- VAL ----
        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        n_val = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)

                bs = x.size(0)
                val_loss_sum += loss.item() * bs
                val_acc_sum += accuracy_from_logits(logits, y) * bs
                n_val += bs

        val_loss = val_loss_sum / n_val
        val_acc = val_acc_sum / n_val

        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    # Save model + history
    torch.save(model.state_dict(), "results/model_centralized_weighted.pth")
    with open("results/history_centralized_weighted.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\n💾 Saved:")
    print("  - results/model_centralized_weighted.pth")
    print("  - results/history_centralized_weighted.json")

if __name__ == "__main__":
    main()
