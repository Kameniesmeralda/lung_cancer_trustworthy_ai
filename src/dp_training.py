# src/dp_training.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from src.model import SimpleLungCNN
from src.config import DEVICE, IMG_SIZE, BATCH_SIZE, LEARNING_RATE, DATA_DIR, NUM_CLASSES


# -------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------
def build_resnet18(num_classes: int, in_channels: int = 1, pretrained: bool = False) -> nn.Module:
    """
    ResNet-18 adapted to grayscale.
    If pretrained=True, we load ImageNet weights and convert conv1 weights from 3ch -> 1ch.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace first conv to accept 1 channel
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    if pretrained:
        # Convert RGB conv weights -> grayscale by averaging channels
        with torch.no_grad():
            model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

    # Replace classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def looks_like_resnet_state_dict(sd: dict) -> bool:
    # Typical ResNet keys
    return any(k.startswith("layer1.") for k in sd.keys()) and any(k.startswith("conv1.") for k in sd.keys())


def load_model_auto(model_path: str, num_classes: int) -> nn.Module:
    """
    Loads either:
    - a torchvision ResNet-18 checkpoint (state_dict)
    - or a SimpleLungCNN checkpoint
    """
    ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint is not a state_dict dict.")

    if looks_like_resnet_state_dict(ckpt):
        model = build_resnet18(num_classes=num_classes, in_channels=1, pretrained=False)
        model.load_state_dict(ckpt, strict=True)
        return model

    # fallback: SimpleLungCNN
    model = SimpleLungCNN(num_classes=num_classes)
    model.load_state_dict(ckpt, strict=True)
    return model


# -------------------------------------------------------------------------
# DP helpers: gradient clipping + Gaussian noise
# NOTE: this is "DP-like" unless you use a full accountant (epsilon/delta).
# -------------------------------------------------------------------------
def dp_clip_and_noise(model: nn.Module, clip_norm: float, noise_multiplier: float):
    """
    1) Clip global grad norm
    2) Add Gaussian noise to grads: N(0, (noise_multiplier*clip_norm)^2)
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

    if noise_multiplier > 0:
        for p in model.parameters():
            if p.grad is None:
                continue
            noise = torch.randn_like(p.grad) * (noise_multiplier * clip_norm)
            p.grad.add_(noise)


# -------------------------------------------------------------------------
# Data loaders (central train/val)
# -------------------------------------------------------------------------
def get_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print("✅ class_to_idx:", train_ds.class_to_idx)
    print("✅ classes:", train_ds.classes)

    return train_loader, val_loader, train_ds.classes


# -------------------------------------------------------------------------
# Train / Eval
# -------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    loss_total, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        loss_total += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = loss_total / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def train_dp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    clip_norm: float,
    noise_multiplier: float,
    lr: float,
    weight_decay: float = 1e-4,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        running_loss, total = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            dp_clip_and_noise(model, clip_norm=clip_norm, noise_multiplier=noise_multiplier)

            optimizer.step()

            running_loss += loss.item() * x.size(0)
            total += y.size(0)

        train_loss = running_loss / total if total > 0 else 0.0
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        best_val = max(best_val, val_acc)
        print(
            f"Epoch {ep:02d}/{epochs} | "
            f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc*100:.2f}% | Best={best_val*100:.2f}%"
        )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="results/model_federated_weighted_global_50_rounds.pth",
        help="Path to starting model (ResNet-18 global is recommended).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--noise_multiplier", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="results/model_dp_like_resnet18_50r.pth")
    parser.add_argument("--save_meta", action="store_true", help="Save a small JSON with DP params.")
    args = parser.parse_args()

    print("🔐 DP-like Training (clip + noise) on a pretrained model (auto-arch: ResNet18 or CNN)")
    print(f"📦 Base model: {args.base_model}")
    print(f"⚙️ epochs={args.epochs}, clip_norm={args.clip_norm}, noise_multiplier={args.noise_multiplier}, lr={args.lr}")
    print(f"💾 Output: {args.out}")

    if not os.path.exists(args.base_model):
        raise FileNotFoundError(f"Base model not found: {args.base_model}")

    train_loader, val_loader, classes = get_loaders(batch_size=args.batch_size)

    model = load_model_auto(args.base_model, num_classes=NUM_CLASSES).to(DEVICE)
    model.eval()
    print(f"✅ Loaded model type: {type(model).__name__}")

    train_dp(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"✅ Saved DP-like model to: {args.out}")

    if args.save_meta:
        meta_path = os.path.splitext(args.out)[0] + "_meta.json"
        meta = {
            "base_model": args.base_model,
            "epochs": args.epochs,
            "clip_norm": args.clip_norm,
            "noise_multiplier": args.noise_multiplier,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "note": "DP-like clipping+noise without privacy accountant (no epsilon/delta computed).",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"🧾 Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
