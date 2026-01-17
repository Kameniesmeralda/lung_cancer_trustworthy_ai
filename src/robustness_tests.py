# src/robustness_tests.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

from src.model import SimpleLungCNN
from src.config import DEVICE, DATA_DIR, IMG_SIZE, NUM_CLASSES


# ----------------------------
# Model loading (auto)
# ----------------------------
def build_resnet18(num_classes: int, in_channels: int = 1, pretrained: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

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
        with torch.no_grad():
            model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def looks_like_resnet_state_dict(sd: dict) -> bool:
    return any(k.startswith("layer1.") for k in sd.keys()) and any(k.startswith("conv1.") for k in sd.keys())


def load_model_auto(model_path: str) -> nn.Module:
    ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint is not a state_dict dict.")

    if looks_like_resnet_state_dict(ckpt):
        model = build_resnet18(num_classes=NUM_CLASSES, in_channels=1, pretrained=False)
        model.load_state_dict(ckpt, strict=True)
        return model

    model = SimpleLungCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt, strict=True)
    return model


# ----------------------------
# Data
# ----------------------------
def load_test_loader(batch_size: int = 16) -> Tuple[DataLoader, List[str]]:
    test_dir = os.path.join(DATA_DIR, "test")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("✅ TEST class_to_idx:", test_dataset.class_to_idx)
    print("✅ TEST classes order:", test_dataset.classes)
    return test_loader, test_dataset.classes


# ----------------------------
# Perturbations (on normalized tensors)
# ----------------------------
def apply_gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return x
    noise = torch.randn_like(x) * std
    y = x + noise
    return torch.clamp(y, -2.5, 2.5)


def apply_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    k = max(3, int(2 * round(3 * sigma) + 1))
    if k % 2 == 0:
        k += 1
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])


def apply_brightness(x: torch.Tensor, factor: float) -> torch.Tensor:
    if abs(factor - 1.0) < 1e-8:
        return x
    x01 = (x * 0.5) + 0.5
    x01 = torch.clamp(x01, 0.0, 1.0)
    x01 = torch.clamp(x01 * factor, 0.0, 1.0)
    x_norm = (x01 - 0.5) / 0.5
    return x_norm


# ----------------------------
# Evaluation
# ----------------------------
@torch.no_grad()
def eval_model_under_perturbation(
    model: nn.Module,
    loader: DataLoader,
    perturbation: str,
    level: float,
) -> float:
    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        if perturbation == "noise":
            x = apply_gaussian_noise(x, std=level)
        elif perturbation == "blur":
            x = apply_blur(x, sigma=level)
        elif perturbation == "brightness":
            x = apply_brightness(x, factor=level)
        elif perturbation == "clean":
            pass
        else:
            raise ValueError(f"Unknown perturbation: {perturbation}")

        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / total if total > 0 else 0.0


def plot_curve(levels: List[float], acc_dict: Dict[str, float], title: str, out_path: str):
    xs = levels
    ys = [acc_dict[str(lv)] for lv in levels]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Severity")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_robustness_suite(
    model_path: str,
    test_loader: DataLoader,
    out_dir: str,
    tag: str,
    noise_levels: List[float],
    blur_levels: List[float],
    bright_levels: List[float],
) -> Dict:
    print(f"\n🧪 Robustness tests for: {model_path}")
    model = load_model_auto(model_path).to(DEVICE)
    print(f"✅ Loaded model type: {type(model).__name__}")

    results = {
        "model_path": model_path,
        "tag": tag,
        "accuracy": {"clean": None, "noise": {}, "blur": {}, "brightness": {}},
        "levels": {"noise": noise_levels, "blur": blur_levels, "brightness": bright_levels},
    }

    acc_clean = eval_model_under_perturbation(model, test_loader, "clean", 0.0)
    results["accuracy"]["clean"] = float(acc_clean)
    print(f"✅ Clean accuracy: {acc_clean*100:.2f}%")

    for lv in noise_levels:
        acc = eval_model_under_perturbation(model, test_loader, "noise", lv)
        results["accuracy"]["noise"][str(lv)] = float(acc)
        print(f"   noise std={lv:<4} -> acc={acc*100:.2f}%")

    for lv in blur_levels:
        acc = eval_model_under_perturbation(model, test_loader, "blur", lv)
        results["accuracy"]["blur"][str(lv)] = float(acc)
        print(f"   blur sigma={lv:<4} -> acc={acc*100:.2f}%")

    for lv in bright_levels:
        acc = eval_model_under_perturbation(model, test_loader, "brightness", lv)
        results["accuracy"]["brightness"][str(lv)] = float(acc)
        print(f"   brightness x{lv:<4} -> acc={acc*100:.2f}%")

    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"robustness_{tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Saved: {json_path}")

    plot_curve(noise_levels, results["accuracy"]["noise"], f"Robustness ({tag}) — Gaussian Noise",
               os.path.join(out_dir, f"robustness_noise_{tag}.png"))
    plot_curve(blur_levels, results["accuracy"]["blur"], f"Robustness ({tag}) — Gaussian Blur",
               os.path.join(out_dir, f"robustness_blur_{tag}.png"))
    plot_curve(bright_levels, results["accuracy"]["brightness"], f"Robustness ({tag}) — Brightness Shift",
               os.path.join(out_dir, f"robustness_brightness_{tag}.png"))

    print(f"📈 Plots saved in: {out_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Robustness tests for models (noise/blur/brightness)")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "results/model_federated_weighted_global_15_rounds.pth",
            "results/model_federated_weighted_global_50_rounds.pth",
            "results/model_dp_like_nm02.pth",
            "results/model_dp_like_nm08.pth",
        ],
        help="List of model paths to evaluate",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="results")

    # Severity levels
    parser.add_argument("--noise_levels", nargs="*", type=float, default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--blur_levels", nargs="*", type=float, default=[0.0, 0.8, 1.5, 2.5])
    parser.add_argument("--brightness_levels", nargs="*", type=float, default=[0.7, 0.85, 1.0, 1.15, 1.3])

    args = parser.parse_args()

    test_loader, _ = load_test_loader(batch_size=args.batch_size)

    all_results = []
    for p in args.models:
        if not os.path.exists(p):
            print(f"⚠️ Skipping (not found): {p}")
            continue
        tag = os.path.splitext(os.path.basename(p))[0]
        all_results.append(
            run_robustness_suite(
                model_path=p,
                test_loader=test_loader,
                out_dir=args.out_dir,
                tag=tag,
                noise_levels=args.noise_levels,
                blur_levels=args.blur_levels,
                bright_levels=args.brightness_levels,
            )
        )

    combined_path = os.path.join(args.out_dir, "robustness_all_models.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Combined robustness report saved to: {combined_path}")


if __name__ == "__main__":
    main()
