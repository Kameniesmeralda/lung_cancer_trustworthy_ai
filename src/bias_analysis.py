import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.model import SimpleLungCNN
from src.config import DEVICE, IMG_SIZE, NUM_CLASSES


# ------------------------------------------------------------
# Config (edit if needed)
# ------------------------------------------------------------
DEFAULT_MODEL_PATH = "results/model_federated_fedprox_mu0.001_rounds50.pth"
CLIENTS_ROOT = "data"  # expects data/client_1/val, ..., data/client_5/val
NUM_CLIENTS = 5
BATCH_SIZE = 16


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def get_transform():
    # IMPORTANT: same normalization as your training
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def load_client_val_loader(client_id: int):
    val_path = os.path.join(CLIENTS_ROOT, f"client_{client_id}", "val")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing folder: {val_path}")

    ds = datasets.ImageFolder(val_path, transform=get_transform())
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return dl, ds


@torch.no_grad()
def eval_on_loader(model, loader):
    model.eval()
    all_logits = []
    all_labels = []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)

        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    preds = logits.argmax(axis=1)

    return labels, preds


def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0


def plot_bar(values_dict, title, ylabel, out_path):
    # values_dict: {"client_1": 0.5, ...}
    keys = list(values_dict.keys())
    vals = [values_dict[k] for k in keys]

    plt.figure()
    plt.bar(keys, vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(model_path: str = DEFAULT_MODEL_PATH):
    os.makedirs("results", exist_ok=True)

    print("📁 Bias/Fairness analysis per client")
    print(f"🔎 Loading model: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"➡️ Put the correct path in DEFAULT_MODEL_PATH or pass your path."
        )

    # Load model
    model = SimpleLungCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # Collect per-client metrics
    results = {
        "model_path": model_path,
        "clients": {},
        "fairness_gaps": {}
    }

    accuracy_per_client = {}
    macrof1_per_client = {}

    # We'll also store per-class F1 to identify class bias
    perclass_f1_per_client = {}

    for cid in range(1, NUM_CLIENTS + 1):
        print(f"\n--- CLIENT {cid} ---")

        loader, dataset = load_client_val_loader(cid)
        class_names = dataset.classes  # order used by ImageFolder

        labels, preds = eval_on_loader(model, loader)

        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

        # Store
        client_key = f"client_{cid}"
        accuracy_per_client[client_key] = float(acc)
        macrof1_per_client[client_key] = float(macro_f1)
        perclass_f1_per_client[client_key] = {
            class_names[i]: float(per_class_f1[i]) for i in range(len(class_names))
        }

        results["clients"][client_key] = {
            "num_val_samples": int(len(dataset)),
            "class_order": class_names,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class_f1": perclass_f1_per_client[client_key],
            "classification_report": classification_report(
                labels, preds, target_names=class_names, digits=4, zero_division=0
            ),
        }

        print(f"✅ Val samples: {len(dataset)}")
        print(f"✅ Accuracy: {acc*100:.2f}%")
        print(f"✅ Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f}")

    # --------------------------------------------------------
    # Task 5.3: Fairness gaps (max-min)
    # --------------------------------------------------------
    acc_vals = list(accuracy_per_client.values())
    f1_vals = list(macrof1_per_client.values())

    acc_gap = (max(acc_vals) - min(acc_vals)) if acc_vals else 0.0
    f1_gap = (max(f1_vals) - min(f1_vals)) if f1_vals else 0.0

    results["fairness_gaps"] = {
        "accuracy_gap": float(acc_gap),
        "macro_f1_gap": float(f1_gap),
        "best_accuracy_client": max(accuracy_per_client, key=accuracy_per_client.get),
        "worst_accuracy_client": min(accuracy_per_client, key=accuracy_per_client.get),
        "best_macro_f1_client": max(macrof1_per_client, key=macrof1_per_client.get),
        "worst_macro_f1_client": min(macrof1_per_client, key=macrof1_per_client.get),
    }

    print("\n==============================")
    print("📌 FAIRNESS GAP SUMMARY ")
    print("==============================")
    print(f"Accuracy gap (max-min): {acc_gap*100:.2f}%")
    print(f"Macro-F1 gap (max-min): {f1_gap:.4f}")
    print("Best/Worst accuracy:",
          results["fairness_gaps"]["best_accuracy_client"],
          "/",
          results["fairness_gaps"]["worst_accuracy_client"])
    print("Best/Worst macro-F1:",
          results["fairness_gaps"]["best_macro_f1_client"],
          "/",
          results["fairness_gaps"]["worst_macro_f1_client"])

    # --------------------------------------------------------
    # Task 5.4: Plots
    # --------------------------------------------------------
    acc_plot_path = "results/bias_accuracy_per_client_fedprox_mu0.001_rounds50.png"
    f1_plot_path = "results/bias_macro_f1_per_client_fedprox_mu0.001_rounds50.png"

    plot_bar(
        accuracy_per_client,
        title="Federated Global Model — Accuracy per Client (Validation) for the fedprox_mu0.001_rounds50",
        ylabel="Accuracy",
        out_path=acc_plot_path,
    )

    plot_bar(
        macrof1_per_client,
        title="Federated Global Model — Macro-F1 per Client (Validation) for the fedprox_mu0.001_rounds50",
        ylabel="Macro-F1",
        out_path=f1_plot_path,
    )

    print("\n📈 Plots saved:")
    print(f"  - {acc_plot_path}")
    print(f"  - {f1_plot_path}")

    # Save JSON summary
    out_json = "results/bias_analysis_per_client_federated_fedprox_mu0.001_rounds50.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Full report saved to: {out_json}")
    print("🎉 Done.")


if __name__ == "__main__":
    # Optional: allow overriding model path via env var or argument later
    main()

