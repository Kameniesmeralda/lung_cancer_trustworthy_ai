import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from src.config import DEVICE, DATA_DIR, IMG_SIZE, NUM_CLASSES


# -----------------------------
# Utils: inverse normalization (for display)
# -----------------------------
def denormalize(img_tensor, mean=0.5, std=0.5):
    img = img_tensor.clone()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    return img


# -----------------------------
# Grad-CAM implementation
# -----------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate(self, x, class_idx=None):
        """
        x: tensor (1,C,H,W)
        returns: cam (H,W) in [0,1], pred_idx
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)

        pred_idx = int(torch.argmax(logits, dim=1).item())
        if class_idx is None:
            class_idx = pred_idx

        score = logits[:, class_idx]
        score.backward(retain_graph=False)

        grads = self.gradients           # (1,C,h,w)
        acts = self.activations          # (1,C,h,w)

        if grads is None or acts is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients. Check target layer.")

        weights = grads.mean(dim=(2, 3), keepdim=True)       # (1,C,1,1)
        cam = (weights * acts).sum(dim=1, keepdim=True)      # (1,1,h,w)
        cam = F.relu(cam)

        cam = cam.squeeze(0).squeeze(0)  # (h,w)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()  # (H,W)

        return cam, pred_idx


# -----------------------------
# Data loader (test split)
# -----------------------------
def load_test_loader(batch_size=1):
    test_dir = os.path.join(DATA_DIR, "test")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    test_ds = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    return test_loader, test_ds.classes


# -----------------------------
# Save Grad-CAM overlay
# -----------------------------
def save_overlay(image_tensor, cam, out_path, title="Grad-CAM"):
    """
    image_tensor: (1,H,W) normalized tensor
    cam: (H,W) numpy in [0,1]
    """
    img = denormalize(image_tensor).squeeze(0).cpu().numpy()  # (H,W)

    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="gray")
    plt.imshow(cam, alpha=0.45)  # heatmap overlay
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Build ResNet-18 (for grayscale)
# -----------------------------
def build_resnet18_for_grayscale(num_classes: int):
    model = resnet18(weights=None)

    # Replace first conv to accept 1-channel
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # Replace classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def extract_state_dict(ckpt):
    """
    Handles different checkpoint formats:
    - pure state_dict
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    - {"model": ...}
    """
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict", "model", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    # fallback: assume ckpt is already a state_dict
    return ckpt


def strip_prefixes(state_dict, prefixes=("module.", "model.", "net.", "backbone.")):
    """
    Removes common prefixes in state_dict keys.
    """
    out = {}
    for k, v in state_dict.items():
        new_k = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
                    changed = True
        out[new_k] = v
    return out


def looks_like_resnet(sd_keys):
    """
    Robust ResNet detection even with key prefixes removed.
    """
    # typical resnet keys contain layer1..layer4 and fc
    has_layers = any(k.startswith("layer1.") or k.startswith("layer2.") or k.startswith("layer3.") or k.startswith("layer4.") for k in sd_keys)
    has_fc = any(k.startswith("fc.") for k in sd_keys)
    has_conv1 = any(k.startswith("conv1.") for k in sd_keys)
    return (has_layers and has_fc) or (has_layers and has_conv1)


def load_model_from_checkpoint(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")
    sd = extract_state_dict(ckpt)
    sd = strip_prefixes(sd)

    # Detect ResNet-style checkpoint
    if looks_like_resnet(sd.keys()):
        model = build_resnet18_for_grayscale(NUM_CLASSES).to(DEVICE)

        missing, unexpected = model.load_state_dict(sd, strict=False)

        # On accepte strict=False, mais on affiche pour debug si besoin
        if len(missing) > 0 or len(unexpected) > 0:
            print("⚠️ load_state_dict(strict=False) report")
            if len(missing) > 0:
                print("  Missing keys:", missing[:20], "..." if len(missing) > 20 else "")
            if len(unexpected) > 0:
                print("  Unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")

        model.eval()
        return model

    # If not ResNet, raise with useful debug
    sample_keys = list(sd.keys())[:30]
    raise RuntimeError(
        "Checkpoint not recognized as ResNet-18 after normalization.\n"
        "Here are the first keys detected (after prefix stripping):\n"
        + "\n".join(sample_keys)
        + "\n\n➡️ This means your training used another model class. "
          "In that case, you must instantiate the SAME architecture used during training."
    )

# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs("results", exist_ok=True)

    model_path = "results/model_federated_weighted_global_50_rounds.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("📁 Loading federated global model (ResNet-18)...")
    model = load_model_from_checkpoint(model_path)

    # Best target layer for ResNet Grad-CAM (last conv in layer4)
    target_layer = model.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)

    test_loader, class_names = load_test_loader(batch_size=1)
    print("✅ Test classes order:", class_names)

    num_samples = 8
    print(f"🔥 Generating Grad-CAM for {num_samples} random test images...")

    for i, (x, y) in enumerate(test_loader):
        if i >= num_samples:
            break

        x = x.to(DEVICE)  # (1,1,H,W)
        true_idx = int(y.item())

        cam, pred_idx = gradcam.generate(x, class_idx=None)

        out_path = f"results/gradcam_resnet18_fed_global_50r_{i}.png"
        title = f"True: {class_names[true_idx]} | Pred: {class_names[pred_idx]}"
        save_overlay(x[0], cam, out_path, title=title)

        print(f"✅ Saved: {out_path} | {title}")

    gradcam.remove_hooks()
    print("🎉 Done. Grad-CAM overlays are in results/.")


if __name__ == "__main__":
    main()
