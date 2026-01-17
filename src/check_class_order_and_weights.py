import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torchvision import datasets, transforms
from src.config import TRAIN_DIR, IMG_SIZE

def get_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

def main():
    transform = get_transforms()
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)

    print("\n✅ class_to_idx (ORDER USED BY ImageFolder):")
    print(train_dataset.class_to_idx)

    # targets = liste des labels (0..C-1) dans l'ordre class_to_idx
    targets = torch.tensor(train_dataset.targets)
    num_classes = len(train_dataset.classes)

    counts = torch.bincount(targets, minlength=num_classes).float()
    print("\n✅ TRAIN counts per class index:")
    for cls_name, idx in train_dataset.class_to_idx.items():
        print(f"  idx={idx} -> {cls_name:45s} : {int(counts[idx].item())}")

    # poids inverses: plus rare -> plus grand
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.mean()  # normalisation (optionnel mais pratique)

    print("\n✅ class weights (SAME ORDER AS class_to_idx):")
    for cls_name, idx in train_dataset.class_to_idx.items():
        print(f"  w[{idx}] for {cls_name:45s} = {weights[idx].item():.4f}")

    print("\n➡️ Copy-paste this tensor into your training if you want:")
    print("class_weights =", weights.tolist())

if __name__ == "__main__":
    main()
