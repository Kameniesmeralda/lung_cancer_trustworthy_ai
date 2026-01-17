import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.config import TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE, NUM_WORKERS, IMG_SIZE


# ---------------------------
# TRANSFORMATIONS IMAGES
# ---------------------------
def get_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # Convertir en niveaux de gris
        transforms.Resize(IMG_SIZE),                   # Redimensionner en 224x224
        transforms.ToTensor(),                         # Convertir en tenseur PyTorch
        transforms.Normalize(mean=[0.5], std=[0.5])    # Normalisation standard
    ])


# ---------------------------
# DATASETS
# ---------------------------
def get_datasets():
    transform = get_transforms()

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

    # Test dataset optionnel
    test_dataset = None
    if os.path.exists(TEST_DIR):
        test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

    return train_dataset, val_dataset, test_dataset


# ---------------------------
# DATALOADERS
# ---------------------------
def get_dataloaders():
    train_dataset, val_dataset, test_dataset = get_datasets()
    print("class_to_idx:", train_dataset.class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS
        )

    return train_loader, val_loader, test_loader
