import torch
from torchvision import models
import os

def download_resnet():
    print("⏳ Téléchargement du modèle ResNet-18 en cours...")
    # Ceci télécharge le fichier dans C:\Users\kamen\.cache\torch\hub\checkpoints\
    try:
        models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        print("✅ Modèle téléchargé avec succès et vérifié.")
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement : {e}")

if __name__ == "__main__":
    download_resnet()