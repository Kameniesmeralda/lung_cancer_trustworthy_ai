import os
import pydicom
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from src.config import DATA_DIR


# ------------------------------------------------------------------------------
# Fonction pour charger un fichier DICOM et le convertir en image numpy
# ------------------------------------------------------------------------------
def dicom_to_array(path):
    dicom = pydicom.dcmread(path)
    array = dicom.pixel_array.astype(np.float32)

    # Normalisation simple (0-255)
    array -= np.min(array)
    array /= np.max(array)
    array *= 255.0

    return array.astype(np.uint8)


# ------------------------------------------------------------------------------
# Fonction pour sauvegarder une image PNG
# ------------------------------------------------------------------------------
def save_png(array, output_path):
    cv2.imwrite(output_path, array)


# ------------------------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------------------------
def preprocess_lidc(root_lidc_folder):
    """
    root_lidc_folder : le chemin où tu vas dézipper le LIDC-IDRI original.
    Exemple: "C:/Users/.../LIDC-IDRI"
    """

    print("🔍 Scan du dataset LIDC-IDRI...")
    dicom_paths = []

    # On parcourt récursivement les dossiers LIDC-IDRI
    for root, dirs, files in os.walk(root_lidc_folder):
        for f in files:
            if f.endswith(".dcm"):
                dicom_paths.append(os.path.join(root, f))

    print(f"📁 {len(dicom_paths)} fichiers DICOM trouvés.")

    # On en prend un sous-échantillon pour Stage 1 si nécessaire (optionnel)
    # dicom_paths = dicom_paths[:2000]

    # Création des répertoires de sortie
    train_dir = os.path.join(DATA_DIR, "train", "nodule")
    nontrain_dir = os.path.join(DATA_DIR, "train", "normal")
    val_dir = os.path.join(DATA_DIR, "val", "nodule")
    nonval_dir = os.path.join(DATA_DIR, "val", "normal")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(nontrain_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(nonval_dir, exist_ok=True)

    # Partition du dataset pour train/val
    train_paths, val_paths = train_test_split(dicom_paths, test_size=0.2, random_state=42)

    print("🔧 Prétraitement des images...")

    # Note : Pour Stage 1, on considère *toute slice contenant un nodule comme 'nodule'*.
    # Pour le moment, nous n'utilisons pas les annotations complexes (XML).
    for idx, path in enumerate(train_paths):
        array = dicom_to_array(path)
        output_path = os.path.join(train_dir, f"img_{idx}.png")
        save_png(array, output_path)

        if idx % 200 == 0:
            print(f"  ➤ {idx} images train converties...")

    for idx, path in enumerate(val_paths):
        array = dicom_to_array(path)
        output_path = os.path.join(val_dir, f"img_{idx}.png")
        save_png(array, output_path)

        if idx % 200 == 0:
            print(f"  ➤ {idx} images val converties...")

    print("🎉 Conversion terminée !")
    print("Les images utilisables par PyTorch sont dans data/train/ et data/val/")
