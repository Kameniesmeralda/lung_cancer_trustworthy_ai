import os
import shutil
import random

# Dossier source
SOURCE_TRAIN = "data/train"
SOURCE_VAL = "data/val"

# Nombre de clients
NUM_CLIENTS = 5

# Dossier destination
DEST_DIR = "data"


def create_structure():
    """Créer la structure data/client_i/train/class et data/client_i/val/class"""
    classes = os.listdir(SOURCE_TRAIN)

    for client_id in range(1, NUM_CLIENTS + 1):
        for split in ["train", "val"]:
            for c in classes:
                path = os.path.join(DEST_DIR, f"client_{client_id}", split, c)
                os.makedirs(path, exist_ok=True)


def distribute_images():
    """Distribue les images de train/ et val/ entre les 5 clients."""
    for split in ["train", "val"]:
        print(f"📁 Distribution des images de {split}/ ...")
        classes = os.listdir(os.path.join("data", split))

        for c in classes:
            class_path = os.path.join("data", split, c)
            images = os.listdir(class_path)
            random.shuffle(images)

            # Découpage en 5 partitions équilibrées
            chunk_size = len(images) // NUM_CLIENTS

            for i in range(NUM_CLIENTS):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < NUM_CLIENTS - 1 else len(images)

                dest_folder = os.path.join(DEST_DIR, f"client_{i+1}", split, c)

                for img in images[start:end]:
                    src_img = os.path.join(class_path, img)
                    dst_img = os.path.join(dest_folder, img)
                    shutil.copy(src_img, dst_img)

                print(f"  → Classe {c} : {len(images[start:end])} images pour client {i+1}")


if __name__ == "__main__":
    create_structure()
    distribute_images()
    print("🎉 Partition terminée : les clients ont leurs données!")
