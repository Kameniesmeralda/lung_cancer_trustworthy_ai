import os

# ----- DIRECTORIES -----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")  # Optionnel

# ----- IMAGE PARAMETERS -----
IMG_SIZE = (224, 224)  # Standard input size for CNNs
NUM_CLASSES = 4        # Lung cancer: adenocarcinoma / large.cell.carcinome / squamous.cell.carcinoma / no cancer

# ----- TRAINING PARAMETERS -----
BATCH_SIZE = 16
NUM_WORKERS = 2

EPOCHS = 5            # Low for testing — you will increase later
LEARNING_RATE = 1e-4

# ----- DEVICE -----
DEVICE = "cpu"

