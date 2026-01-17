import os
from collections import Counter

from src.dataset import get_datasets  # utilise ImageFolder(TRAIN_DIR/VAL_DIR/TEST_DIR)
# dataset.py construit train_dataset, val_dataset, test_dataset :contentReference[oaicite:2]{index=2}

def count_classes(ds):
    # ds.targets = liste des labels (int) pour ImageFolder
    counts = Counter(ds.targets)
    # mapping id -> nom de classe
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    # retourne dict lisible {class_name: count}
    return {idx_to_class[i]: counts.get(i, 0) for i in sorted(idx_to_class.keys())}

def print_split(name, ds):
    if ds is None:
        print(f"\n{name}: (aucun dataset)")
        return
    counts = count_classes(ds)
    total = sum(counts.values())
    print(f"\n{name} (total={total})")
    for cls, n in counts.items():
        pct = 100 * n / total if total > 0 else 0
        print(f"  - {cls:25s}: {n:4d}  ({pct:5.1f}%)")

if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets()
    print_split("TRAIN", train_ds)
    print_split("VAL", val_ds)
    print_split("TEST", test_ds)

