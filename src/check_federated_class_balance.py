import os
from collections import Counter, defaultdict
from torchvision import datasets, transforms

# Adapte si besoin (chez toi: data/client_1, ..., data/client_5)
DATA_DIR = "data"
CLIENTS = [f"client_{i}" for i in range(1, 6)]
SPLITS = ["train", "val", "test"]  # si un split n'existe pas pour un client, on skip

IMG_SIZE = (224, 224)

def get_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

def count_classes(imagefolder_ds):
    # targets = liste d'index (0..C-1)
    counts = Counter(imagefolder_ds.targets)
    total = len(imagefolder_ds)
    # map idx -> class name
    idx_to_class = {v: k for k, v in imagefolder_ds.class_to_idx.items()}
    # format in class order
    rows = []
    for idx in range(len(idx_to_class)):
        name = idx_to_class[idx]
        n = counts.get(idx, 0)
        pct = (100.0 * n / total) if total > 0 else 0.0
        rows.append((idx, name, n, pct))
    return total, rows, imagefolder_ds.class_to_idx

def main():
    transform = get_transform()

    # Pour vérifier cohérence d'ordre des classes entre clients/splits
    reference_class_to_idx = None
    mismatches = []

    # Pour un mini résumé global
    global_counts_by_split = {s: defaultdict(int) for s in SPLITS}
    global_total_by_split = {s: 0 for s in SPLITS}

    print("\n==============================")
    print("FEDERATED CLASS IMBALANCE CHECK")
    print("==============================\n")

    for client in CLIENTS:
        print(f"--- {client.upper()} ---")
        client_path = os.path.join(DATA_DIR, client)

        for split in SPLITS:
            split_path = os.path.join(client_path, split)
            if not os.path.exists(split_path):
                continue

            ds = datasets.ImageFolder(split_path, transform=transform)
            total, rows, class_to_idx = count_classes(ds)

            # check class order consistency
            if reference_class_to_idx is None:
                reference_class_to_idx = class_to_idx
            else:
                if class_to_idx != reference_class_to_idx:
                    mismatches.append((client, split, class_to_idx))

            print(f"\n{split.upper()} (total={total})")
            for idx, name, n, pct in rows:
                print(f"  - idx={idx:<2} {name:<45}: {n:>4} ({pct:>5.1f}%)")

            # update global
            global_total_by_split[split] += total
            for idx, name, n, pct in rows:
                global_counts_by_split[split][name] += n

        print("\n")

    print("=== GLOBAL (ALL CLIENTS MERGED) ===")
    for split in SPLITS:
        if global_total_by_split[split] == 0:
            continue
        print(f"\n{split.upper()} (total={global_total_by_split[split]})")
        for name, n in sorted(global_counts_by_split[split].items(), key=lambda x: -x[1]):
            pct = 100.0 * n / global_total_by_split[split]
            print(f"  - {name:<45}: {n:>4} ({pct:>5.1f}%)")

    print("\n=== CLASS ORDER CHECK ===")
    if reference_class_to_idx is None:
        print("No client data found. Check your folders under data/client_i/...")
    else:
        print("Reference class_to_idx:")
        print(reference_class_to_idx)

    if len(mismatches) == 0:
        print("\n✅ All clients/splits share the SAME class_to_idx order.")
    else:
        print("\n⚠️ Mismatch detected in class_to_idx across clients/splits:")
        for client, split, c2i in mismatches:
            print(f"- {client}/{split}: {c2i}")

    print("\nDone.\n")

if __name__ == "__main__":
    main()
