import matplotlib.pyplot as plt

# -----------------------
# Federated loss (FedAvg)
# -----------------------
rounds = [1, 2, 3, 4, 5]
fed_loss = [
    1.3314091662565868,
    1.0961499214172363,
    0.9266105112102296,
    0.9608227651980188,
    0.9503287052114805
]

plt.figure(figsize=(7, 5))
plt.plot(rounds, fed_loss, marker='o', linewidth=2)
plt.title("Federated Training Loss Curve (5 rounds)")
plt.xlabel("Round")
plt.ylabel("Loss (FedAvg Aggregated)")
plt.grid(True)
plt.xticks(rounds)
plt.show()

# Federated accuracy
# -----------------------
fed_acc = [
    0.2361111111111111,
    0.4861111111111111,
    0.5694444444444444,
    0.5833333333333334,
    0.6388888888888888
]

plt.figure(figsize=(7, 5))
plt.plot(rounds, fed_acc, marker='o', linewidth=2)
plt.title("Federated Validation Accuracy Curve (5 rounds)")
plt.xlabel("Round")
plt.ylabel("Accuracy (FedAvg Aggregated)")
plt.grid(True)
plt.xticks(rounds)
plt.show()