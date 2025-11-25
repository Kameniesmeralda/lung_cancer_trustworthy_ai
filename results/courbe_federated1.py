import matplotlib.pyplot as plt

# Federated Learning metrics
rounds = [1, 2, 3]
fit_loss = [1.404061610897058, 1.1154114060456368, 0.9182547190838887]
eval_acc = [0.5277777777777778, 0.4861111111111111, 0.4861111111111111]

# --- Loss curve ---
plt.figure(figsize=(6,4))
plt.plot(rounds, fit_loss, marker='o')
plt.xlabel("Round")
plt.ylabel("Federated Loss")
plt.title("Federated Training Loss (FedAvg)")
plt.grid(True)
plt.show()

# --- Accuracy curve ---
plt.figure(figsize=(6,4))
plt.plot(rounds, eval_acc, marker='o')
plt.xlabel("Round")
plt.ylabel("Federated Accuracy")
plt.title("Federated Evaluation Accuracy (FedAvg)")
plt.grid(True)
plt.show()
