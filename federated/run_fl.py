import subprocess
import time

# -------------------------------------------------------------------------
# Script qui lance le serveur FL puis les clients FL automatiquement.
# -------------------------------------------------------------------------

def run_federated_learning():
    print(" Démarrage du Federated Learning...")

    # 1. Lancer le serveur FL
    print(" Lancement du serveur...")
    server_process = subprocess.Popen(
        ["python", "federated/server.py"],
    )

    # Pause pour laisser le serveur démarrer
    time.sleep(3)

    # 2. Lancer les clients FL
    client_processes = []
    num_clients = 5

    print(" Lancement des 5 clients...")

    for i in range(num_clients):
        cmd = [
            "python", "federated/client.py",
            "--client_id", str(i+1)
        ]
        p = subprocess.Popen(cmd)
        client_processes.append(p)
        time.sleep(1)  # petit délai pour éviter les conflits

    # 3. Attendre la fin de tous les clients
    for p in client_processes:
        p.wait()

    # 4. Arrêter le serveur
    server_process.terminate()
    print(" Federated Learning terminé.")


if __name__ == "__main__":
    run_federated_learning()
