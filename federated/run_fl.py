import subprocess
import time
import socket
import sys


def wait_for_server(host="127.0.0.1", port=8080, timeout_s=60):
    """Attend que le serveur écoute vraiment sur host:port."""
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def run_federated_learning():
    print("🚀 Démarrage du Federated Learning...")

    # 1) Lancer le serveur
    print("🟦 Lancement du serveur...")
    server_process = subprocess.Popen(
        [sys.executable, "federated/server.py"],
    )

    # 2) Attendre que le port soit ouvert (pas un sleep fixe)
    print("⏳ Attente que le serveur soit prêt (port 8080)...")
    ok = wait_for_server(timeout_s=90)
    if not ok:
        print("❌ Serveur non prêt après 90s. Stop.")
        server_process.terminate()
        return

    print("✅ Serveur prêt. Lancement des clients...")

    # 3) Lancer les clients
    client_processes = []
    num_clients = 5

    for cid in range(1, num_clients + 1):
        cmd = [sys.executable, "federated/client.py", "--client_id", str(cid)]
        p = subprocess.Popen(cmd)
        client_processes.append(p)
        time.sleep(5.0)  # petit délai pour éviter un burst

    # 4) Attendre la fin des clients
    for p in client_processes:
        p.wait()

    # 5) IMPORTANT: laisser le serveur finir (il s'arrête après num_rounds)
    print("⏳ Attente fin serveur...")
    server_process.wait()

    print("🎉 Federated Learning terminé.")


if __name__ == "__main__":
    run_federated_learning()
