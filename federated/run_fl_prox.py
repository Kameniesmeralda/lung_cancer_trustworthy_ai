import subprocess
import time
import socket
import sys


def wait_for_server(host="127.0.0.1", port=8080, timeout_s=90):
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def run_federated_learning():
    print("🚀 Starting Federated Learning (FedProx Option A)...")

    # 1) Start server
    print("🟦 Launching FedProx server...")
    server_process = subprocess.Popen([sys.executable, "federated/server_prox.py"])

    # 2) Wait server ready
    print("⏳ Waiting for server (port 8080)...")
    ok = wait_for_server()
    if not ok:
        print("❌ Server not ready. Stop.")
        server_process.terminate()
        return

    print("✅ Server ready. Launching clients...")

    # 3) Launch clients
    client_processes = []
    num_clients = 5

    for cid in range(1, num_clients + 1):
        cmd = [sys.executable, "federated/client_prox.py", "--client_id", str(cid)]
        p = subprocess.Popen(cmd)
        client_processes.append(p)
        time.sleep(0.5)

    # 4) Wait clients
    for p in client_processes:
        p.wait()

    # 5) Wait server finish
    print("⏳ Waiting server to finish...")
    server_process.wait()

    print("🎉 FedProx training finished.")


if __name__ == "__main__":
    run_federated_learning()
