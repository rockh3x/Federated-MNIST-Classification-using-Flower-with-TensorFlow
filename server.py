import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.server import ServerConfig
import matplotlib.pyplot as plt

# To store accuracies
accuracies = []

class LoggingFedAvg(FedAvg):
    def aggregate_evaluate(self, rnd, results, failures):
        metrics_aggregated = super().aggregate_evaluate(rnd, results, failures)
        if metrics_aggregated is not None:
            loss, metrics = metrics_aggregated
            accuracy = metrics.get("accuracy", None)
            print(f"Round {rnd} accuracy: {accuracy}")
            if accuracy is not None:
                accuracies.append((rnd, accuracy))
        return metrics_aggregated

def plot_accuracy(accuracies):
    rounds, accs = zip(*accuracies)
    plt.figure(figsize=(6,4))
    plt.plot(rounds, accs, marker='o')
    plt.title("Federated Model Accuracy per Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("federated_accuracy.png", dpi=300)
    plt.savefig("federated_accuracy.svg")
    plt.show()

if __name__ == "__main__":
    strategy = LoggingFedAvg(
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    fl.server.start_server(
        strategy=strategy,
        config=ServerConfig(num_rounds=5),
        server_address="localhost:8080",
    )
    # After running all rounds, plot accuracy
    if accuracies:  # If you want to plot after server ends, persist accuracies and run this block separately
        plot_accuracy(accuracies)
