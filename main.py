from core.federated.engine import FederatedAggregator, AdaptiveOptimizer
import numpy as np
import time

def run_deepflow_demo():
    print("--- 🧠 DeepFlow-ML: Distributed Optimization & Federated Learning ---")
    
    # 1. Initialize Global Engine
    aggregator = FederatedAggregator(algorithm="FedAvg")
    optimizer = AdaptiveOptimizer(learning_rate=0.005, beta=0.95)
    
    # 2. Simulate Decentralized Model Updates (e.g., from different Scania trucks)
    print("🚀 Simulating local training on decentralized edge nodes...")
    num_clients = 5
    # Simulating 5 client update vectors (weights)
    client_updates = [np.random.randn(10) for _ in range(num_clients)]
    client_data_sizes = [100, 250, 150, 300, 200] # Weighting by local dataset size
    
    # 3. Federated Aggregation Phase
    print("📡 Performing secure weight aggregation...")
    global_model = aggregator.aggregate(client_updates, client_data_sizes)
    print(f"✅ Global Model Vector Head: {global_model[:3]}")
    
    # 4. Global Optimization Phase
    print("📉 Applying adaptive global optimization step...")
    # Simulate a gradient for global adjustment
    mock_gradient = np.random.randn(10) * 0.1
    optimized_global_model = optimizer.step(mock_gradient, global_model)
    
    # 5. Summary & Metrics
    stats = optimizer.get_summary()
    print("\n--- ⚡ DeepFlow Execution Metrics ---")
    print(f"Aggregation Algorithm: {aggregator.algorithm}")
    print(f"Current Learning Rate: {stats['current_lr']}")
    print(f"Total Nodes Synchronized: {num_clients}")
    print("✅ Federated training round completed successfully.")

if __name__ == "__main__":
    run_deepflow_demo()
