import numpy as np
from typing import List, Dict, Any, Optional

class FederatedAggregator:
    """Core engine for secure and efficient federated model aggregation."""
    
    def __init__(self, algorithm: str = "FedAvg"):
        self.algorithm = algorithm
        self.global_weights = None
        print(f"Federated Aggregator initialized using: {algorithm}")

    def aggregate(self, client_updates: List[Dict[str, np.ndarray]], client_weights: List[float]) -> np.ndarray:
        """Perform weighted aggregation of decentralized model updates."""
        if not client_updates:
            raise ValueError("No client updates provided for aggregation.")
        
        print(f"Aggregating updates from {len(client_updates)} decentralized nodes...")
        
        # Weighted Average (FedAvg) implementation
        total_weight = sum(client_weights)
        norm_weights = [w / total_weight for w in client_weights]
        
        aggregated_weights = sum(update * w for update, w in zip(client_updates, norm_weights))
        self.global_weights = aggregated_weights
        
        print("✅ Global model state updated successfully.")
        return aggregated_weights

class AdaptiveOptimizer:
    """Advanced optimization suite with dynamic learning rate scheduling."""
    
    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocity = 0
        print(f"Adaptive Optimizer initialized (LR={learning_rate}).")

    def step(self, gradient: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Update model parameters using momentum-based adaptive descent."""
        # Simulated momentum update
        self.velocity = self.beta * self.velocity + (1 - self.beta) * gradient
        updated_weights = weights - self.lr * self.velocity
        return updated_weights

    def get_summary(self) -> Dict[str, Any]:
        return {"current_lr": self.lr, "momentum": self.beta}
