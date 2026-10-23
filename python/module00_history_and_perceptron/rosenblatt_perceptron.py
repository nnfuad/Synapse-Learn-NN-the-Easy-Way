"""
Module 00: The Genesis & The Atomic Neuron
Part 2: The 1957 Rosenblatt Perceptron & Learning Rule

===============================================================================
HISTORICAL CONTEXT:
In 1957, Frank Rosenblatt at the Cornell Aeronautical Laboratory invented the
PERCEPTRON. Unlike McCulloch-Pitts neurons whose parameters had to be manually
wired, the Perceptron was the first artificial entity in history that could
LEARN its own weights from trial-and-error experience!

In 1960, the U.S. Navy unveiled the "Mark I Perceptron" — a physical room-sized
machine connected to a 20x20 grid of cadmium sulfide photocells (a 400-pixel camera).
Motor-driven potentiometers physically rotated dials to update weights!

THE PHYSICAL ANALOGY:
Imagine a blindfolded archer tuning a bow's tension:
- Target (y) is either Bullseye (+1 or 1) or Miss (0).
- Archer shoots (predicts y_hat).
- If the archer hits correctly (y - y_hat = 0): No adjustment needed.
- If the archer shoots TOO LOW (y = 1, y_hat = 0, error = +1):
  Tighten the bow string proportional to the wind direction: w += η * x.
- If the archer shoots TOO HIGH (y = 0, y_hat = 1, error = -1):
  Loosen the bow string proportional to the wind direction: w -= η * x.

GEOMETRIC MEANING:
In 2D space (inputs x1, x2), the Perceptron draws a straight line:
       w1 * x1 + w2 * x2 + b = 0  ===>  x2 = -(w1 / w2) * x1 - (b / w2)
The weight vector w = [w1, w2] is the normal vector perpendicular to the line,
pointing into the positive territory.
===============================================================================
"""

import numpy as np
from typing import Tuple, List, Dict, Any


class RosenblattPerceptron:
    """
    Frank Rosenblatt's 1957 Single-Layer Perceptron.
    
    Formula:
        y_hat = 1 if (w · x + b) >= 0 else 0
        
    Update Rule (Novikoff 1962 guaranteed convergence on linearly separable data):
        error = y - y_hat
        w = w + learning_rate * error * x
        b = b + learning_rate * error
    """

    def __init__(self, n_inputs: int, learning_rate: float = 0.1, max_epochs: int = 100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        # Initialize weights to small values or zero (historically zero or random)
        self.weights = np.zeros(n_inputs, dtype=np.float64)
        self.bias = 0.0
        self.history: List[Dict[str, Any]] = []

    def predict(self, x: np.ndarray) -> int:
        """Heaviside step activation."""
        linear_output = np.dot(x, self.weights) + self.bias
        return 1 if linear_output >= 0.0 else 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Train the perceptron using the historical Perceptron Learning Rule.
        
        Returns:
            True if converged to 0 errors; False if max_epochs reached without convergence.
        """
        n_samples = X.shape[0]

        for epoch in range(1, self.max_epochs + 1):
            total_errors = 0
            step_logs = []

            for i in range(n_samples):
                xi = X[i]
                target = y[i]
                y_hat = self.predict(xi)
                error = target - y_hat

                if error != 0:
                    total_errors += 1
                    # Perceptron Learning Rule update:
                    delta_w = self.learning_rate * error * xi
                    delta_b = self.learning_rate * error

                    self.weights += delta_w
                    self.bias += delta_b

                    step_logs.append({
                        "sample_idx": i,
                        "x": xi.tolist(),
                        "target": int(target),
                        "pred": int(y_hat),
                        "weights_now": self.weights.copy().tolist(),
                        "bias_now": float(self.bias),
                    })

            self.history.append({
                "epoch": epoch,
                "errors": total_errors,
                "weights": self.weights.copy(),
                "bias": self.bias,
                "steps": step_logs
            })

            # Perfect convergence check
            if total_errors == 0:
                print(f"🎉 Perceptron converged in {epoch} epochs with 0 errors!")
                return True

        print(f"⚠️ Reached max epochs ({self.max_epochs}) without full convergence.")
        return False

    def decision_boundary_line(self, x1_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute (x1, x2) coordinates for the decision boundary:
        w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b) / w2
        """
        w1, w2 = self.weights
        if abs(w2) < 1e-9:
            raise ZeroDivisionError("w2 is zero; boundary is vertical x1 = -b/w1")
        x1 = np.linspace(x1_range[0], x1_range[1], 100)
        x2 = -(w1 * x1 + self.bias) / w2
        return x1, x2


# ============================================================================
# Production PyTorch Formulation (Modern Equivalent)
# ============================================================================

import torch
import torch.nn as nn

class PyTorchPerceptron(nn.Module):
    """
    Modern PyTorch formulation of the single-layer perceptron.
    Instead of step activation (which has zero gradient almost everywhere),
    modern networks use smooth activations (like Sigmoid) to allow gradient descent.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sigmoid acts as a smooth, differentiable step function
        return torch.sigmoid(self.linear(x))


# ============================================================================
# Demonstration & Verification
# ============================================================================

def demonstrate_rosenblatt():
    print("=" * 75)
    print("🤖 1957 ROSENBLATT PERCEPTRON: LEARNING THE 'OR' GATE FROM EXPERIENCE")
    print("=" * 75)

    # Dataset: 2D inputs for logical OR
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    y = np.array([0, 1, 1, 1])

    perceptron = RosenblattPerceptron(n_inputs=2, learning_rate=0.2, max_epochs=20)
    print("Initial weights:", perceptron.weights, "Initial bias:", perceptron.bias)
    print("\nStarting Training...\n")
    
    converged = perceptron.fit(X, y)

    print("\nFinal Learned Parameters:")
    print(f"Weight w1 : {perceptron.weights[0]:.4f}")
    print(f"Weight w2 : {perceptron.weights[1]:.4f}")
    print(f"Bias b    : {perceptron.bias:.4f}")

    print("\nVerification of Learned Predictions:")
    print(" x1  | x2  | Linear (w·x+b) | Step Output | Expected | Correct?")
    print("-----+-----+----------------+-------------+----------+---------")
    all_correct = True
    for xi, target in zip(X, y):
        linear_val = np.dot(xi, perceptron.weights) + perceptron.bias
        pred = perceptron.predict(xi)
        correct = (pred == target)
        if not correct:
            all_correct = False
        print(f" {xi[0]:.0f}   | {xi[1]:.0f}   |    {linear_val:+7.3f}     |      {pred}      |    {target}     | {'✅' if correct else '❌'}")

    print(f"\nOverall Result: {'SUCCESS' if all_correct else 'FAILURE'}")
    
    w1, w2 = perceptron.weights
    b = perceptron.bias
    print(f"Decision Boundary Equation: {w1:.3f}*x1 + {w2:.3f}*x2 + ({b:.3f}) = 0")
    print(f"Or explicitly: x2 = {-w1/w2:.3f}*x1 + {-b/w2:.3f}")


if __name__ == "__main__":
    demonstrate_rosenblatt()
