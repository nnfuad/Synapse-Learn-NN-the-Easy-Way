"""
Module 00: The Genesis & The Atomic Neuron
Part 4: Solving the XOR Crisis with a Multilayer Perceptron (MLP)

===============================================================================
HISTORICAL CONTEXT:
The XOR crisis froze neural network research for over a decade.
The question was: "How do we train intermediate (hidden) layers when we only know
the target output at the very end?"

In 1986, David Rumelhart, Geoffrey Hinton, and Ronald Williams published:
"Learning representations by back-propagating errors" in Nature.
By replacing the non-differentiable step function with a smooth, differentiable
activation (Sigmoid / ReLU), gradients could flow backward through hidden layers
via the calculus chain rule!

THE GEOMETRIC SECRET: THE DIMENSIONAL WARP
Why does a hidden layer solve XOR?
A single layer can only draw ONE straight slice across input space (x1, x2).
A hidden layer of 2 neurons transforms the input coordinates into a NEW
space (h1, h2):
       [x1, x2] ===(Hidden Layer Matrix + ReLU)===> [h1, h2]

In this new (h1, h2) coordinate space, the diagonally trapped points are
"unfolded" and pushed apart! Now, a single straight line from the final neuron
can trivially separate the classes!
===============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class XORResolutionNet(nn.Module):
    """
    A minimal 2-layer Multilayer Perceptron (MLP) that conquers XOR.
    
    Architecture:
      Input (2) ---> Linear(2 -> 4) ---> ReLU ---> Linear(4 -> 1) ---> Sigmoid
    """
    def __init__(self, hidden_dim: int = 4):
        super().__init__()
        self.layer1 = nn.Linear(2, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.layer1(x))
        out = torch.sigmoid(self.layer2(h))
        return out

    def get_hidden_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Inspect how the hidden layer warps input space."""
        with torch.no_grad():
            return self.activation(self.layer1(x))


def train_xor_mlp():
    print("=" * 75)
    print("🚀 BREAKING THE AI WINTER: MULTILAYER PERCEPTRON (MLP) SOLVING XOR")
    print("=" * 75)

    # XOR dataset as PyTorch tensors
    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ], dtype=torch.float32)

    y = torch.tensor([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ], dtype=torch.float32)

    # Set seed for guaranteed deterministic pedagogical demonstration
    torch.manual_seed(42)

    model = XORResolutionNet(hidden_dim=4)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.08)

    print("\nModel Architecture:")
    print(model)
    print("\nInitial untrained predictions:")
    with torch.no_grad():
        preds = model(X)
        for i in range(4):
            print(f"  Input: {X[i].tolist()} -> Target: {y[i].item():.0f} | Raw Pred: {preds[i].item():.4f}")

    print("\nTraining MLP using Gradient Descent & Backpropagation...")
    epochs = 400
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward() # Backpropagation: computes dLoss/dW for all layers!
        optimizer.step()

        if epoch % 100 == 0 or epoch == 1:
            acc = ((predictions > 0.5).float() == y).float().mean().item() * 100
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.5f} | Accuracy: {acc:.1f}%")

    print("\nFinal Post-Training Results:")
    print(" x1 | x2 | Target | Model Output | Binary Decision | Status")
    print("----+----+--------+--------------+-----------------+-------")
    with torch.no_grad():
        final_preds = model(X)
        all_passed = True
        for i in range(4):
            val = final_preds[i].item()
            decision = 1 if val >= 0.5 else 0
            target_val = int(y[i].item())
            is_correct = (decision == target_val)
            if not is_correct:
                all_passed = False
            print(f" {X[i][0]:.0f}  | {X[i][1]:.0f}  |   {target_val}    |    {val:.4f}    |        {decision}        | {'✅ SUCCESS' if is_correct else '❌ FAIL'}")

    print(f"\nFinal Result: {'ALL XOR CASES RESOLVED WITH 100% ACCURACY!' if all_passed else 'FAILED'}")

    # Inspect the hidden layer space transformation
    print("\n" + "=" * 75)
    print("🔍 HOW THE HIDDEN LAYER WARPED THE SPACE (Hidden Activations):")
    hidden_reps = model.get_hidden_representation(X)
    for i in range(4):
        print(f"  Input {X[i].tolist()} (Class {int(y[i].item())}) mapped to 4D Hidden Vector: {np.round(hidden_reps[i].numpy(), 3)}")
    print("=" * 75)


if __name__ == "__main__":
    train_xor_mlp()
