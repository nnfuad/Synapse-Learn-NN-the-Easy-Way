"""
Module 10: PyTorch Reference Implementation and Gradient Parity Verification
Compares pure NumPy manual analytical gradients against PyTorch Autograd.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn

try:
    from .hand_neural_network import HandNeuralNetwork
except ImportError:
    from hand_neural_network import HandNeuralNetwork


# =============================================================================
# 1. PyTorch Reference Architecture
# =============================================================================

class PyTorchNeuralNetwork(nn.Module):
    """
    PyTorch nn.Module matching the exact architecture of HandNeuralNetwork:
        input_dim (2) -> hidden_dim (H) -> output_dim (1)
    Constructed in double precision (torch.float64) for exact machine-precision
    parity testing with NumPy float64 calculations.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 2,
        output_dim: int = 1,
        activation: str = 'sigmoid',
        activation2: str = 'sigmoid',
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.activation_name = activation.lower()
        self.activation2_name = activation2.lower()

        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden_dim, output_dim, dtype=torch.float64)

    def _apply_activation(self, z: torch.Tensor, act_name: str) -> torch.Tensor:
        if act_name == 'sigmoid':
            return torch.sigmoid(z)
        elif act_name == 'relu':
            return torch.relu(z)
        elif act_name == 'tanh':
            return torch.tanh(z)
        elif act_name == 'linear':
            return z
        else:
            raise ValueError(f"Unsupported activation: {act_name}")

    def forward(self, x: torch.Tensor, return_trace: bool = False) -> Any:
        """
        Forward pass through PyTorch layers with optional intermediate trace.
        """
        z1 = self.fc1(x)
        a1 = self._apply_activation(z1, self.activation_name)
        z2 = self.fc2(a1)
        a2 = self._apply_activation(z2, self.activation2_name)

        if return_trace:
            return {
                'x': x,
                'z1': z1,
                'a1': a1,
                'z2': z2,
                'a2': a2,
            }
        return a2

    def sync_weights_from_numpy(self, np_model: HandNeuralNetwork) -> None:
        """
        Synchronize PyTorch layer weights and biases directly from NumPy model.
        PyTorch stores weight as (out_features, in_features) and bias as (out_features,).
        NumPy stores W1 as (in_features, out_features) and b1 as (1, out_features).
        """
        self.fc1.weight.data.copy_(torch.tensor(np_model.W1.T, dtype=torch.float64))
        self.fc1.bias.data.copy_(torch.tensor(np_model.b1.flatten(), dtype=torch.float64))
        self.fc2.weight.data.copy_(torch.tensor(np_model.W2.T, dtype=torch.float64))
        self.fc2.bias.data.copy_(torch.tensor(np_model.b2.flatten(), dtype=torch.float64))

    def sync_weights_to_numpy(self, np_model: HandNeuralNetwork) -> None:
        """Copy current PyTorch weights back into NumPy model."""
        np_model.set_weights(
            W1=self.fc1.weight.data.detach().cpu().numpy().T,
            b1=self.fc1.bias.data.detach().cpu().numpy().reshape(1, -1),
            W2=self.fc2.weight.data.detach().cpu().numpy().T,
            b2=self.fc2.bias.data.detach().cpu().numpy().reshape(1, -1),
        )


# =============================================================================
# 2. Gradient Parity Comparison Function
# =============================================================================

def compare_gradients_with_pytorch(
    np_model: Optional[HandNeuralNetwork] = None,
    torch_model: Optional[PyTorchNeuralNetwork] = None,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    loss_type: str = 'bce',
    atol: float = 1e-6,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Validate that NumPy exact analytical chain rule gradients match PyTorch
    autograd reverse-mode gradients within numerical tolerance (atol <= 1e-6).

    Args:
        np_model: Optional pre-configured HandNeuralNetwork.
        torch_model: Optional pre-configured PyTorchNeuralNetwork.
        X: Input data batch of shape (N, 2).
        y: True target labels of shape (N, 1).
        loss_type: 'bce' or 'mse'.
        atol: Maximum allowable absolute difference.
        verbose: If True, prints comparison table.

    Returns:
        results: Dictionary with differences, gradients, and parity pass status.
    """
    loss_type = loss_type.lower()
    if loss_type not in ('bce', 'mse'):
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    # Set up default models if not provided
    if np_model is None:
        act2 = 'sigmoid' if loss_type == 'bce' else 'linear'
        np_model = HandNeuralNetwork(
            input_dim=2,
            hidden_dim=3,
            output_dim=1,
            activation='tanh',
            activation2=act2,
            loss=loss_type,
            seed=42,
        )

    if torch_model is None:
        torch_model = PyTorchNeuralNetwork(
            input_dim=np_model.input_dim,
            hidden_dim=np_model.hidden_dim,
            output_dim=np_model.output_dim,
            activation=np_model.activation,
            activation2=np_model.activation2,
        )
        torch_model.sync_weights_from_numpy(np_model)

    # Set up default data if not provided
    if X is None or y is None:
        X = np.array([
            [0.2, 0.8],
            [0.5, -0.4],
            [-0.7, 0.3],
            [0.9, 0.1],
        ], dtype=np.float64)
        if loss_type == 'bce':
            y = np.array([[1.0], [0.0], [1.0], [0.0]], dtype=np.float64)
        else:
            y = np.array([[0.75], [-0.25], [0.40], [0.10]], dtype=np.float64)

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    # -------------------------------------------------------------------------
    # 1. NumPy Forward & Backward Pass
    # -------------------------------------------------------------------------
    trace_np = np_model.forward(X_arr)
    loss_np = np_model.compute_loss(trace_np['a2'], y_arr)
    grads_np = np_model.backward(trace_np, y_arr)

    # -------------------------------------------------------------------------
    # 2. PyTorch Forward & Backward Pass
    # -------------------------------------------------------------------------
    torch_model.zero_grad()
    X_torch = torch.tensor(X_arr, dtype=torch.float64, requires_grad=False)
    y_torch = torch.tensor(y_arr, dtype=torch.float64, requires_grad=False)

    out_torch = torch_model(X_torch)

    if loss_type == 'bce':
        criterion = nn.BCELoss(reduction='mean')
    else:
        criterion = nn.MSELoss(reduction='mean')

    loss_torch = criterion(out_torch, y_torch)
    loss_torch.backward()

    # PyTorch gradient extraction (aligning shapes with NumPy)
    torch_dW1 = torch_model.fc1.weight.grad.detach().cpu().numpy().T
    torch_db1 = torch_model.fc1.bias.grad.detach().cpu().numpy().reshape(1, -1)
    torch_dW2 = torch_model.fc2.weight.grad.detach().cpu().numpy().T
    torch_db2 = torch_model.fc2.bias.grad.detach().cpu().numpy().reshape(1, -1)

    # -------------------------------------------------------------------------
    # 3. Discrepancy Calculations
    # -------------------------------------------------------------------------
    diff_loss = abs(loss_np - float(loss_torch.item()))
    diff_forward = float(np.max(np.abs(trace_np['a2'] - out_torch.detach().cpu().numpy())))

    diff_dW1 = float(np.max(np.abs(grads_np['dW1'] - torch_dW1)))
    diff_db1 = float(np.max(np.abs(grads_np['db1'] - torch_db1)))
    diff_dW2 = float(np.max(np.abs(grads_np['dW2'] - torch_dW2)))
    diff_db2 = float(np.max(np.abs(grads_np['db2'] - torch_db2)))

    max_grad_diff = max(diff_dW1, diff_db1, diff_dW2, diff_db2)
    passed = bool(max_grad_diff < atol and diff_loss < atol)

    results = {
        'loss_np': loss_np,
        'loss_torch': float(loss_torch.item()),
        'diff_loss': diff_loss,
        'diff_forward': diff_forward,
        'diff_dW1': diff_dW1,
        'diff_db1': diff_db1,
        'diff_dW2': diff_dW2,
        'diff_db2': diff_db2,
        'max_grad_diff': max_grad_diff,
        'passed': passed,
        'grads_np': {
            'dW1': grads_np['dW1'],
            'db1': grads_np['db1'],
            'dW2': grads_np['dW2'],
            'db2': grads_np['db2'],
        },
        'grads_torch': {
            'dW1': torch_dW1,
            'db1': torch_db1,
            'dW2': torch_dW2,
            'db2': torch_db2,
        },
    }

    if verbose:
        print("=" * 70)
        print("       NUMPY SCRATCH VS PYTORCH AUTOGRAD PARITY VERIFICATION")
        print("=" * 70)
        print(f"Loss Function: {loss_type.upper()} | Hidden Act: {np_model.activation} | Output Act: {np_model.activation2}")
        print(f"NumPy Loss:    {loss_np:.10f}")
        print(f"PyTorch Loss:  {float(loss_torch.item()):.10f} (Diff: {diff_loss:.2e})")
        print(f"Forward Output Max Diff: {diff_forward:.2e}")
        print("-" * 70)
        print(f"dL/dW1 Max Diff: {diff_dW1:.2e} {'[PASS]' if diff_dW1 < atol else '[FAIL]'}")
        print(f"dL/db1 Max Diff: {diff_db1:.2e} {'[PASS]' if diff_db1 < atol else '[FAIL]'}")
        print(f"dL/dW2 Max Diff: {diff_dW2:.2e} {'[PASS]' if diff_dW2 < atol else '[FAIL]'}")
        print(f"dL/db2 Max Diff: {diff_db2:.2e} {'[PASS]' if diff_db2 < atol else '[FAIL]'}")
        print(f"Overall Max Gradient Discrepancy: {max_grad_diff:.2e}")
        print(f"Status: {'PASSED PARITY (< ' + str(atol) + ')' if passed else 'FAILED'}")
        print("=" * 70)

    return results


# =============================================================================
# 3. Demonstration Script (__main__)
# =============================================================================

if __name__ == '__main__':
    print("\n" + "#" * 78)
    print("MODULE 10 DEMO: HAND NEURAL NETWORK & PYTORCH AUTOGRAD COMPARISON")
    print("#" * 78 + "\n")

    # Step 1: Explain single step arithmetic
    demo_nn = HandNeuralNetwork(
        input_dim=2,
        hidden_dim=2,
        output_dim=1,
        activation='sigmoid',
        activation2='sigmoid',
        loss='bce',
        seed=1337,
    )
    print(">>> 1. Step-by-Step Arithmetic Calculation Breakdown:")
    demo_nn.explain_step_by_step(x=[0.6, 0.2], y=1.0, lr=0.1, print_output=True)

    # Step 2: Compare gradients with PyTorch on BCE loss
    print("\n>>> 2. BCE Loss Gradient Parity Check:")
    compare_gradients_with_pytorch(loss_type='bce', verbose=True)

    # Step 3: Compare gradients with PyTorch on MSE loss
    print("\n>>> 3. MSE Loss Gradient Parity Check:")
    compare_gradients_with_pytorch(loss_type='mse', verbose=True)

    # Step 4: Compare across activations (ReLU, Sigmoid, Tanh)
    print("\n>>> 4. Activation Flow Check across ReLU, Sigmoid, Tanh:")
    for act in ['relu', 'sigmoid', 'tanh']:
        nn_act = HandNeuralNetwork(
            input_dim=2,
            hidden_dim=3,
            output_dim=1,
            activation=act,
            activation2='sigmoid',
            loss='bce',
            seed=42,
        )
        res = compare_gradients_with_pytorch(np_model=nn_act, loss_type='bce', verbose=False)
        print(f"Activation: {act:8s} | Max Diff vs PyTorch: {res['max_grad_diff']:.2e} | Passed: {res['passed']}")
    print("\nAll gradient checks completed successfully!\n")
