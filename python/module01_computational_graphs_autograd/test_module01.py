"""
Pytest Suite for Module 01: Computational Graphs & Autodiff
Verifies:
1. Analytical gradients vs Finite Difference Numerical Approximations (eps=1e-5).
2. Reverse chain rule gradient accumulation over branching graphs (Diamond DAG).
3. Exact mathematical parity against PyTorch 2.x autograd engine.
4. Custom torch.autograd.Function implementation via torch.autograd.gradcheck.
5. Micrograd MLP forward and backward convergence on non-linear XOR data.
"""

import sys
import os
import math
import pytest
import numpy as np
import torch

# Ensure local module directory is in path for standalone pytest execution
sys.path.insert(0, os.path.dirname(__file__))

try:
    from .autograd_engine import Value, MLP
    from .pytorch_autograd_comparison import CustomSigmoid, CustomGELU, compare_scalar_graph
except ImportError:
    from autograd_engine import Value, MLP
    from pytorch_autograd_comparison import CustomSigmoid, CustomGELU, compare_scalar_graph


# =============================================================================
# 1. Finite Difference Numerical Gradient Approximations
# =============================================================================

def numerical_gradient(f, x_val: float, eps: float = 1e-5) -> float:
    """Central finite difference formula: (f(x + eps) - f(x - eps)) / (2 * eps)."""
    return (f(x_val + eps) - f(x_val - eps)) / (2.0 * eps)


def test_finite_difference_elementary_ops():
    """Verify +, -, *, /, ** against central finite differences."""
    eps = 1e-5

    # 1. Addition: f(x) = x + 7.5
    vx = Value(3.2)
    vz = vx + 7.5
    vz.backward()
    num_grad = numerical_gradient(lambda x: x + 7.5, 3.2, eps)
    assert math.isclose(vx.grad, num_grad, rel_tol=1e-4)

    # 2. Multiplication: f(x) = 4.2 * x
    vx = Value(2.5)
    vz = vx * 4.2
    vz.backward()
    num_grad = numerical_gradient(lambda x: 4.2 * x, 2.5, eps)
    assert math.isclose(vx.grad, num_grad, rel_tol=1e-4)

    # 3. Power: f(x) = x^3
    vx = Value(2.0)
    vz = vx ** 3
    vz.backward()
    num_grad = numerical_gradient(lambda x: x ** 3, 2.0, eps)
    assert math.isclose(vx.grad, num_grad, rel_tol=1e-4)

    # 4. Division: f(x) = 10 / x
    vx = Value(4.0)
    vz = 10.0 / vx
    vz.backward()
    num_grad = numerical_gradient(lambda x: 10.0 / x, 4.0, eps)
    assert math.isclose(vx.grad, num_grad, rel_tol=1e-4)


def test_finite_difference_activations():
    """Verify non-linear activations (relu, sigmoid, tanh, exp, log) against finite differences."""
    eps = 1e-5

    # ReLU (at non-zero point)
    vx = Value(1.7)
    vz = vx.relu()
    vz.backward()
    num_grad = numerical_gradient(lambda x: max(0.0, x), 1.7, eps)
    assert math.isclose(vx.grad, num_grad, rel_tol=1e-4)

    # Sigmoid
    vx = Value(-0.8)
    vz = vx.sigmoid()
    vz.backward()
    num_grad = numerical_gradient(lambda x: 1.0 / (1.0 + math.exp(-x)), -0.8, eps)
    assert math.isclose(vx.grad, num_grad, rel_tol=1e-4)

    # Tanh
    vx = Value(0.65)
    vz = vx.tanh()
    vz.backward()
    num_grad = numerical_gradient(lambda x: math.tanh(x), 0.65, eps)
    assert math.isclose(vx.grad, num_grad, rel_tol=1e-4)

    # Exp
    vx = Value(1.2)
    vz = vx.exp()
    vz.backward()
    num_grad = numerical_gradient(lambda x: math.exp(x), 1.2, eps)
    assert math.isclose(vx.grad, num_grad, rel_tol=1e-4)

    # Log
    vx = Value(3.5)
    vz = vx.log()
    vz.backward()
    num_grad = numerical_gradient(lambda x: math.log(x), 3.5, eps)
    assert math.isclose(vx.grad, num_grad, rel_tol=1e-4)


# =============================================================================
# 2. Branching DAG Gradient Accumulation (Water Pipe Backpressure Superposition)
# =============================================================================

def test_branching_addition():
    """f(x) = x + x => df/dx = 2.0"""
    x = Value(3.0)
    z = x + x
    z.backward()
    assert math.isclose(z.data, 6.0)
    assert math.isclose(x.grad, 2.0)


def test_branching_multiplication():
    """f(x) = x * x => df/dx = 2 * x = 2 * 4.0 = 8.0"""
    x = Value(4.0)
    z = x * x
    z.backward()
    assert math.isclose(z.data, 16.0)
    assert math.isclose(x.grad, 8.0)


def test_diamond_graph_accumulation():
    """
    Diamond graph:
          x
        /   \\
       a     b
        \\   /
          c
    a = x^2
    b = 3 * x
    c = a + b
    dc/dx = 2x + 3
    For x = 5.0 => dc/dx = 2(5) + 3 = 13.0
    """
    x = Value(5.0)
    a = x ** 2
    b = 3.0 * x
    c = a + b
    c.backward()
    assert math.isclose(c.data, 25.0 + 15.0)
    assert math.isclose(x.grad, 13.0)


def test_quadruple_branch():
    """f(x) = x + x + x + x => df/dx = 4.0"""
    x = Value(2.5)
    z = x + x + x + x
    z.backward()
    assert math.isclose(z.data, 10.0)
    assert math.isclose(x.grad, 4.0)


# =============================================================================
# 3. PyTorch Autograd Parity Tests
# =============================================================================

def test_pytorch_scalar_parity():
    """Verify custom Value vs PyTorch tensor across multiple random values."""
    for (x, y, w, b) in [
        (1.5, -2.0, 0.4, 0.9),
        (-0.8, 1.2, -1.5, 2.1),
        (0.3, -0.4, 3.2, -0.5),
    ]:
        res = compare_scalar_graph(x_val=x, y_val=y, w_val=w, b_val=b)
        assert res["forward"]["diff"] < 1e-9
        assert res["grad_x"]["diff"] < 1e-9
        assert res["grad_y"]["diff"] < 1e-9
        assert res["grad_w"]["diff"] < 1e-9
        assert res["grad_b"]["diff"] < 1e-9


def test_pytorch_custom_functions_gradcheck():
    """Verify CustomSigmoid and CustomGELU via PyTorch's gradcheck."""
    test_tensor = torch.randn(3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(CustomSigmoid.apply, test_tensor, eps=1e-6, atol=1e-4)
    assert torch.autograd.gradcheck(CustomGELU.apply, test_tensor, eps=1e-6, atol=1e-4)


# =============================================================================
# 4. Micrograd MLP Training Convergence
# =============================================================================

def test_micrograd_mlp_learns_xor():
    """
    Verify that an MLP constructed purely with Value nodes can train
    via gradient descent on the classic 1969 XOR benchmark.
    """
    import random
    random.seed(42)
    np.random.seed(42)

    # Inputs and targets for XOR
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    Y = [0.0, 1.0, 1.0, 0.0]

    # Model: 2 inputs -> 4 hidden neurons -> 1 output neuron
    model = MLP(nin=2, nouts=[4, 1], nonlin="tanh")

    initial_loss = sum(((model(x) - y) ** 2 for x, y in zip(X, Y)), Value(0.0)).data

    # Train for 120 epochs with pure gradient descent
    learning_rate = 0.5
    for epoch in range(120):
        # Forward pass
        loss = sum(((model(x) - y) ** 2 for x, y in zip(X, Y)), Value(0.0))

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update weights (Robbins-Monro SGD step)
        for p in model.parameters():
            p.data -= learning_rate * p.grad

    final_loss = loss.data
    # Verify significant loss reduction (> 80% decrease)
    assert final_loss < initial_loss * 0.20, f"Loss didn't drop enough: initial={initial_loss}, final={final_loss}"
