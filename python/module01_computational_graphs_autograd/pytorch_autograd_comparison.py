"""
Module 01: Computational Graphs, Autograd & The Reverse Chain Rule
Part 2: PyTorch Autograd Deep Dive & Scratch vs Production Parity Verification

===============================================================================
HISTORICAL EVOLUTION (1986 -> 2026):
From Rumelhart, Hinton & Williams' manual matrix calculus derivations (1986)
to modern dynamic Define-by-Run tape-based autograd engines (Torch7 -> PyTorch 2.x).

HOW PYTORCH AUTOGRAD WORKS INTERNALLY:
1. Every `torch.Tensor` with `requires_grad=True` acts like our `Value` class.
2. When operations are performed on tensors, PyTorch dynamically creates a DAG of
   C++ `torch::autograd::Node` objects.
3. Each tensor holds a `.grad_fn` attribute pointing to the function node that
   produced it (e.g., `<AddBackward0>`, `<MulBackward0>`, `<SigmoidBackward0>`).
4. Leaf tensors have `grad_fn=None` and accumulate gradients into `.grad` via
   internal `AccumulateGrad` nodes.
5. Calling `.backward()` executes a reverse topological sort on the C++ DAG,
   identical to Linnainmaa's 1970 reverse accumulation algorithm.
===============================================================================
"""

from __future__ import annotations
import math
import numpy as np
import torch
from torch.autograd import Function

try:
    from .autograd_engine import Value, MLP
except ImportError:
    from autograd_engine import Value, MLP


class CustomSigmoid(Function):
    """
    Custom autograd function demonstrating how PyTorch allows extending its
    differentiation engine with custom C++ or Python operators.

    Mathematical Definition:
        sigma(x) = 1 / (1 + exp(-x))
    Adjoint Derivative:
        d(sigma(x))/dx = sigma(x) * (1 - sigma(x))
    Reverse Chain Rule:
        grad_input = grad_output * d(sigma(x))/dx
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # Save output for backward (more efficient than recomputing exp)
        output = 1.0 / (1.0 + torch.exp(-x))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (output,) = ctx.saved_tensors
        # Reverse chain rule: dL/dx = (dL/dy) * (dy/dx)
        grad_input = grad_output * output * (1.0 - output)
        return grad_input


class CustomGELU(Function):
    """
    Custom autograd function for Gaussian Error Linear Unit (GELU, Hendrycks & Gimpel 2016),
    the standard non-linear activation used in modern Transformers (GPT-4, BERT, LLaMA).

    Approximation formula:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        inner = sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))
        tanh_val = torch.tanh(inner)
        ctx.save_for_backward(x, tanh_val, inner)
        return 0.5 * x * (1.0 + tanh_val)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, tanh_val, inner = ctx.saved_tensors
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        sech2 = 1.0 - torch.pow(tanh_val, 2)
        d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * torch.pow(x, 2))
        d_gelu = 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner
        return grad_output * d_gelu


def compare_scalar_graph(
    x_val: float = 2.0,
    y_val: float = -3.0,
    w_val: float = 0.5,
    b_val: float = 1.2
) -> dict:
    """
    Constructs an identical composite non-linear computation graph in both
    our scratch Value engine and PyTorch's native autograd engine, comparing
    forward values and reverse gradients to machine precision.

    Expression:
        z = (x * w + y)^2 / (b + exp(x))
        L = sigmoid(z) + relu(x * y)
    """
    # -------------------------------------------------------------------------
    # 1. Scratch Value Engine computation
    # -------------------------------------------------------------------------
    vx = Value(x_val, label="x")
    vy = Value(y_val, label="y")
    vw = Value(w_val, label="w")
    vb = Value(b_val, label="b")

    vz = ((vx * vw + vy) ** 2) / (vb + vx.exp())
    vL = vz.sigmoid() + (vx * vy).relu()
    vL.backward()

    # -------------------------------------------------------------------------
    # 2. PyTorch Native Autograd computation
    # -------------------------------------------------------------------------
    tx = torch.tensor(x_val, dtype=torch.float64, requires_grad=True)
    ty = torch.tensor(y_val, dtype=torch.float64, requires_grad=True)
    tw = torch.tensor(w_val, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b_val, dtype=torch.float64, requires_grad=True)

    tz = ((tx * tw + ty) ** 2) / (tb + torch.exp(tx))
    tL = torch.sigmoid(tz) + torch.relu(tx * ty)
    tL.backward()

    # -------------------------------------------------------------------------
    # 3. Precision Check
    # -------------------------------------------------------------------------
    results = {
        "forward": {
            "scratch": vL.data,
            "pytorch": float(tL.item()),
            "diff": abs(vL.data - float(tL.item())),
        },
        "grad_x": {
            "scratch": vx.grad,
            "pytorch": float(tx.grad.item()),
            "diff": abs(vx.grad - float(tx.grad.item())),
        },
        "grad_y": {
            "scratch": vy.grad,
            "pytorch": float(ty.grad.item()),
            "diff": abs(vy.grad - float(ty.grad.item())),
        },
        "grad_w": {
            "scratch": vw.grad,
            "pytorch": float(tw.grad.item()),
            "diff": abs(vw.grad - float(tw.grad.item())),
        },
        "grad_b": {
            "scratch": vb.grad,
            "pytorch": float(tb.grad.item()),
            "diff": abs(vb.grad - float(tb.grad.item())),
        },
    }
    return results


def run_mlp_parity_check():
    """
    Verifies that a multi-layer perceptron built with our scratch Value class
    computes identical forward loss and weight gradients as an equivalent PyTorch model.
    """
    # Fix seed for reproducible initialization
    np.random.seed(42)
    torch.manual_seed(42)

    # 2 inputs, 1 hidden layer with 3 neurons, 1 output neuron
    nin, nhid, nout = 2, 3, 1

    # Weights and biases
    W1 = np.random.uniform(-0.5, 0.5, (nhid, nin))
    b1 = np.random.uniform(-0.1, 0.1, (nhid,))
    W2 = np.random.uniform(-0.5, 0.5, (nout, nhid))
    b2 = np.random.uniform(-0.1, 0.1, (nout,))

    x_input = [1.2, -0.7]
    target = 0.85

    # --- Scratch Forward & Backward ---
    # Layer 1
    h_scratch = []
    w1_values = []
    b1_values = []
    for j in range(nhid):
        w_row = [Value(W1[j, k], label=f"W1_{j}_{k}") for k in range(nin)]
        b_val = Value(b1[j], label=f"b1_{j}")
        w1_values.append(w_row)
        b1_values.append(b_val)
        act = sum((w * x for w, x in zip(w_row, x_input)), b_val)
        h_scratch.append(act.tanh())

    # Layer 2
    w2_values = [Value(W2[0, k], label=f"W2_0_{k}") for k in range(nhid)]
    b2_val = Value(b2[0], label="b2_0")
    act2 = sum((w * h for w, h in zip(w2_values, h_scratch)), b2_val)
    out_scratch = act2.sigmoid()

    # Mean Squared Error Loss: (out - target)^2
    loss_scratch = (out_scratch - target) ** 2
    loss_scratch.backward()

    # --- PyTorch Forward & Backward ---
    t_W1 = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
    t_b1 = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
    t_W2 = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
    t_b2 = torch.tensor(b2, dtype=torch.float64, requires_grad=True)
    t_x = torch.tensor(x_input, dtype=torch.float64)
    t_target = torch.tensor(target, dtype=torch.float64)

    t_h = torch.tanh(t_W1 @ t_x + t_b1)
    t_out = torch.sigmoid(t_W2 @ t_h + t_b2)
    t_loss = (t_out - t_target) ** 2
    t_loss.backward()

    # Verification
    loss_diff = abs(loss_scratch.data - t_loss.item())
    w1_max_diff = max(
        abs(w1_values[j][k].grad - t_W1.grad[j, k].item())
        for j in range(nhid)
        for k in range(nin)
    )
    w2_max_diff = max(
        abs(w2_values[k].grad - t_W2.grad[0, k].item())
        for k in range(nhid)
    )

    print("\n--- MLP Scratch vs PyTorch Parity Report ---")
    print(f"Scratch Loss : {loss_scratch.data:.8f}")
    print(f"PyTorch Loss : {t_loss.item():.8f}")
    print(f"Loss Delta   : {loss_diff:.2e}")
    print(f"W1 Max ΔGrad : {w1_max_diff:.2e}")
    print(f"W2 Max ΔGrad : {w2_max_diff:.2e}")

    assert loss_diff < 1e-10, f"Loss mismatch: {loss_diff}"
    assert w1_max_diff < 1e-10, f"W1 gradient mismatch: {w1_max_diff}"
    assert w2_max_diff < 1e-10, f"W2 gradient mismatch: {w2_max_diff}"
    print("Exact mathematical parity confirmed down to 1e-10!")


if __name__ == "__main__":
    print("=" * 70)
    print("Module 01: PyTorch Autograd Comparison & Verification")
    print("=" * 70)

    # 1. Custom torch.autograd.Function test with gradcheck
    print("\n[1] Running torch.autograd.gradcheck on CustomSigmoid...")
    test_input = torch.randn(4, dtype=torch.float64, requires_grad=True)
    gradcheck_passed = torch.autograd.gradcheck(CustomSigmoid.apply, test_input, eps=1e-6, atol=1e-4)
    print(f"CustomSigmoid gradcheck passed: {gradcheck_passed}")

    print("\n[2] Running torch.autograd.gradcheck on CustomGELU...")
    gradcheck_gelu = torch.autograd.gradcheck(CustomGELU.apply, test_input, eps=1e-6, atol=1e-4)
    print(f"CustomGELU gradcheck passed: {gradcheck_gelu}")

    # 2. Scalar graph parity comparison
    print("\n[3] Comparing Scalar Non-linear Computational Graph...")
    res = compare_scalar_graph()
    for var, vals in res.items():
        print(f"  {var:10s} | Scratch: {vals['scratch']:+.6f} | PyTorch: {vals['pytorch']:+.6f} | Δ: {vals['diff']:.2e}")

    # 3. Multi-layer Perceptron parity check
    run_mlp_parity_check()
