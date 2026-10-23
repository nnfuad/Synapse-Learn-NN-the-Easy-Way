"""
Pytest Verification Suite for Module 03: Generalization, Regularization & Normalization

Verifies:
1. Pure NumPy BatchNorm1d forward and analytical backward against torch.nn.BatchNorm1d (1e-5).
2. Pure NumPy LayerNorm forward and analytical backward against torch.nn.LayerNorm (1e-5).
3. Pure NumPy RMSNorm forward and analytical backward against PyTorch autograd reference (1e-5).
4. Inverted Dropout expectation preservation and zero-inference-overhead.
5. SGD vs Decoupled SGD equivalence and AdamW decoupled dynamics.
6. Pre-LN Transformer block execution.
"""

import os
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure local imports work reliably
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalization_scratch import BatchNorm1dScratch, LayerNormScratch, RMSNormScratch
from dropout_and_weight_decay import InvertedDropoutScratch, DecoupledWeightDecaySGD, AdamWScratch
from pytorch_sota_norm import RMSNormPyTorch, PreLNTransformerBlock, compare_pre_vs_post_ln_gradients


def test_batchnorm1d_forward_and_backward():
    torch.manual_seed(42)
    np.random.seed(42)

    B, D = 8, 5
    eps = 1e-5
    momentum = 0.1

    # Generate synthetic input
    x_np = np.random.randn(B, D).astype(np.float64) * 2.5 + 1.2
    x_torch = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)

    # Initialize scratch and PyTorch modules
    bn_scratch = BatchNorm1dScratch(num_features=D, eps=eps, momentum=momentum)
    bn_torch = nn.BatchNorm1d(num_features=D, eps=eps, momentum=momentum, dtype=torch.float64)

    # Align initial weights and biases
    gamma_init = np.random.uniform(0.5, 1.5, size=D).astype(np.float64)
    beta_init = np.random.uniform(-0.5, 0.5, size=D).astype(np.float64)

    bn_scratch.gamma = gamma_init.copy()
    bn_scratch.beta = beta_init.copy()

    with torch.no_grad():
        bn_torch.weight.copy_(torch.tensor(gamma_init))
        bn_torch.bias.copy_(torch.tensor(beta_init))

    # 1. Forward Pass (Training Mode)
    out_scratch = bn_scratch.forward(x_np)
    out_torch = bn_torch(x_torch)

    np.testing.assert_allclose(
        out_scratch,
        out_torch.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="BatchNorm1d forward training output mismatch",
    )

    # Verify running statistics
    np.testing.assert_allclose(
        bn_scratch.running_mean,
        bn_torch.running_mean.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="BatchNorm1d running_mean mismatch",
    )
    np.testing.assert_allclose(
        bn_scratch.running_var,
        bn_torch.running_var.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="BatchNorm1d running_var mismatch",
    )

    # 2. Backward Pass
    grad_output = np.random.randn(B, D).astype(np.float64)
    grad_torch = torch.tensor(grad_output, dtype=torch.float64)

    # Backward PyTorch
    out_torch.backward(grad_torch)

    # Backward Scratch
    dx_scratch = bn_scratch.backward(grad_output)

    # Verify dL/dx
    np.testing.assert_allclose(
        dx_scratch,
        x_torch.grad.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="BatchNorm1d dL/dx analytical backward mismatch",
    )

    # Verify dL/d_gamma and dL/d_beta
    np.testing.assert_allclose(
        bn_scratch.grad_gamma,
        bn_torch.weight.grad.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="BatchNorm1d dL/d_gamma mismatch",
    )
    np.testing.assert_allclose(
        bn_scratch.grad_beta,
        bn_torch.bias.grad.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="BatchNorm1d dL/d_beta mismatch",
    )

    # 3. Eval Mode
    bn_scratch.training = False
    bn_torch.eval()

    x_test = np.random.randn(B, D).astype(np.float64)
    out_scratch_eval = bn_scratch.forward(x_test)
    with torch.no_grad():
        out_torch_eval = bn_torch(torch.tensor(x_test, dtype=torch.float64))

    np.testing.assert_allclose(
        out_scratch_eval,
        out_torch_eval.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="BatchNorm1d forward eval mode mismatch",
    )


def test_layernorm_forward_and_backward():
    torch.manual_seed(42)
    np.random.seed(42)

    B, T, D = 4, 6, 8
    eps = 1e-5

    x_np = np.random.randn(B, T, D).astype(np.float64) * 3.0 + 0.5
    x_torch = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)

    ln_scratch = LayerNormScratch(normalized_shape=D, eps=eps)
    ln_torch = nn.LayerNorm(normalized_shape=D, eps=eps, dtype=torch.float64)

    gamma_init = np.random.uniform(0.8, 1.2, size=D).astype(np.float64)
    beta_init = np.random.uniform(-0.3, 0.3, size=D).astype(np.float64)

    ln_scratch.gamma = gamma_init.copy()
    ln_scratch.beta = beta_init.copy()

    with torch.no_grad():
        ln_torch.weight.copy_(torch.tensor(gamma_init))
        ln_torch.bias.copy_(torch.tensor(beta_init))

    # Forward
    out_scratch = ln_scratch.forward(x_np)
    out_torch = ln_torch(x_torch)

    np.testing.assert_allclose(
        out_scratch,
        out_torch.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="LayerNorm forward mismatch",
    )

    # Backward
    grad_output = np.random.randn(B, T, D).astype(np.float64)
    out_torch.backward(torch.tensor(grad_output, dtype=torch.float64))
    dx_scratch = ln_scratch.backward(grad_output)

    np.testing.assert_allclose(
        dx_scratch,
        x_torch.grad.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="LayerNorm dL/dx analytical backward mismatch",
    )
    np.testing.assert_allclose(
        ln_scratch.grad_gamma,
        ln_torch.weight.grad.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="LayerNorm dL/d_gamma mismatch",
    )
    np.testing.assert_allclose(
        ln_scratch.grad_beta,
        ln_torch.bias.grad.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="LayerNorm dL/d_beta mismatch",
    )


def test_rmsnorm_forward_and_backward():
    torch.manual_seed(42)
    np.random.seed(42)

    B, T, D = 4, 5, 12
    eps = 1e-6

    x_np = np.random.randn(B, T, D).astype(np.float64)
    x_torch = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)

    rms_scratch = RMSNormScratch(dim=D, eps=eps)
    rms_torch = RMSNormPyTorch(dim=D, eps=eps).to(dtype=torch.float64)

    gamma_init = np.random.uniform(0.7, 1.3, size=D).astype(np.float64)
    rms_scratch.gamma = gamma_init.copy()
    with torch.no_grad():
        rms_torch.weight.copy_(torch.tensor(gamma_init, dtype=torch.float64))

    # Forward
    out_scratch = rms_scratch.forward(x_np)
    out_torch = rms_torch(x_torch)

    np.testing.assert_allclose(
        out_scratch,
        out_torch.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="RMSNorm forward output mismatch",
    )

    # Backward
    grad_output = np.random.randn(B, T, D).astype(np.float64)
    out_torch.backward(torch.tensor(grad_output, dtype=torch.float64))
    dx_scratch = rms_scratch.backward(grad_output)

    np.testing.assert_allclose(
        dx_scratch,
        x_torch.grad.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="RMSNorm dL/dx analytical backward mismatch",
    )
    np.testing.assert_allclose(
        rms_scratch.grad_gamma,
        rms_torch.weight.grad.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="RMSNorm dL/d_gamma mismatch",
    )


def test_inverted_dropout():
    np.random.seed(42)
    p = 0.4
    dropout = InvertedDropoutScratch(p=p, seed=42)

    # Forward in training mode
    x = np.ones((1000, 100))
    y = dropout.forward(x)

    # Check that approx p fraction is 0
    zero_fraction = np.mean(y == 0.0)
    assert abs(zero_fraction - p) < 0.03, f"Expected zero fraction ~{p}, got {zero_fraction}"

    # Check expectation preservation E[y] == E[x] = 1.0
    mean_val = np.mean(y)
    assert abs(mean_val - 1.0) < 0.05, f"Expected mean ~1.0, got {mean_val}"

    # Eval mode: must be exact identity
    dropout.training = False
    y_eval = dropout.forward(x)
    np.testing.assert_array_equal(y_eval, x)

    # Backward check: non-zero elements scaled by 1/(1-p)
    dropout.training = True
    _ = dropout.forward(x)
    grad_out = np.ones_like(x)
    dx = dropout.backward(grad_out)
    expected_dx = np.where(dropout.mask > 0, 1.0 / (1.0 - p), 0.0)
    np.testing.assert_allclose(dx, expected_dx)


def test_decoupled_weight_decay_and_adamw():
    # 1. Check SGD L2 vs Decoupled equivalence
    sgd = DecoupledWeightDecaySGD(lr=0.01, weight_decay=0.05)
    w = np.array([1.5, -2.0, 0.5])
    grad = np.array([0.1, -0.3, 0.2])

    w_l2 = sgd.step_l2(w.copy(), grad)
    w_decoupled = sgd.step_decoupled(w.copy(), grad)
    np.testing.assert_allclose(w_l2, w_decoupled, rtol=1e-7)

    # 2. Check AdamW vs Adam L2 divergence
    w_init = np.array([1.0, 10.0])
    # Parameter 1 gets large gradient, parameter 2 gets small gradient
    grad = np.array([100.0, 0.01])

    adam_l2 = AdamWScratch({"w": w_init.copy()}, lr=1e-2, weight_decay=0.1, decoupled=False)
    adam_w = AdamWScratch({"w": w_init.copy()}, lr=1e-2, weight_decay=0.1, decoupled=True)

    for _ in range(5):
        adam_l2.step({"w": grad})
        adam_w.step({"w": grad})

    # The decoupled weight decay should decay parameter 2 steadily by lr * weight_decay,
    # whereas Adam with L2 couples to the moment v_t. They will diverge.
    assert not np.allclose(adam_l2.params["w"], adam_w.params["w"])


def test_pre_ln_transformer_execution():
    block = PreLNTransformerBlock(d_model=32, n_heads=4)
    x = torch.randn(2, 8, 32)
    out = block(x)
    assert out.shape == (2, 8, 32)
    assert not torch.isnan(out).any()

    # Verify Pre-LN vs Post-LN gradient stability benchmark executes cleanly
    pre_norm, post_norm = compare_pre_vs_post_ln_gradients(depth=8, d_model=32, n_heads=4)
    assert pre_norm > 0 and post_norm > 0
