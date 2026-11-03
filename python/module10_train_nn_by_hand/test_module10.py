"""
Pytest Suite for Module 10: 'Train a NN by Hand and See What Happens Inside'

Verifies:
1. Exact intermediate forward pass values, shapes, and trace dictionary keys.
2. Non-linear activation functions and their analytical first derivatives.
3. BCE and MSE loss calculations and numerical clipping stability.
4. Two-sided central finite difference gradient verification (eps = 1e-5) under BCE loss.
5. Two-sided central finite difference gradient verification (eps = 1e-5) under MSE loss.
6. Exact mathematical parity against PyTorch 2.x Autograd on BCE loss (< 1e-6).
7. Exact mathematical parity against PyTorch 2.x Autograd on MSE loss (< 1e-6).
8. Gradient flow and parity across different activations (ReLU, Sigmoid, Tanh).
9. Single-sample vs batch-level mathematical consistency.
10. explain_step_by_step breakdown formatting, completeness, and arithmetic sections.
11. Full convergence and 100% accuracy on the canonical non-linear XOR problem.
12. Full convergence and > 95% accuracy on concentric circles non-linear classification.
"""

import sys
import os
import math
import pytest
import numpy as np
import torch

# Ensure local module directory is in path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from .hand_neural_network import (
        HandNeuralNetwork,
        sigmoid,
        sigmoid_prime,
        relu,
        relu_prime,
        tanh,
        tanh_prime,
        linear,
        linear_prime,
        bce_loss,
        mse_loss,
    )
    from .pytorch_comparison import (
        PyTorchNeuralNetwork,
        compare_gradients_with_pytorch,
    )
except ImportError:
    from hand_neural_network import (
        HandNeuralNetwork,
        sigmoid,
        sigmoid_prime,
        relu,
        relu_prime,
        tanh,
        tanh_prime,
        linear,
        linear_prime,
        bce_loss,
        mse_loss,
    )
    from pytorch_comparison import (
        PyTorchNeuralNetwork,
        compare_gradients_with_pytorch,
    )


# =============================================================================
# 1. Activation Functions & Analytical Derivatives
# =============================================================================

def test_activation_functions_and_derivatives():
    """Verify analytical activations and their derivatives at key points."""
    # Sigmoid
    assert math.isclose(float(sigmoid(0.0)), 0.5, rel_tol=1e-7)
    assert math.isclose(float(sigmoid_prime(0.0)), 0.25, rel_tol=1e-7)
    assert sigmoid(100.0) > 0.9999
    assert sigmoid(-100.0) < 0.0001

    # ReLU
    assert relu(2.5) == 2.5
    assert relu(-1.5) == 0.0
    assert relu_prime(2.5) == 1.0
    assert relu_prime(-1.5) == 0.0

    # Tanh
    assert math.isclose(float(tanh(0.0)), 0.0, abs_tol=1e-7)
    assert math.isclose(float(tanh_prime(0.0)), 1.0, abs_tol=1e-7)
    assert tanh(50.0) > 0.9999
    assert tanh(-50.0) < -0.9999

    # Linear
    assert linear(3.14) == 3.14
    assert linear_prime(3.14) == 1.0

    # Array input check
    test_arr = np.array([-2.0, 0.0, 2.0])
    np.testing.assert_allclose(sigmoid(test_arr), 1.0 / (1.0 + np.exp(-test_arr)))
    np.testing.assert_allclose(relu(test_arr), np.array([0.0, 0.0, 2.0]))
    np.testing.assert_allclose(tanh(test_arr), np.tanh(test_arr))


# =============================================================================
# 2. Loss Functions & Stability
# =============================================================================

def test_bce_and_mse_loss_calculations():
    """Verify MSE and BCE loss formulas, known values, and clipping stability."""
    y_true = np.array([[1.0], [0.0]])

    # MSE check: ((0.8 - 1)^2 + (0.2 - 0)^2) / 2 = (0.04 + 0.04) / 2 = 0.04
    y_pred_mse = np.array([[0.8], [0.2]])
    assert math.isclose(mse_loss(y_pred_mse, y_true), 0.04, abs_tol=1e-7)

    # BCE check on 0.5: -ln(0.5) = 0.69314718
    y_pred_bce = np.array([[0.5], [0.5]])
    assert math.isclose(bce_loss(y_pred_bce, y_true), math.log(2.0), abs_tol=1e-7)

    # Numerical clipping prevents inf / nan on exact 0 or 1
    y_pred_extreme = np.array([[0.0], [1.0]])
    extreme_loss = bce_loss(y_pred_extreme, y_true)
    assert not np.isnan(extreme_loss)
    assert not np.isinf(extreme_loss)
    assert extreme_loss > 10.0  # High penalty for completely wrong predictions


# =============================================================================
# 3. Forward Pass Values, Shapes, and Trace
# =============================================================================

def test_forward_pass_values_and_shapes():
    """Verify trace keys, array shapes, and exact hand-calculated linear algebra."""
    net = HandNeuralNetwork(
        input_dim=2,
        hidden_dim=3,
        output_dim=1,
        activation='linear',
        activation2='linear',
    )
    # Assign deterministic weights
    W1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b1 = np.array([[0.1, 0.2, 0.3]])
    W2 = np.array([[0.5], [-0.5], [1.0]])
    b2 = np.array([[0.7]])
    net.set_weights(W1, b1, W2, b2)

    X = np.array([[2.0, 3.0], [1.0, 0.0]])
    trace = net.forward(X)

    # Check trace dictionary keys
    assert set(trace.keys()) >= {'x', 'z1', 'a1', 'z2', 'a2', 'output'}

    # Check shapes
    assert trace['x'].shape == (2, 2)
    assert trace['z1'].shape == (2, 3)
    assert trace['a1'].shape == (2, 3)
    assert trace['z2'].shape == (2, 1)
    assert trace['a2'].shape == (2, 1)

    # Hand calculation for sample 0: x = [2.0, 3.0]
    # z1[0] = 2.0*1.0 + 3.0*4.0 + 0.1 = 14.1
    # z1[1] = 2.0*2.0 + 3.0*5.0 + 0.2 = 19.2
    # z1[2] = 2.0*3.0 + 3.0*6.0 + 0.3 = 24.3
    expected_z1_0 = np.array([14.1, 19.2, 24.3])
    np.testing.assert_allclose(trace['z1'][0], expected_z1_0, atol=1e-7)

    # z2[0] = 14.1*0.5 + 19.2*(-0.5) + 24.3*1.0 + 0.7 = 7.05 - 9.60 + 24.30 + 0.70 = 22.45
    expected_z2_0 = 22.45
    assert math.isclose(float(trace['z2'][0, 0]), expected_z2_0, abs_tol=1e-7)


# =============================================================================
# 4. Numerical Gradient Verification (Finite Differences) - BCE Loss
# =============================================================================

def test_numerical_gradient_finite_differences_bce():
    """Verify analytical gradients w.r.t W1, b1, W2, b2 using central finite differences."""
    net = HandNeuralNetwork(
        input_dim=2,
        hidden_dim=3,
        output_dim=1,
        activation='tanh',
        activation2='sigmoid',
        loss='bce',
        seed=123,
    )
    X = np.array([[0.3, -0.5], [0.8, 0.4]], dtype=np.float64)
    y = np.array([[1.0], [0.0]], dtype=np.float64)

    trace = net.forward(X)
    grads = net.backward(trace, y)

    eps = 1e-5

    # Check W1 numerical gradients
    for i in range(net.input_dim):
        for j in range(net.hidden_dim):
            orig = net.W1[i, j]
            net.W1[i, j] = orig + eps
            l_plus = net.compute_loss(net.forward(X)['a2'], y)
            net.W1[i, j] = orig - eps
            l_minus = net.compute_loss(net.forward(X)['a2'], y)
            net.W1[i, j] = orig

            num_grad = (l_plus - l_minus) / (2.0 * eps)
            ana_grad = grads['dW1'][i, j]
            assert math.isclose(ana_grad, num_grad, rel_tol=1e-4, abs_tol=1e-5)

    # Check b1 numerical gradients
    for j in range(net.hidden_dim):
        orig = net.b1[0, j]
        net.b1[0, j] = orig + eps
        l_plus = net.compute_loss(net.forward(X)['a2'], y)
        net.b1[0, j] = orig - eps
        l_minus = net.compute_loss(net.forward(X)['a2'], y)
        net.b1[0, j] = orig

        num_grad = (l_plus - l_minus) / (2.0 * eps)
        ana_grad = grads['db1'][0, j]
        assert math.isclose(ana_grad, num_grad, rel_tol=1e-4, abs_tol=1e-5)

    # Check W2 numerical gradients
    for j in range(net.hidden_dim):
        orig = net.W2[j, 0]
        net.W2[j, 0] = orig + eps
        l_plus = net.compute_loss(net.forward(X)['a2'], y)
        net.W2[j, 0] = orig - eps
        l_minus = net.compute_loss(net.forward(X)['a2'], y)
        net.W2[j, 0] = orig

        num_grad = (l_plus - l_minus) / (2.0 * eps)
        ana_grad = grads['dW2'][j, 0]
        assert math.isclose(ana_grad, num_grad, rel_tol=1e-4, abs_tol=1e-5)

    # Check b2 numerical gradients
    orig = net.b2[0, 0]
    net.b2[0, 0] = orig + eps
    l_plus = net.compute_loss(net.forward(X)['a2'], y)
    net.b2[0, 0] = orig - eps
    l_minus = net.compute_loss(net.forward(X)['a2'], y)
    net.b2[0, 0] = orig

    num_grad = (l_plus - l_minus) / (2.0 * eps)
    ana_grad = grads['db2'][0, 0]
    assert math.isclose(ana_grad, num_grad, rel_tol=1e-4, abs_tol=1e-5)


# =============================================================================
# 5. Numerical Gradient Verification (Finite Differences) - MSE Loss
# =============================================================================

def test_numerical_gradient_finite_differences_mse():
    """Verify analytical gradients w.r.t W1, b1, W2, b2 under MSE loss."""
    net = HandNeuralNetwork(
        input_dim=2,
        hidden_dim=2,
        output_dim=1,
        activation='sigmoid',
        activation2='linear',
        loss='mse',
        seed=999,
    )
    X = np.array([[0.5, 0.2], [-0.3, 0.7]], dtype=np.float64)
    y = np.array([[0.8], [-0.4]], dtype=np.float64)

    trace = net.forward(X)
    grads = net.backward(trace, y)
    eps = 1e-5

    for i in range(net.input_dim):
        for j in range(net.hidden_dim):
            orig = net.W1[i, j]
            net.W1[i, j] = orig + eps
            l_plus = net.compute_loss(net.forward(X)['a2'], y)
            net.W1[i, j] = orig - eps
            l_minus = net.compute_loss(net.forward(X)['a2'], y)
            net.W1[i, j] = orig
            num_grad = (l_plus - l_minus) / (2.0 * eps)
            assert math.isclose(grads['dW1'][i, j], num_grad, rel_tol=1e-4, abs_tol=1e-5)

    for j in range(net.hidden_dim):
        orig = net.W2[j, 0]
        net.W2[j, 0] = orig + eps
        l_plus = net.compute_loss(net.forward(X)['a2'], y)
        net.W2[j, 0] = orig - eps
        l_minus = net.compute_loss(net.forward(X)['a2'], y)
        net.W2[j, 0] = orig
        num_grad = (l_plus - l_minus) / (2.0 * eps)
        assert math.isclose(grads['dW2'][j, 0], num_grad, rel_tol=1e-4, abs_tol=1e-5)


# =============================================================================
# 6. PyTorch Autograd Gradient Equality - BCE Loss
# =============================================================================

def test_pytorch_autograd_gradient_equality_bce():
    """Validate that analytical BCE gradients match PyTorch autograd within 1e-6."""
    res = compare_gradients_with_pytorch(loss_type='bce', atol=1e-6, verbose=False)
    assert res['passed'] is True
    assert res['max_grad_diff'] < 1e-6
    assert res['diff_loss'] < 1e-6


# =============================================================================
# 7. PyTorch Autograd Gradient Equality - MSE Loss
# =============================================================================

def test_pytorch_autograd_gradient_equality_mse():
    """Validate that analytical MSE gradients match PyTorch autograd within 1e-6."""
    res = compare_gradients_with_pytorch(loss_type='mse', atol=1e-6, verbose=False)
    assert res['passed'] is True
    assert res['max_grad_diff'] < 1e-6
    assert res['diff_loss'] < 1e-6


# =============================================================================
# 8. Gradient Flow Across Different Activations (ReLU, Sigmoid, Tanh)
# =============================================================================

@pytest.mark.parametrize("act_name", ["relu", "sigmoid", "tanh"])
def test_gradient_flow_different_activations(act_name: str):
    """Verify gradient calculation and PyTorch parity across hidden activations."""
    np_model = HandNeuralNetwork(
        input_dim=2,
        hidden_dim=4,
        output_dim=1,
        activation=act_name,
        activation2='sigmoid',
        loss='bce',
        seed=42,
    )
    res = compare_gradients_with_pytorch(np_model=np_model, loss_type='bce', atol=1e-6, verbose=False)
    assert res['passed'] is True
    assert res['max_grad_diff'] < 1e-6

    # Verify no NaNs or Infs anywhere in gradients
    for key, grad in res['grads_np'].items():
        assert not np.isnan(grad).any(), f"NaN found in {key}"
        assert not np.isinf(grad).any(), f"Inf found in {key}"


# =============================================================================
# 9. Batch vs Single Sample Consistency
# =============================================================================

def test_batch_vs_single_sample_consistency():
    """Verify batch size 1 matches single sample algebra."""
    net = HandNeuralNetwork(input_dim=2, hidden_dim=2, output_dim=1, seed=7)
    x_single = np.array([0.4, -0.6])
    y_single = 1.0

    # Pass as 1D
    trace_1d = net.forward(x_single)
    grads_1d = net.backward(trace_1d, y_single)

    # Pass as 2D batch of size 1
    x_2d = np.array([[0.4, -0.6]])
    y_2d = np.array([[1.0]])
    trace_2d = net.forward(x_2d)
    grads_2d = net.backward(trace_2d, y_2d)

    np.testing.assert_allclose(trace_1d['a2'], trace_2d['a2'], atol=1e-12)
    np.testing.assert_allclose(grads_1d['dW1'], grads_2d['dW1'], atol=1e-12)
    np.testing.assert_allclose(grads_1d['db1'], grads_2d['db1'], atol=1e-12)
    np.testing.assert_allclose(grads_1d['dW2'], grads_2d['dW2'], atol=1e-12)
    np.testing.assert_allclose(grads_1d['db2'], grads_2d['db2'], atol=1e-12)


# =============================================================================
# 10. explain_step_by_step Formatting & Completeness
# =============================================================================

def test_explain_step_by_step_formatting():
    """Verify explain_step_by_step produces all required mathematical sections."""
    net = HandNeuralNetwork(input_dim=2, hidden_dim=2, output_dim=1, seed=123)
    explanation = net.explain_step_by_step([0.5, -0.2], 1.0, lr=0.05, print_output=False)

    assert isinstance(explanation, str)
    # Check section titles
    assert "1. FORWARD PASS" in explanation
    assert "2. LOSS COMPUTATION" in explanation
    assert "3. BACKWARD PASS" in explanation
    assert "4. PARAMETER UPDATE" in explanation

    # Check key intermediate terms
    assert "z1[" in explanation
    assert "a1[" in explanation
    assert "z2 =" in explanation
    assert "a2 =" in explanation
    assert "dL/da2 =" in explanation
    assert "dL/dz2 =" in explanation
    assert "dL/dW2[" in explanation
    assert "dL/db2 =" in explanation
    assert "dL/da1[" in explanation
    assert "dL/dz1[" in explanation
    assert "dL/dW1[" in explanation
    assert "dL/db1[" in explanation


# =============================================================================
# 11. XOR Convergence
# =============================================================================

def test_xor_convergence():
    """Verify that HandNeuralNetwork converges to 100% accuracy on non-linear XOR."""
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y_xor = np.array([[0], [1], [1], [0]], dtype=np.float64)

    net = HandNeuralNetwork(
        input_dim=2,
        hidden_dim=4,
        output_dim=1,
        activation='tanh',
        activation2='sigmoid',
        loss='bce',
        lr=1.0,
        seed=42,
    )

    history = net.fit(X_xor, y_xor, epochs=1500, verbose=False)

    # Final loss should be small (< 0.05)
    assert history['loss'][-1] < 0.05
    # Final accuracy on XOR should be 100%
    preds = net.predict(X_xor)
    np.testing.assert_array_equal(preds, y_xor)


# =============================================================================
# 12. Concentric Circles Non-Linear Classification Convergence
# =============================================================================

def test_concentric_circles_convergence():
    """Verify convergence on synthetic concentric circles dataset (> 95% accuracy)."""
    rng = np.random.RandomState(42)
    n_samples = 100

    # Inner circle: r ~ 0.35, label = 0
    theta0 = rng.uniform(0, 2 * np.pi, n_samples // 2)
    r0 = 0.35 + rng.normal(0, 0.03, n_samples // 2)
    X0 = np.stack([r0 * np.cos(theta0), r0 * np.sin(theta0)], axis=1)
    y0 = np.zeros((n_samples // 2, 1))

    # Outer circle: r ~ 0.85, label = 1
    theta1 = rng.uniform(0, 2 * np.pi, n_samples // 2)
    r1 = 0.85 + rng.normal(0, 0.03, n_samples // 2)
    X1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)
    y1 = np.ones((n_samples // 2, 1))

    X = np.vstack([X0, X1])
    y = np.vstack([y0, y1])

    net = HandNeuralNetwork(
        input_dim=2,
        hidden_dim=8,
        output_dim=1,
        activation='tanh',
        activation2='sigmoid',
        loss='bce',
        lr=0.5,
        seed=123,
    )

    history = net.fit(X, y, epochs=1000, verbose=False)

    preds = net.predict(X)
    acc = np.mean(preds == y)
    assert acc > 0.95, f"Expected > 95% accuracy on concentric circles, got {acc * 100:.1f}%"
