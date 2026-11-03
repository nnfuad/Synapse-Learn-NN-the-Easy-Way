"""
Module 10: Train a NN by Hand and See What Happens Inside
Pure NumPy manual neural network, analytical chain rule backpropagation,
PyTorch parity validation, and step-by-step arithmetic inspection.
"""

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

try:
    from .pytorch_comparison import (
        PyTorchNeuralNetwork,
        compare_gradients_with_pytorch,
    )
except ImportError:
    pass

__all__ = [
    "HandNeuralNetwork",
    "sigmoid",
    "sigmoid_prime",
    "relu",
    "relu_prime",
    "tanh",
    "tanh_prime",
    "linear",
    "linear_prime",
    "bce_loss",
    "mse_loss",
    "PyTorchNeuralNetwork",
    "compare_gradients_with_pytorch",
]
