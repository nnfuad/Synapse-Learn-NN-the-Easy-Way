"""
Module 10: Train a NN by Hand and See What Happens Inside
Pure NumPy Implementation of a 2-Layer Neural Network with Full Forward/Backward Trace
and Step-by-Step Analytical Arithmetic Explanations.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Union, Callable
import numpy as np


# =============================================================================
# 1. Activation Functions & Analytical Derivatives
# =============================================================================

def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Numerically stable Logistic Sigmoid activation function:
        sigma(z) = 1 / (1 + exp(-z))
    """
    z_arr = np.asarray(z, dtype=np.float64)
    pos_mask = (z_arr >= 0)
    neg_mask = ~pos_mask
    out = np.empty_like(z_arr, dtype=np.float64)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-z_arr[pos_mask]))
    exp_neg = np.exp(z_arr[neg_mask])
    out[neg_mask] = exp_neg / (1.0 + exp_neg)
    return float(out) if np.isscalar(z) else out


def sigmoid_prime(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Analytical first derivative of Sigmoid:
        sigma'(z) = sigma(z) * (1 - sigma(z))
    """
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Rectified Linear Unit (ReLU) activation:
        relu(z) = max(0, z)
    """
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.maximum(0.0, z_arr)
    return float(out) if np.isscalar(z) else out


def relu_prime(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Analytical sub-derivative of ReLU:
        relu'(z) = 1 if z > 0 else 0
    """
    z_arr = np.asarray(z, dtype=np.float64)
    out = (z_arr > 0.0).astype(np.float64)
    return float(out) if np.isscalar(z) else out


def tanh(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Hyperbolic Tangent activation:
        tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
    """
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.tanh(z_arr)
    return float(out) if np.isscalar(z) else out


def tanh_prime(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Analytical first derivative of Tanh:
        tanh'(z) = 1 - tanh(z)^2
    """
    t = tanh(z)
    return 1.0 - t ** 2


def linear(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Linear (Identity) activation:
        linear(z) = z
    """
    z_arr = np.asarray(z, dtype=np.float64)
    return float(z_arr) if np.isscalar(z) else z_arr.copy()


def linear_prime(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Analytical derivative of Identity:
        linear'(z) = 1
    """
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.ones_like(z_arr, dtype=np.float64)
    return float(out) if np.isscalar(z) else out


ACTIVATION_MAP: Dict[str, Tuple[Callable, Callable]] = {
    'sigmoid': (sigmoid, sigmoid_prime),
    'relu': (relu, relu_prime),
    'tanh': (tanh, tanh_prime),
    'linear': (linear, linear_prime),
}


# =============================================================================
# 2. Loss Functions
# =============================================================================

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Mean Squared Error (MSE) loss:
        L = (1 / N) * sum((y_pred - y_true)^2)
    """
    y_p = np.asarray(y_pred, dtype=np.float64)
    y_t = np.asarray(y_true, dtype=np.float64)
    return float(np.mean((y_p - y_t) ** 2))


def bce_loss(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-15) -> float:
    """
    Binary Cross-Entropy (BCE) loss with numerical clipping:
        L = - (1 / N) * sum(y_true * ln(y_pred) + (1 - y_true) * ln(1 - y_pred))
    """
    y_p = np.clip(np.asarray(y_pred, dtype=np.float64), eps, 1.0 - eps)
    y_t = np.asarray(y_true, dtype=np.float64)
    return float(-np.mean(y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p)))


# =============================================================================
# 3. HandNeuralNetwork (Pure NumPy 2-Layer Neural Network)
# =============================================================================

class HandNeuralNetwork:
    """
    Pedagogical pure NumPy 2-Layer Neural Network:
        2 inputs -> H hidden neurons -> 1 output neuron
    Supports full forward trace, exact analytical backpropagation,
    weight updates, and transparent step-by-step arithmetic explanations.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 2,
        output_dim: int = 1,
        activation: str = 'sigmoid',
        activation2: str = 'sigmoid',
        loss: str = 'bce',
        lr: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize the 2-layer neural network parameters and activations.

        Args:
            input_dim: Number of input features (default 2).
            hidden_dim: Number of hidden neurons (H).
            output_dim: Number of output neurons (default 1).
            activation: Activation function for hidden layer ('sigmoid', 'relu', 'tanh', 'linear').
            activation2: Activation function for output layer ('sigmoid', 'relu', 'tanh', 'linear').
            loss: Loss function ('bce' or 'mse').
            lr: Default learning rate.
            seed: Optional random seed for reproducible weight initialization.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.activation = activation.lower()
        self.activation2 = activation2.lower()
        self.loss_fn = loss.lower()
        self.lr = lr

        if self.activation not in ACTIVATION_MAP:
            raise ValueError(f"Unsupported hidden activation: {activation}. Choose from {list(ACTIVATION_MAP.keys())}")
        if self.activation2 not in ACTIVATION_MAP:
            raise ValueError(f"Unsupported output activation: {activation2}. Choose from {list(ACTIVATION_MAP.keys())}")
        if self.loss_fn not in ('bce', 'mse'):
            raise ValueError(f"Unsupported loss function: {loss}. Choose 'bce' or 'mse'.")

        self.act1_fn, self.act1_prime = ACTIVATION_MAP[self.activation]
        self.act2_fn, self.act2_prime = ACTIVATION_MAP[self.activation2]

        self.rng = np.random.RandomState(seed)
        self._initialize_weights()

        self.last_trace: Optional[Dict[str, np.ndarray]] = None

    def _initialize_weights(self) -> None:
        """Initialize layer weights and biases using scaled Xavier / He initialization."""
        # Scale for layer 1
        scale1 = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        self.W1 = self.rng.randn(self.input_dim, self.hidden_dim).astype(np.float64) * scale1
        self.b1 = np.zeros((1, self.hidden_dim), dtype=np.float64)

        # Scale for layer 2
        scale2 = np.sqrt(2.0 / (self.hidden_dim + self.output_dim))
        self.W2 = self.rng.randn(self.hidden_dim, self.output_dim).astype(np.float64) * scale2
        self.b2 = np.zeros((1, self.output_dim), dtype=np.float64)

    def set_weights(
        self,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
    ) -> None:
        """
        Manually assign exact weight and bias matrices.

        Args:
            W1: Hidden layer weights (input_dim, hidden_dim).
            b1: Hidden layer bias (1, hidden_dim) or (hidden_dim,).
            W2: Output layer weights (hidden_dim, output_dim).
            b2: Output layer bias (1, output_dim) or (output_dim,).
        """
        W1_arr = np.asarray(W1, dtype=np.float64)
        b1_arr = np.asarray(b1, dtype=np.float64).reshape(1, self.hidden_dim)
        W2_arr = np.asarray(W2, dtype=np.float64)
        b2_arr = np.asarray(b2, dtype=np.float64).reshape(1, self.output_dim)

        assert W1_arr.shape == (self.input_dim, self.hidden_dim), (
            f"W1 shape mismatch: expected ({self.input_dim}, {self.hidden_dim}), got {W1_arr.shape}"
        )
        assert W2_arr.shape == (self.hidden_dim, self.output_dim), (
            f"W2 shape mismatch: expected ({self.hidden_dim}, {self.output_dim}), got {W2_arr.shape}"
        )

        self.W1 = W1_arr.copy()
        self.b1 = b1_arr.copy()
        self.W2 = W2_arr.copy()
        self.b2 = b2_arr.copy()

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return copies of the current weight and bias matrices."""
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W2': self.W2.copy(),
            'b2': self.b2.copy(),
        }

    def forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Comprehensive forward pass returning a trace dictionary with every
        intermediate computation:
            x: Input batch (N, input_dim)
            z1: Layer 1 pre-activation (N, hidden_dim)
            a1: Layer 1 post-activation (N, hidden_dim)
            z2: Layer 2 pre-activation (N, output_dim)
            a2: Layer 2 post-activation / output (N, output_dim)

        Args:
            X: Input array of shape (N, input_dim) or (input_dim,).

        Returns:
            trace: Dictionary containing {'x', 'z1', 'a1', 'z2', 'a2', 'output'}.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        assert X_arr.shape[1] == self.input_dim, (
            f"Input features mismatch: expected {self.input_dim}, got {X_arr.shape[1]}"
        )

        # Layer 1: Hidden layer
        z1 = X_arr @ self.W1 + self.b1
        a1 = self.act1_fn(z1)

        # Layer 2: Output layer
        z2 = a1 @ self.W2 + self.b2
        a2 = self.act2_fn(z2)

        trace = {
            'x': X_arr,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'a2': a2,
            'output': a2,
            'y_hat': a2,
        }
        self.last_trace = trace
        return trace

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Callable shorthand executing forward pass and returning output a2."""
        return self.forward(X)['a2']

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute scalar loss (BCE or MSE) between predicted and true targets."""
        y_p = np.atleast_1d(np.asarray(y_pred, dtype=np.float64))
        y_t = np.atleast_1d(np.asarray(y_true, dtype=np.float64))
        if y_p.ndim == 1:
            y_p = y_p.reshape(-1, 1)
        if y_t.ndim == 1:
            y_t = y_t.reshape(-1, 1)

        if self.loss_fn == 'bce':
            return bce_loss(y_p, y_t)
        elif self.loss_fn == 'mse':
            return mse_loss(y_p, y_t)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")

    def backward(
        self,
        X_or_trace: Union[np.ndarray, Dict[str, np.ndarray]],
        y: Optional[Union[float, int, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Comprehensive backward pass calculating exact analytical gradients via
        the reverse chain rule:
            dL/da2, dL/dz2, dL/dW2, dL/db2, dL/da1, dL/dz1, dL/dW1, dL/db1

        Args:
            X_or_trace: Input data X or trace dictionary from forward pass.
            y: True target labels of shape (N, output_dim), (N,), or scalar.

        Returns:
            grads: Dictionary containing exact analytical gradients.
        """
        if isinstance(X_or_trace, dict):
            trace = X_or_trace
        else:
            trace = self.forward(X_or_trace)

        if y is None:
            raise ValueError("Target y must be provided for backward pass.")

        y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        X = trace['x']
        z1 = trace['z1']
        a1 = trace['a1']
        z2 = trace['z2']
        a2 = trace['a2']

        N = len(y_arr)

        # ---------------------------------------------------------------------
        # 1. Output Layer Gradients (dL/da2 and dL/dz2)
        # ---------------------------------------------------------------------
        if self.loss_fn == 'bce':
            eps = 1e-15
            a2_clipped = np.clip(a2, eps, 1.0 - eps)
            # dL/da2 = (1/N) * (a2 - y) / (a2 * (1 - a2))
            dL_da2 = (1.0 / N) * ((a2_clipped - y_arr) / (a2_clipped * (1.0 - a2_clipped)))

            # If output activation is Sigmoid, da2/dz2 = a2 * (1 - a2) perfectly
            # cancels the denominator, avoiding numerical instability:
            if self.activation2 == 'sigmoid':
                dL_dz2 = (a2 - y_arr) / N
            else:
                dL_dz2 = dL_da2 * self.act2_prime(z2)

        elif self.loss_fn == 'mse':
            # L = (1/N) * sum((a2 - y)^2)
            # dL/da2 = (2/N) * (a2 - y)
            dL_da2 = (2.0 / N) * (a2 - y_arr)
            dL_dz2 = dL_da2 * self.act2_prime(z2)
        else:
            raise ValueError(f"Unknown loss: {self.loss_fn}")

        # ---------------------------------------------------------------------
        # 2. Output Layer Weights and Bias Gradients
        # ---------------------------------------------------------------------
        # dL/dW2 = a1^T @ dL/dz2 -> shape (hidden_dim, output_dim)
        dL_dW2 = a1.T @ dL_dz2
        # dL/db2 = sum_N(dL/dz2) -> shape (1, output_dim)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        # ---------------------------------------------------------------------
        # 3. Hidden Layer Gradients (dL/da1 and dL/dz1)
        # ---------------------------------------------------------------------
        # dL/da1 = dL/dz2 @ W2^T -> shape (N, hidden_dim)
        dL_da1 = dL_dz2 @ self.W2.T
        # dL/dz1 = dL/da1 * sigma1'(z1) -> shape (N, hidden_dim)
        dL_dz1 = dL_da1 * self.act1_prime(z1)

        # ---------------------------------------------------------------------
        # 4. Hidden Layer Weights and Bias Gradients
        # ---------------------------------------------------------------------
        # dL/dW1 = X^T @ dL/dz1 -> shape (input_dim, hidden_dim)
        dL_dW1 = X.T @ dL_dz1
        # dL/db1 = sum_N(dL/dz1) -> shape (1, hidden_dim)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        grads = {
            # Canonical chain rule notations
            'dL/da2': dL_da2,
            'dL/dz2': dL_dz2,
            'dL/dW2': dL_dW2,
            'dL/db2': dL_db2,
            'dL/da1': dL_da1,
            'dL/dz1': dL_dz1,
            'dL/dW1': dL_dW1,
            'dL/db1': dL_db1,
            # Convenient shorthand aliases
            'da2': dL_da2,
            'dz2': dL_dz2,
            'dW2': dL_dW2,
            'db2': dL_db2,
            'da1': dL_da1,
            'dz1': dL_dz1,
            'dW1': dL_dW1,
            'db1': dL_db1,
            # Alternative named aliases
            'grad_W1': dL_dW1,
            'grad_b1': dL_db1,
            'grad_W2': dL_dW2,
            'grad_b2': dL_db2,
        }
        return grads

    def step(self, grads: Dict[str, np.ndarray], lr: Optional[float] = None) -> None:
        """
        Update model parameters using computed analytical gradients:
            W = W - lr * dL/dW
            b = b - lr * dL/db

        Args:
            grads: Gradient dictionary from backward pass.
            lr: Learning rate (uses self.lr if None).
        """
        learning_rate = self.lr if lr is None else lr

        dW1 = grads.get('dL/dW1', grads.get('dW1', grads.get('grad_W1')))
        db1 = grads.get('dL/db1', grads.get('db1', grads.get('grad_b1')))
        dW2 = grads.get('dL/dW2', grads.get('dW2', grads.get('grad_W2')))
        db2 = grads.get('dL/db2', grads.get('db2', grads.get('grad_b2')))

        if dW1 is None or db1 is None or dW2 is None or db2 is None:
            raise KeyError("Gradient dictionary must contain dW1, db1, dW2, db2.")

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1.reshape(self.b1.shape)
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2.reshape(self.b2.shape)

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
        raw: bool = False,
    ) -> np.ndarray:
        """
        Compute model predictions.

        Args:
            X: Input samples of shape (N, input_dim) or (input_dim,).
            threshold: Decision boundary threshold for classification.
            raw: If True, return continuous output a2; otherwise return discrete classes.

        Returns:
            Array of predictions.
        """
        trace = self.forward(X)
        a2 = trace['a2']
        if raw or (self.activation2 != 'sigmoid' and self.loss_fn != 'bce'):
            return a2
        return (a2 >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute predicted continuous output probabilities."""
        return self.forward(X)['a2']

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: Optional[float] = None,
        verbose: bool = False,
    ) -> Dict[str, list]:
        """
        Train the network on data (X, y) via full-batch gradient descent.

        Args:
            X: Input features (N, input_dim).
            y: Target values (N, output_dim) or (N,).
            epochs: Number of gradient descent iterations.
            lr: Learning rate override (optional).
            verbose: If True, print progress periodically.

        Returns:
            history: Dictionary tracking 'loss' and 'accuracy' per epoch.
        """
        learning_rate = self.lr if lr is None else lr
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        history: Dict[str, list] = {'loss': [], 'accuracy': []}

        for epoch in range(epochs):
            # Forward pass
            trace = self.forward(X)
            y_pred = trace['a2']

            # Loss computation
            loss_val = self.compute_loss(y_pred, y_arr)
            history['loss'].append(loss_val)

            # Accuracy (if classification)
            if self.activation2 == 'sigmoid' or self.loss_fn == 'bce':
                preds = (y_pred >= 0.5).astype(int)
                acc = float(np.mean(preds == y_arr.astype(int)))
                history['accuracy'].append(acc)

            # Backward pass
            grads = self.backward(trace, y_arr)

            # Step parameter update
            self.step(grads, lr=learning_rate)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                acc_str = f" | Acc: {history['accuracy'][-1] * 100:.1f}%" if history['accuracy'] else ""
                print(f"Epoch {epoch:4d}/{epochs} - Loss: {loss_val:.6f}{acc_str}")

        return history

    def explain_step_by_step(
        self,
        x: Union[list, np.ndarray],
        y: Union[float, int, np.ndarray],
        lr: Optional[float] = None,
        print_output: bool = True,
    ) -> str:
        """
        Generate a comprehensive, crystal-clear step-by-step arithmetic breakdown
        showing every single formula, substitution, and exact number calculated
        in both forward and backward passes.

        Args:
            x: Single input sample [x0, x1].
            y: Single target value.
            lr: Optional learning rate for parameter update demonstration.
            print_output: If True, prints explanation to standard output.

        Returns:
            The complete explanation string.
        """
        learning_rate = self.lr if lr is None else lr
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        assert len(x_arr) == self.input_dim, (
            f"x must have length {self.input_dim}, got {len(x_arr)}"
        )
        y_val = float(np.asarray(y).flatten()[0])

        x_2d = x_arr.reshape(1, -1)
        y_2d = np.array([[y_val]], dtype=np.float64)

        # Run forward pass trace
        trace = self.forward(x_2d)
        z1 = trace['z1'][0]  # shape (H,)
        a1 = trace['a1'][0]  # shape (H,)
        z2_val = float(trace['z2'][0, 0])
        a2_val = float(trace['a2'][0, 0])

        loss_val = self.compute_loss(trace['a2'], y_2d)
        grads = self.backward(trace, y_2d)

        lines = []
        sep = "=" * 78
        sub_sep = "-" * 78

        lines.append(sep)
        lines.append("     TRAIN A NN BY HAND: STEP-BY-STEP ARITHMETIC BREAKDOWN")
        lines.append(sep)
        lines.append(f"Input features  x = {x_arr.tolist()}")
        lines.append(f"Target label    y = {y_val}")
        lines.append(f"Hidden Act:     {self.activation.upper()} | Output Act: {self.activation2.upper()} | Loss: {self.loss_fn.upper()}")
        lines.append("")

        # ---------------------------------------------------------------------
        # 1. Forward Pass Breakdown
        # ---------------------------------------------------------------------
        lines.append(sub_sep)
        lines.append("1. FORWARD PASS: LINEAR COMBINATIONS & ACTIVATIONS")
        lines.append(sub_sep)
        lines.append("[Hidden Layer Pre-activations: z1 = x @ W1 + b1]")
        for j in range(self.hidden_dim):
            terms = [f"({x_arr[i]:.4f} * {self.W1[i, j]:.4f})" for i in range(self.input_dim)]
            terms_str = " + ".join(terms) + f" + ({self.b1[0, j]:.4f})"
            lines.append(f"  z1[{j}] = {terms_str} = {z1[j]:.6f}")

        lines.append("")
        lines.append(f"[Hidden Layer Activations: a1 = {self.activation}(z1)]")
        for j in range(self.hidden_dim):
            lines.append(f"  a1[{j}] = {self.activation}({z1[j]:.6f}) = {a1[j]:.6f}")

        lines.append("")
        lines.append("[Output Layer Pre-activation: z2 = a1 @ W2 + b2]")
        w2_terms = [f"({a1[j]:.4f} * {self.W2[j, 0]:.4f})" for j in range(self.hidden_dim)]
        w2_terms_str = " + ".join(w2_terms) + f" + ({self.b2[0, 0]:.4f})"
        lines.append(f"  z2 = {w2_terms_str} = {z2_val:.6f}")

        lines.append("")
        lines.append(f"[Output Layer Activation: a2 = {self.activation2}(z2)]")
        lines.append(f"  a2 = {self.activation2}({z2_val:.6f}) = {a2_val:.6f}  (Predicted y_hat)")

        # ---------------------------------------------------------------------
        # 2. Loss Computation
        # ---------------------------------------------------------------------
        lines.append("")
        lines.append(sub_sep)
        lines.append("2. LOSS COMPUTATION")
        lines.append(sub_sep)
        if self.loss_fn == 'bce':
            lines.append("Formula: L = - [y * ln(a2) + (1 - y) * ln(1 - a2)]")
            lines.append(f"  L = - [{y_val} * ln({a2_val:.6f}) + (1 - {y_val}) * ln({1.0 - a2_val:.6f})]")
            lines.append(f"  L = {loss_val:.6f}")
        else:
            lines.append("Formula: L = (a2 - y)^2")
            lines.append(f"  L = ({a2_val:.6f} - {y_val})^2 = {loss_val:.6f}")

        # ---------------------------------------------------------------------
        # 3. Backward Pass (Chain Rule Breakdown)
        # ---------------------------------------------------------------------
        lines.append("")
        lines.append(sub_sep)
        lines.append("3. BACKWARD PASS (REVERSE CHAIN RULE DERIVATIVES)")
        lines.append(sub_sep)

        dL_da2_val = float(grads['dL/da2'][0, 0])
        dL_dz2_val = float(grads['dL/dz2'][0, 0])
        lines.append("[Output Layer Error Signal]:")
        lines.append(f"  dL/da2 = {dL_da2_val:.6f}")
        lines.append(f"  da2/dz2 = {self.activation2}_prime({z2_val:.6f}) = {self.act2_prime(z2_val):.6f}")
        lines.append(f"  dL/dz2 = dL/da2 * da2/dz2 = {dL_dz2_val:.6f}")

        lines.append("")
        lines.append("[Output Layer Weights & Bias Gradients]:")
        for j in range(self.hidden_dim):
            lines.append(
                f"  dL/dW2[{j}, 0] = a1[{j}] * dL/dz2 = {a1[j]:.6f} * {dL_dz2_val:.6f} = {grads['dL/dW2'][j, 0]:.6f}"
            )
        lines.append(f"  dL/db2 = dL/dz2 = {grads['dL/db2'][0, 0]:.6f}")

        lines.append("")
        lines.append("[Hidden Layer Gradients (Backpropagated)]: dL/da1 = dL/dz2 * W2^T")
        for j in range(self.hidden_dim):
            lines.append(
                f"  dL/da1[{j}] = dL/dz2 * W2[{j}, 0] = {dL_dz2_val:.6f} * {self.W2[j, 0]:.6f} = {grads['dL/da1'][0, j]:.6f}"
            )

        lines.append("")
        lines.append("[Hidden Layer Pre-activation Gradients]: dL/dz1 = dL/da1 * act1_prime(z1)")
        for j in range(self.hidden_dim):
            act1_p = float(self.act1_prime(z1[j]))
            lines.append(
                f"  dL/dz1[{j}] = {grads['dL/da1'][0, j]:.6f} * {act1_p:.6f} = {grads['dL/dz1'][0, j]:.6f}"
            )

        lines.append("")
        lines.append("[Hidden Layer Weights & Bias Gradients]: dL/dW1 = x^T @ dL/dz1")
        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                lines.append(
                    f"  dL/dW1[{i}, {j}] = x[{i}] * dL/dz1[{j}] = {x_arr[i]:.6f} * {grads['dL/dz1'][0, j]:.6f} = {grads['dL/dW1'][i, j]:.6f}"
                )
        for j in range(self.hidden_dim):
            lines.append(f"  dL/db1[{j}] = dL/dz1[{j}] = {grads['dL/db1'][0, j]:.6f}")

        # ---------------------------------------------------------------------
        # 4. Parameter Update Demonstration
        # ---------------------------------------------------------------------
        lines.append("")
        lines.append(sub_sep)
        lines.append(f"4. PARAMETER UPDATE STEP (Learning Rate lr = {learning_rate})")
        lines.append(sub_sep)
        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                new_w = self.W1[i, j] - learning_rate * grads['dL/dW1'][i, j]
                lines.append(
                    f"  W1[{i}, {j}]_new = {self.W1[i, j]:.6f} - ({learning_rate} * {grads['dL/dW1'][i, j]:.6f}) = {new_w:.6f}"
                )
        for j in range(self.hidden_dim):
            new_b = self.b1[0, j] - learning_rate * grads['dL/db1'][0, j]
            lines.append(
                f"  b1[{j}]_new = {self.b1[0, j]:.6f} - ({learning_rate} * {grads['dL/db1'][0, j]:.6f}) = {new_b:.6f}"
            )
        for j in range(self.hidden_dim):
            new_w2 = self.W2[j, 0] - learning_rate * grads['dL/dW2'][j, 0]
            lines.append(
                f"  W2[{j}, 0]_new = {self.W2[j, 0]:.6f} - ({learning_rate} * {grads['dL/dW2'][j, 0]:.6f}) = {new_w2:.6f}"
            )
        new_b2 = self.b2[0, 0] - learning_rate * grads['dL/db2'][0, 0]
        lines.append(
            f"  b2_new = {self.b2[0, 0]:.6f} - ({learning_rate} * {grads['dL/db2'][0, 0]:.6f}) = {new_b2:.6f}"
        )
        lines.append(sep)

        full_text = "\n".join(lines)
        if print_output:
            print(full_text)
        return full_text
