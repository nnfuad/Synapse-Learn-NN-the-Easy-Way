"""
Module 03: Generalization, Regularization & Normalization
Tier 1: Scratch NumPy Implementations of Modern Normalization Layers

Historical Context & Citations:
1. Batch Normalization:
   - Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network
     Training by Reducing Internal Covariate Shift." ICML 2015.
   - Santurkar, S., Tsipras, D., Tran, A., & Madry, A. (2018). "How Does Batch Normalization
     Help Optimization?" NeurIPS 2018.
     *Key Insight*: Debunked the original "Internal Covariate Shift" (ICS) hypothesis.
     BatchNorm's primary superpower is smoothing the loss landscape (making the gradient
     Lipschitz continuous), which enables dramatically higher learning rates and stable training.

2. Layer Normalization:
   - Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). "Layer Normalization." arXiv:1607.06450.
     *Key Insight*: Normalizes across features for each individual sample, eliminating
     batch-size dependency. Crucial for recurrent networks and Transformer sequence models.

3. Root Mean Square Normalization (RMSNorm):
   - Zhang, B., & Sennrich, R. (2019). "Root Mean Square Normalization." NeurIPS 2019.
     *Key Insight*: Discovered that mean-centering in LayerNorm is redundant; the scaling
     property alone stabilizes the gradient dynamics. RMSNorm reduces 10-50% computational
     overhead and is the standard in modern LLMs (LLaMA 1/2/3, Mistral, Gemma, DeepSeek).

Physical Analogy:
- The Assembly Line Quality Control:
  Imagine a car manufacturing assembly line with 100 consecutive stations. If Station 5
  calibrates bolts 2mm too wide, Station 6 adjusts, but by Station 50 the tolerances
  have wildly drifted, stalling the factory. Normalization acts as a precision jig at
  every single station, snapping dimensions back to a calibrated zero-mean / unit-variance
  standard, preventing upstream chaos from breaking downstream tools.
"""

from typing import Tuple, Optional, Union
import numpy as np


class BatchNorm1dScratch:
    """
    Pure NumPy implementation of 1D Batch Normalization (Ioffe & Szegedy, 2015).
    
    Normalizes activations across the mini-batch dimension:
        y = gamma * (x - mu_B) / sqrt(var_B + eps) + beta
    
    Tracks running statistics for inference:
        running_mean = (1 - momentum) * running_mean + momentum * mu_B
        running_var  = (1 - momentum) * running_var  + momentum * var_unbiased
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # Learnable parameters (scale and shift)
        if self.affine:
            self.gamma = np.ones(num_features, dtype=np.float64)
            self.beta = np.zeros(num_features, dtype=np.float64)
        else:
            self.gamma = np.ones(num_features, dtype=np.float64)
            self.beta = np.zeros(num_features, dtype=np.float64)

        # Non-trainable running statistics for inference
        self.running_mean = np.zeros(num_features, dtype=np.float64)
        self.running_var = np.ones(num_features, dtype=np.float64)

        # Gradients
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

        # Mode flag
        self.training = True

        # Cache for analytical backward pass
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        Args:
            x: Input array of shape (B, D) where B is batch size and D is num_features.
        Returns:
            out: Normalized, scaled, and shifted array of shape (B, D).
        """
        x = np.asarray(x, dtype=np.float64)
        B, D = x.shape
        assert D == self.num_features, f"Expected {self.num_features} features, got {D}"

        if self.training:
            # Step 1: Mini-batch mean (shape: [D])
            mean = np.mean(x, axis=0)

            # Step 2: Mini-batch variance (biased for normalization, shape: [D])
            x_centered = x - mean
            var_biased = np.mean(x_centered ** 2, axis=0)

            # Step 3: Standard deviation with numerical stability
            inv_std = 1.0 / np.sqrt(var_biased + self.eps)

            # Step 4: Normalized activation
            x_hat = x_centered * inv_std

            # Step 5: Scale and shift
            out = self.gamma * x_hat + self.beta

            # Update running statistics (PyTorch uses unbiased sample variance for running_var)
            if B > 1:
                var_unbiased = var_biased * (B / (B - 1))
            else:
                var_unbiased = var_biased

            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var_unbiased

            # Cache for backward pass
            self._cache = {
                "x_centered": x_centered,
                "inv_std": inv_std,
                "x_hat": x_hat,
                "B": B,
            }
        else:
            # Inference mode: use accumulated population statistics
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_hat + self.beta

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Analytical backward pass for Batch Normalization.
        Derivation:
            dL/d_gamma = sum(grad_output * x_hat, axis=0)
            dL/d_beta  = sum(grad_output, axis=0)
            dL/d_x     = (gamma / (B * std)) * [B * dy - sum(dy) - x_hat * sum(dy * x_hat)]
        
        Args:
            grad_output: Upstream gradient dL/dy of shape (B, D).
        Returns:
            grad_input: Gradient w.r.t input dL/dx of shape (B, D).
        """
        assert self.training, "Backward pass should only be called during training mode"
        grad_output = np.asarray(grad_output, dtype=np.float64)

        x_hat = self._cache["x_hat"]
        inv_std = self._cache["inv_std"]
        B = self._cache["B"]

        # Gradients w.r.t affine parameters
        self.grad_gamma = np.sum(grad_output * x_hat, axis=0)
        self.grad_beta = np.sum(grad_output, axis=0)

        # Gradient w.r.t normalized activations
        dx_hat = grad_output * self.gamma

        # Fast, numerically stable analytical gradient w.r.t input x
        # dL/dx = (1 / (B * std)) * [B * dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat)]
        sum_dx_hat = np.sum(dx_hat, axis=0, keepdims=True)
        sum_dx_hat_x_hat = np.sum(dx_hat * x_hat, axis=0, keepdims=True)

        grad_input = (inv_std / B) * (B * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x_hat)

        return grad_input


class LayerNormScratch:
    """
    Pure NumPy implementation of Layer Normalization (Ba, Kiros, & Hinton, 2016).
    
    Normalizes activations across the channel/feature dimension for each sample independently:
        mu = (1 / D) * sum(x_i)
        var = (1 / D) * sum((x_i - mu)^2)
        y = gamma * (x - mu) / sqrt(var + eps) + beta
    
    Supports 2D (B, D) and 3D (B, T, D) sequence tensors.
    """

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gamma = np.ones(self.normalized_shape, dtype=np.float64)
            self.beta = np.zeros(self.normalized_shape, dtype=np.float64)
        else:
            self.gamma = np.ones(self.normalized_shape, dtype=np.float64)
            self.beta = np.zeros(self.normalized_shape, dtype=np.float64)

        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for LayerNorm.
        Normalizes over the axes specified by normalized_shape (typically the trailing dimensions).
        """
        x = np.asarray(x, dtype=np.float64)
        norm_dims = len(self.normalized_shape)
        norm_axes = tuple(range(x.ndim - norm_dims, x.ndim))
        D = int(np.prod(self.normalized_shape))

        mean = np.mean(x, axis=norm_axes, keepdims=True)
        x_centered = x - mean
        var = np.mean(x_centered ** 2, axis=norm_axes, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)

        x_hat = x_centered * inv_std
        out = self.gamma * x_hat + self.beta

        self._cache = {
            "x_hat": x_hat,
            "inv_std": inv_std,
            "norm_axes": norm_axes,
            "batch_axes": tuple(range(x.ndim - norm_dims)),
            "D": D,
        }
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Analytical backward pass for Layer Normalization.
        Args:
            grad_output: Upstream gradient dL/dy of shape matching x.
        Returns:
            grad_input: Gradient w.r.t input x.
        """
        grad_output = np.asarray(grad_output, dtype=np.float64)

        x_hat = self._cache["x_hat"]
        inv_std = self._cache["inv_std"]
        norm_axes = self._cache["norm_axes"]
        batch_axes = self._cache["batch_axes"]
        D = self._cache["D"]

        # Gradients w.r.t gamma and beta
        if self.elementwise_affine:
            self.grad_gamma = np.sum(grad_output * x_hat, axis=batch_axes)
            self.grad_beta = np.sum(grad_output, axis=batch_axes)

        # Gradient w.r.t x_hat
        dx_hat = grad_output * self.gamma

        # Analytical backward over normalized feature dimensions
        sum_dx_hat = np.sum(dx_hat, axis=norm_axes, keepdims=True)
        sum_dx_hat_x_hat = np.sum(dx_hat * x_hat, axis=norm_axes, keepdims=True)

        grad_input = (inv_std / D) * (D * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x_hat)

        return grad_input


class RMSNormScratch:
    """
    Pure NumPy implementation of Root Mean Square Normalization (Zhang & Sennrich, 2019).
    
    Standard in modern LLMs (LLaMA 3, Mistral, Gemma, DeepSeek).
    Dispenses with mean-centering, enforcing only scale invariance:
        RMS(x) = sqrt( (1 / D) * sum(x_i^2) + eps )
        x_hat  = x / RMS(x)
        y      = gamma * x_hat
    
    Computational benefit: Saves ~10-50% wall-clock time compared to LayerNorm
    by eliminating the mean-centering pass and bias addition.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps

        # Learnable scale parameter (no bias parameter is used in RMSNorm!)
        self.gamma = np.ones(dim, dtype=np.float64)
        self.grad_gamma = np.zeros_like(self.gamma)

        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for RMSNorm.
        Args:
            x: Input array of shape (..., D) where D == self.dim.
        Returns:
            out: Scaled RMS-normalized array of same shape.
        """
        x = np.asarray(x, dtype=np.float64)
        assert x.shape[-1] == self.dim, f"Expected trailing dimension {self.dim}, got {x.shape[-1]}"

        # Compute Root Mean Square along feature axis
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        inv_rms = 1.0 / rms

        x_hat = x * inv_rms
        out = self.gamma * x_hat

        self._cache = {
            "x_hat": x_hat,
            "inv_rms": inv_rms,
            "batch_axes": tuple(range(x.ndim - 1)),
        }
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Analytical backward pass for RMSNorm.
        Derivation:
            dL/d_gamma = sum(grad_output * x_hat, axis=batch_axes)
            dL/d_x_hat = grad_output * gamma
            dL/d_x_i   = inv_rms * [ dL/d_x_hat_i - (x_hat_i / D) * sum(dL/d_x_hat_j * x_hat_j) ]
        
        Args:
            grad_output: Upstream gradient dL/dy.
        Returns:
            grad_input: Gradient w.r.t input x.
        """
        grad_output = np.asarray(grad_output, dtype=np.float64)

        x_hat = self._cache["x_hat"]
        inv_rms = self._cache["inv_rms"]
        batch_axes = self._cache["batch_axes"]
        D = self.dim

        # Gradient w.r.t learnable scale gamma
        self.grad_gamma = np.sum(grad_output * x_hat, axis=batch_axes)

        # Gradient w.r.t normalized activations
        dx_hat = grad_output * self.gamma

        # Analytical backward formula
        # sum_j (dx_hat_j * x_hat_j) along feature axis
        sum_dx_hat_x_hat = np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
        grad_input = inv_rms * (dx_hat - (x_hat / D) * sum_dx_hat_x_hat)

        return grad_input
