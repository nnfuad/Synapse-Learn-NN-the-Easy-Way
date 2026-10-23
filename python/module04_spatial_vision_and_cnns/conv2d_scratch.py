"""
Module 04: Spatial Vision & Convolutions (CNNs)
Tier 1: Scratch NumPy Implementation of Conv2D and MaxPool2D

Historical Context & Citations:
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
  "Gradient-Based Learning Applied to Document Recognition."
  Proceedings of the IEEE, 86(11):2278-2324. (LeNet-5)
  *Key Insight*: Fully connected layers on images suffer from parameter explosion and
  total disregard for 2D spatial locality. Convolutions introduce two foundational inductive biases:
  1. Local Receptive Fields (neurons respond only to neighboring pixels).
  2. Weight Sharing (the same feature detector slides across the entire image, guaranteeing
     translation equivariance: f(g(x)) = g(f(x))).

Physical Analogy:
- The Stenciled Flashlight on an Art Canvas:
  Imagine a massive dark museum canvas. Instead of turning on a blinding floodlight that
  connects every brushstroke to every receptor in your brain at once (Dense MLP), you hold
  a small cutout stencil (the kernel) with a narrow flashlight beam. You systematically
  slide the stencil across each coordinate (striding), illuminating only local edges, textures,
  and color contrasts regardless of where they appear on the wall.
"""

from typing import Union, Tuple, Optional
import numpy as np


def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, (tuple, list)):
        return int(x[0]), int(x[1])
    return int(x), int(x)


class Conv2dScratch:
    """
    Pure NumPy implementation of 2D Convolution using vectorized im2col.
    
    Supports:
    - Arbitrary input channels (C_in) and output filters (C_out)
    - Kernel size (k_h, k_w)
    - Stride (s_h, s_w)
    - Zero-padding (p_h, p_w)
    - Dilation (d_h, d_w)
    - Analytical forward and backward passes matching PyTorch autograd to machine precision.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.use_bias = bias

        kh, kw = self.kernel_size

        # Kaiming / He uniform initialization
        bound = 1.0 / np.sqrt(in_channels * kh * kw)
        self.weight = np.random.uniform(
            -bound, bound, size=(out_channels, in_channels, kh, kw)
        ).astype(np.float64)

        if self.use_bias:
            self.bias = np.random.uniform(-bound, bound, size=(out_channels,)).astype(np.float64)
        else:
            self.bias = np.zeros(out_channels, dtype=np.float64)

        # Gradients
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

        # Cache for backward pass
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Vectorized forward pass using im2col matrix multiplication.
        Args:
            x: Input tensor of shape (N, C_in, H_in, W_in).
        Returns:
            out: Convolved tensor of shape (N, C_out, H_out, W_out).
        """
        x = np.asarray(x, dtype=np.float64)
        N, Cin, Hin, Win = x.shape
        assert Cin == self.in_channels, f"Expected {self.in_channels} channels, got {Cin}"

        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation

        # Compute output dimensions
        Hout = (Hin + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        Wout = (Win + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        assert Hout > 0 and Wout > 0, f"Calculated non-positive output spatial size: ({Hout}, {Wout})"

        # Apply zero-padding
        if ph > 0 or pw > 0:
            x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
        else:
            x_pad = x

        # Build im2col indices
        i0 = np.repeat(np.arange(kh) * dh, kw)
        i1 = np.repeat(np.arange(Hout) * sh, Wout)
        j0 = np.tile(np.arange(kw) * dw, kh)
        j1 = np.tile(np.arange(Wout) * sw, Hout)

        i = np.tile(i0.reshape(-1, 1) + i1.reshape(1, -1), (Cin, 1))
        j = np.tile(j0.reshape(-1, 1) + j1.reshape(1, -1), (Cin, 1))
        k = np.repeat(np.arange(Cin), kh * kw).reshape(-1, 1)

        # Extract column matrix: shape (N, Cin * kh * kw, Hout * Wout)
        cols = x_pad[:, k, i, j]

        # Reshape for matrix multiplication: (Cin * kh * kw, N * Hout * Wout)
        cols_mat = cols.transpose(1, 2, 0).reshape(Cin * kh * kw, -1)

        # Weight matrix: (C_out, Cin * kh * kw)
        weight_mat = self.weight.reshape(self.out_channels, -1)

        # Matrix multiplication: W * X_col + b
        out_mat = weight_mat @ cols_mat
        if self.use_bias:
            out_mat += self.bias[:, None]

        # Reshape back to (N, C_out, H_out, W_out)
        out = out_mat.reshape(self.out_channels, Hout, Wout, N).transpose(3, 0, 1, 2)

        # Cache variables for analytical backward pass
        self._cache = {
            "x_pad_shape": x_pad.shape,
            "Hin": Hin,
            "Win": Win,
            "cols_mat": cols_mat,
            "i": i,
            "j": j,
            "k": k,
            "Hout": Hout,
            "Wout": Wout,
            "N": N,
        }

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Analytical backward pass for 2D Convolution via col2im.
        Args:
            grad_output: Upstream gradient dL/dy of shape (N, C_out, H_out, W_out).
        Returns:
            grad_input: Gradient dL/dx of shape (N, C_in, H_in, W_in).
        """
        grad_output = np.asarray(grad_output, dtype=np.float64)
        Cout = self.out_channels
        kh, kw = self.kernel_size
        ph, pw = self.padding
        Cin = self.in_channels

        cols_mat = self._cache["cols_mat"]
        i = self._cache["i"]
        j = self._cache["j"]
        k = self._cache["k"]
        Hout = self._cache["Hout"]
        Wout = self._cache["Wout"]
        N = self._cache["N"]
        x_pad_shape = self._cache["x_pad_shape"]
        Hin = self._cache["Hin"]
        Win = self._cache["Win"]

        # 1. Gradient w.r.t Bias: sum across batch, height, and width
        if self.use_bias:
            self.grad_bias = np.sum(grad_output, axis=(0, 2, 3))
        else:
            self.grad_bias.fill(0.0)

        # 2. Reshape grad_output to (C_out, H_out * W_out * N)
        grad_mat = grad_output.transpose(1, 2, 3, 0).reshape(Cout, -1)

        # 3. Gradient w.r.t Weight: grad_mat @ cols_mat.T
        dW_mat = grad_mat @ cols_mat.T
        self.grad_weight = dW_mat.reshape(self.weight.shape)

        # 4. Gradient w.r.t Input Columns: W.T @ grad_mat
        weight_mat = self.weight.reshape(Cout, -1)
        dcols_mat = weight_mat.T @ grad_mat  # shape: (Cin * kh * kw, Hout * Wout * N)

        # Reshape back to per-sample columns: (N, Cin * kh * kw, Hout * Wout)
        dcols = dcols_mat.reshape(Cin * kh * kw, Hout * Wout, N).transpose(2, 0, 1)

        # 5. col2im accumulation into dX_pad
        dX_pad = np.zeros(x_pad_shape, dtype=np.float64)
        for n in range(N):
            np.add.at(dX_pad[n], (k, i, j), dcols[n])

        # 6. Slice away padding to retrieve gradient w.r.t unpadded input x
        if ph > 0 and pw > 0:
            grad_input = dX_pad[:, :, ph:-ph, pw:-pw]
        elif ph > 0:
            grad_input = dX_pad[:, :, ph:-ph, :]
        elif pw > 0:
            grad_input = dX_pad[:, :, :, pw:-pw]
        else:
            grad_input = dX_pad

        return grad_input


class MaxPool2dScratch:
    """
    Pure NumPy implementation of 2D Max Pooling.
    
    Subsamples spatial dimensions by routing only the maximum activation
    within each (k_h, k_w) window. Tracks spatial argmax for exact backward routing.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 2,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self.padding = _pair(padding)

        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        N, C, Hin, Win = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        if ph > 0 or pw > 0:
            x_pad = np.pad(
                x,
                ((0, 0), (0, 0), (ph, ph), (pw, pw)),
                mode="constant",
                constant_values=-np.inf,
            )
        else:
            x_pad = x

        Hout = (Hin + 2 * ph - kh) // sh + 1
        Wout = (Win + 2 * pw - kw) // sw + 1
        assert Hout > 0 and Wout > 0, f"Invalid output size ({Hout}, {Wout})"

        out = np.zeros((N, C, Hout, Wout), dtype=np.float64)
        argmax_indices = np.zeros((N, C, Hout, Wout, 2), dtype=np.int64)

        for n in range(N):
            for c in range(C):
                for i in range(Hout):
                    for j in range(Wout):
                        h_start = i * sh
                        w_start = j * sw
                        patch = x_pad[n, c, h_start : h_start + kh, w_start : w_start + kw]
                        max_val = np.max(patch)
                        out[n, c, i, j] = max_val

                        # Relative spatial coordinates of the max
                        max_rel = np.unravel_index(np.argmax(patch), patch.shape)
                        argmax_indices[n, c, i, j] = [h_start + max_rel[0], w_start + max_rel[1]]

        self._cache = {
            "x_pad_shape": x_pad.shape,
            "argmax_indices": argmax_indices,
            "ph": ph,
            "pw": pw,
            "Hin": Hin,
            "Win": Win,
        }
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_output = np.asarray(grad_output, dtype=np.float64)
        argmax_indices = self._cache["argmax_indices"]
        x_pad_shape = self._cache["x_pad_shape"]
        ph = self._cache["ph"]
        pw = self._cache["pw"]

        N, C, Hout, Wout = grad_output.shape
        dX_pad = np.zeros(x_pad_shape, dtype=np.float64)

        for n in range(N):
            for c in range(C):
                for i in range(Hout):
                    for j in range(Wout):
                        h_pos, w_pos = argmax_indices[n, c, i, j]
                        dX_pad[n, c, h_pos, w_pos] += grad_output[n, c, i, j]

        if ph > 0 and pw > 0:
            grad_input = dX_pad[:, :, ph:-ph, pw:-pw]
        elif ph > 0:
            grad_input = dX_pad[:, :, ph:-ph, :]
        elif pw > 0:
            grad_input = dX_pad[:, :, :, pw:-pw]
        else:
            grad_input = dX_pad

        return grad_input
