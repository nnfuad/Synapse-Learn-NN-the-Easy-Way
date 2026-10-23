"""
Module 03: Generalization, Regularization & Normalization
Tier 2: Production SOTA PyTorch Pre-LN RMSNorm and Modern Transformer Block

Architecture & Citations:
- Zhang, B., & Sennrich, R. (2019). "Root Mean Square Normalization." NeurIPS 2019.
- Touvron et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." arXiv:2302.13971.
- Xiong et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML 2020.
  *Key Insight*: In original Post-LN Transformers (Vaswani et al. 2017), the gradient scale
  at the output layers diverges from that at input layers, requiring delicate warmup.
  In Pre-LN Transformers, gradient scale is uniform across layers, enabling instant convergence.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormPyTorch(nn.Module):
    """
    State-of-the-art RMSNorm in PyTorch matching LLaMA 3 and Mistral architectures.
    
    Mathematically:
        RMS(x) = sqrt( mean(x^2, dim=-1, keepdim=True) + eps )
        y = (x / RMS(x)) * gamma
    
    Numerical Precision Tip:
        Calculations are cast internally to torch.float32 to avoid catastrophic
        underflow/overflow when operating under bfloat16 or float16 mixed precision.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Learnable gain parameter initialized to 1.0
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for maximum numerical stability
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # Upcast to float32 for normalization, then cast back to original tensor dtype
        output = self._norm(x.float()).to(input_dtype)
        return output * self.weight


class MLPBlock(nn.Module):
    """
    Standard Feed-Forward Network (FFN) with SwiGLU / GELU activation.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PreLNTransformerBlock(nn.Module):
    """
    Modern Pre-LN Transformer Block equipped with RMSNorm (LLaMA-style).
    
    Structure:
        x_norm1 = norm1(x)
        x = x + SelfAttention(x_norm1)
        x_norm2 = norm2(x)
        x = x + MLP(x_norm2)
    
    The identity residual path is left completely unhindered by normalization,
    forming an uninterrupted gradient superhighway from layer N to layer 1.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.d_model = d_model
        self.n_heads = n_heads

        # Pre-LN Normalizers
        self.norm1 = RMSNormPyTorch(d_model, eps=eps)
        self.norm2 = RMSNormPyTorch(d_model, eps=eps)

        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

        # Feed-Forward Network
        self.mlp = MLPBlock(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            out: Transformed tensor of shape (batch_size, seq_len, d_model)
        """
        # Sub-layer 1: Pre-LN Attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Sub-layer 2: Pre-LN MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class PostLNTransformerBlock(nn.Module):
    """
    Classic Post-LN Transformer Block (Vaswani et al., 2017).
    
    Structure:
        x = norm1(x + SelfAttention(x))
        x = norm2(x + MLP(x))
    
    Demonstrates gradient decay / instability in deep stacks without warmup.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: Optional[int] = None, eps: float = 1e-6):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.mlp = MLPBlock(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model, eps=eps)
        self.norm2 = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


def compare_pre_vs_post_ln_gradients(
    depth: int = 16,
    batch_size: int = 4,
    seq_len: int = 16,
    d_model: int = 64,
    n_heads: int = 4,
) -> Tuple[float, float]:
    """
    Empirical gradient benchmark: Measures the gradient norm at the first layer
    in a 16-layer Pre-LN vs Post-LN stack.
    
    Returns:
        (pre_ln_grad_norm, post_ln_grad_norm)
    """
    torch.manual_seed(42)

    # Build Pre-LN stack
    pre_layers = nn.ModuleList([PreLNTransformerBlock(d_model, n_heads) for _ in range(depth)])
    # Build Post-LN stack
    post_layers = nn.ModuleList([PostLNTransformerBlock(d_model, n_heads) for _ in range(depth)])

    x_pre = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    x_post = x_pre.detach().clone().requires_grad_(True)

    # Forward Pre-LN
    h = x_pre
    for layer in pre_layers:
        h = layer(h)
    loss_pre = h.sum()
    loss_pre.backward()

    # Forward Post-LN
    h = x_post
    for layer in post_layers:
        h = layer(h)
    loss_post = h.sum()
    loss_post.backward()

    # Measure gradient norm at input / first layer
    pre_grad_norm = float(x_pre.grad.norm().item())
    post_grad_norm = float(x_post.grad.norm().item())

    return pre_grad_norm, post_grad_norm
