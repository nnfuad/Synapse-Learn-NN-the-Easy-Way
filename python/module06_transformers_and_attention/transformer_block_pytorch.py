"""
Module 06: The Modern Transformer Revolution & Attention Mechanics
Implementation 3: Production PyTorch 2.x Transformer Decoder Block (FlashAttention, RMSNorm, SwiGLU, GQA)

Historical & SOTA Context:
- Vaswani et al. (2017): Attention Is All You Need.
- Zhang, B., & Sennrich, R. (2019): Root Mean Square Layer Normalization (RMSNorm).
  NeurIPS 2019.
- Shazeer, N. (2020): GLU Variants Improve Transformer (SwiGLU). arXiv:2002.05202.
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022):
  FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
  NeurIPS 2022.
- Ainslie, J., et al. (2023): GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints. EMNLP 2023.

Architectural Innovations:
1. RMSNorm: Replaces standard LayerNorm by omitting the mean-centering step,
   relying purely on root mean square scaling. Reduces GPU memory traffic by ~15%.
2. SwiGLU: Replaces standard ReLU/GELU feedforward layers with a gated linear unit:
   SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down.
3. Grouped-Query Attention (GQA): Allows multiple Query heads to share fewer Key/Value
   heads, dramatically compressing KV-Cache VRAM footprint during decoding.
4. FlashAttention: Exploits GPU SRAM tiling to compute exact softmax without
   materializing the massive N x N attention matrix to high-bandwidth memory (HBM).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from rope_and_kv_cache import RotaryPositionalEmbedding, KVCache


class RMSNorm(nn.Module):
    r"""
    Root Mean Square Layer Normalization.
    
    Formula:
      \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean square along the last dimension
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        # Scale and multiply by learnable parameter gamma (weight)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return x_norm * self.weight


class SwiGLU(nn.Module):
    r"""
    SwiGLU Feedforward Network.
    
    Formula:
      \text{SwiGLU}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        # Standard LLaMA / DeepSeek scaling: ~8/3 * dim rounded to multiple of 64
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate path with Swish/SiLU activation multiplied elementwise by Up path
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands Key/Value heads to match Query heads for Grouped-Query Attention (GQA).
    Shape in:  (batch, num_kv_heads, seq_len, head_dim)
    Shape out: (batch, num_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with RoPE and FlashAttention (SDPA) integration.
    Supports MHA (num_kv_heads == num_heads), GQA (1 < num_kv_heads < num_heads),
    and MQA (num_kv_heads == 1).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_rep = num_heads // self.num_kv_heads

        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        # Rotary Positional Embedding
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        Args:
          x: Input tensor (batch, seq_len, dim)
          position_ids: Token position IDs (batch, seq_len)
          kv_cache: Optional KVCache object
          layer_idx: Layer index for cache storage
          is_causal: Whether to apply causal masking
          
        Returns:
          attn_output: (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 2. Apply Rotary Position Embedding (RoPE)
        q, k = self.rope(q, k, position_ids)

        # 3. Update / fetch from KV-Cache if provided
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, layer_idx=layer_idx)

        # 4. Repeat KV heads for GQA if necessary: (batch, num_heads, total_seq_len, head_dim)
        k = repeat_kv(k, self.num_rep)
        v = repeat_kv(v, self.num_rep)

        # 5. Production FlashAttention via PyTorch 2.x Scaled Dot-Product Attention (SDPA)
        # Note: If generating token-by-token (q seq_len == 1 but k seq_len > 1), causal mask is not needed
        # because the query is at the very last position!
        causal_flag = is_causal if seq_len > 1 else False
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal_flag
        )

        # 6. Transpose back and combine heads: (batch, seq_len, dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        # 7. Output projection
        return self.out_proj(attn_out)


class TransformerDecoderBlock(nn.Module):
    r"""
    Complete SOTA Transformer Decoder Block with Pre-LayerNorm (RMSNorm) Architecture:
    
      x_{attn} = x + \text{GQA}(\text{RMSNorm}_1(x))
      x_{out}  = x_{attn} + \text{SwiGLU}(\text{RMSNorm}_2(x_{attn}))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        max_seq_len: int = 2048,
        eps: float = 1e-6
    ):
        super().__init__()
        self.dim = dim
        self.attn_norm = RMSNorm(dim, eps=eps)
        self.attention = GroupedQueryAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len
        )
        self.ffn_norm = RMSNorm(dim, eps=eps)
        self.feed_forward = SwiGLU(dim=dim, hidden_dim=ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        is_causal: bool = True
    ) -> torch.Tensor:
        # 1. Attention with pre-RMSNorm and residual skip connection
        residual = x
        x_norm = self.attn_norm(x)
        attn_out = self.attention(
            x_norm,
            position_ids=position_ids,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            is_causal=is_causal
        )
        x = residual + attn_out

        # 2. SwiGLU FFN with pre-RMSNorm and residual skip connection
        residual = x
        x_norm = self.ffn_norm(x)
        ffn_out = self.feed_forward(x_norm)
        x = residual + ffn_out

        return x


if __name__ == "__main__":
    print("=" * 78)
    print("🚀 Module 06: Production Transformer Decoder Block (PyTorch 2.x SDPA)")
    print("=" * 78)

    torch.manual_seed(42)

    dim = 64
    num_heads = 4
    num_kv_heads = 2  # GQA: 4 query heads share 2 KV heads!
    seq_len = 6
    batch_size = 2

    block = TransformerDecoderBlock(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads
    )

    x = torch.randn(batch_size, seq_len, dim)
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    # 1. Full Sequence Forward Pass
    out = block(x, position_ids, is_causal=True)

    print("\n[Architecture Specifications]")
    print(f"  Model Dimension (d_model)   : {dim}")
    print(f"  Attention Heads             : {num_heads} (Query), {num_kv_heads} (Key/Value - GQA 2:1)")
    print(f"  Normalization               : RMSNorm (Root Mean Square, no mean-centering)")
    print(f"  Feed-Forward Network        : SwiGLU (Gated SiLU projection)")
    print(f"  Attention Engine            : F.scaled_dot_product_attention (FlashAttention backend)")
    print(f"\n[Forward Pass Execution]")
    print(f"  Input Shape  : {x.shape}")
    print(f"  Output Shape : {out.shape} (Preserved d_model)")

    # 2. Autoregressive KV-Cache Correctness Verification
    print("\n" + "-" * 78)
    print("🧪 Verifying KV-Cache Autoregressive Output vs Full-Sequence Output")
    print("-" * 78)

    cache = KVCache(num_layers=1)
    # Generate token-by-token
    cached_outputs = []
    for t in range(seq_len):
        x_token = x[:, t : t + 1, :]
        pos_token = position_ids[:, t : t + 1]
        out_token = block(x_token, pos_token, kv_cache=cache, layer_idx=0)
        cached_outputs.append(out_token)

    cached_full = torch.cat(cached_outputs, dim=1)

    max_diff = torch.max(torch.abs(out - cached_full)).item()
    print(f"  Maximum absolute difference between Cached & Full forward: {max_diff:.2e}")
    if max_diff < 1e-4:
        print("  ✅ KV-Cache generates IDENTICAL tokens to full-sequence forward pass!")
    else:
        print("  ⚠️ Deviation detected.")
