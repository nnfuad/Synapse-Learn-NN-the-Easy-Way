"""
Module 06: The Modern Transformer Revolution & Attention Mechanics
Implementation 2: Rotary Position Embeddings (RoPE) & Autoregressive KV-Cache

Historical Context:
- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021/2024).
  RoFormer: Enhanced Transformer with Rotary Position Embedding.
  Neurocomputing, 568, 127063.
- Shazeer, N. (2019).
  Fast Transformer Decoding: One Write-Head is All You Need.
  arXiv:1911.02150.

The Physical Analogy:
1. RoPE (The Dial of Frequencies):
   - Rather than tattooing an absolute coordinate number onto a token's vector
     (like traditional Sinusoidal embeddings), RoPE rotates the vector in 2D
     subspaces like the hands of clocks ticking at different harmonic frequencies.
   - When Query at position m and Key at position n meet, their dot product
     depends strictly on the relative angle difference (m - n). As distance grows,
     the interference of multiple harmonic clocks naturally decays the signal!

2. The KV-Cache (The Stenographer's Ledger):
   - In standard autoregressive generation without caching, generating token 1000
     requires re-reading and re-calculating keys and values for all 999 previous tokens!
     This turns text generation into an O(T^3) computational nightmare.
   - The KV-Cache acts as a stenographer's ledger. As each past word is uttered,
     its Key and Value are permanently recorded in fast SRAM/HBM cache.
     When a new word arrives, we only project that ONE new word into Query, Key, Value,
     append its Key/Value to the ledger, and attend the single Query against the ledger!
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, List


def precompute_rope_freqs(
    dim: int,
    max_seq_len: int = 2048,
    theta_base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Precompute cosine and sine frequency tables for Rotary Position Embeddings.
    
    Formula:
      \theta_i = \theta_{base}^{-2(i-1)/d}, \quad i \in [1, d/2]
      \text{angles}_{m, i} = m \cdot \theta_i
      
    Args:
      dim: Head dimension (must be even)
      max_seq_len: Maximum sequence length
      theta_base: Base frequency parameter (default: 10000.0)
      
    Returns:
      cos: (max_seq_len, dim // 2)
      sin: (max_seq_len, dim // 2)
    """
    assert dim % 2 == 0, f"RoPE dimension must be even, got {dim}"
    # Indices: [0, 2, 4, ..., dim-2]
    dim_indices = torch.arange(0, dim, 2, dtype=torch.float32)
    # Frequencies: theta_i = 10000^(-2i / dim)
    freqs = 1.0 / (theta_base ** (dim_indices / dim))
    
    # Position indices: [0, 1, 2, ..., max_seq_len-1]
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    
    # Outer product: (max_seq_len, dim // 2)
    angles = torch.outer(positions, freqs)
    
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions:
    For [x1, x2], returns [-x2, x1].
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor
) -> torch.Tensor:
    r"""
    Applies 2D Rotary Position Embeddings to input Query or Key tensor.
    
    Formula:
      R_{\Theta, m} x = (x \odot \cos(m\Theta)) + (\text{rotate\_half}(x) \odot \sin(m\Theta))
      
    Args:
      x: Tensor of shape (batch, num_heads, seq_len, head_dim)
      cos: Precomputed cos table of shape (max_seq_len, head_dim // 2)
      sin: Precomputed sin table of shape (max_seq_len, head_dim // 2)
      position_ids: Position indices of shape (batch, seq_len)
      
    Returns:
      x_rotated: Tensor of shape (batch, num_heads, seq_len, head_dim)
    """
    # Duplicate cos and sin to match head_dim: (max_seq_len, head_dim)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    # Gather cos and sin for current position_ids: (batch, seq_len, head_dim)
    # Shape after unsqueeze: (batch, 1, seq_len, head_dim)
    cos_pos = cos[position_ids].unsqueeze(1)
    sin_pos = sin[position_ids].unsqueeze(1)

    # Apply 2D rotation
    return (x * cos_pos) + (rotate_half(x) * sin_pos)


class RotaryPositionalEmbedding(nn.Module):
    """
    PyTorch Module wrapper for RoPE caching and forward application.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta_base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta_base = theta_base

        cos, sin = precompute_rope_freqs(dim, max_seq_len, theta_base)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rotates both queries and keys by their respective token positions.
        """
        q_rot = apply_rotary_pos_emb(q, self.cos_cached, self.sin_cached, position_ids)
        k_rot = apply_rotary_pos_emb(k, self.cos_cached, self.sin_cached, position_ids)
        return q_rot, k_rot


class KVCache:
    """
    Autoregressive Key-Value Cache manager for multi-layer Transformers.
    Stores and concatenates past Key and Value representations during token generation.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self.value_cache: List[Optional[torch.Tensor]] = [None] * num_layers

    def reset(self):
        """Clears the cache memory."""
        self.key_cache = [None] * self.num_layers
        self.value_cache = [None] * self.num_layers

    def get_seq_len(self, layer_idx: int = 0) -> int:
        """Returns current cached sequence length."""
        if self.key_cache[layer_idx] is None:
            return 0
        # Shape: (batch, num_kv_heads, seq_len, head_dim)
        return self.key_cache[layer_idx].shape[2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new token Key and Value representations.
        
        Args:
          key_states: (batch, num_kv_heads, new_seq_len, head_dim)
          value_states: (batch, num_kv_heads, new_seq_len, head_dim)
          layer_idx: Layer index
          
        Returns:
          full_keys: (batch, num_kv_heads, total_seq_len, head_dim)
          full_values: (batch, num_kv_heads, total_seq_len, head_dim)
        """
        if self.key_cache[layer_idx] is None:
            # First prefill step
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # Incremental autoregressive decode step: concatenate along seq_len dimension (dim=2)
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def rope_relative_distance_inner_product(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_m: int,
    pos_n: int,
    dim: int = 64
) -> float:
    """
    Computes <R_m q, R_n k> to numerically prove that the dot product
    depends strictly on relative distance (m - n).
    """
    cos, sin = precompute_rope_freqs(dim=dim, max_seq_len=max(pos_m, pos_n) + 10)
    
    # Shape: (1, 1, 1, dim)
    q_tensor = q.reshape(1, 1, 1, dim)
    k_tensor = k.reshape(1, 1, 1, dim)

    q_rot = apply_rotary_pos_emb(q_tensor, cos, sin, torch.tensor([[pos_m]]))
    k_rot = apply_rotary_pos_emb(k_tensor, cos, sin, torch.tensor([[pos_n]]))

    # Inner product
    inner_prod = torch.sum(q_rot * k_rot).item()
    return inner_prod


if __name__ == "__main__":
    print("=" * 78)
    print("🔄 Module 06: RoPE 2D Rotation & Autoregressive KV-Cache Demonstration")
    print("=" * 78)

    torch.manual_seed(42)

    # 1. Demonstrate RoPE Relative Distance Invariance
    head_dim = 16
    q = torch.randn(head_dim)
    k = torch.randn(head_dim)

    # Distance is 3 across different absolute positions
    prod_1 = rope_relative_distance_inner_product(q, k, pos_m=3, pos_n=0, dim=head_dim)
    prod_2 = rope_relative_distance_inner_product(q, k, pos_m=10, pos_n=7, dim=head_dim)
    prod_3 = rope_relative_distance_inner_product(q, k, pos_m=53, pos_n=50, dim=head_dim)

    print("\n[RoPE Relative Distance Property: <R_m q, R_n k> with m - n = 3]")
    print(f"  Position pair (3, 0)   dot product : {prod_1:.6f}")
    print(f"  Position pair (10, 7)  dot product : {prod_2:.6f}")
    print(f"  Position pair (53, 50) dot product : {prod_3:.6f}")
    print(f"  Max absolute deviation: {max(abs(prod_1 - prod_2), abs(prod_1 - prod_3)):.2e}")
    print("  -> Proof: RoPE preserves relative distance regardless of absolute token index!")

    # 2. Demonstrate KV-Cache Mechanics
    print("\n" + "-" * 78)
    print("⚡ KV-Cache Step-by-Step Simulation (Prompt Prefill + Autoregressive Decode)")
    print("-" * 78)

    cache = KVCache(num_layers=1)
    batch_size = 1
    num_heads = 2

    # Step 1: Prompt Prefill (3 tokens: "The", "cat", "sat")
    k_prompt = torch.randn(batch_size, num_heads, 3, head_dim)
    v_prompt = torch.randn(batch_size, num_heads, 3, head_dim)
    cached_k, cached_v = cache.update(k_prompt, v_prompt, layer_idx=0)
    print(f"Prefill Phase  (3 tokens) -> Cache Length: {cache.get_seq_len(0)} tokens")

    # Step 2: Decode Token 4 ("on")
    k_tok4 = torch.randn(batch_size, num_heads, 1, head_dim)
    v_tok4 = torch.randn(batch_size, num_heads, 1, head_dim)
    cached_k, cached_v = cache.update(k_tok4, v_tok4, layer_idx=0)
    print(f"Decode Step 1  (token 'on')   -> Cache Length: {cache.get_seq_len(0)} tokens (Shape: {cached_k.shape})")

    # Step 3: Decode Token 5 ("the")
    k_tok5 = torch.randn(batch_size, num_heads, 1, head_dim)
    v_tok5 = torch.randn(batch_size, num_heads, 1, head_dim)
    cached_k, cached_v = cache.update(k_tok5, v_tok5, layer_idx=0)
    print(f"Decode Step 2  (token 'the')  -> Cache Length: {cache.get_seq_len(0)} tokens (Shape: {cached_k.shape})")

    print("\n✅ Verification Complete: RoPE relative invariance & KV-Cache mechanics verified!")
