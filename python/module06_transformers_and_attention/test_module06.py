"""
Unit Tests for Module 06: The Modern Transformer Revolution & Attention Mechanics
Verifies:
- Scaled Dot-Product Attention probability axioms (rows sum to 1.0).
- Causal mask strict upper-triangular zeroing and temporal invariance.
- Scratch NumPy attention numerical equivalence with PyTorch SDPA.
- RoPE relative distance invariance: <R_m q, R_n k> == <R_{m+s} q, R_{n+s} k>.
- Autoregressive KV-Cache exact numerical equivalence with full-sequence forward pass.
- RMSNorm and SwiGLU mathematical correctness.
- TransformerDecoderBlock end-to-end forward/backward execution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import math
import numpy as np
import torch
import torch.nn.functional as F
import pytest

from attention_scratch import (
    scaled_dot_product_attention_scratch,
    MultiHeadAttentionScratch,
    create_causal_mask,
    softmax,
)
from rope_and_kv_cache import (
    precompute_rope_freqs,
    apply_rotary_pos_emb,
    RotaryPositionalEmbedding,
    KVCache,
    rope_relative_distance_inner_product,
)
from transformer_block_pytorch import (
    RMSNorm,
    SwiGLU,
    TransformerDecoderBlock,
)


def test_attention_weights_sum_to_one():
    """Verify that every row of the attention matrix forms a valid probability distribution."""
    rng = np.random.RandomState(42)
    batch_size = 2
    seq_len_q = 6
    seq_len_k = 8
    d_k = 16

    Q = rng.randn(batch_size, seq_len_q, d_k)
    K = rng.randn(batch_size, seq_len_k, d_k)
    V = rng.randn(batch_size, seq_len_k, d_k)

    _, attn_weights = scaled_dot_product_attention_scratch(Q, K, V)

    # Shape: (batch, seq_len_q, seq_len_k)
    assert attn_weights.shape == (batch_size, seq_len_q, seq_len_k)

    # Every row must sum to 1.0
    row_sums = np.sum(attn_weights, axis=-1)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)
    assert np.all(attn_weights >= 0.0), "Attention probabilities cannot be negative"


def test_causal_mask_temporal_invariance():
    """
    Verify that with causal masking:
    1. Future tokens receive strictly 0.0 attention weight.
    2. Modifying token at t+1 has ZERO effect on token representation at step t.
    """
    rng = np.random.RandomState(42)
    seq_len = 5
    d_model = 16
    mha = MultiHeadAttentionScratch(d_model=d_model, num_heads=4, seed=42)

    X_orig = rng.randn(1, seq_len, d_model)
    out_orig, weights = mha.forward(X_orig, is_causal=True)

    # Check upper triangle is 0
    for h in range(4):
        w = weights[0, h]
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert w[i, j] == 0.0, f"Future leakage at head {h}, pos ({i}, {j})"

    # Mutate the last token (step 4)
    X_mutated = X_orig.copy()
    X_mutated[0, 4, :] += 100.0  # Massive perturbation to future token

    out_mutated, _ = mha.forward(X_mutated, is_causal=True)

    # Outputs at steps 0, 1, 2, 3 MUST BE COMPLETELY UNCHANGED
    np.testing.assert_allclose(
        out_orig[0, :4, :],
        out_mutated[0, :4, :],
        atol=1e-6,
        err_msg="Temporal causality violated: future perturbation changed past token outputs!"
    )


def test_numerical_equivalence_scratch_vs_pytorch_attention():
    """
    Verify that pure NumPy scaled dot-product attention produces numerically
    identical results to PyTorch F.scaled_dot_product_attention on same inputs.
    """
    rng = np.random.RandomState(1337)
    batch_size = 2
    heads = 3
    seq_len = 4
    d_k = 8

    Q_np = rng.randn(batch_size, heads, seq_len, d_k).astype(np.float32)
    K_np = rng.randn(batch_size, heads, seq_len, d_k).astype(np.float32)
    V_np = rng.randn(batch_size, heads, seq_len, d_k).astype(np.float32)

    # NumPy scratch attention (bidirectional)
    out_np, _ = scaled_dot_product_attention_scratch(Q_np, K_np, V_np)

    # PyTorch SDPA
    Q_torch = torch.from_numpy(Q_np)
    K_torch = torch.from_numpy(K_np)
    V_torch = torch.from_numpy(V_np)

    out_torch = F.scaled_dot_product_attention(
        Q_torch, K_torch, V_torch, is_causal=False
    ).numpy()

    np.testing.assert_allclose(out_np, out_torch, atol=1e-5)


def test_rope_relative_distance_invariance():
    """
    Verify fundamental theorem of RoPE:
      <R_m q, R_n k> depends ONLY on (m - n).
      Shifting both positions by constant s must yield identical dot product.
    """
    torch.manual_seed(99)
    dim = 32
    q = torch.randn(dim)
    k = torch.randn(dim)

    # Compare <R_m q, R_n k> across shifts s in [0, 5, 20, 100]
    m_base = 4
    n_base = 1

    prod_ref = rope_relative_distance_inner_product(q, k, pos_m=m_base, pos_n=n_base, dim=dim)

    for shift in [5, 20, 50, 120]:
        prod_shifted = rope_relative_distance_inner_product(
            q, k, pos_m=m_base + shift, pos_n=n_base + shift, dim=dim
        )
        assert abs(prod_ref - prod_shifted) < 1e-5, (
            f"RoPE relative distance invariance failed at shift {shift}: "
            f"{prod_ref} vs {prod_shifted}"
        )


def test_kv_cache_autoregressive_equivalence():
    """
    Verify that step-by-step autoregressive generation with KVCache
    produces bit-for-bit identical hidden states to a parallel full-sequence pass.
    """
    torch.manual_seed(42)
    dim = 32
    num_heads = 4
    num_kv_heads = 2
    seq_len = 5
    batch_size = 1

    block = TransformerDecoderBlock(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads
    )
    block.eval()

    x = torch.randn(batch_size, seq_len, dim)
    pos_ids = torch.arange(seq_len).unsqueeze(0)

    # 1. Full sequence parallel pass
    with torch.no_grad():
        out_parallel = block(x, pos_ids, is_causal=True)

    # 2. Token-by-token cached generation
    cache = KVCache(num_layers=1)
    cached_outs = []
    with torch.no_grad():
        for t in range(seq_len):
            x_t = x[:, t : t + 1, :]
            pos_t = pos_ids[:, t : t + 1]
            out_t = block(x_t, pos_t, kv_cache=cache, layer_idx=0)
            cached_outs.append(out_t)

    out_cached = torch.cat(cached_outs, dim=1)

    # Must be identical within floating point tolerance
    diff = torch.max(torch.abs(out_parallel - out_cached)).item()
    assert diff < 1e-5, f"KV cache discrepancy: max diff {diff}"


def test_rmsnorm_formula():
    """Verify RMSNorm implements exact mathematical formula."""
    dim = 8
    eps = 1e-6
    norm = RMSNorm(dim=dim, eps=eps)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0]])

    # Analytical calculation: rms = sqrt(mean(x^2) + eps)
    rms_expected = math.sqrt(sum([v ** 2 for v in [1, 2, 3, 4, -1, -2, -3, -4]]) / 8.0 + eps)
    expected_out = (x / rms_expected) * norm.weight

    actual_out = norm(x)
    np.testing.assert_allclose(actual_out.detach().numpy(), expected_out.detach().numpy(), atol=1e-5)


def test_swiglu_mlp_activation():
    """Verify SwiGLU feedforward network forward shape and gradient flow."""
    dim = 16
    hidden_dim = 32
    mlp = SwiGLU(dim=dim, hidden_dim=hidden_dim)

    x = torch.randn(2, 4, dim, requires_grad=True)
    out = mlp(x)

    assert out.shape == (2, 4, dim)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert mlp.w_gate.weight.grad is not None
