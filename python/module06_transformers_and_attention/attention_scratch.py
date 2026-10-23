"""
Module 06: The Modern Transformer Revolution & Attention Mechanics
Implementation 1: Scaled Dot-Product & Multi-Head Attention from Pure NumPy Scratch

Historical Context:
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need.
  Advances in Neural Information Processing Systems (NeurIPS 2017).

The Physical Analogy:
- The Room of Diplomats:
  Imagine a summit of international diplomats. Each diplomat holds three envelopes:
  1. Query (Q): "What question or piece of information am I currently looking for?"
  2. Key (K): "What topic or area of expertise do I represent?"
  3. Value (V): "What actual facts and knowledge do I contain?"
  When a diplomat steps forward, they present their Query to every other diplomat's
  Key envelope. The dot product measures the resonance (similarity).
  Softmax converts these resonance scores into attention probabilities that sum to 1.0.
  Finally, the diplomat collects a blended aggregate of everyone's Value envelopes,
  weighted precisely by those attention probabilities!

Why Scale by 1 / sqrt(d_k)?
- If components of Q and K are independent random variables with zero mean and
  unit variance, their inner product q · k = sum_{i=1}^{d_k} q_i k_i has mean 0
  and variance d_k.
- For large d_k (e.g., 64 or 128), values grow to large magnitudes, pushing
  the Softmax function into regions with near-zero gradients (Softmax saturation).
- Dividing by sqrt(d_k) normalizes the variance back to 1.0, preserving healthy
  gradient flow during training!
"""

import numpy as np
from typing import Tuple, Optional


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable Softmax along the specified axis.
    Subtracts max(x) to prevent floating-point overflow in exp(x).
    """
    max_x = np.max(x, axis=axis, keepdims=True)
    # Handle rows that might be completely masked (-inf)
    exp_x = np.exp(x - max_x)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    # Protect against 0 division in completely masked rows
    sum_exp = np.where(sum_exp == 0, 1.0, sum_exp)
    return exp_x / sum_exp


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Constructs an autoregressive causal mask where future positions (j > i)
    are filled with -inf so they receive exactly 0 attention probability.
    
    Returns:
      mask: Shape (seq_len, seq_len) with 0.0 in lower triangle and -1e9 in upper triangle.
    """
    mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float64), k=1)
    return mask


def scaled_dot_product_attention_scratch(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Pure NumPy Scaled Dot-Product Attention.
    
    Formula:
      Attention(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
      
    Args:
      Q: Query matrix of shape (..., seq_len_q, d_k)
      K: Key matrix of shape (..., seq_len_k, d_k)
      V: Value matrix of shape (..., seq_len_k, d_v)
      mask: Optional mask broadcastable to (..., seq_len_q, seq_len_k)
      
    Returns:
      output: Attended representations of shape (..., seq_len_q, d_v)
      attn_weights: Normalized attention weights of shape (..., seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]
    
    # 1. Compute raw dot-product resonance: Q @ K^T
    # Shape: (..., seq_len_q, seq_len_k)
    scores = np.matmul(Q, np.swapaxes(K, -2, -1))
    
    # 2. Scale by 1 / sqrt(d_k) to stabilize variance
    scores = scores / np.sqrt(d_k)
    
    # 3. Apply mask (e.g. causal upper triangle set to -inf)
    if mask is not None:
        scores = scores + mask
        
    # 4. Normalize rows into probabilities via Softmax
    attn_weights = softmax(scores, axis=-1)
    
    # 5. Weighted combination of Value vectors: A @ V
    # Shape: (..., seq_len_q, d_v)
    output = np.matmul(attn_weights, V)
    
    return output, attn_weights


class MultiHeadAttentionScratch:
    r"""
    Multi-Head Attention (MHA) from scratch in pure NumPy.
    
    Formula:
      \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
      \text{where head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        seed: int = 42
    ):
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (d_model + d_model))

        # Projection weights
        self.W_q = rng.randn(d_model, d_model) * scale
        self.W_k = rng.randn(d_model, d_model) * scale
        self.W_v = rng.randn(d_model, d_model) * scale
        self.W_o = rng.randn(d_model, d_model) * scale

        # Optional biases
        self.b_q = np.zeros(d_model)
        self.b_k = np.zeros(d_model)
        self.b_v = np.zeros(d_model)
        self.b_o = np.zeros(d_model)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split last dimension into (num_heads, d_k) and transpose to (batch, num_heads, seq_len, d_k).
        """
        batch_size, seq_len, _ = x.shape
        # (batch, seq_len, num_heads, d_k)
        x_reshaped = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        # (batch, num_heads, seq_len, d_k)
        return np.transpose(x_reshaped, (0, 2, 1, 3))

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Transpose (batch, num_heads, seq_len, d_k) back to (batch, seq_len, d_model).
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        x_transposed = np.transpose(x, (0, 2, 1, 3))
        return x_transposed.reshape(batch_size, seq_len, self.d_model)

    def forward(
        self,
        X_q: np.ndarray,
        X_k: Optional[np.ndarray] = None,
        X_v: Optional[np.ndarray] = None,
        is_causal: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for Multi-Head Attention.
        
        Args:
          X_q: Query input (batch_size, seq_len_q, d_model)
          X_k: Key input (batch_size, seq_len_k, d_model) (defaults to X_q for self-attention)
          X_v: Value input (batch_size, seq_len_k, d_model) (defaults to X_k for self-attention)
          is_causal: If True, applies lower-triangular causal mask
          
        Returns:
          output: (batch_size, seq_len_q, d_model)
          attn_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        if X_k is None:
            X_k = X_q
        if X_v is None:
            X_v = X_k

        batch_size, seq_len_q, _ = X_q.shape
        seq_len_k = X_k.shape[1]

        # 1. Project inputs: (batch, seq_len, d_model)
        Q = X_q @ self.W_q + self.b_q
        K = X_k @ self.W_k + self.b_k
        V = X_v @ self.W_v + self.b_v

        # 2. Split into multiple attention heads: (batch, num_heads, seq_len, d_k)
        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)

        # 3. Create causal mask if requested
        mask = None
        if is_causal:
            # (1, 1, seq_len_q, seq_len_k)
            mask = create_causal_mask(seq_len_q).reshape(1, 1, seq_len_q, seq_len_k)

        # 4. Scaled Dot-Product Attention across all heads in parallel
        # head_outputs: (batch, num_heads, seq_len_q, d_k)
        # attn_weights: (batch, num_heads, seq_len_q, seq_len_k)
        head_outputs, attn_weights = scaled_dot_product_attention_scratch(
            Q_heads, K_heads, V_heads, mask=mask
        )

        # 5. Concatenate heads back to d_model: (batch, seq_len_q, d_model)
        combined = self._combine_heads(head_outputs)

        # 6. Final linear projection: W_o
        output = combined @ self.W_o + self.b_o

        return output, attn_weights


if __name__ == "__main__":
    print("=" * 78)
    print("⚡ Module 06: Scaled Dot-Product & Multi-Head Attention Scratch")
    print("=" * 78)

    np.random.seed(42)

    tokens = ["The", "bank", "robber", "fled", "down", "the", "river", "bank"]
    seq_len = len(tokens)
    d_model = 16
    num_heads = 4

    # Generate synthetic token embeddings
    X = np.random.randn(1, seq_len, d_model)

    # Initialize MHA
    mha = MultiHeadAttentionScratch(d_model=d_model, num_heads=num_heads, seed=42)

    # Compute Bidirectional Self-Attention
    out_bidi, weights_bidi = mha.forward(X, is_causal=False)
    # Compute Causal (Autoregressive) Self-Attention
    out_causal, weights_causal = mha.forward(X, is_causal=True)

    print(f"\n[Sequence Setup]")
    print(f"  Tokens: {tokens}")
    print(f"  d_model = {d_model}, num_heads = {num_heads}, head_dim (d_k) = {d_model // num_heads}")
    print(f"  Output shape: {out_bidi.shape}")
    print(f"  Attention weights shape: {weights_bidi.shape} (batch, heads, seq, seq)")

    # Verify probability distribution: each row must sum to 1.0
    row_sums = np.sum(weights_causal[0, 0], axis=-1)
    print(f"\n[Causal Mask Probability Verification - Head 0]")
    for i, s in enumerate(row_sums):
        print(f"  Row {i} ('{tokens[i]}') attention sum: {s:.6f}")

    print("\n[Causal Attention Matrix - Head 0 (Upper triangle must be 0.0)]:")
    header = f"{'Token':<10} | " + " | ".join([f"{t:>6}" for t in tokens])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(weights_causal[0, 0]):
        row_str = " | ".join([f"{val:.3f}" if val > 1e-6 else "  .   " for val in row])
        print(f"{tokens[i]:<10} | {row_str}")

    print("\n✅ Verification Complete: Scaled Dot-Product & Multi-Head Attention")
    print("   operate with mathematical fidelity and strict causal bounds!")
