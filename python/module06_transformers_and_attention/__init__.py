"""
Module 06: The Modern Transformer Revolution & Attention Mechanics
Project Synapse: The Graphical Neural Network Explainer

Historical Era: 2017 - 2024
- Vaswani et al. (2017): Attention Is All You Need (Multi-Head Self-Attention)
- Shazeer (2019): Fast Transformer Decoding: One Write-Head is All You Need (MQA/KV-Cache)
- Su et al. (2021): RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)
- Dao et al. (2022): FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- Ainslie et al. (2023): GQA: Training Generalized Multi-Query Transformer Models
"""

from .attention_scratch import (
    scaled_dot_product_attention_scratch,
    MultiHeadAttentionScratch,
)
from .rope_and_kv_cache import (
    RotaryPositionalEmbedding,
    KVCache,
    apply_rotary_pos_emb,
)
from .transformer_block_pytorch import (
    RMSNorm,
    SwiGLU,
    TransformerDecoderBlock,
)

__all__ = [
    "scaled_dot_product_attention_scratch",
    "MultiHeadAttentionScratch",
    "RotaryPositionalEmbedding",
    "KVCache",
    "apply_rotary_pos_emb",
    "RMSNorm",
    "SwiGLU",
    "TransformerDecoderBlock",
]
