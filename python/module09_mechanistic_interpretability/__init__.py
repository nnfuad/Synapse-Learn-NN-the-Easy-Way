"""
Module 09: Mechanistic Interpretability & Frontiers
"""

from .sparse_autoencoder_sae import (
    SparseAutoencoder,
    SyntheticSuperpositionData,
)
from .circuit_induction_heads import (
    AttentionHead,
    MultiHeadAttentionLayer,
    TwoLayerToyTransformer,
    generate_induction_sequences,
    train_induction_circuit,
)

__all__ = [
    "SparseAutoencoder",
    "SyntheticSuperpositionData",
    "AttentionHead",
    "MultiHeadAttentionLayer",
    "TwoLayerToyTransformer",
    "generate_induction_sequences",
    "train_induction_circuit",
]
