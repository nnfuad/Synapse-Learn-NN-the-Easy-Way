"""
Unit Tests for Module 09: Mechanistic Interpretability & Frontiers
Verifies:
1. Sparse Autoencoder (SAE) forward pass, reconstruction, and L1 sparsity loss.
2. SAE unit norm column projection constraint on decoder weights.
3. Recovery of ground-truth monosemantic feature vectors from 2D superposition.
4. Causal attention masking and attention pattern row stochasticity (summing to 1).
5. 2-layer Attention-Only Transformer forward pass.
6. Induction head mechanism ([A][B] ... [A] -> [B]) copying and attention spike.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import pytest

from sparse_autoencoder_sae import (
    SparseAutoencoder,
    SyntheticSuperpositionData,
)
from circuit_induction_heads import (
    AttentionHead,
    MultiHeadAttentionLayer,
    TwoLayerToyTransformer,
    generate_induction_sequences,
    train_induction_circuit,
)


# ---------------------------------------------------------------------------
# Test 1: Sparse Autoencoder Architecture & Constraints
# ---------------------------------------------------------------------------

def test_sae_forward_shapes_and_losses():
    d_in = 4
    d_sae = 16
    sae = SparseAutoencoder(d_in=d_in, d_sae=d_sae, l1_coefficient=0.05)

    x = torch.randn(8, d_in)
    x_hat, f, loss, metrics = sae(x)

    assert x_hat.shape == (8, d_in)
    assert f.shape == (8, d_sae)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    assert "recon_mse" in metrics
    assert "l1_loss" in metrics
    assert "l0_active_features" in metrics
    assert "explained_variance_r2" in metrics


def test_sae_decoder_column_normalization():
    sae = SparseAutoencoder(d_in=3, d_sae=12)
    # Mutate weights arbitrarily
    sae.w_dec.data = torch.randn_like(sae.w_dec.data) * 5.0
    sae.normalize_decoder_columns()

    # Column norms must be 1.0
    col_norms = torch.norm(sae.w_dec.data, p=2, dim=0)
    assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5)


def test_sae_superposition_disentanglement():
    """
    Tests that SAE disentangles 5 features in 2D space,
    achieving high explained variance and recovering ground-truth directions.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    toy_data = SyntheticSuperpositionData(d_in=2, num_features=5, sparsity=0.7)
    train_x, _ = toy_data.generate_batch(1500)
    test_x, _ = toy_data.generate_batch(300)

    sae = SparseAutoencoder(d_in=2, d_sae=8, l1_coefficient=0.03)
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.03)

    for _ in range(200):
        optimizer.zero_grad()
        _, _, loss, _ = sae(train_x)
        loss.backward()
        optimizer.step()
        sae.normalize_decoder_columns()

    _, _, _, test_metrics = sae(test_x)
    assert test_metrics["explained_variance_r2"] > 0.85, "SAE must achieve >85% explained variance"

    max_sims = toy_data.compute_feature_recovery_cosine_sim(sae)
    assert np.mean(max_sims) > 0.85, "Learned decoder columns must align with ground truth features"


# ---------------------------------------------------------------------------
# Test 2: Attention Circuits & Causal Masking
# ---------------------------------------------------------------------------

def test_attention_causal_mask_and_stochasticity():
    d_model = 16
    d_head = 8
    head = AttentionHead(d_model=d_model, d_head=d_head)

    seq_len = 5
    x = torch.randn(2, seq_len, d_model)
    causal_mask = torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0).repeat(2, 1, 1)

    _, attn = head(x, causal_mask)

    # 1. Shape check (B, S, S)
    assert attn.shape == (2, seq_len, seq_len)

    # 2. Rows sum to 1.0
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    # 3. Future tokens have strictly 0 attention
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert torch.all(attn[:, i, j] == 0.0), f"Position {i} attended to future token {j}"


# ---------------------------------------------------------------------------
# Test 3: 2-Layer Transformer & Induction Head Mechanism
# ---------------------------------------------------------------------------

def test_two_layer_transformer_shapes():
    model = TwoLayerToyTransformer(vocab_size=16, d_model=24, num_heads=2)
    tokens = torch.randint(0, 16, (3, 7))
    logits, attn0, attn1 = model(tokens)

    assert logits.shape == (3, 7, 16)
    assert attn0.shape == (3, 2, 7, 7)
    assert attn1.shape == (3, 2, 7, 7)


def test_induction_circuit_training_and_copy_mechanism():
    """
    Verifies that a 2-layer transformer trained on repeated sequences
    learns to attend to token B given second trigger token A.
    """
    torch.manual_seed(99)
    np.random.seed(99)

    vocab_size = 12
    d_model = 24
    model = TwoLayerToyTransformer(vocab_size=vocab_size, d_model=d_model, num_heads=2)

    # Quick training for 180 steps
    losses = train_induction_circuit(model, num_steps=180, lr=0.015)
    assert losses[-1] < losses[0], "Circuit loss must decrease during training"

    # Test on a specific induction sequence: [A=4, B=9, distractor_1=2, distractor_2=6, A=4]
    test_seq = torch.tensor([[4, 9, 2, 6, 4]], dtype=torch.long)
    logits, attn0, attn1 = model(test_seq)

    # Check that predicting next token produces target token B=9
    top_pred = torch.argmax(logits[0, -1, :]).item()
    assert top_pred == 9, f"Expected induction prediction 9 (Token B), got {top_pred}"

    # Check that in Layer 1, at least one head strongly attends from position 4 to position 1 (token B)
    # attn1 shape: (B, H, S, S)
    head_attns_to_b = [attn1[0, h, 4, 1].item() for h in range(2)]
    max_attn = max(head_attns_to_b)
    assert max_attn > 0.40, f"Expected induction head attention to token B > 40%, got {max_attn:.1%}"
