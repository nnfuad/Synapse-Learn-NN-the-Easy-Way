"""
Unit Tests for Module 05: Sequential Processing & Recurrent Dynamics
Verifies LSTM cell gate activations, Constant Error Carousel (CEC) arithmetic,
analytical backward gradients, Vanilla RNN vanishing gradient decay,
Bahdanau additive attention probability properties, and Seq2Seq end-to-end flow.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import pytest

from lstm_scratch import LSTMCellScratch, LSTMScratch, VanillaRNNScratch
from bahdanau_attention_pytorch import (
    BahdanauAttention,
    Seq2SeqEncoder,
    Seq2SeqDecoder,
    Seq2SeqModel,
)


def test_lstm_cell_scratch_gate_bounds():
    """Verify all gate activations conform to theoretical range bounds."""
    cell = LSTMCellScratch(input_dim=5, hidden_dim=8, seed=42)
    x = np.random.randn(5, 3)
    h_prev = np.random.randn(8, 3)
    c_prev = np.random.randn(8, 3)

    h_t, c_t, cache = cell.forward(x, h_prev, c_prev)

    # Sigmoid gates must lie strictly in (0, 1)
    assert np.all(cache["f_t"] >= 0.0) and np.all(cache["f_t"] <= 1.0), "Forget gate out of [0, 1]"
    assert np.all(cache["i_t"] >= 0.0) and np.all(cache["i_t"] <= 1.0), "Input gate out of [0, 1]"
    assert np.all(cache["o_t"] >= 0.0) and np.all(cache["o_t"] <= 1.0), "Output gate out of [0, 1]"

    # Tanh candidate must lie in [-1, 1]
    assert np.all(cache["c_tilde"] >= -1.0) and np.all(cache["c_tilde"] <= 1.0), "Candidate cell out of [-1, 1]"

    # Hidden state h_t = o_t * tanh(c_t) must lie in [-1, 1]
    assert np.all(h_t >= -1.0) and np.all(h_t <= 1.0), "Hidden state out of [-1, 1]"


def test_lstm_constant_error_carousel_preservation():
    """Verify that when forget gate = 1 and input gate = 0, memory is preserved perfectly."""
    cell = LSTMCellScratch(input_dim=3, hidden_dim=4, seed=1)
    c_initial = np.array([[1.5], [-2.0], [0.75], [3.14]])
    h_prev = np.zeros((4, 1))
    x = np.zeros((3, 1))

    # Force forget gate = 1 and input gate = 0
    cell.b_f = np.ones((4, 1)) * 50.0   # sigmoid(50) = 1.0
    cell.b_i = np.ones((4, 1)) * -50.0  # sigmoid(-50) = 0.0

    _, c_t, cache = cell.forward(x, h_prev, c_initial)

    # Cell state update: c_t = 1.0 * c_initial + 0.0 * c_tilde = c_initial
    np.testing.assert_allclose(c_t, c_initial, atol=1e-5)

    # In backward pass, dc_prev = total_dc_t * f_t = total_dc_t * 1.0 (Gradient highway!)
    dh_t = np.zeros((4, 1))
    dc_t = np.array([[1.0], [1.0], [1.0], [1.0]])
    _, _, dc_prev, _ = cell.backward(dh_t, dc_t, cache)
    np.testing.assert_allclose(dc_prev, dc_t, atol=1e-5)


def test_lstm_scratch_unrolled_shapes():
    """Verify multi-step LSTM sequence processing output shapes."""
    seq_len = 7
    batch_size = 4
    input_dim = 5
    hidden_dim = 10

    lstm = LSTMScratch(input_dim=input_dim, hidden_dim=hidden_dim, seed=42)
    X = np.random.randn(seq_len, batch_size, input_dim)

    H, (h_T, c_T), caches = lstm.forward(X)

    assert H.shape == (seq_len, batch_size, hidden_dim)
    assert h_T.shape == (hidden_dim, batch_size)
    assert c_T.shape == (hidden_dim, batch_size)
    assert len(caches) == seq_len

    # Test backward pass shapes
    dH = np.random.randn(*H.shape)
    dX, grads = lstm.backward(dH, caches)

    assert dX.shape == X.shape
    for k in ["dW_f", "dU_f", "db_f", "dW_i", "dU_i", "db_i", "dW_c", "dU_c", "db_c", "dW_o", "dU_o", "db_o"]:
        assert k in grads
        assert grads[k].shape == getattr(lstm.cell, k.replace("d", "")).shape


def test_lstm_cell_numerical_gradient_check():
    """
    Finite difference numerical gradient check against analytical backward pass
    for LSTMCellScratch parameters.
    """
    cell = LSTMCellScratch(input_dim=2, hidden_dim=2, seed=7)
    x = np.random.randn(2, 1)
    h_prev = np.random.randn(2, 1)
    c_prev = np.random.randn(2, 1)

    # Forward
    h_t, c_t, cache = cell.forward(x, h_prev, c_prev)

    # Let scalar loss L = sum(h_t) + sum(c_t)
    dh_t = np.ones_like(h_t)
    dc_t = np.ones_like(c_t)
    _, _, _, analytical_grads = cell.backward(dh_t, dc_t, cache)

    # Check gradient w.r.t W_f numerically
    eps = 1e-6
    W_f_orig = cell.W_f.copy()
    num_grad_W_f = np.zeros_like(cell.W_f)

    for i in range(cell.W_f.shape[0]):
        for j in range(cell.W_f.shape[1]):
            cell.W_f = W_f_orig.copy()
            cell.W_f[i, j] += eps
            h_plus, c_plus, _ = cell.forward(x, h_prev, c_prev)
            loss_plus = np.sum(h_plus) + np.sum(c_plus)

            cell.W_f = W_f_orig.copy()
            cell.W_f[i, j] -= eps
            h_minus, c_minus, _ = cell.forward(x, h_prev, c_prev)
            loss_minus = np.sum(h_minus) + np.sum(c_minus)

            num_grad_W_f[i, j] = (loss_plus - loss_minus) / (2.0 * eps)

    cell.W_f = W_f_orig
    rel_error = np.linalg.norm(analytical_grads["dW_f"] - num_grad_W_f) / (
        np.linalg.norm(analytical_grads["dW_f"]) + np.linalg.norm(num_grad_W_f) + 1e-12
    )
    assert rel_error < 1e-4, f"Gradient check failed with relative error {rel_error}"


def test_vanilla_rnn_vs_lstm_vanishing_gradient():
    """Verify that Vanilla RNN gradients decay exponentially while LSTM maintains signal."""
    vanilla = VanillaRNNScratch(input_dim=2, hidden_dim=6, spectral_radius=0.85, seed=123)
    seq_len = 20
    vanilla_curve, lstm_curve = vanilla.forward_and_track_gradients(seq_len=seq_len)

    # Gradient at step 0 compared to step 19
    vanilla_decay_ratio = vanilla_curve[0] / (vanilla_curve[-1] + 1e-12)
    lstm_decay_ratio = lstm_curve[0] / (lstm_curve[-1] + 1e-12)

    # Vanilla RNN decays significantly more than LSTM
    assert vanilla_decay_ratio < 0.1, "Vanilla RNN did not show expected gradient decay"
    assert lstm_decay_ratio > vanilla_decay_ratio * 3.0, "LSTM did not significantly out-preserve gradient"


def test_bahdanau_attention_weights_sum_to_one():
    """Verify Bahdanau attention produces valid probability distribution summing to 1.0."""
    torch.manual_seed(42)
    batch_size = 4
    enc_seq_len = 8
    enc_dim = 16
    dec_dim = 24

    attn = BahdanauAttention(enc_hidden_dim=enc_dim, dec_hidden_dim=dec_dim, attn_dim=32)

    dec_hidden = torch.randn(batch_size, dec_dim)
    enc_outputs = torch.randn(batch_size, enc_seq_len, enc_dim)

    context, weights = attn(dec_hidden, enc_outputs)

    assert context.shape == (batch_size, enc_dim)
    assert weights.shape == (batch_size, enc_seq_len)

    # Sum along sequence dimension must be 1.0
    sums = weights.sum(dim=-1)
    np.testing.assert_allclose(sums.detach().numpy(), np.ones(batch_size), atol=1e-5)


def test_bahdanau_attention_masking():
    """Verify that masked tokens receive 0 attention weight."""
    torch.manual_seed(42)
    attn = BahdanauAttention(enc_hidden_dim=8, dec_hidden_dim=8, attn_dim=16)

    dec_hidden = torch.randn(2, 8)
    enc_outputs = torch.randn(2, 5, 8)

    # Mask: 1st sequence has 5 real tokens, 2nd sequence has 3 real tokens and 2 pad tokens
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])

    _, weights = attn(dec_hidden, enc_outputs, mask=mask)

    # Padded tokens in batch 1 must have 0.0 weight
    assert torch.all(weights[1, 3:] == 0.0)
    assert weights[1, :3].sum().item() == pytest.approx(1.0, abs=1e-5)


def test_seq2seq_model_end_to_end_forward_backward():
    """Verify complete Seq2Seq forward and backward pass with cross entropy loss."""
    torch.manual_seed(42)
    vocab_size = 15
    embed_dim = 16
    enc_dim = 32
    dec_dim = 32

    encoder = Seq2SeqEncoder(vocab_size, embed_dim, enc_hidden_dim=enc_dim, dec_hidden_dim=dec_dim)
    decoder = Seq2SeqDecoder(vocab_size, embed_dim, enc_hidden_dim=enc_dim, dec_hidden_dim=dec_dim)
    model = Seq2SeqModel(encoder, decoder)

    src = torch.randint(1, vocab_size, (3, 6))
    trg = torch.randint(1, vocab_size, (3, 5))

    outputs, attentions = model(src, trg, teacher_forcing_ratio=0.5)

    assert outputs.shape == (3, 5, vocab_size)
    assert attentions.shape == (3, 5, 6)

    # Compute loss and check backprop
    loss = torch.nn.functional.cross_entropy(
        outputs[:, 1:].reshape(-1, vocab_size),
        trg[:, 1:].reshape(-1)
    )
    loss.backward()

    # Check that gradients were populated
    assert encoder.embedding.weight.grad is not None
    assert decoder.attention.v_a.weight.grad is not None
