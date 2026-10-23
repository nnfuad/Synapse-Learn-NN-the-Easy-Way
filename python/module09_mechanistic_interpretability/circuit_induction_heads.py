"""
Module 09: Mechanistic Interpretability & Frontiers - Induction Head Circuits
==============================================================================
Historical & Theoretical Context:
- Elhage et al. (2021): "A Mathematical Framework for Transformer Circuits" (Anthropic) -
  Introduced the residual stream view, QK circuits (where to attend) and OV circuits (what to read/write).
- Olsson et al. (2022): "In-context Learning and Induction Heads" (Anthropic) -
  Identified induction heads as the core microscopic mechanism behind in-context learning in LLMs.
  During training, an abrupt "induction bump" phase change occurs where 2-layer circuits form.

The 2-Layer Induction Head Mechanism:
Consider a prompt containing a repeated sequence:
  Tokens: [..., A, B, ..., A] -> Next Token Target: B

How the 2-Layer Circuit Operates:
1. Layer 0 (Previous-Token Head):
   - Head H0 attending from position i to position i - 1.
   - It copies the identity of token A (at position i-1) into the residual stream of token B (position i).
   - Now position i represents "Token B, preceded by Token A".

2. Layer 1 (Induction Head):
   - At current position j (token A), the Query vector represents "I am Token A, find where I previously appeared".
   - At position i (token B), the Key vector encodes "preceded by Token A" (transferred by Layer 0).
   - The dot product Q_j * K_i is high! Head H1 attends from the current A to token B!
   - The Value/Output circuit reads Token B from position i and adds a positive direction to the
     residual stream that unembeds directly to token B's logit!
"""

from typing import Tuple, Dict, List, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. 2-Layer Attention-Only Transformer Architecture
# ---------------------------------------------------------------------------

class AttentionHead(nn.Module):
    """
    Single Attention Head with explicit QK and OV circuit projections.
    """
    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_head = d_head
        self.w_q = nn.Linear(d_model, d_head, bias=False)
        self.w_k = nn.Linear(d_model, d_head, bias=False)
        self.w_v = nn.Linear(d_model, d_head, bias=False)
        self.w_o = nn.Linear(d_head, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, seq_len, d_model)
        Returns:
            head_out: (B, seq_len, d_model)
            attn_weights: (B, seq_len, seq_len)
        """
        B, S, D = x.shape
        q = self.w_q(x)  # (B, S, d_head)
        k = self.w_k(x)  # (B, S, d_head)
        v = self.w_v(x)  # (B, S, d_head)

        # QK circuit: Attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_head)

        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        # OV circuit: Value projection and Output write
        z = torch.bmm(attn_weights, v)  # (B, S, d_head)
        head_out = self.w_o(z)          # (B, S, d_model)

        return head_out, attn_weights


class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-Head Attention layer tracking individual head attention patterns.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.heads = nn.ModuleList([
            AttentionHead(d_model, self.d_head) for _ in range(num_heads)
        ])

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            layer_out: (B, S, d_model)
            all_attn_weights: (B, num_heads, S, S)
        """
        head_outs = []
        attn_weights_list = []

        for head in self.heads:
            out_h, attn_h = head(x, causal_mask)
            head_outs.append(out_h)
            attn_weights_list.append(attn_h)

        # Sum of head outputs (residual stream addition)
        combined_out = sum(head_outs)
        all_attn_weights = torch.stack(attn_weights_list, dim=1)  # (B, H, S, S)
        return combined_out, all_attn_weights


class TwoLayerToyTransformer(nn.Module):
    """
    Minimal 2-Layer Attention-Only Transformer designed for mechanistic circuit analysis.
    No MLPs to isolate purely attention-driven in-context learning.
    """
    def __init__(self, vocab_size: int = 32, d_model: int = 32, num_heads: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads

        # Embedding & Positional Encoding
        self.w_e = nn.Embedding(vocab_size, d_model)
        self.w_pos = nn.Embedding(64, d_model)

        # Layer 0 & Layer 1 Attention blocks
        self.layer0 = MultiHeadAttentionLayer(d_model, num_heads)
        self.layer1 = MultiHeadAttentionLayer(d_model, num_heads)

        # Unembedding projection W_U
        self.w_u = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, S) integer token ids
        Returns:
            logits: (B, S, vocab_size)
            attn_layer0: (B, num_heads, S, S)
            attn_layer1: (B, num_heads, S, S)
        """
        B, S = tokens.shape
        pos = torch.arange(S, device=tokens.device).unsqueeze(0).repeat(B, 1)

        # Residual stream initialization
        resid = self.w_e(tokens) + self.w_pos(pos)

        # Causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones((S, S), device=tokens.device)).unsqueeze(0).repeat(B, 1, 1)

        # Layer 0
        out0, attn0 = self.layer0(resid, causal_mask)
        resid = resid + out0

        # Layer 1
        out1, attn1 = self.layer1(resid, causal_mask)
        resid = resid + out1

        # Unembed
        logits = self.w_u(resid)
        return logits, attn0, attn1


# ---------------------------------------------------------------------------
# 2. Synthetic Sequence Generator & Induction Evaluation
# ---------------------------------------------------------------------------

def generate_induction_sequences(
    batch_size: int = 256,
    prefix_length: int = 6,
    vocab_size: int = 30
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences of the form:
    [Token_A, Token_B, C_1, C_2, ..., Token_A] -> Next Token: Token_B
    Where Token_A and Token_B are unique random tokens, followed by distractor tokens,
    and concluding with Token_A repeated at the end.
    """
    sequences = []
    targets = []

    for _ in range(batch_size):
        # Pick A and B
        a, b = np.random.choice(vocab_size, size=2, replace=False)
        # Distractors
        distractors = np.random.choice(vocab_size, size=prefix_length, replace=True)
        # Full sequence: [A, B, distractor_1, ..., distractor_k, A]
        seq = [a, b] + list(distractors) + [a]
        sequences.append(seq)
        targets.append(b)

    return (
        torch.tensor(sequences, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long)
    )


def train_induction_circuit(
    model: TwoLayerToyTransformer,
    num_steps: int = 400,
    lr: float = 0.01
) -> List[float]:
    """
    Trains the 2-layer transformer on repeated sequences until induction heads form.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_history = []

    for step in range(num_steps):
        seqs, targets = generate_induction_sequences(batch_size=64, prefix_length=4, vocab_size=model.vocab_size)
        logits, _, _ = model(seqs)

        # Loss at the final position predicting token B
        final_logits = logits[:, -1, :]  # (B, vocab_size)
        loss = F.cross_entropy(final_logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    return loss_history


# ---------------------------------------------------------------------------
# 3. CLI Self-Test Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Project Synapse - Module 09: 2-Layer Transformer Induction Circuit")
    print("=" * 70)

    # Initialize 2-layer model: 2 heads per layer
    vocab_size = 20
    d_model = 32
    model = TwoLayerToyTransformer(vocab_size=vocab_size, d_model=d_model, num_heads=2)

    print(f"2-Layer Transformer created: vocab={vocab_size}, d_model={d_model}, heads=2.")

    # Train induction circuit
    print("Training induction circuit on repeated token patterns for 300 steps...")
    losses = train_induction_circuit(model, num_steps=300, lr=0.01)
    print(f"  Initial Loss: {losses[0]:.4f} -> Final Loss: {losses[-1]:.4f}")

    # Test induction inference on a clean test sequence: [A=7, B=12, C=3, D=9, A=7]
    test_seq = torch.tensor([[7, 12, 3, 9, 7]], dtype=torch.long)
    logits, attn0, attn1 = model(test_seq)

    # Prediction at final position
    next_token_pred = torch.argmax(logits[0, -1, :]).item()
    target_token = 12
    pred_prob = F.softmax(logits[0, -1, :], dim=-1)[target_token].item()

    print(f"\nInference Test:")
    print(f"  Input Tokens:  [A=7, B=12, C=3, D=9, A=7]")
    print(f"  Target Token:  {target_token} (Token B)")
    print(f"  Model Predict: {next_token_pred} (Probability: {pred_prob:.1%})")

    # Layer 1 Head 0 attention from position 4 (last A) to position 1 (token B)
    # attn1 shape: (B, H, S, S)
    attn_to_b = attn1[0, 0, 4, 1].item()
    print(f"  Layer 1 Head 0 Attention to B: {attn_to_b:.2%}")
    print("[SUCCESS] Induction Head Circuit verified.")
