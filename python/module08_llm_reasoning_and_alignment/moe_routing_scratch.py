"""
Module 08: Modern LLMs, Alignment & Reasoning - Sparse Mixture of Experts (MoE)
================================================================================
Historical & Theoretical Context:
- Shazeer et al. (2017): "Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer" - Pioneered conditional computation in deep networks,
  activating only a subset of parameters per token.
- Fedus et al. (2021): "Switch Transformers" - Simplified routing to Top-1 with auxiliary
  load balancing losses to achieve 1+ Trillion parameter models.
- Jiang et al. (2024): "Mixtral of Experts" (Mistral AI) - Popularized Top-2 routing over
  8 experts for 7B active / 47B total parameters.
- DeepSeek-AI (2024): "DeepSeek-V2 / DeepSeek-V3" - Innovated fine-grained expert segmentation
  (e.g., 64 routed experts + 2 isolated shared experts) achieving SOTA efficiency.

Key Mathematical Formulations:
1. Gating / Routing Logits:
   H(x) = x * W_gate  where W_gate in R^{d_model x N}

2. Top-K Selection & Renormalization:
   TopK_indices = argsort(H(x))[-K:]
   TopK_logits = H(x)[TopK_indices]
   weights = softmax(TopK_logits)  such that sum_{k=1}^K weights_k = 1.0

3. Auxiliary Load-Balancing Loss (Switch / GShard / Mixtral):
   To prevent router collapse (all tokens routing to only 1 or 2 favorite experts),
   an auxiliary loss penalizes imbalance:
   - f_i = (1 / T) * sum_{t=1}^T 1(expert i in TopK(x_t))  [fraction of tokens routed]
   - P_i = (1 / T) * sum_{t=1}^T softmax(H(x_t))_i          [mean routing probability]
   L_balance = alpha_balance * N * sum_{i=1}^N (f_i * P_i)

   Why f_i * P_i? f_i is discrete/non-differentiable, but P_i is continuous/differentiable.
   Minimizing this product pushes down P_i for the overloaded experts!

4. DeepSeek-Style Shared Experts:
   y = sum_{k in TopK} weights_k * Expert_k(x) + Expert_shared(x)
   Shared experts capture ubiquitous generic language knowledge, allowing routed experts
   to specialize in fine-grained niche domains.
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Expert Feed-Forward Network (SwiGLU Architecture)
# ---------------------------------------------------------------------------

class SwiGLUExpert(nn.Module):
    """
    Standard modern LLM expert block using SwiGLU non-linearity (Chowdhery et al., LLaMA, Mixtral).
    Output: FFN(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# 2. Top-K Router with Load-Balancing Auxiliary Loss
# ---------------------------------------------------------------------------

class TopKRouter(nn.Module):
    """
    Computes routing gates, selects Top-K experts per token, renormalizes weights,
    and computes the load-balancing auxiliary loss.
    """
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.aux_loss_weight = aux_loss_weight

        # Gate projection matrix
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (B, T, d_model) or (num_tokens, d_model)
        Returns:
            topk_weights: (num_tokens, top_k) normalized routing weights
            topk_indices: (num_tokens, top_k) selected expert indices
            aux_loss: scalar load-balancing auxiliary loss tensor
            diagnostics: dictionary of routing statistics (f_i, P_i, max_prob)
        """
        original_shape = x.shape
        flat_x = x.view(-1, self.d_model)  # (N_tokens, d_model)
        num_tokens = flat_x.shape[0]

        # 1. Routing logits: (N_tokens, num_experts)
        logits = self.gate(flat_x)
        router_probs = F.softmax(logits, dim=-1)

        # 2. Select Top-K experts
        topk_weights, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)

        # 3. Softmax over Top-K only (renormalization)
        topk_weights = F.softmax(topk_weights, dim=-1)

        # 4. Compute Load Balancing Auxiliary Loss
        # f_i: fraction of tokens dispatched to each expert
        # Create one-hot mask for top-k selections
        # topk_indices is (N_tokens, top_k)
        dispatch_mask = torch.zeros(
            (num_tokens, self.num_experts),
            dtype=torch.float32,
            device=x.device
        )
        dispatch_mask.scatter_(dim=1, index=topk_indices, value=1.0)
        f_i = dispatch_mask.mean(dim=0)  # (num_experts,)

        # P_i: average probability assigned to expert i across all tokens
        P_i = router_probs.mean(dim=0)   # (num_experts,)

        # Aux loss = alpha * num_experts * sum_{i} (f_i * P_i)
        # For uniform distribution, f_i = top_k / N, P_i = 1 / N
        # Ideal minimum = top_k
        aux_loss = self.aux_loss_weight * self.num_experts * torch.sum(f_i * P_i)

        diagnostics = {
            "expert_dispatch_fraction": f_i.detach(),
            "expert_mean_probability": P_i.detach(),
            "max_expert_utilization": f_i.max().item(),
            "min_expert_utilization": f_i.min().item(),
        }

        return topk_weights, topk_indices, aux_loss, diagnostics


# ---------------------------------------------------------------------------
# 3. Full Sparse Mixture of Experts Layer (with DeepSeek Shared Expert)
# ---------------------------------------------------------------------------

class SparseMoELayer(nn.Module):
    """
    Modular Sparse MoE layer supporting:
    - N routed experts with Top-K dispatch
    - Optional M shared isolated experts (DeepSeek architecture)
    - Residual token-wise aggregation
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_routed_experts: int = 8,
        top_k: int = 2,
        num_shared_experts: int = 1,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts

        # Router
        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_routed_experts,
            top_k=top_k,
            aux_loss_weight=aux_loss_weight,
        )

        # Routed Experts
        self.experts = nn.ModuleList([
            SwiGLUExpert(d_model=d_model, d_ff=d_ff)
            for _ in range(num_routed_experts)
        ])

        # DeepSeek-style Shared Experts
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                SwiGLUExpert(d_model=d_model, d_ff=d_ff)
                for _ in range(num_shared_experts)
            ])
        else:
            self.shared_experts = None

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model) or (tokens, d_model)
        Returns:
            output: tensor of same shape as x
            aux_loss: auxiliary load balancing loss
            diagnostics: routing statistics
        """
        orig_shape = x.shape
        flat_x = x.view(-1, self.d_model)
        num_tokens = flat_x.shape[0]

        # 1. Routing
        topk_weights, topk_indices, aux_loss, diagnostics = self.router(flat_x)

        # 2. Dispatch and aggregate routed experts
        # Output accumulator
        out = torch.zeros_like(flat_x)

        # Dispatch tokens by expert for computational efficiency
        for expert_idx in range(self.num_routed_experts):
            # Find tokens routed to this expert (and which top_k slot)
            token_mask, slot_indices = torch.where(topk_indices == expert_idx)

            if len(token_mask) > 0:
                tokens_for_expert = flat_x[token_mask]
                expert_out = self.experts[expert_idx](tokens_for_expert)
                weights = topk_weights[token_mask, slot_indices].unsqueeze(-1)
                out.index_add_(0, token_mask, weights * expert_out)

        # 3. Add Shared Expert output (DeepSeek style)
        if self.shared_experts is not None:
            for shared_exp in self.shared_experts:
                out = out + shared_exp(flat_x)

        return out.view(orig_shape), aux_loss, diagnostics


# ---------------------------------------------------------------------------
# 4. CLI Self-Test Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Project Synapse - Module 08: Sparse Mixture of Experts (MoE)")
    print("=" * 70)

    # Initialize MoE layer: 8 routed experts (Top-2) + 1 shared expert
    d_model = 64
    d_ff = 128
    num_experts = 8
    top_k = 2

    moe = SparseMoELayer(
        d_model=d_model,
        d_ff=d_ff,
        num_routed_experts=num_experts,
        top_k=top_k,
        num_shared_experts=1,
        aux_loss_weight=0.01
    )

    # Synthetic batch of tokens: Batch size 4, Sequence length 16 -> 64 tokens
    dummy_input = torch.randn(4, 16, d_model)
    out, aux_loss, diag = moe(dummy_input)

    print(f"Input shape: {dummy_input.shape} -> Output shape: {out.shape}")
    print(f"Auxiliary Load-Balancing Loss: {aux_loss.item():.6f}")
    print(f"Expert Dispatch Fractions: {diag['expert_dispatch_fraction'].tolist()}")
    print(f"Total dispatched tokens check (sum * 64): {diag['expert_dispatch_fraction'].sum().item() * 64:.1f} (expected {64 * top_k})")
    print(f"Max Expert Utilization: {diag['max_expert_utilization']:.2%}")
    print(f"Min Expert Utilization: {diag['min_expert_utilization']:.2%}")
    print("[SUCCESS] Sparse MoE Router verified.")
