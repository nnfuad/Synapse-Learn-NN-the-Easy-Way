"""
Unit Tests for Module 08: Modern LLMs, Alignment & Reasoning
Verifies:
1. SwiGLU expert feed-forward activations.
2. Top-K Router selection, weight normalization, and load-balancing auxiliary loss.
3. Sparse MoE layer dispatch and DeepSeek-style shared expert integration.
4. Direct Preference Optimization (DPO) analytical loss, reward margin, and gradients.
5. Group Relative Policy Optimization (GRPO) advantage normalization, PPO clipping, and KL regularizer.
6. Test-time compute rollout search tree traversal and pruning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import pytest

from moe_routing_scratch import (
    SwiGLUExpert,
    TopKRouter,
    SparseMoELayer,
)
from dpo_and_grpo_loss import (
    DPOLoss,
    GRPOLoss,
    TestTimeReasoningSimulator,
)


# ---------------------------------------------------------------------------
# Test 1: SwiGLU Expert Block
# ---------------------------------------------------------------------------

def test_swiglu_expert_shapes():
    d_model, d_ff = 32, 64
    expert = SwiGLUExpert(d_model=d_model, d_ff=d_ff)
    x = torch.randn(10, d_model)
    y = expert(x)
    assert y.shape == (10, d_model)
    assert not torch.isnan(y).any()


# ---------------------------------------------------------------------------
# Test 2: Top-K Router & Load Balancing Loss
# ---------------------------------------------------------------------------

def test_topk_router_weights_and_indices():
    d_model = 32
    num_experts = 6
    top_k = 2

    router = TopKRouter(d_model=d_model, num_experts=num_experts, top_k=top_k)
    x = torch.randn(20, d_model)

    weights, indices, aux_loss, diag = router(x)

    # 1. Shape checks
    assert weights.shape == (20, top_k)
    assert indices.shape == (20, top_k)

    # 2. Indices in range
    assert torch.all(indices >= 0)
    assert torch.all(indices < num_experts)

    # 3. Softmax weights sum to 1 for each token
    weights_sum = weights.sum(dim=-1)
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)

    # 4. Aux loss is scalar and differentiable
    assert aux_loss.dim() == 0
    assert aux_loss.item() > 0.0
    aux_loss.backward()
    assert router.gate.weight.grad is not None


def test_topk_router_perfect_balance():
    """
    When routing is balanced, the product sum(f_i * P_i) is at its theoretical minimum top_k / N.
    """
    d_model = 16
    num_experts = 4
    top_k = 1
    router = TopKRouter(d_model=d_model, num_experts=num_experts, top_k=top_k, aux_loss_weight=1.0)

    # With zero weights and zero bias, router outputs equal probabilities 1/4 to all experts
    with torch.no_grad():
        router.gate.weight.zero_()

    # Create dummy tokens
    x = torch.randn(100, d_model)
    _, _, aux_loss, diag = router(x)

    # Expected: P_i = 0.25 for all i.
    # sum(f_i * P_i) = 0.25 * sum(f_i) = 0.25 * 1.0 = 0.25
    # aux_loss = 1.0 * 4 * 0.25 = 1.0
    assert torch.isclose(aux_loss, torch.tensor(1.0), atol=1e-3)


# ---------------------------------------------------------------------------
# Test 3: Full Sparse MoE Layer (Routed + Shared Experts)
# ---------------------------------------------------------------------------

def test_sparse_moe_layer_forward_and_backward():
    d_model = 24
    d_ff = 48
    moe = SparseMoELayer(
        d_model=d_model,
        d_ff=d_ff,
        num_routed_experts=4,
        top_k=2,
        num_shared_experts=1,
        aux_loss_weight=0.01,
    )

    x = torch.randn(2, 8, d_model, requires_grad=True)
    out, aux_loss, diag = moe(x)

    assert out.shape == (2, 8, d_model)
    total_loss = out.sum() + aux_loss
    total_loss.backward()

    assert x.grad is not None
    # Check that both routed and shared expert weights received gradients
    assert moe.experts[0].w_gate.weight.grad is not None
    assert moe.shared_experts[0].w_gate.weight.grad is not None


# ---------------------------------------------------------------------------
# Test 4: Direct Preference Optimization (DPO)
# ---------------------------------------------------------------------------

def test_dpo_loss_preference_behavior():
    dpo = DPOLoss(beta=0.1)

    # Scenario A: Policy prefers chosen completion over rejected
    policy_chosen = torch.tensor([-1.0, -1.0])
    policy_rejected = torch.tensor([-5.0, -5.0])
    ref_chosen = torch.tensor([-2.0, -2.0])
    ref_rejected = torch.tensor([-2.0, -2.0])

    loss_aligned, metrics_aligned = dpo(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

    # Scenario B: Policy prefers rejected completion over chosen (misaligned)
    policy_chosen_bad = torch.tensor([-5.0, -5.0])
    policy_rejected_bad = torch.tensor([-1.0, -1.0])

    loss_misaligned, metrics_misaligned = dpo(
        policy_chosen_bad, policy_rejected_bad, ref_chosen, ref_rejected
    )

    assert loss_aligned.item() < loss_misaligned.item()
    assert metrics_aligned["accuracy"].item() == 1.0
    assert metrics_misaligned["accuracy"].item() == 0.0
    assert torch.all(metrics_aligned["reward_margin"] > 0)
    assert torch.all(metrics_misaligned["reward_margin"] < 0)


def test_dpo_loss_gradient_direction():
    """
    Verifies that DPO gradient decreases loss by increasing chosen log-prob
    and decreasing rejected log-prob.
    """
    dpo = DPOLoss(beta=0.2)
    chosen = torch.tensor([-2.0], requires_grad=True)
    rejected = torch.tensor([-2.0], requires_grad=True)
    ref = torch.tensor([-2.0])

    loss, _ = dpo(chosen, rejected, ref, ref)
    loss.backward()

    # dLoss/dChosen should be negative (increasing chosen lowers loss)
    assert chosen.grad.item() < 0
    # dLoss/dRejected should be positive (decreasing rejected lowers loss)
    assert rejected.grad.item() > 0


# ---------------------------------------------------------------------------
# Test 5: Group Relative Policy Optimization (GRPO)
# ---------------------------------------------------------------------------

def test_grpo_group_advantage_normalization():
    grpo = GRPOLoss()
    rewards = torch.tensor([10.0, 20.0, 30.0, 40.0])
    advantages = grpo.compute_group_advantages(rewards)

    # Mean should be 0, std should be 1
    assert torch.isclose(advantages.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(advantages.std(unbiased=False), torch.tensor(1.0), atol=1e-3)


def test_grpo_ppo_clipping_and_gradient():
    grpo = GRPOLoss(clip_epsilon=0.2, kl_weight=0.01)

    # Two rollouts: 1 successful (reward 1), 1 failed (reward 0)
    rewards = torch.tensor([1.0, 0.0])
    policy_logp = torch.tensor([-1.0, -1.0], requires_grad=True)
    old_logp = torch.tensor([-1.0, -1.0])
    ref_logp = torch.tensor([-1.0, -1.0])

    loss, metrics = grpo(policy_logp, old_logp, ref_logp, rewards)
    loss.backward()

    # Winning rollout should have negative gradient (encouraging higher probability)
    assert policy_logp.grad[0].item() < 0
    # Losing rollout should have positive gradient (discouraging probability)
    assert policy_logp.grad[1].item() > 0


# ---------------------------------------------------------------------------
# Test 6: Reasoning Tree Search Simulator
# ---------------------------------------------------------------------------

def test_reasoning_tree_simulator():
    sim = TestTimeReasoningSimulator()
    tree = sim.build_toy_search_tree()
    root = tree["root"]

    assert len(root.children) == 2
    assert root.children[0].is_pruned is True  # Branch A pruned
    assert root.children[1].is_pruned is False # Branch B kept
    assert len(tree["best_path"]) == 4
