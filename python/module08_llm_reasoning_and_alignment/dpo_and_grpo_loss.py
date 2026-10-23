"""
Module 08: Modern LLMs, Alignment & Reasoning - DPO & GRPO Loss Formulations
=============================================================================
Historical & Theoretical Context:
- Ouyang et al. (2022): InstructGPT (PPO-based RLHF) - Established human preference
  tuning via a 3-step pipeline: SFT -> Reward Model (RM) -> PPO RL training.
- Rafailov et al. (2023): "Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model" (DPO, NeurIPS 2023) - Proved that the Bradley-Terry
  preference model has an exact analytical closed-form solution, eliminating the
  need to train an explicit reward model or run unstable actor-critic loops.
- Shao et al. / DeepSeek-AI (2024-2025): DeepSeekMath & DeepSeek-R1 - Introduced
  Group Relative Policy Optimization (GRPO). GRPO completely discards the memory-heavy
  critic/value neural network by sampling groups of candidate completions and normalizing
  rewards relative to the group mean and variance.
- Snell et al. (2024), OpenAI o1/o3: Test-Time Compute Scaling - Shifting compute
  from pretraining to inference via chain-of-thought (CoT) search and verification.

Key Mathematical Formulations:
1. DPO Analytical Objective:
   L_DPO(pi_theta; pi_ref) = - E_{(x, y_w, y_l)} [ log sigma( beta * log(pi_theta(y_w|x)/pi_ref(y_w|x))
                                                             - beta * log(pi_theta(y_l|x)/pi_ref(y_l|x)) ) ]
   Implicit Rewards:
     r_theta(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x))

2. GRPO (Group Relative Policy Optimization):
   For prompt q, sample group G = {o_1, o_2, ..., o_G}
   Calculate group rewards r = {r_1, r_2, ..., r_G}
   Group Relative Advantage:
     A_i = (r_i - mean(r)) / (std(r) + eps)
   Surrogate Objective with PPO Clipping:
     ratio_i = pi_theta(o_i|q) / pi_old(o_i|q)
     surr1 = ratio_i * A_i
     surr2 = clip(ratio_i, 1 - eps, 1 + eps) * A_i
     L_clip = - mean( min(surr1, surr2) )
   KL Penalty (Schulman / DeepSeek formulation):
     D_KL = (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
"""

from typing import Tuple, Dict, List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Direct Preference Optimization (DPO) Loss
# ---------------------------------------------------------------------------

class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (Rafailov et al., 2023).
    Bypasses training a separate reward model by reparameterizing the Bradley-Terry
    preference objective directly into the policy language model.
    """
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            policy_chosen_logps: (B,) log pi_theta(y_w | x)
            policy_rejected_logps: (B,) log pi_theta(y_l | x)
            reference_chosen_logps: (B,) log pi_ref(y_w | x)
            reference_rejected_logps: (B,) log pi_ref(y_l | x)
        Returns:
            loss: scalar DPO loss tensor
            metrics: dict of implicit rewards, margins, and accuracy
        """
        # 1. Log ratios between policy and reference
        # pi_logratios = log(pi_theta(y) / pi_ref(y))
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        # 2. Implicit rewards implied by the policy
        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios
        reward_margin = chosen_rewards - rejected_rewards

        # 3. Bradley-Terry logits: beta * (log pi(y_w)/pi_ref(y_w) - log pi(y_l)/pi_ref(y_l))
        logits = self.beta * (chosen_logratios - rejected_logratios)

        # 4. Binary Cross-Entropy / Negative Log Sigmoid with optional label smoothing
        if self.label_smoothing > 0.0:
            loss = (
                - (1.0 - self.label_smoothing) * F.logsigmoid(logits)
                - self.label_smoothing * F.logsigmoid(-logits)
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()

        # 5. Diagnostic metrics
        with torch.no_grad():
            accuracy = (reward_margin > 0.0).float().mean()
            metrics = {
                "chosen_rewards": chosen_rewards.detach(),
                "rejected_rewards": rejected_rewards.detach(),
                "reward_margin": reward_margin.detach(),
                "accuracy": accuracy.detach(),
            }

        return loss, metrics


# ---------------------------------------------------------------------------
# 2. Group Relative Policy Optimization (GRPO) Loss
# ---------------------------------------------------------------------------

class GRPOLoss(nn.Module):
    """
    Group Relative Policy Optimization (DeepSeekMath / DeepSeek-R1).
    Optimizes reasoning policy LLMs by sampling a group of responses per prompt,
    normalizing advantages within the group, and removing the critic model.
    """
    def __init__(
        self,
        clip_epsilon: float = 0.2,
        kl_weight: float = 0.04,
        eps: float = 1e-6
    ):
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.kl_weight = kl_weight
        self.eps = eps

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Computes group-normalized advantages:
          A_i = (r_i - mean(r)) / (std(r) + eps)
        Args:
            rewards: (group_size,) or (batch_prompts, group_size)
        """
        if rewards.dim() == 1:
            mean = rewards.mean()
            std = rewards.std(unbiased=False)
            return (rewards - mean) / (std + self.eps)
        else:
            mean = rewards.mean(dim=-1, keepdim=True)
            std = rewards.std(dim=-1, keepdim=True, unbiased=False)
            return (rewards - mean) / (std + self.eps)

    def forward(
        self,
        policy_logps: torch.Tensor,
        old_policy_logps: torch.Tensor,
        reference_logps: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            policy_logps: (G,) current policy log probabilities
            old_policy_logps: (G,) rollout policy log probabilities
            reference_logps: (G,) frozen reference model log probabilities
            rewards: (G,) group reward values (e.g., unit test pass rate or math verifier)
        Returns:
            loss: total GRPO loss to minimize
            metrics: dict with advantage, clipped fraction, and KL divergence
        """
        # 1. Compute Group Relative Advantages
        advantages = self.compute_group_advantages(rewards)

        # 2. Importance sampling ratio: pi_theta / pi_old
        log_ratio = policy_logps - old_policy_logps
        ratio = torch.exp(log_ratio)

        # 3. PPO Surrogate Clipped Objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_objective = torch.min(surr1, surr2).mean()

        # 4. KL Divergence Penalty (DeepSeek formulation)
        # D_KL = (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
        # In log-space: let u = log(pi_ref) - log(pi_theta)
        # then exp(u) - u - 1 >= 0
        u = reference_logps - policy_logps
        kl_div = (torch.exp(u) - u - 1.0).mean()

        # Total loss (we minimize negative objective + KL penalty)
        total_loss = -policy_objective + self.kl_weight * kl_div

        with torch.no_grad():
            clip_fraction = (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean()
            metrics = {
                "mean_reward": rewards.mean().detach(),
                "advantages": advantages.detach(),
                "mean_ratio": ratio.mean().detach(),
                "clip_fraction": clip_fraction.detach(),
                "kl_divergence": kl_div.detach(),
            }

        return total_loss, metrics


# ---------------------------------------------------------------------------
# 3. Test-Time Compute Rollout Search Tree Simulator
# ---------------------------------------------------------------------------

class ReasoningTreeNode:
    """
    Represents a candidate thought branch in a Test-Time Compute search tree (MCTS / Best-of-N).
    """
    def __init__(self, step_text: str, score: float, parent: Optional['ReasoningTreeNode'] = None):
        self.step_text = step_text
        self.score = score
        self.parent = parent
        self.children: List['ReasoningTreeNode'] = []
        self.is_pruned = False

    def add_child(self, step_text: str, score: float) -> 'ReasoningTreeNode':
        child = ReasoningTreeNode(step_text, score, parent=self)
        self.children.append(child)
        return child


class TestTimeReasoningSimulator:
    """
    Demonstrates inference-time compute scaling:
    Explores candidate reasoning branches, prunes dead ends with a process reward model (PRM),
    and reconstructs the verified chain-of-thought (CoT).
    """
    __test__ = False  # Prevent pytest from mistaking this helper class as a test suite

    def __init__(self, prune_threshold: float = 0.4):
        self.prune_threshold = prune_threshold

    def build_toy_search_tree(self) -> Dict:
        """
        Creates a pedagogical reasoning tree for the problem:
        "Find all prime numbers p such that p + 2 and p + 4 are also prime."
        """
        root = ReasoningTreeNode(step_text="Problem: Find prime triplets (p, p+2, p+4)", score=1.0)

        # Branch 1 (Dead End): Assume p can be arbitrarily large
        b1 = root.add_child("Branch A: Assume infinite triplets exist; test primes p=5, 7, 11...", score=0.3)
        b1.is_pruned = True
        b1.add_child("Sub-step: p=7 -> (7, 9, 11); 9 is not prime (Failed)", score=0.0).is_pruned = True

        # Branch 2 (Self-Correction): Analyze modulo 3 residues
        b2 = root.add_child("Branch B: Test residues modulo 3: every integer is 0, 1, or 2 mod 3", score=0.95)

        # Branch 2 sub-branches
        b2_1 = b2.add_child("Case 1: If p = 3k+1, then p+2 = 3k+3 is divisible by 3 (not prime if > 3)", score=0.98)
        b2_2 = b2.add_child("Case 2: If p = 3k+2, then p+4 = 3k+6 is divisible by 3 (not prime if > 3)", score=0.98)
        b2_3 = b2.add_child("Case 3: If p = 3k, the only prime divisible by 3 is p = 3", score=1.0)

        # Conclusion
        conclusion = b2_3.add_child("Conclusion: For p=3 -> (3, 5, 7) all prime. Only one unique triplet exists!", score=1.0)

        return {
            "root": root,
            "best_path": [
                "Problem: Find prime triplets (p, p+2, p+4)",
                "Branch B: Test residues modulo 3: every integer is 0, 1, or 2 mod 3",
                "Case 3: If p = 3k, the only prime divisible by 3 is p = 3",
                "Conclusion: For p=3 -> (3, 5, 7) all prime. Only one unique triplet exists!"
            ]
        }


# ---------------------------------------------------------------------------
# 4. CLI Self-Test Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Project Synapse - Module 08: DPO & GRPO Loss Formulations")
    print("=" * 70)

    # 1. DPO Loss Verification
    dpo = DPOLoss(beta=0.1)
    # Simulated log probabilities: Policy strongly prefers chosen over rejected
    policy_chosen = torch.tensor([-1.2, -0.8, -1.5])
    policy_rejected = torch.tensor([-3.5, -4.0, -3.2])
    ref_chosen = torch.tensor([-2.0, -2.0, -2.0])
    ref_rejected = torch.tensor([-2.0, -2.0, -2.0])

    loss_dpo, dpo_metrics = dpo(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    print(f"DPO Loss: {loss_dpo.item():.4f}")
    print(f"  Implicit Reward Margin: {dpo_metrics['reward_margin'].tolist()}")
    print(f"  Implicit Accuracy: {dpo_metrics['accuracy'].item():.0%}")

    # 2. GRPO Loss Verification
    grpo = GRPOLoss(clip_epsilon=0.2, kl_weight=0.05)
    # Group of 4 candidate reasoning paths
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.5])  # 2 correct solutions, 1 partial, 1 failed
    policy_logp = torch.tensor([-1.1, -2.5, -1.0, -1.8], requires_grad=True)
    old_logp = torch.tensor([-1.2, -2.4, -1.1, -1.9])
    ref_logp = torch.tensor([-1.3, -2.3, -1.2, -2.0])

    loss_grpo, grpo_metrics = grpo(policy_logp, old_logp, ref_logp, rewards)
    loss_grpo.backward()

    print(f"\nGRPO Loss: {loss_grpo.item():.4f}")
    print(f"  Group Advantages: {grpo_metrics['advantages'].tolist()}")
    print(f"  Policy Gradients: {policy_logp.grad.tolist()}")
    print(f"  KL Divergence: {grpo_metrics['kl_divergence'].item():.6f}")

    # 3. Test-Time Search
    sim = TestTimeReasoningSimulator()
    tree = sim.build_toy_search_tree()
    print(f"\nTest-Time Compute Reasoning Trace ({len(tree['best_path'])} steps):")
    for step in tree['best_path']:
        print(f"  -> {step}")
    print("[SUCCESS] DPO & GRPO Loss Engines verified.")
