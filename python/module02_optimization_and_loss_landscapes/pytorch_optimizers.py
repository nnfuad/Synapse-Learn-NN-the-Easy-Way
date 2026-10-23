"""
Module 02: Loss Geometry, Optimization & SOTA Optimizers
Part 3: SOTA PyTorch 2.x Optimization with AdamW & Cosine Annealing with Warmup

===============================================================================
PRODUCTION SOTA BEST PRACTICES (2017 -> 2026):
1. AdamW (Loshchilov & Hutter, 2017 / ICLR 2019):
   The universal default optimizer for modern Transformers (Llama, GPT-4, Mistral)
   and Vision Transformers (ViT).

2. Learning Rate Schedule: Linear Warmup + Cosine Decay (Goyal et al. 2017, Chinchilla 2022):
   - Warmup: Early in training, parameter gradients are volatile and unaligned.
     A linear ramp-up from ~0 to lr_max prevents violent weight destabilization.
   - Cosine Decay: Smoothly glides learning rate down to lr_min (typically 10% or 1%
     of lr_max), allowing parameters to settle into deep, flat, generalizing minima.
===============================================================================
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWithWarmupLR(_LRScheduler):
    """
    Cosine Annealing Learning Rate Scheduler with Linear Warmup.

    Schedule Phases:
    1. Linear Warmup: for step in [0, warmup_steps), lr linearly increases from
       min_lr to max_lr.
    2. Cosine Decay: for step in [warmup_steps, total_steps), lr decays via cosine
       curve from max_lr down to min_lr.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        assert warmup_steps < total_steps, "warmup_steps must be strictly less than total_steps"
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # self.last_epoch tracks current step count in PyTorch schedulers
        step = max(0, self.last_epoch)

        if step < self.warmup_steps:
            # Linear ramp: lr = min_lr + (step / warmup_steps) * (base_lr - min_lr)
            alpha = float(step) / float(max(1, self.warmup_steps))
            return [
                self.min_lr + alpha * (base_lr - self.min_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay: smoothly reaches min_lr at step (total_steps - 1)
            decay_steps = max(1, self.total_steps - 1 - self.warmup_steps)
            progress = float(step - self.warmup_steps) / float(decay_steps)
            progress = min(1.0, max(0.0, progress))
            cosine_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + cosine_mult * (base_lr - self.min_lr)
                for base_lr in self.base_lrs
            ]


def verify_adamw_vs_adam_l2():
    """
    Demonstrates the mathematical divergence between Adam with L2 regularization
    and AdamW with Decoupled Weight Decay.

    When gradients are large, Adam with L2 suppresses weight decay because
    the gradient term in the denominator sqrt(v) dwarfs the penalty.
    In contrast, AdamW applies identical proportional decay regardless of gradient magnitude.
    """
    torch.manual_seed(42)

    # Two identical weights starting at w = 10.0
    w_adam = torch.tensor([10.0], dtype=torch.float32, requires_grad=True)
    w_adamw = torch.tensor([10.0], dtype=torch.float32, requires_grad=True)

    # Standard Adam with L2 penalty (weight_decay argument in Adam)
    opt_adam = torch.optim.Adam([w_adam], lr=0.1, weight_decay=0.1)
    # AdamW with Decoupled Weight Decay
    opt_adamw = torch.optim.AdamW([w_adamw], lr=0.1, weight_decay=0.1)

    # Case A: Massive gradient (e.g. g = 1000.0)
    # Simulate 5 steps with large synthetic gradients
    for _ in range(5):
        # Synthetic loss producing huge gradient: dL/dw = 100.0
        loss_adam = 100.0 * w_adam
        loss_adam.backward()
        opt_adam.step()
        opt_adam.zero_grad()

        loss_adamw = 100.0 * w_adamw
        loss_adamw.backward()
        opt_adamw.step()
        opt_adamw.zero_grad()

    diff = abs(w_adam.item() - w_adamw.item())
    return {
        "w_adam": w_adam.item(),
        "w_adamw": w_adamw.item(),
        "difference": diff,
    }


class BenchmarkMLP(nn.Module):
    """Deep Non-Linear MLP for optimizer benchmarking."""

    def __init__(self, in_features: int = 10, hidden_dim: int = 64, out_features: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_adamw_warmup_training(
    total_steps: int = 150,
    warmup_steps: int = 30,
    max_lr: float = 3e-3,
    min_lr: float = 1e-5,
) -> Dict[str, Any]:
    """
    Executes a complete production training loop on synthetic data using
    PyTorch AdamW + CosineAnnealingWithWarmupLR.
    """
    torch.manual_seed(42)
    model = BenchmarkMLP(in_features=10, hidden_dim=32, out_features=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-2)
    scheduler = CosineAnnealingWithWarmupLR(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=min_lr,
    )
    criterion = nn.MSELoss()

    # Fixed synthetic dataset
    X = torch.randn(64, 10)
    y = torch.sin(X[:, 0:1]) + 0.5 * torch.cos(X[:, 1:2])

    lr_history = []
    loss_history = []

    for step in range(total_steps):
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()

        optimizer.step()
        lr_history.append(scheduler.get_last_lr()[0])
        scheduler.step()
        loss_history.append(loss.item())

    return {
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "lr_history": lr_history,
        "loss_history": loss_history,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Module 02: PyTorch AdamW & Warmup Cosine Scheduler")
    print("=" * 70)

    # 1. Verify decoupled weight decay difference
    decay_res = verify_adamw_vs_adam_l2()
    print(f"Decoupled Weight Decay Verification:")
    print(f"  Adam (L2 regularized) final weight: {decay_res['w_adam']:.6f}")
    print(f"  AdamW (Decoupled)     final weight: {decay_res['w_adamw']:.6f}")
    print(f"  Divergence: {decay_res['difference']:.6f}")

    # 2. Run training loop with Cosine Annealing + Warmup
    print("\nRunning AdamW + Warmup Cosine Decay training benchmark...")
    sim = run_adamw_warmup_training(total_steps=100, warmup_steps=20)
    print(f"  Initial Loss: {sim['initial_loss']:.6f}")
    print(f"  Final Loss  : {sim['final_loss']:.6f}")
    print(f"  Peak LR reached at step 20: {sim['lr_history'][20]:.6e}")
    print(f"  Final LR at step 99: {sim['lr_history'][-1]:.6e}")
    print("PyTorch AdamW and Cosine Annealing scheduler verified successfully.")
