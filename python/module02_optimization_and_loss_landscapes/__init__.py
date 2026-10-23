"""
Module 02: Loss Geometry, Optimization & SOTA Optimizers
Pure NumPy & PyTorch implementations of optimization algorithms and non-convex landscapes.
"""

from .optimizers_scratch import Optimizer, SGD, Momentum, RMSprop, Adam, AdamW
from .loss_landscapes import (
    LossLandscape,
    IllConditionedQuadratic,
    RosenbrockLandscape,
    SaddlePointLandscape,
    RastriginLandscape,
    TrajectoryTracker,
)
from .pytorch_optimizers import (
    CosineAnnealingWithWarmupLR,
    verify_adamw_vs_adam_l2,
    run_adamw_warmup_training,
)

__all__ = [
    "Optimizer",
    "SGD",
    "Momentum",
    "RMSprop",
    "Adam",
    "AdamW",
    "LossLandscape",
    "IllConditionedQuadratic",
    "RosenbrockLandscape",
    "SaddlePointLandscape",
    "RastriginLandscape",
    "TrajectoryTracker",
    "CosineAnnealingWithWarmupLR",
    "verify_adamw_vs_adam_l2",
    "run_adamw_warmup_training",
]
