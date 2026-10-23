"""
Pytest Suite for Module 02: Optimization & Loss Landscapes
Verifies:
1. Convergence of pure NumPy optimizers (SGD, Momentum, RMSprop, Adam, AdamW) on quadratic surfaces.
2. Momentum suppression of orthogonal oscillations in narrow ravines.
3. Adam / AdamW escaping saddle points where SGD stalls.
4. Mathematical divergence between Adam L2 regularization and AdamW decoupled weight decay.
5. PyTorch CosineAnnealingWithWarmupLR scheduler dynamics and boundary conditions.
6. End-to-end PyTorch AdamW training on non-linear MLP benchmark.
"""

import sys
import os
import math
import pytest
import numpy as np
import torch

# Ensure local module directory is in path for standalone pytest execution
sys.path.insert(0, os.path.dirname(__file__))

try:
    from .optimizers_scratch import SGD, Momentum, RMSprop, Adam, AdamW
    from .loss_landscapes import (
        IllConditionedQuadratic,
        RosenbrockLandscape,
        SaddlePointLandscape,
        TrajectoryTracker,
    )
    from .pytorch_optimizers import (
        CosineAnnealingWithWarmupLR,
        verify_adamw_vs_adam_l2,
        run_adamw_warmup_training,
    )
except ImportError:
    from optimizers_scratch import SGD, Momentum, RMSprop, Adam, AdamW
    from loss_landscapes import (
        IllConditionedQuadratic,
        RosenbrockLandscape,
        SaddlePointLandscape,
        TrajectoryTracker,
    )
    from pytorch_optimizers import (
        CosineAnnealingWithWarmupLR,
        verify_adamw_vs_adam_l2,
        run_adamw_warmup_training,
    )


# =============================================================================
# 1. Convergence on Quadratic Landscapes
# =============================================================================

@pytest.mark.parametrize("opt_cls,lr", [
    (SGD, 0.05),
    (Momentum, 0.03),
    (RMSprop, 0.05),
    (Adam, 0.1),
    (AdamW, 0.1),
])
def test_optimizers_converge_on_isotropic_bowl(opt_cls, lr):
    """Verify that all scratch optimizers reduce loss by at least 90% on f(x, y) = x^2 + y^2."""
    param = np.array([3.0, -4.0], dtype=np.float64)
    initial_loss = float(np.sum(param ** 2))

    opt = opt_cls([param], lr=lr)
    for _ in range(80):
        grad = 2.0 * param  # d(x^2 + y^2)/d[x, y] = 2 * [x, y]
        opt.step([grad])

    final_loss = float(np.sum(param ** 2))
    assert final_loss < initial_loss * 0.10, (
        f"{opt_cls.__name__} failed to converge: initial={initial_loss}, final={final_loss}"
    )


# =============================================================================
# 2. Ravine Oscillation & Momentum Advantage (Polyak 1964)
# =============================================================================

def test_momentum_outperforms_sgd_on_ill_conditioned_ravine():
    """
    On an anisotropic ravine f(x, y) = 0.5*(x^2 + 50*y^2), vanilla SGD with large learning rate
    suffers violent orthogonal oscillations along y, whereas Polyak Momentum averages them out
    and advances faster down x.
    """
    ravine = IllConditionedQuadratic(gamma=50.0)
    start_pos = (-5.0, 1.0)
    steps = 40
    lr = 0.03

    sgd_sim = TrajectoryTracker.run_simulation(ravine, SGD, start_pos, n_steps=steps, lr=lr)
    mom_sim = TrajectoryTracker.run_simulation(ravine, Momentum, start_pos, n_steps=steps, lr=lr, beta=0.85)

    # Momentum should achieve lower final loss than vanilla SGD after 40 steps
    assert mom_sim["final_loss"] < sgd_sim["final_loss"], (
        f"Momentum ({mom_sim['final_loss']}) did not outperform SGD ({sgd_sim['final_loss']})"
    )


# =============================================================================
# 3. Saddle Point Escape (Dauphin et al. 2014)
# =============================================================================

def test_adam_escapes_saddle_point():
    """
    On saddle landscape f(x, y) = 0.5*x^2 - 0.5*y^2 starting near stationary point (0.01, 0.001),
    Adam's adaptive step size accelerates descent along the negative curvature axis y.
    """
    saddle = SaddlePointLandscape()
    start_pos = (0.05, 0.01)

    adam_sim = TrajectoryTracker.run_simulation(saddle, Adam, start_pos, n_steps=50, lr=0.05)
    # The negative curvature direction y decreases loss towards -infinity
    assert adam_sim["final_loss"] < -0.1, f"Adam failed to escape saddle: final_loss={adam_sim['final_loss']}"


# =============================================================================
# 4. Decoupled Weight Decay Divergence (Adam vs AdamW)
# =============================================================================

def test_scratch_adam_vs_adamw_weight_decay():
    """
    Verify that in our scratch implementations, AdamW decouples weight decay
    from adaptive second moment calculation, producing divergent parameter values
    when gradients are large.
    """
    p_adam = np.array([10.0])
    p_adamw = np.array([10.0])

    opt_adam = Adam([p_adam], lr=0.1, weight_decay=0.05)
    opt_adamw = AdamW([p_adamw], lr=0.1, weight_decay=0.05)

    # Apply 10 steps of a large gradient: g = 50.0
    for _ in range(10):
        large_grad = np.array([50.0])
        opt_adam.step([large_grad])
        opt_adamw.step([large_grad])

    diff = abs(p_adam[0] - p_adamw[0])
    # The two must mathematically diverge under high gradient conditions
    assert diff > 0.01, f"Adam and AdamW should diverge under high gradient, got diff={diff}"


def test_pytorch_adamw_decoupling_verification():
    """Verify PyTorch Adam vs AdamW decoupled weight decay parity logic."""
    res = verify_adamw_vs_adam_l2()
    assert res["difference"] > 0.1, "PyTorch Adam and AdamW should diverge under heavy gradients"


# =============================================================================
# 5. Cosine Annealing with Linear Warmup Scheduler
# =============================================================================

def test_cosine_annealing_with_warmup_schedule():
    """Verify scheduler boundary values and monotonic properties."""
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([dummy_param], lr=0.1)

    warmup_steps = 20
    total_steps = 100
    min_lr = 1e-4

    scheduler = CosineAnnealingWithWarmupLR(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=min_lr,
    )

    lrs = []
    for step in range(total_steps):
        lrs.append(scheduler.get_lr()[0])
        optimizer.step()
        scheduler.step()

    # Step 0 should be min_lr
    assert math.isclose(lrs[0], min_lr, rel_tol=1e-3)

    # Step warmup_steps should reach peak lr (0.1)
    assert math.isclose(lrs[warmup_steps], 0.1, rel_tol=1e-3)

    # End of warmup: lr should be strictly increasing during warmup
    for i in range(1, warmup_steps):
        assert lrs[i] > lrs[i - 1], f"LR not increasing at warmup step {i}"

    # During cosine decay: lr should be monotonically decreasing
    for i in range(warmup_steps + 1, total_steps):
        assert lrs[i] < lrs[i - 1], f"LR not decreasing at cosine decay step {i}"

    # Final step should be close to min_lr
    assert math.isclose(lrs[-1], min_lr, rel_tol=1e-2)


# =============================================================================
# 6. End-to-End PyTorch Training Benchmark
# =============================================================================

def test_pytorch_adamw_training_convergence():
    """Verify full PyTorch training loop convergence with AdamW and scheduler."""
    train_result = run_adamw_warmup_training(total_steps=80, warmup_steps=15)
    init_loss = train_result["initial_loss"]
    final_loss = train_result["final_loss"]

    assert final_loss < init_loss * 0.1, (
        f"Training did not converge sufficiently: initial={init_loss}, final={final_loss}"
    )
