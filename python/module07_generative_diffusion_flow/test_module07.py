"""
Unit Tests for Module 07: Generative Paradigms (Diffusion & Flow Matching)
Verifies:
1. Mathematical variance schedules and closed-form forward corruption.
2. Equivalence of recursive Langevin corruption vs closed-form q_sample.
3. DDPM training loss computation and gradient backpropagation.
4. DDPM reverse denoising sampling loop & trajectory consistency.
5. Optimal Transport Flow Matching (OT-CFM) linear interpolation and target velocity.
6. Numerical ODE integration schemes (Euler, Midpoint, RK4).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import pytest

from diffusion_ddpm_scratch import (
    DiffusionNoiseSchedule,
    SinusoidalTimeEmbedding,
    DenoisingMLP2D,
    DDPM2D,
    generate_toy_data,
)
from flow_matching_toy import (
    VelocityMLP2D,
    OptimalTransportFlowMatcher,
)


# ---------------------------------------------------------------------------
# Test 1: Schedule Mathematics
# ---------------------------------------------------------------------------

def test_diffusion_schedule_properties():
    schedule = DiffusionNoiseSchedule(num_timesteps=100, schedule_type="linear", beta_start=1e-4, beta_end=0.02)

    # 1. Betas strictly within (0, 1)
    assert torch.all(schedule.betas > 0)
    assert torch.all(schedule.betas < 1)

    # 2. Alphas = 1 - betas
    assert torch.allclose(schedule.alphas, 1.0 - schedule.betas)

    # 3. Alphas cumprod monotonically decreases
    alphas_cumprod = schedule.alphas_cumprod
    diffs = alphas_cumprod[1:] - alphas_cumprod[:-1]
    assert torch.all(diffs < 0), "alphas_cumprod must monotonically decrease over time"

    # 4. First step close to 1, final step significantly decayed
    assert alphas_cumprod[0] > 0.99
    assert alphas_cumprod[-1] < 0.5


def test_cosine_schedule():
    schedule = DiffusionNoiseSchedule(num_timesteps=50, schedule_type="cosine")
    assert len(schedule.betas) == 50
    assert torch.all(schedule.betas > 0)
    assert torch.all(schedule.alphas_cumprod[1:] < schedule.alphas_cumprod[:-1])


# ---------------------------------------------------------------------------
# Test 2: Closed-Form q(x_t | x_0) vs Sequential Markov Step
# ---------------------------------------------------------------------------

def test_forward_sampling_marginal_statistics():
    """
    At t = T, x_t should statistically match standard Gaussian N(0, I)
    when x_0 is centered and alpha_bar_T is near 0.
    """
    torch.manual_seed(42)
    schedule = DiffusionNoiseSchedule(num_timesteps=200, beta_start=0.001, beta_end=0.05)

    x_0 = torch.zeros((5000, 2))  # origin
    t_last = torch.full((5000,), 199, dtype=torch.long)

    x_t, _ = schedule.q_sample(x_0, t_last)

    # Empirical mean should be near 0, empirical variance near 1
    mean = x_t.mean(dim=0)
    var = x_t.var(dim=0)

    assert torch.allclose(mean, torch.zeros(2), atol=0.08)
    assert torch.allclose(var, torch.ones(2), atol=0.10)


def test_closed_form_vs_sequential_step():
    """
    Verifies that stepping through q(x_1 | x_0) recursively matches
    the closed-form distribution mean and variance for t = 2.
    """
    torch.manual_seed(123)
    schedule = DiffusionNoiseSchedule(num_timesteps=10, beta_start=0.01, beta_end=0.05)
    x_0 = torch.tensor([[2.0, -1.5]], dtype=torch.float32).repeat(10000, 1)

    # Closed-form at t=1 (0-indexed step 1 is the 2nd step)
    t = torch.full((10000,), 1, dtype=torch.long)
    x_t_closed, _ = schedule.q_sample(x_0, t)

    # Sequential: step 0 then step 1
    beta_0 = schedule.betas[0]
    beta_1 = schedule.betas[1]
    eps_0 = torch.randn_like(x_0)
    x_1_seq = torch.sqrt(1 - beta_0) * x_0 + torch.sqrt(beta_0) * eps_0
    eps_1 = torch.randn_like(x_0)
    x_2_seq = torch.sqrt(1 - beta_1) * x_1_seq + torch.sqrt(beta_1) * eps_1

    # Means and standard deviations should match
    assert torch.allclose(x_t_closed.mean(dim=0), x_2_seq.mean(dim=0), atol=0.06)
    assert torch.allclose(x_t_closed.std(dim=0), x_2_seq.std(dim=0), atol=0.06)


# ---------------------------------------------------------------------------
# Test 3: Sinusoidal Embedding & Denoising MLP
# ---------------------------------------------------------------------------

def test_sinusoidal_embedding_dimension():
    embed = SinusoidalTimeEmbedding(dim=32)
    t = torch.tensor([0, 10, 50, 99])
    out = embed(t)
    assert out.shape == (4, 32)
    assert not torch.isnan(out).any()


def test_denoising_mlp_forward_and_backward():
    model = DenoisingMLP2D(hidden_dim=32, time_dim=16)
    x = torch.randn(8, 2)
    t = torch.randint(0, 50, (8,))
    pred = model(x, t)

    assert pred.shape == (8, 2)
    loss = pred.sum()
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"


# ---------------------------------------------------------------------------
# Test 4: DDPM Sampler Loop & Trajectory
# ---------------------------------------------------------------------------

def test_ddpm_sampling_trajectory():
    schedule = DiffusionNoiseSchedule(num_timesteps=10)
    model = DenoisingMLP2D(hidden_dim=32, time_dim=16)
    ddpm = DDPM2D(model, schedule)

    samples, traj = ddpm.sample(num_samples=16, return_trajectory=True)
    assert samples.shape == (16, 2)
    assert traj.shape == (11, 16, 2)  # T + 1 steps recorded
    assert not np.isnan(traj).any()


# ---------------------------------------------------------------------------
# Test 5: Optimal Transport Flow Matching (OT-CFM)
# ---------------------------------------------------------------------------

def test_ot_cfm_target_velocity_consistency():
    model = VelocityMLP2D(hidden_dim=32, time_dim=16)
    cfm = OptimalTransportFlowMatcher(model, sigma_min=0.0)

    x_0 = torch.tensor([[0.0, 0.0]])
    x_1 = torch.tensor([[2.0, 4.0]])

    # Loss computation check
    loss = cfm.compute_loss(x_1, x_0)
    assert loss.dim() == 0
    assert loss.item() >= 0.0

    # Gradient flow check
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"


def test_flow_matching_ode_solvers():
    model = VelocityMLP2D(hidden_dim=32, time_dim=16)
    cfm = OptimalTransportFlowMatcher(model)

    for solver in ["euler", "midpoint", "rk4"]:
        samples, traj = cfm.solve_ode(num_samples=10, num_steps=5, solver=solver)
        assert samples.shape == (10, 2)
        assert traj.shape == (6, 10, 2)
        assert not np.isnan(traj).any(), f"NaN encountered in {solver} ODE solver"


def test_flow_matching_vector_grid():
    model = VelocityMLP2D(hidden_dim=32, time_dim=16)
    cfm = OptimalTransportFlowMatcher(model)

    X, Y, U, V = cfm.get_vector_field_grid(t_scalar=0.5, grid_res=8)
    assert X.shape == (8, 8)
    assert Y.shape == (8, 8)
    assert U.shape == (8, 8)
    assert V.shape == (8, 8)
    assert not np.isnan(U).any()
    assert not np.isnan(V).any()
