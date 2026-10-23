"""
Module 07: Generative Paradigms - Optimal Transport Flow Matching (OT-CFM)
===========================================================================
Historical & Theoretical Context:
- Lipman et al. (2023): "Flow Matching for Generative Modeling" (ICLR 2023)
- Albergo & Vanden-Eijnden (2023): "Building Normalizing Flows with Stochastic Interpolants"
- Liu, Gong, Liu (2023): "Flow Straight and Fast: Learning to Generate and Transfer Data with Flow"

The Core Paradigm Shift (Diffusion vs Flow Matching):
- Diffusion models create curved, stochastic trajectories driven by Brownian motion.
  Sampling requires dozens or hundreds of discrete reverse Langevin steps.
- Flow Matching directly learns a time-dependent vector field v_theta(x, t) that generates
  a continuous probability path from base distribution p_0 (standard Gaussian noise) to
  data distribution p_1 along straight, deterministic trajectories.
- Optimal Transport Conditional Flow Matching (OT-CFM) defines the conditional path:
    x_t = (1 - t) * x_0 + t * x_1   where t in [0, 1]
  The instantaneous conditional velocity vector field is simply:
    u_t(x_t | x_0, x_1) = d/dt [ x_t ] = x_1 - x_0
- The training objective is an unweighted, regression MSE:
    L_CFM(theta) = E_{t ~ U(0,1), x_0 ~ p_0, x_1 ~ p_1} [ || v_theta(x_t, t) - (x_1 - x_0) ||^2 ]
- Generation: Solve the ordinary differential equation (ODE):
    dx/dt = v_theta(x, t)  from t = 0 to t = 1
  with standard numerical integrators (Euler, Midpoint, RK4) in just 10-20 steps!
"""

from typing import Tuple, List, Optional, Callable
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .diffusion_ddpm_scratch import SinusoidalTimeEmbedding, generate_toy_data
except (ImportError, ValueError):
    from diffusion_ddpm_scratch import SinusoidalTimeEmbedding, generate_toy_data


# ---------------------------------------------------------------------------
# 1. Flow Matching Velocity Network
# ---------------------------------------------------------------------------

class VelocityMLP2D(nn.Module):
    """
    Learns the deterministic velocity vector field v_theta(x, t) in R^2.
    """
    def __init__(self, hidden_dim: int = 128, time_dim: int = 64):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_proj = nn.Linear(2, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(3)
        ])

        self.out_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2) position coordinates in continuous space
            t: (B,) continuous timesteps in [0.0, 1.0]
        Returns:
            v: (B, 2) instantaneous velocity vectors dx/dt
        """
        # Embed continuous time scalar into frequency vector
        # Multiply by 1000 so sinusoidal frequencies operate in standard wavelength range
        t_scaled = t * 1000.0
        t_emb = self.time_embed(t_scaled)

        h = self.input_proj(x) + t_emb
        for block in self.blocks:
            h = h + block(h)

        velocity = self.out_proj(h)
        return velocity


# ---------------------------------------------------------------------------
# 2. Optimal Transport Conditional Flow Matching (OT-CFM) Engine
# ---------------------------------------------------------------------------

class OptimalTransportFlowMatcher:
    """
    Implements OT-CFM loss computation, ODE solvers, and vector field evaluators.
    """
    def __init__(
        self,
        model: nn.Module,
        sigma_min: float = 1e-4,
        device: torch.device = torch.device("cpu")
    ):
        self.model = model
        self.sigma_min = sigma_min
        self.device = device
        self.model.to(device)

    def compute_loss(
        self,
        x_1: torch.Tensor,
        x_0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Samples t ~ Uniform(0, 1) and base noise x_0 ~ N(0, I).
        Computes conditional trajectory:
          x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        Target vector field:
          u_t = x_1 - (1 - sigma_min) * x_0
        Loss:
          || v_theta(x_t, t) - u_t ||^2
        """
        batch_size = x_1.shape[0]

        if x_0 is None:
            x_0 = torch.randn_like(x_1)

        # Uniform continuous time in [0, 1]
        t = torch.rand((batch_size,), device=self.device)
        t_exp = t.view(-1, 1)

        # Optimal transport linear displacement interpolant
        scale_x0 = 1.0 - (1.0 - self.sigma_min) * t_exp
        x_t = scale_x0 * x_0 + t_exp * x_1

        # Target velocity
        target_velocity = x_1 - (1.0 - self.sigma_min) * x_0

        # Model velocity prediction
        pred_velocity = self.model(x_t, t)

        return F.mse_loss(pred_velocity, target_velocity)

    @torch.no_grad()
    def solve_ode(
        self,
        num_samples: int = 500,
        num_steps: int = 20,
        solver: str = "midpoint",
        x_start: Optional[torch.Tensor] = None,
        return_trajectory: bool = True
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """
        Integrates dx/dt = v_theta(x, t) from t = 0 to t = 1.
        Solvers supported:
          - 'euler': 1st order Forward Euler
          - 'midpoint': 2nd order Runge-Kutta (RK2)
          - 'rk4': 4th order Runge-Kutta
        """
        self.model.eval()

        if x_start is None:
            x = torch.randn((num_samples, 2), device=self.device)
        else:
            x = x_start.clone().to(self.device)

        dt = 1.0 / num_steps
        trajectory = [x.detach().cpu().numpy()] if return_trajectory else None

        for step in range(num_steps):
            t_curr = step * dt
            t_tensor = torch.full((num_samples,), t_curr, device=self.device)

            if solver == "euler":
                v = self.model(x, t_tensor)
                x = x + dt * v

            elif solver == "midpoint":
                # Half-step
                v_start = self.model(x, t_tensor)
                t_mid = t_curr + 0.5 * dt
                t_mid_tensor = torch.full((num_samples,), t_mid, device=self.device)
                x_mid = x + 0.5 * dt * v_start
                # Full-step with midpoint velocity
                v_mid = self.model(x_mid, t_mid_tensor)
                x = x + dt * v_mid

            elif solver == "rk4":
                t_mid = t_curr + 0.5 * dt
                t_end = t_curr + dt
                t_mid_t = torch.full((num_samples,), t_mid, device=self.device)
                t_end_t = torch.full((num_samples,), t_end, device=self.device)

                k1 = self.model(x, t_tensor)
                k2 = self.model(x + 0.5 * dt * k1, t_mid_t)
                k3 = self.model(x + 0.5 * dt * k2, t_mid_t)
                k4 = self.model(x + dt * k3, t_end_t)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"Unsupported solver: {solver}")

            if return_trajectory:
                trajectory.append(x.detach().cpu().numpy())

        final_samples = x.detach().cpu()
        traj_array = np.array(trajectory) if return_trajectory else None
        return final_samples, traj_array

    @torch.no_grad()
    def get_vector_field_grid(
        self,
        t_scalar: float,
        bounds: Tuple[float, float, float, float] = (-2.5, 2.5, -2.5, 2.5),
        grid_res: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates the learned velocity field across a 2D meshgrid at a fixed time t.
        Returns:
            X, Y: 2D meshgrid arrays
            U, V: horizontal and vertical velocity components
        """
        self.model.eval()
        x_min, x_max, y_min, y_max = bounds
        xs = np.linspace(x_min, x_max, grid_res)
        ys = np.linspace(y_min, y_max, grid_res)
        X, Y = np.meshgrid(xs, ys)

        grid_points = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
        grid_tensor = torch.from_numpy(grid_points).to(self.device)
        t_tensor = torch.full((len(grid_points),), t_scalar, device=self.device)

        velocities = self.model(grid_tensor, t_tensor).detach().cpu().numpy()
        U = velocities[:, 0].reshape(X.shape)
        V = velocities[:, 1].reshape(Y.shape)
        return X, Y, U, V


# ---------------------------------------------------------------------------
# 3. CLI Self-Test Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Project Synapse - Module 07: Optimal Transport Flow Matching (OT-CFM)")
    print("=" * 70)

    # 1. Dataset & Model
    data = generate_toy_data("two_moons", 1000)
    tensor_data = torch.from_numpy(data)

    model = VelocityMLP2D(hidden_dim=64, time_dim=32)
    cfm = OptimalTransportFlowMatcher(model)

    print(f"VelocityMLP2D parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Mini-train loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print("Training Flow Matching on 2D Moons for 100 iterations...")
    for step in range(100):
        optimizer.zero_grad()
        loss = cfm.compute_loss(tensor_data[:128])
        loss.backward()
        optimizer.step()
        if (step + 1) % 25 == 0:
            print(f"  Step {step + 1:3d} | CFM Regression Loss: {loss.item():.4f}")

    # 3. ODE Integration with Midpoint Solver
    samples, traj = cfm.solve_ode(num_samples=200, num_steps=15, solver="midpoint")
    print(f"ODE Solve completed: {samples.shape[0]} samples generated in 15 steps.")
    print(f"Trajectory recorded: {traj.shape} steps.")

    # 4. Vector Field Grid
    X, Y, U, V = cfm.get_vector_field_grid(t_scalar=0.5, grid_res=10)
    print(f"Vector field evaluated at t=0.5 on {X.shape} grid.")
    print("[SUCCESS] OT-CFM Engine verified.")
