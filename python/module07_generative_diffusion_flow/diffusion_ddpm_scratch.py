"""
Module 07: Generative Paradigms - Denoising Diffusion Probabilistic Models (DDPM)
=================================================================================
Historical & Theoretical Context:
- Sohl-Dickstein et al. (2015): "Deep Unsupervised Learning using Nonequilibrium
  Thermodynamics" - First introduced the idea of slowly destroying structure in data
  via a forward diffusion process and learning a reverse Markov chain to restore it.
- Ho, Jain, Abbeel (2020): "Denoising Diffusion Probabilistic Models" (DDPM) -
  Demonstrated high-fidelity generative modeling by connecting diffusion to
  score matching (Song & Ermon) and parameterizing the reverse model to predict the
  injected Gaussian noise epsilon.

Key Mathematical Formulations:
1. Forward Variance Schedule:
   beta_t in (0, 1) for t in {1, ..., T}
   alpha_t = 1 - beta_t
   alpha_bar_t = prod_{s=1}^t alpha_s

2. Forward Marginal Distribution:
   q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon, where epsilon ~ N(0, I)

3. Score Matching Connection (Tweedie's Formula):
   grad_{x_t} log p_t(x_t) = - epsilon_theta(x_t, t) / sqrt(1 - alpha_bar_t)

4. Reverse Denoising Transition:
   p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
   mu_theta(x_t, t) = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_theta(x_t, t))
   sigma_t = sqrt(beta_tilde_t), where beta_tilde_t = ((1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * beta_t

5. Simplified Training Objective:
   L_simple(theta) = E_{t, x_0, epsilon} [ || epsilon - epsilon_theta(x_t, t) ||^2 ]
"""

from typing import Tuple, List, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Noise Schedules (NumPy & PyTorch)
# ---------------------------------------------------------------------------

class DiffusionNoiseSchedule:
    """
    Computes and caches closed-form diffusion schedule constants:
    beta_t, alpha_t, alpha_bar_t, and posterior variance beta_tilde_t.
    """
    def __init__(
        self,
        num_timesteps: int = 100,
        schedule_type: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device = torch.device("cpu")
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif schedule_type == "cosine":
            # Cosine schedule from Nichol & Dhariwal (2021)
            steps = num_timesteps + 1
            s = 0.008
            t = torch.linspace(0, num_timesteps, steps, dtype=torch.float32) / num_timesteps
            alphas_cumprod = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, min=1e-5, max=0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute quantities for forward process q(x_t | x_0)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)

        # Precompute posterior q(x_{t-1} | x_t, x_0) quantities
        # beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)
        # Clamped log variance because posterior variance is 0 at t = 0
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        ).to(device)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion step (reparameterization trick):
        Sample x_t directly from x_0 at arbitrary step t without running 1...t sequentially.
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise


# ---------------------------------------------------------------------------
# 2. Sinusoidal Time Embedding & Denoising MLP
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Projects scalar integer timestep t to a continuous frequency vector.
    Similar to positional encodings in Transformer (Vaswani et al. 2017).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb_scale)
        emb = t.float().view(-1, 1) * emb.view(1, -1)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode='constant', value=0.0)
        return emb


class DenoisingMLP2D(nn.Module):
    """
    A lightweight, expressive MLP with residual skips designed to predict
    the injected noise epsilon_theta(x_t, t) for 2D points.
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

        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2) noisy 2D points
            t: (B,) discrete timestep integers
        Returns:
            eps_pred: (B, 2) predicted noise vector
        """
        t_emb = self.time_embed(t)
        h = self.input_proj(x) + t_emb
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# 3. Full DDPM Sampler & Loss Pipeline
# ---------------------------------------------------------------------------

class DDPM2D:
    """
    Full Toy 2D DDPM pipeline providing training loss and reverse sampling trajectories.
    """
    def __init__(
        self,
        model: nn.Module,
        schedule: DiffusionNoiseSchedule,
        device: torch.device = torch.device("cpu")
    ):
        self.model = model
        self.schedule = schedule
        self.device = device
        self.model.to(device)

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Computes L_simple = E_{t, eps} [ || eps - eps_theta(x_t, t) ||^2 ]
        """
        batch_size = x_0.shape[0]
        # Uniformly sample random timestep for each example in the batch
        t = torch.randint(0, self.schedule.num_timesteps, (batch_size,), device=self.device).long()
        noise = torch.randn_like(x_0)

        # Forward noising
        x_t, _ = self.schedule.q_sample(x_0, t, noise)

        # Predict noise
        pred_noise = self.model(x_t, t)
        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample_step(self, x_t: torch.Tensor, t_int: int) -> torch.Tensor:
        """
        Single reverse denoising step from x_t to x_{t-1}.
        Formula:
          x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta) + sigma_t * z
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t_int, device=self.device, dtype=torch.long)

        # Predict epsilon
        eps_theta = self.model(x_t, t_tensor)

        beta_t = self.schedule.betas[t_int]
        alpha_t = self.schedule.alphas[t_int]
        sqrt_one_minus_alpha_bar_t = self.schedule.sqrt_one_minus_alphas_cumprod[t_int]

        # Model mean
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta
        )

        if t_int == 0:
            return mean
        else:
            # Inject Langevin noise with variance beta_tilde_t
            variance = self.schedule.posterior_variance[t_int]
            sigma = torch.sqrt(variance)
            noise = torch.randn_like(x_t)
            return mean + sigma * noise

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 500,
        return_trajectory: bool = True
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """
        Runs the full reverse Markov chain from t = T-1 down to t = 0.
        Returns:
            final_samples: (N, 2)
            trajectory: (T+1, N, 2) numpy array if return_trajectory is True
        """
        self.model.eval()
        # Step T: Start from isotropic Gaussian noise
        x = torch.randn((num_samples, 2), device=self.device)

        trajectory = [x.detach().cpu().numpy()] if return_trajectory else None

        for t in reversed(range(self.schedule.num_timesteps)):
            x = self.p_sample_step(x, t)
            if return_trajectory:
                trajectory.append(x.detach().cpu().numpy())

        final_samples = x.detach().cpu()
        traj_array = np.array(trajectory) if return_trajectory else None
        return final_samples, traj_array


# ---------------------------------------------------------------------------
# 4. 2D Toy Data Distributions
# ---------------------------------------------------------------------------

def generate_toy_data(dataset_type: str = "two_moons", n_samples: int = 2000) -> np.ndarray:
    """
    Generates pedagogical 2D distributions:
    - 'two_moons': Two interleaving crescents
    - 'swiss_roll': 2D spiral
    - 'circle_ring': Concentric circles
    - 'eight_gaussians': Symmetric mixture of 8 Gaussians in a ring
    """
    np.random.seed(42)
    if dataset_type == "two_moons":
        n_half = n_samples // 2
        theta = np.linspace(0, np.pi, n_half)
        # Moon 1
        x1 = np.cos(theta) + np.random.normal(0, 0.06, n_half)
        y1 = np.sin(theta) + np.random.normal(0, 0.06, n_half)
        # Moon 2
        x2 = 1.0 - np.cos(theta) + np.random.normal(0, 0.06, n_samples - n_half)
        y2 = 1.0 - np.sin(theta) - 0.5 + np.random.normal(0, 0.06, n_samples - n_half)
        pts = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
        pts = (pts - pts.mean(axis=0)) / (pts.std(axis=0) * 1.5)
        return pts.astype(np.float32)

    elif dataset_type == "swiss_roll":
        theta = np.sqrt(np.random.uniform(0, 1, n_samples)) * 3 * np.pi
        r = theta
        x = r * np.cos(theta) + np.random.normal(0, 0.2, n_samples)
        y = r * np.sin(theta) + np.random.normal(0, 0.2, n_samples)
        pts = np.column_stack([x, y])
        pts = (pts - pts.mean(axis=0)) / (pts.std(axis=0) * 1.5)
        return pts.astype(np.float32)

    elif dataset_type == "circle_ring":
        r_inner = 0.6 + np.random.normal(0, 0.05, n_samples // 2)
        theta_inner = np.random.uniform(0, 2 * np.pi, n_samples // 2)
        inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])

        r_outer = 1.4 + np.random.normal(0, 0.05, n_samples - n_samples // 2)
        theta_outer = np.random.uniform(0, 2 * np.pi, n_samples - n_samples // 2)
        outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])
        pts = np.vstack([inner, outer])
        return pts.astype(np.float32)

    elif dataset_type == "eight_gaussians":
        centers = [
            (math.cos(2 * math.pi * i / 8) * 1.5, math.sin(2 * math.pi * i / 8) * 1.5)
            for i in range(8)
        ]
        pts = []
        for i in range(n_samples):
            c = centers[np.random.randint(0, 8)]
            pts.append([c[0] + np.random.normal(0, 0.12), c[1] + np.random.normal(0, 0.12)])
        return np.array(pts, dtype=np.float32)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


# ---------------------------------------------------------------------------
# 5. CLI Self-Test Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Project Synapse - Module 07: Toy 2D DDPM Denoising Diffusion")
    print("=" * 70)

    # 1. Schedule verification
    schedule = DiffusionNoiseSchedule(num_timesteps=50, schedule_type="linear")
    print(f"Schedule initialized: {schedule.num_timesteps} steps.")
    print(f"  alpha_bar_0: {schedule.alphas_cumprod[0].item():.4f}")
    print(f"  alpha_bar_T: {schedule.alphas_cumprod[-1].item():.6f}")

    # 2. Model & Toy Data
    data = generate_toy_data("two_moons", 1000)
    tensor_data = torch.from_numpy(data)
    print(f"Dataset 'two_moons' loaded with {len(data)} samples.")

    model = DenoisingMLP2D(hidden_dim=64, time_dim=32)
    ddpm = DDPM2D(model, schedule)

    # 3. Quick mini-train check
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    print("Training mini-batch loop for 100 iterations...")
    for step in range(100):
        optimizer.zero_grad()
        loss = ddpm.compute_loss(tensor_data[:128])
        loss.backward()
        optimizer.step()
        if (step + 1) % 25 == 0:
            print(f"  Step {step + 1:3d} | Denoising MSE Loss: {loss.item():.4f}")

    # 4. Sampling reverse trajectory
    samples, traj = ddpm.sample(num_samples=200, return_trajectory=True)
    print(f"Sampling completed: {samples.shape[0]} points generated.")
    print(f"Trajectory recorded: {traj.shape} (timesteps, samples, dims).")
    print("[SUCCESS] DDPM Scratch Engine verified.")
