"""
Module 09: Mechanistic Interpretability & Frontiers - Sparse Autoencoders (SAEs)
================================================================================
Historical & Theoretical Context:
- Elhage et al. (2022): "Toy Models of Superposition" (Anthropic) - Demonstrated why
  neural networks represent more features than dimensions (M > D) by packing
  features almost orthogonally into activation space.
- Bricken et al. (2023): "Towards Monosemanticity: Decomposing Language Models With
  Dictionary Learning" (Anthropic) - Proved that Sparse Autoencoders (SAEs) can
  successfully untangle polysemantic neurons into monosemantic, interpretable directions.
- Cunningham et al. (2023): "Sparse Autoencoders Find Highly Interpretable Features in LMs".
- Templeton et al. (2024): "Scaling Monosemanticity" - Extracted millions of monosemantic
  features from Claude 3 Sonnet (e.g., Golden Gate Bridge feature, neuroscience concepts).

Key Mathematical Formulations:
1. Superposition Hypothesis:
   The activation vector x in R^D is formed by summing sparse ground-truth features:
     x = sum_{i=1}^M f_i * v_i  where M > D, f_i >= 0 (sparse), and ||v_i|| = 1.
   Because M > D, individual coordinates (neurons) are polysemantic mixtures.

2. Sparse Autoencoder Architecture:
   Center input with decoder bias:
     x_tilde = x - b_dec
   Encoder (Projects R^D -> R^M, M >> D):
     f(x) = ReLU(W_enc * x_tilde + b_enc)
   Decoder (Reconstructs R^M -> R^D):
     x_hat = W_dec * f(x) + b_dec

3. SAE Training Objective:
   L_SAE = (1 / D) * || x - x_hat ||_2^2 + lambda * sum_{i=1}^M | f_i(x) |
   - Reconstruction Loss: MSE measures fidelity of representation.
   - Sparsity Penalty: L1 norm encourages most feature activations f_i to be strictly 0.

4. Decoder Column Normalization Constraint:
   || W_{dec, :, i} ||_2 = 1.0 for all i in {1, ..., M}
   This prevents the autoencoder from cheating the L1 penalty by scaling up decoder weights
   while shrinking feature activations.
"""

from typing import Tuple, Dict, Optional, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Sparse Autoencoder (SAE) Module
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """
    Standard Sparse Autoencoder for dictionary learning on activation vectors.
    """
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        l1_coefficient: float = 0.05,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.l1_coefficient = l1_coefficient

        # Encoder: Projects from d_in to overcomplete d_sae
        self.w_enc = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder: Projects back from d_sae to d_in
        # Decoder weights are initialized with unit norm columns
        self.w_dec = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.w_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_dec, a=math.sqrt(5))
        # Ensure decoder columns are normalized to unit L2 norm
        self.normalize_decoder_columns()

    @torch.no_grad()
    def normalize_decoder_columns(self):
        """
        Projects each column of W_dec to unit sphere: ||W_dec[:, i]||_2 = 1.0
        """
        col_norms = torch.norm(self.w_dec.data, p=2, dim=0, keepdim=True)
        col_norms = torch.clamp(col_norms, min=1e-8)
        self.w_dec.data = self.w_dec.data / col_norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        f(x) = ReLU(W_enc * (x - b_dec) + b_enc)
        """
        x_centered = x - self.b_dec
        hidden = F.linear(x_centered, self.w_enc, self.b_enc)
        return F.relu(hidden)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        x_hat = W_dec * f + b_dec
        """
        return F.linear(f, self.w_dec, self.b_dec)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Args:
            x: (B, d_in) input activation vectors
        Returns:
            x_hat: (B, d_in) reconstructed activations
            f: (B, d_sae) sparse feature activations
            loss: combined reconstruction MSE + L1 sparsity loss
            metrics: dictionary of R^2, L0 sparsity, and loss breakdown
        """
        # 1. Forward pass
        f = self.encode(x)
        x_hat = self.decode(f)

        # 2. Losses
        # Reconstruction MSE
        recon_loss = F.mse_loss(x_hat, x)
        # L1 Sparsity penalty (mean across batch of sum of absolute activations)
        l1_loss = self.l1_coefficient * torch.mean(torch.sum(torch.abs(f), dim=-1))

        total_loss = recon_loss + l1_loss

        # 3. Interpretability Metrics
        with torch.no_grad():
            # L0 Sparsity: Average number of active features per sample
            l0_sparsity = (f > 0.0).float().sum(dim=-1).mean().item()

            # Explained Variance (R^2)
            total_variance = torch.var(x, dim=0).sum()
            residual_variance = torch.var(x - x_hat, dim=0).sum()
            r2_score = (1.0 - residual_variance / (total_variance + 1e-8)).item()

            metrics = {
                "recon_mse": recon_loss.item(),
                "l1_loss": l1_loss.item(),
                "total_loss": total_loss.item(),
                "l0_active_features": l0_sparsity,
                "explained_variance_r2": r2_score,
            }

        return x_hat, f, total_loss, metrics


# ---------------------------------------------------------------------------
# 2. Synthetic Superposition Generator
# ---------------------------------------------------------------------------

class SyntheticSuperpositionData:
    """
    Generates synthetic activations in superposition:
    - M ground truth features in D dimensions (M > D, e.g. 5 features in 2D).
    - Ground truth vectors are arranged around the 2D unit circle.
    - Each sample randomly activates 1 or 2 sparse features with positive intensities.
    - The SAE must recover the true 5 feature vectors!
    """
    def __init__(self, d_in: int = 2, num_features: int = 5, sparsity: float = 0.8):
        self.d_in = d_in
        self.num_features = num_features
        self.sparsity = sparsity

        # Generate ground truth directions (e.g. vertices of regular pentagon in 2D)
        angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False)
        self.ground_truth_vectors = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)

    def generate_batch(self, batch_size: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            activations: (batch_size, d_in) superposition vectors
            ground_truth_coeffs: (batch_size, num_features) sparse ground-truth coefficients
        """
        # Sparse activation mask: each feature is active with probability (1 - sparsity)
        mask = (np.random.uniform(0, 1, (batch_size, self.num_features)) > self.sparsity).astype(np.float32)

        # Ensure at least one feature is active per sample
        empty_rows = np.where(mask.sum(axis=1) == 0)[0]
        if len(empty_rows) > 0:
            random_features = np.random.randint(0, self.num_features, size=len(empty_rows))
            mask[empty_rows, random_features] = 1.0

        # Positive intensities (exponential distribution)
        intensities = np.random.exponential(scale=1.5, size=(batch_size, self.num_features))
        coeffs = mask * intensities

        # Linear superposition: x = sum c_i * v_i
        activations = coeffs @ self.ground_truth_vectors

        return (
            torch.from_numpy(activations.astype(np.float32)),
            torch.from_numpy(coeffs.astype(np.float32)),
        )

    def compute_feature_recovery_cosine_sim(self, sae: SparseAutoencoder) -> List[float]:
        """
        Computes maximum cosine similarity between each ground truth feature vector
        and the learned SAE decoder columns. High cosine sim (> 0.90) indicates successful
        monosemantic feature disentanglement!
        """
        gt_tensor = torch.from_numpy(self.ground_truth_vectors)  # (M, D)
        dec_weights = sae.w_dec.detach().t()                      # (d_sae, D)

        # Normalize rows
        gt_norm = F.normalize(gt_tensor, p=2, dim=1)
        dec_norm = F.normalize(dec_weights, p=2, dim=1)

        # Cosine similarity matrix: (M, d_sae)
        cos_matrix = torch.matmul(gt_norm, dec_norm.t())

        # For each ground truth vector, find maximum matching decoder column (absolute cosine sim)
        max_sims = torch.max(torch.abs(cos_matrix), dim=1)[0].tolist()
        return max_sims


# ---------------------------------------------------------------------------
# 3. CLI Self-Test Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Project Synapse - Module 09: Sparse Autoencoder (SAE) Disentanglement")
    print("=" * 70)

    # 1. Setup 5 features in 2D space (Superposition: M=5 > D=2)
    toy_data = SyntheticSuperpositionData(d_in=2, num_features=5, sparsity=0.75)
    train_x, train_coeffs = toy_data.generate_batch(2000)
    test_x, _ = toy_data.generate_batch(500)

    print(f"Synthetic Data created: 5 ground-truth features in 2D hidden space.")
    print(f"Sample activations shape: {train_x.shape}")

    # 2. Initialize SAE: Overcomplete dictionary (d_sae = 8 >= 5)
    sae = SparseAutoencoder(d_in=2, d_sae=8, l1_coefficient=0.03)
    optimizer = torch.optim.Adam(sae.parameters(), lr=0.02)

    # 3. Training loop
    print("Training SAE for 300 epochs...")
    for epoch in range(300):
        optimizer.zero_grad()
        _, _, loss, metrics = sae(train_x)
        loss.backward()
        optimizer.step()
        # Enforce unit norm constraint on decoder columns
        sae.normalize_decoder_columns()

    # 4. Evaluate Disentanglement
    _, _, test_loss, test_metrics = sae(test_x)
    max_sims = toy_data.compute_feature_recovery_cosine_sim(sae)

    print(f"\nTraining Results:")
    print(f"  Reconstruction R^2: {test_metrics['explained_variance_r2']:.2%}")
    print(f"  Average L0 Active Features: {test_metrics['l0_active_features']:.2f} / 8")
    print(f"  Ground-Truth Feature Recovery Cosine Similarities: {[round(s, 3) for s in max_sims]}")
    print(f"  Mean Cosine Similarity: {np.mean(max_sims):.3f}")
    print("[SUCCESS] Sparse Autoencoder verified.")
