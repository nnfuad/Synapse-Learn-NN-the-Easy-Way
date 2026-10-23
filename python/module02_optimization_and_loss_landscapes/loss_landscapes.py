"""
Module 02: Loss Geometry, Optimization & SOTA Optimizers
Part 2: Non-Convex Loss Landscapes, Ravines, Saddles & Trajectory Simulation

===============================================================================
GEOMETRIC FOUNDATIONS:
Optimization in deep learning does not occur on smooth, isotropic bowls.
Modern neural network loss surfaces feature:
1. Ill-Conditioned Ravines:
   Eigenvalues of the Hessian Hessian(L) satisfy lambda_max >> lambda_min.
   The condition number kappa = lambda_max / lambda_min is large (e.g. 100 to 10,000).
   Gradients point almost orthogonally to the direction of the minimum.

2. Saddle Points & Flat Plateaus (Dauphin et al. 2014):
   Points where grad(L) = 0, but the Hessian has both positive and negative eigenvalues.
   In high dimensions, saddle points exponentially outnumber local minima.

3. Multimodal Non-Convexity:
   Landscapes packed with local minima, deceptive basins, and barrier ridges.
===============================================================================
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List, Dict, Any, Type, Optional

try:
    from .optimizers_scratch import Optimizer, SGD, Momentum, RMSprop, Adam, AdamW
except ImportError:
    from optimizers_scratch import Optimizer, SGD, Momentum, RMSprop, Adam, AdamW


class LossLandscape:
    """Base class for 2D mathematical test loss landscapes."""

    def __call__(self, x: float, y: float) -> float:
        raise NotImplementedError

    def grad(self, x: float, y: float) -> np.ndarray:
        raise NotImplementedError

    def hessian(self, x: float, y: float) -> np.ndarray:
        raise NotImplementedError

    def grid(
        self,
        x_min: float = -2.0,
        x_max: float = 2.0,
        y_min: float = -2.0,
        y_max: float = 2.0,
        resolution: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate meshgrid X, Y and evaluated loss grid Z for contour plotting."""
        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = self(X[i, j], Y[i, j])
        return X, Y, Z


class IllConditionedQuadratic(LossLandscape):
    """
    Anisotropic Quadratic Bowl (The Narrow Ravine):
        f(x, y) = 0.5 * (x^2 + gamma * y^2)
    Condition Number: kappa = gamma (default 50.0).
    The gradient in y is 50x steeper than in x, causing SGD to violently oscillate
    across walls while momentum or Adam glides smoothly down the ravine floor.
    """

    def __init__(self, gamma: float = 50.0):
        self.gamma = gamma

    def __call__(self, x: float, y: float) -> float:
        return 0.5 * (x**2 + self.gamma * (y**2))

    def grad(self, x: float, y: float) -> np.ndarray:
        return np.array([x, self.gamma * y], dtype=np.float64)

    def hessian(self, x: float, y: float) -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, self.gamma]], dtype=np.float64)


class RosenbrockLandscape(LossLandscape):
    """
    The Rosenbrock "Banana" Valley (Rosenbrock, 1960):
        f(x, y) = (a - x)^2 + b * (y - x^2)^2
    Standard parameters: a = 1.0, b = 100.0.
    Global minimum: (1.0, 1.0) with f(1, 1) = 0.
    A notoriously challenging test for optimizers because finding the valley is easy,
    but navigating the twisting, flat, parabolic canyon floor requires sophisticated steps.
    """

    def __init__(self, a: float = 1.0, b: float = 100.0):
        self.a = a
        self.b = b

    def __call__(self, x: float, y: float) -> float:
        return (self.a - x) ** 2 + self.b * ((y - x**2) ** 2)

    def grad(self, x: float, y: float) -> np.ndarray:
        dx = -2.0 * (self.a - x) - 4.0 * self.b * x * (y - x**2)
        dy = 2.0 * self.b * (y - x**2)
        return np.array([dx, dy], dtype=np.float64)

    def hessian(self, x: float, y: float) -> np.ndarray:
        dxx = 2.0 - 4.0 * self.b * (y - 3.0 * (x**2))
        dxy = -4.0 * self.b * x
        dyy = 2.0 * self.b
        return np.array([[dxx, dxy], [dxy, dyy]], dtype=np.float64)


class SaddlePointLandscape(LossLandscape):
    """
    Hyperbolic Paraboloid Saddle:
        f(x, y) = 0.5 * x^2 - 0.5 * y^2
    Stationary point at (0, 0) with grad(0, 0) = [0, 0].
    Hessian eigenvalues: +1 (convex along x) and -1 (concave along y).
    Tests whether optimizers can escape saddle points and follow the negative curvature direction.
    """

    def __call__(self, x: float, y: float) -> float:
        return 0.5 * (x**2) - 0.5 * (y**2)

    def grad(self, x: float, y: float) -> np.ndarray:
        return np.array([x, -y], dtype=np.float64)

    def hessian(self, x: float, y: float) -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)


class RastriginLandscape(LossLandscape):
    """
    Rastrigin Highly Multimodal Landscape (Rastrigin, 1974):
        f(x, y) = 20 + (x^2 - 10*cos(2*pi*x)) + (y^2 - 10*cos(2*pi*y))
    Global minimum: (0, 0) with f(0, 0) = 0.
    Features hundreds of regular local minima traps to test escaping local basins.
    """

    def __call__(self, x: float, y: float) -> float:
        return (
            20.0
            + (x**2 - 10.0 * math.cos(2.0 * math.pi * x))
            + (y**2 - 10.0 * math.cos(2.0 * math.pi * y))
        )

    def grad(self, x: float, y: float) -> np.ndarray:
        dx = 2.0 * x + 20.0 * math.pi * math.sin(2.0 * math.pi * x)
        dy = 2.0 * y + 20.0 * math.pi * math.sin(2.0 * math.pi * y)
        return np.array([dx, dy], dtype=np.float64)

    def hessian(self, x: float, y: float) -> np.ndarray:
        dxx = 2.0 + 40.0 * (math.pi**2) * math.cos(2.0 * math.pi * x)
        dyy = 2.0 + 40.0 * (math.pi**2) * math.cos(2.0 * math.pi * y)
        return np.array([[dxx, 0.0], [0.0, dyy]], dtype=np.float64)


# =============================================================================
# Trajectory Simulation & Comparative Optimizer Benchmark
# =============================================================================

class TrajectoryTracker:
    """Records and evaluates optimization paths across 2D loss surfaces."""

    @staticmethod
    def run_simulation(
        landscape: LossLandscape,
        optimizer_cls: Type[Optimizer],
        start_pos: Tuple[float, float],
        n_steps: int = 100,
        clip_grad: Optional[float] = 100.0,
        **opt_kwargs,
    ) -> Dict[str, Any]:
        """
        Simulate an optimizer trajectory across a loss landscape.
        Returns coordinates, loss history, and gradient norm history.
        """
        # Create parameter array [x, y]
        param = np.array([float(start_pos[0]), float(start_pos[1])], dtype=np.float64)
        optimizer = optimizer_cls([param], **opt_kwargs)

        path = [[float(param[0]), float(param[1])]]
        losses = [float(landscape(param[0], param[1]))]
        grad_norms = []

        for step in range(n_steps):
            g = landscape.grad(param[0], param[1])
            g_norm = float(np.linalg.norm(g))
            grad_norms.append(g_norm)

            # Optional gradient clipping to prevent extreme numerical overflow
            if clip_grad is not None and g_norm > clip_grad:
                g = (g / g_norm) * clip_grad

            optimizer.step([g])

            # Safety clamp for diverging trajectories
            param[0] = np.clip(param[0], -100.0, 100.0)
            param[1] = np.clip(param[1], -100.0, 100.0)

            path.append([float(param[0]), float(param[1])])
            losses.append(float(landscape(param[0], param[1])))

        return {
            "name": optimizer_cls.__name__,
            "path": path,
            "losses": losses,
            "grad_norms": grad_norms,
            "final_pos": [float(param[0]), float(param[1])],
            "final_loss": losses[-1],
        }

    @classmethod
    def compare_all(
        cls,
        landscape: LossLandscape,
        start_pos: Tuple[float, float],
        n_steps: int = 80,
        learning_rate: float = 0.01,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Runs SGD, Momentum, RMSprop, Adam, and AdamW from identical starting positions
        for side-by-side behavioral comparison.
        """
        configs = [
            (SGD, {"lr": learning_rate}),
            (Momentum, {"lr": learning_rate, "beta": 0.9}),
            (RMSprop, {"lr": learning_rate, "alpha": 0.99}),
            (Adam, {"lr": learning_rate, "beta1": 0.9, "beta2": 0.999}),
            (AdamW, {"lr": learning_rate, "beta1": 0.9, "beta2": 0.999, "weight_decay": 0.01}),
        ]

        results = {}
        for opt_cls, kwargs in configs:
            results[opt_cls.__name__] = cls.run_simulation(
                landscape=landscape,
                optimizer_cls=opt_cls,
                start_pos=start_pos,
                n_steps=n_steps,
                **kwargs,
            )
        return results


if __name__ == "__main__":
    print("=" * 70)
    print("Module 02: Loss Landscapes & Trajectory Simulation Demo")
    print("=" * 70)

    # 1. Test on Narrow Ravine (Ill-Conditioned Quadratic)
    ravine = IllConditionedQuadratic(gamma=50.0)
    start = (-3.0, 1.5)
    print(f"\nSimulating Ravine traversal from start {start}...")
    sim_ravine = TrajectoryTracker.compare_all(ravine, start_pos=start, n_steps=60, learning_rate=0.03)

    for name, data in sim_ravine.items():
        print(f"  {name:10s} | Final Loss: {data['final_loss']:10.5f} | Final (x, y): ({data['final_pos'][0]:+.3f}, {data['final_pos'][1]:+.3f})")

    # 2. Test on Rosenbrock Banana
    rosen = RosenbrockLandscape()
    start_rosen = (-1.2, 1.0)
    print(f"\nSimulating Rosenbrock Valley traversal from start {start_rosen}...")
    sim_rosen = TrajectoryTracker.compare_all(rosen, start_pos=start_rosen, n_steps=100, learning_rate=0.002)

    for name, data in sim_rosen.items():
        print(f"  {name:10s} | Final Loss: {data['final_loss']:10.5f} | Final (x, y): ({data['final_pos'][0]:+.3f}, {data['final_pos'][1]:+.3f})")
