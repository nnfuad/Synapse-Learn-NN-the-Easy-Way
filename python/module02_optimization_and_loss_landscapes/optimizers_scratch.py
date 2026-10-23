"""
Module 02: Loss Geometry, Optimization & SOTA Optimizers
Part 1: Pure NumPy Implementations of Foundational & Modern Optimizers

===============================================================================
HISTORICAL LINEAGE & GROUNDING:
1. Herbert Robbins & Sutton Monro (1951):
   "A Stochastic Approximation Method."
   The Annals of Mathematical Statistics, 22(3), 400-407.
   -> Introduced Stochastic Gradient Descent (SGD) under noisy observations.

2. Boris T. Polyak (1964):
   "Some methods of speeding up the convergence of iteration methods."
   USSR Computational Mathematics and Mathematical Physics, 4(5), 1-17.
   -> Heavy-Ball Momentum method: added kinetic inertia to escape oscillations
      and accelerate along low-curvature ravines.

3. Geoffrey Hinton, Nitish Srivastava, Kevin Swersky (2012):
   "Neural Networks for Machine Learning - Lecture 6a: Overview of mini-batch
   gradient descent." Coursera.
   -> RMSprop: Exponential moving average of squared gradients to normalize
      per-parameter step sizes without vanishing like AdaGrad (Duchi et al. 2011).

4. Diederik P. Kingma & Jimmy Ba (2014 / ICLR 2015):
   "Adam: A Method for Stochastic Optimization."
   -> Combined first moment (Polyak momentum) and second moment (RMSprop variance)
      with initialization bias correction terms.

5. Ilya Loshchilov & Frank Hutter (2017 / ICLR 2019):
   "Decoupled Weight Decay Regularization."
   -> Solved Adam's fundamental flaw with L2 regularization: AdamW decouples
      weight decay from the adaptive gradient moment updates, enabling SOTA
      generalization in Transformers and deep networks.

===============================================================================
PHYSICAL ANALOGY: Skiing Down a Foggy Mountain Ravine
- The Terrain: A steep, narrow alpine ravine (ill-conditioned curvature).
- The Fog: Dense fog limits visibility to a 1-meter radius around the skier's boots
  (local gradient only; no global map).
- Vanilla SGD: The skier turns directly in the direction of steepest descent.
  They violently ping-pong between the steep canyon walls (large gradient),
  making almost zero forward progress down the gentle canyon floor (small gradient).
- Polyak Momentum: The skier acts as a heavy bowling ball with mass and inertia.
  The left-right wall bounces cancel each other out over time, while forward
  velocity steadily builds down the ravine.
- RMSprop: Skier wears smart crampons that automatically increase friction on
  the high-vibration axis (steep walls) and decrease friction along the gentle slope.
- Adam / AdamW: The skier combines high-speed aerodynamic momentum with adaptive
  crampons, plus a constant gentle tether (weight decay) pulling back toward the
  lodge to prevent runaway speeds.
===============================================================================
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional


class Optimizer:
    """Base class for scratch optimizers operating on NumPy parameter arrays."""

    def __init__(self, params: List[np.ndarray], lr: float = 1e-3, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_count = 0

    def step(self, grads: List[np.ndarray]) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        # NumPy doesn't retain implicit gradient graphs; handled externally
        pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (Robbins & Monro, 1951) with optional L2 weight decay.

    Update Rule:
        theta_{t+1} = theta_t - eta * (g_t + lambda * theta_t)
    """

    def __init__(self, params: List[np.ndarray], lr: float = 1e-2, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)

    def step(self, grads: List[np.ndarray]) -> None:
        self.step_count += 1
        for p, g in zip(self.params, grads):
            grad = g.copy()
            if self.weight_decay != 0.0:
                grad += self.weight_decay * p
            p -= self.lr * grad


class Momentum(Optimizer):
    """
    Polyak Heavy-Ball Momentum (Polyak, 1964).

    Physical Metaphor: A heavy mass rolling through the loss landscape.

    Update Rule:
        v_{t+1} = beta * v_t + (g_t + lambda * theta_t)
        theta_{t+1} = theta_t - eta * v_{t+1}
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-2,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(params, lr, weight_decay)
        self.beta = beta
        self.nesterov = nesterov
        # Velocity buffers initialized to zeros
        self.velocities = [np.zeros_like(p) for p in params]

    def step(self, grads: List[np.ndarray]) -> None:
        self.step_count += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            grad = g.copy()
            if self.weight_decay != 0.0:
                grad += self.weight_decay * p

            # Update velocity
            self.velocities[i] = self.beta * self.velocities[i] + grad

            if self.nesterov:
                # Nesterov Accelerated Gradient (NAG, Nesterov 1983)
                p_update = self.beta * self.velocities[i] + grad
            else:
                p_update = self.velocities[i]

            p -= self.lr * p_update


class RMSprop(Optimizer):
    """
    Root Mean Square Propagation (Hinton et al., 2012).

    Adapts learning rate per-parameter by dividing by the running root-mean-square
    of historical gradients.

    Update Rule:
        s_{t+1} = alpha * s_t + (1 - alpha) * g_t^2
        theta_{t+1} = theta_t - (eta / (sqrt(s_{t+1}) + eps)) * g_t
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-3,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr, weight_decay)
        self.alpha = alpha
        self.eps = eps
        # Second-moment running average buffer
        self.square_avgs = [np.zeros_like(p) for p in params]

    def step(self, grads: List[np.ndarray]) -> None:
        self.step_count += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            grad = g.copy()
            if self.weight_decay != 0.0:
                grad += self.weight_decay * p

            self.square_avgs[i] = (
                self.alpha * self.square_avgs[i] + (1.0 - self.alpha) * (grad ** 2)
            )
            denom = np.sqrt(self.square_avgs[i]) + self.eps
            p -= (self.lr / denom) * grad


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Kingma & Ba, 2014).

    Combines 1st-moment momentum (Polyak) and 2nd-moment variance scaling (RMSprop)
    with initialization bias correction. Note: In standard Adam, weight decay
    is implemented as L2 regularization on the objective (added to the gradient).

    Update Rule:
        g_tilde = g_t + lambda * theta_t   (L2 regularization)
        m_{t+1} = beta_1 * m_t + (1 - beta_1) * g_tilde
        v_{t+1} = beta_2 * v_t + (1 - beta_2) * (g_tilde^2)
        m_hat   = m_{t+1} / (1 - beta_1^{t+1})
        v_hat   = v_{t+1} / (1 - beta_2^{t+1})
        theta_{t+1} = theta_t - (eta / (sqrt(v_hat) + eps)) * m_hat
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]  # 1st moment
        self.v = [np.zeros_like(p) for p in params]  # 2nd moment

    def step(self, grads: List[np.ndarray]) -> None:
        self.step_count += 1
        t = self.step_count

        for i, (p, g) in enumerate(zip(self.params, grads)):
            grad = g.copy()
            if self.weight_decay != 0.0:
                # Standard Adam couples L2 penalty into gradient
                grad += self.weight_decay * p

            # Update biased 1st and 2nd moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)

            # Compute bias-corrected estimates
            m_hat = self.m[i] / (1.0 - self.beta1 ** t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** t)

            # Update parameters
            denom = np.sqrt(v_hat) + self.eps
            p -= (self.lr / denom) * m_hat


class AdamW(Optimizer):
    """
    Decoupled Weight Decay Adam (Loshchilov & Hutter, 2017 / ICLR 2019).

    WHY ADAMW IS REVOLUTIONARY:
    In standard Adam with L2 regularization, weights with large gradients receive
    much SMALLER weight decay than weights with small gradients because the
    denominator sqrt(v_hat) suppresses the penalty:
        grad_penalty = lambda * theta / sqrt(v_hat)
    AdamW decouples weight decay completely from moment computation:
        theta_{t+1} = theta_t - eta * lambda * theta_t - (eta / (sqrt(v_hat) + eps)) * m_hat
    This restores true proportional weight decay across all weights regardless of
    gradient scale, leading to dramatically better generalization.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(params, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]  # 1st moment
        self.v = [np.zeros_like(p) for p in params]  # 2nd moment

    def step(self, grads: List[np.ndarray]) -> None:
        self.step_count += 1
        t = self.step_count

        for i, (p, g) in enumerate(zip(self.params, grads)):
            grad = g.copy()  # Pure unregularized gradient

            # 1. Decoupled weight decay step
            if self.weight_decay != 0.0:
                p -= self.lr * self.weight_decay * p

            # 2. Update biased 1st and 2nd moment estimates purely on task gradient
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)

            # 3. Compute bias-corrected moments
            m_hat = self.m[i] / (1.0 - self.beta1 ** t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** t)

            # 4. Adaptive gradient step
            denom = np.sqrt(v_hat) + self.eps
            p -= (self.lr / denom) * m_hat


if __name__ == "__main__":
    print("=" * 70)
    print("Module 02: Pure NumPy Optimizers Test")
    print("=" * 70)

    # Simple quadratic test: f(x) = x^2, target x* = 0
    test_optimizers = {
        "SGD": SGD([np.array([5.0])], lr=0.1),
        "Momentum": Momentum([np.array([5.0])], lr=0.1, beta=0.9),
        "RMSprop": RMSprop([np.array([5.0])], lr=0.1),
        "Adam": Adam([np.array([5.0])], lr=0.5),
        "AdamW": AdamW([np.array([5.0])], lr=0.5, weight_decay=0.01),
    }

    for name, opt in test_optimizers.items():
        for _ in range(25):
            x = opt.params[0]
            grad = 2.0 * x  # d(x^2)/dx = 2x
            opt.step([grad])
        print(f"Optimizer: {name:10s} | Final x: {opt.params[0][0]:+.6f}")
