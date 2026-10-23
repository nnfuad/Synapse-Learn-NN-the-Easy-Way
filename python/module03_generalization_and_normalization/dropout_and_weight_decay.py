"""
Module 03: Generalization, Regularization & Normalization
Tier 1: Scratch Implementations of Inverted Dropout and Decoupled Weight Decay (AdamW)

Historical Context & Citations:
1. Dropout:
   - Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).
     "Dropout: A Simple Way to Prevent Neural Networks from Overfitting."
     Journal of Machine Learning Research (JMLR), 15(56):1929-1958.
   - Physical Metaphor (The Musician Ensemble):
     If a jazz band always plays together, musicians co-adapt: the bassist gets lazy because
     the pianist always covers the low notes. If you randomly mute 50% of the musicians on
     each rehearsal, every musician is forced to master the rhythm and harmony independently.
     At concert time (inference), all musicians play together, forming a robust ensemble.

2. Decoupled Weight Decay (AdamW):
   - Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization."
     International Conference on Learning Representations (ICLR 2019).
   - Core Discovery:
     In SGD with momentum, L2 regularization and weight decay are identical up to a constant factor.
     In adaptive optimizers (Adam, RMSProp), adding L2 regularization to the loss function couples
     the decay penalty to the historical gradient variance (v_t), drastically diluting regularization
     for frequently updated weights. Decoupling the decay restored Adam's generalization performance,
     enabling modern LLM pretraining (GPT-3, LLaMA, BERT).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class InvertedDropoutScratch:
    """
    Pure NumPy implementation of Inverted Dropout (Srivastava et al., 2014).
    
    Standard Dropout scales activations at test time by (1 - p).
    Inverted Dropout inverts the scaling factor to training time:
        Training:
            m ~ Bernoulli(1 - p)
            y = (x * m) / (1 - p)
        Inference:
            y = x
    
    Advantage: Zero computational overhead or state modifications at test time.
    """

    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        """
        Args:
            p: Probability of dropping an activation (must be in [0, 1)).
            seed: Optional random seed for deterministic testing.
        """
        assert 0.0 <= p < 1.0, f"Dropout probability p must be in [0, 1), got {p}"
        self.p = p
        self.keep_prob = 1.0 - p
        self.training = True
        self.rng = np.random.default_rng(seed)
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        Args:
            x: Input array of arbitrary shape.
        Returns:
            out: Dropout-applied array of identical shape.
        """
        x = np.asarray(x, dtype=np.float64)

        if not self.training or self.p == 0.0:
            self.mask = None
            return x

        # Sample binary Bernoulli mask: 1 with probability (1 - p), 0 with probability p
        self.mask = (self.rng.random(size=x.shape) < self.keep_prob).astype(np.float64)

        # Scale by 1 / (1 - p) so expected value E[y] == E[x]
        return (x * self.mask) / self.keep_prob

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        Args:
            grad_output: Upstream gradient dL/dy.
        Returns:
            grad_input: Gradient dL/dx.
        """
        grad_output = np.asarray(grad_output, dtype=np.float64)

        if not self.training or self.p == 0.0 or self.mask is None:
            return grad_output

        return (grad_output * self.mask) / self.keep_prob


class DecoupledWeightDecaySGD:
    """
    Comparative implementation of SGD with L2 Regularization vs Decoupled Weight Decay.
    
    For vanilla SGD without momentum:
        w_l2    = w - lr * (grad + lambda * w) = w * (1 - lr * lambda) - lr * grad
        w_decay = w * (1 - lr * lambda) - lr * grad
    They are mathematically equivalent for standard SGD!
    """

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.01):
        self.lr = lr
        self.weight_decay = weight_decay

    def step_l2(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """SGD step with L2 regularization added directly to gradient."""
        effective_grad = grad + self.weight_decay * w
        return w - self.lr * effective_grad

    def step_decoupled(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """SGD step with decoupled weight decay."""
        return w * (1.0 - self.lr * self.weight_decay) - self.lr * grad


class AdamWScratch:
    """
    Pure NumPy implementation of Decoupled Weight Decay Adam (AdamW, Loshchilov & Hutter 2019).
    
    Contrasts directly with Adam with L2 regularization:
    
    1. Standard Adam with L2:
       g_t = grad + lambda * w_t
       m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
       v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t)^2
       w_{t+1} = w_t - (lr / (sqrt(v_hat) + eps)) * m_hat
       *Problem*: Large gradient dimensions blow up v_t, dividing and reducing the decay penalty!
    
    2. AdamW (Decoupled Weight Decay):
       g_t = grad  <-- gradient is clean, NO weight decay term inside moments
       m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
       v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t)^2
       w_{t+1} = w_t - (lr * lambda * w_t) - (lr / (sqrt(v_hat) + eps)) * m_hat
       *Fix*: Every parameter decays proportionally, regardless of gradient scale.
    """

    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        decoupled: bool = True,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.decoupled = decoupled

        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: Dict[str, np.ndarray]):
        """
        Perform a single optimization step across all parameters.
        """
        self.t += 1

        for name, param in self.params.items():
            grad = grads[name]

            if not self.decoupled and self.weight_decay != 0.0:
                # Coupled L2 regularization: add lambda * w into gradient
                grad = grad + self.weight_decay * param

            # Update biased 1st and 2nd moment estimates
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[name] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1.0 - self.beta2 ** self.t)

            # Adaptive gradient update step
            step_update = (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat

            if self.decoupled and self.weight_decay != 0.0:
                # Decoupled weight decay: applied directly to weights
                param -= self.lr * self.weight_decay * param

            param -= step_update
