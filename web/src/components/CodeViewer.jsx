import React, { useState } from 'react';
import { Copy, Check, Terminal, FileCode2 } from 'lucide-react';

const ALL_SNIPPETS = {
  // Module 00 Snippets
  rosenblatt: {
    module: '00',
    title: "1957 Rosenblatt Perceptron from Scratch",
    language: "Python (NumPy)",
    path: "python/module00_history_and_perceptron/rosenblatt_perceptron.py",
    description: "Pure mathematical implementation with zero black boxes. Direct Heaviside step function and Novikoff-guaranteed learning rule.",
    code: `import numpy as np

class RosenblattPerceptron:
    """Frank Rosenblatt's 1957 Single-Layer Perceptron."""
    def __init__(self, n_inputs: int, lr: float = 0.1, max_epochs: int = 50):
        self.lr = lr
        self.max_epochs = max_epochs
        self.weights = np.zeros(n_inputs)
        self.bias = 0.0

    def predict(self, x: np.ndarray) -> int:
        """Heaviside step activation: 1 if w·x + b >= 0 else 0"""
        linear = np.dot(x, self.weights) + self.bias
        return 1 if linear >= 0.0 else 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Perceptron Learning Rule: Δw = η(y - ŷ)x, Δb = η(y - ŷ)"""
        for epoch in range(1, self.max_epochs + 1):
            errors = 0
            for xi, target in zip(X, y):
                y_hat = self.predict(xi)
                err = target - y_hat
                if err != 0:
                    errors += 1
                    self.weights += self.lr * err * xi
                    self.bias += self.lr * err
            if errors == 0:
                print(f"Converged in {epoch} epochs!")
                return True
        return False`
  },
  mcculloch: {
    module: '00',
    title: "1943 McCulloch-Pitts Artificial Neuron",
    language: "Python",
    path: "python/module00_history_and_perceptron/mcculloch_pitts.py",
    description: "The first computational model of a neuron. Biological excitatory integration with absolute inhibitory veto.",
    code: `class McCullochPittsNeuron:
    """The 1943 McCulloch-Pitts Biological Logic Neuron."""
    def __init__(self, threshold: int, num_excitatory: int):
        self.threshold = threshold
        self.weights = [1] * num_excitatory

    def forward(self, excitatory: list, inhibitory: list = None) -> int:
        if inhibitory and any(inh == 1 for inh in inhibitory):
            return 0  # Absolute veto
        total_excitation = sum(w * x for w, x in zip(self.weights, excitatory))
        return 1 if total_excitation >= self.threshold else 0`
  },
  pytorch_mlp: {
    module: '00',
    title: "Modern PyTorch 2-Layer Multilayer Perceptron",
    language: "PyTorch 2.x",
    path: "python/module00_history_and_perceptron/mlp_solution_pytorch.py",
    description: "Conquering XOR by warping space with hidden non-linear layers and backpropagation.",
    code: `import torch
import torch.nn as nn

class XORResolutionNet(nn.Module):
    """Multilayer Perceptron (MLP) with non-linear hidden representations."""
    def __init__(self, hidden_dim: int = 4):
        super().__init__()
        self.layer1 = nn.Linear(2, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.layer1(x))
        return torch.sigmoid(self.layer2(h))`
  },

  // Module 01 Snippets
  autograd_engine: {
    module: '01',
    title: "Pure Scalar Autodiff Engine from Scratch (Value Class)",
    language: "Python",
    path: "python/module01_computational_graphs_autograd/autograd_engine.py",
    description: "Linnainmaa 1970 reverse mode automatic differentiation with topological sort DAG resolution and gradient accumulation.",
    code: `class Value:
    """Scalar Value node in a dynamic Directed Acyclic Graph (DAG)."""
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, self.data))))
        out = Value(s, (self,), 'sigmoid')
        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0  # Seed output gradient dL/dL = 1.0
        for node in reversed(topo):
            node._backward()`
  },
  custom_autograd_pytorch: {
    module: '01',
    title: "Custom PyTorch Function & Parity Comparison",
    language: "PyTorch 2.x",
    path: "python/module01_computational_graphs_autograd/pytorch_autograd_comparison.py",
    description: "Custom torch.autograd.Function implementation verified via gradcheck and exact parity against Value class.",
    code: `import torch
from torch.autograd import Function

class CustomSigmoid(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        output = 1.0 / (1.0 + torch.exp(-x))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (output,) = ctx.saved_tensors
        return grad_output * output * (1.0 - output)

# Numerical gradient verification
test_tensor = torch.randn(4, dtype=torch.float64, requires_grad=True)
passed = torch.autograd.gradcheck(CustomSigmoid.apply, test_tensor, eps=1e-6)
assert passed is True`
  },

  // Module 02 Snippets
  optimizers_scratch: {
    module: '02',
    title: "Pure NumPy Optimizers (SGD, Momentum, Adam, AdamW)",
    language: "Python (NumPy)",
    path: "python/module02_optimization_and_loss_landscapes/optimizers_scratch.py",
    description: "Raw equations for Robbins-Monro (1951), Polyak Momentum (1964), Kingma & Ba (2014), and Loshchilov & Hutter (2017).",
    code: `class AdamW:
    """Decoupled Weight Decay Adam (Loshchilov & Hutter 2017 / ICLR 2019)."""
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            # 1. Decoupled weight decay step
            if self.weight_decay != 0:
                p -= self.lr * self.weight_decay * p

            # 2. Update biased first and second moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # 3. Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # 4. Adaptive gradient step
            p -= (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat`
  },
  loss_landscapes: {
    module: '02',
    title: "Non-Convex Loss Landscapes & Trajectory Simulation",
    language: "Python (NumPy)",
    path: "python/module02_optimization_and_loss_landscapes/loss_landscapes.py",
    description: "Analytical gradients and Hessians for Ill-Conditioned Ravines, Rosenbrock Banana, and Saddles.",
    code: `class IllConditionedQuadratic:
    """f(x, y) = 0.5 * (x^2 + gamma * y^2). Condition Number = gamma."""
    def __init__(self, gamma=50.0):
        self.gamma = gamma

    def __call__(self, x, y):
        return 0.5 * (x**2 + self.gamma * y**2)

    def grad(self, x, y):
        return np.array([x, self.gamma * y])

class RosenbrockLandscape:
    """f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2."""
    def __call__(self, x, y):
        return (1.0 - x)**2 + 100.0 * (y - x**2)**2

    def grad(self, x, y):
        dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
        dy = 200.0 * (y - x**2)
        return np.array([dx, dy])`
  },
  pytorch_sota_opt: {
    module: '02',
    title: "PyTorch AdamW + Cosine Annealing with Warmup",
    language: "PyTorch 2.x",
    path: "python/module02_optimization_and_loss_landscapes/pytorch_optimizers.py",
    description: "Modern SOTA training regimen combining AdamW with Linear Warmup and Cosine Decay.",
    code: `class CosineAnnealingWithWarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = max(0, self.last_epoch)
        if step < self.warmup_steps:
            alpha = float(step) / float(max(1, self.warmup_steps))
            return [self.min_lr + alpha * (b - self.min_lr) for b in self.base_lrs]
        else:
            decay_steps = max(1, self.total_steps - 1 - self.warmup_steps)
            progress = min(1.0, max(0.0, float(step - self.warmup_steps) / float(decay_steps)))
            cosine_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + cosine_mult * (b - self.min_lr) for b in self.base_lrs]`
  },

  // Module 03 Snippets
  batchnorm_scratch: {
    module: '03',
    title: "BatchNorm1d from Scratch (NumPy)",
    language: "NumPy",
    path: "python/module03_generalization_and_normalization/normalization_scratch.py",
    description: "Ioffe & Szegedy 2015 mini-batch standardization with running mean/variance and analytical backward pass.",
    code: `class BatchNorm1dScratch:
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.gamma = np.ones(num_features, dtype=np.float64)
        self.beta = np.zeros(num_features, dtype=np.float64)
        self.running_mean = np.zeros(num_features, dtype=np.float64)
        self.running_var = np.ones(num_features, dtype=np.float64)
        self.eps = eps
        self.momentum = momentum
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            mean = np.mean(x, axis=0)
            x_centered = x - mean
            var_biased = np.mean(x_centered ** 2, axis=0)
            inv_std = 1.0 / np.sqrt(var_biased + self.eps)
            x_hat = x_centered * inv_std
            out = self.gamma * x_hat + self.beta

            # Update running stats for inference
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * (var_biased * x.shape[0] / (x.shape[0] - 1))
            self._cache = (x_hat, inv_std, x.shape[0])
            return out
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * x_hat + self.beta`
  },
  rmsnorm_scratch: {
    module: '03',
    title: "RMSNorm Scratch (Modern LLM SOTA)",
    language: "NumPy",
    path: "python/module03_generalization_and_normalization/normalization_scratch.py",
    description: "Zhang & Sennrich 2019 Root Mean Square Normalization. Eliminates mean centering, saving 10-50% computational overhead.",
    code: `class RMSNormScratch:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.gamma = np.ones(dim, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        x_hat = x / rms
        self._cache = (x_hat, 1.0 / rms)
        return self.gamma * x_hat

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        x_hat, inv_rms = self._cache
        dx_hat = grad_output * self.gamma
        sum_dx_hat_x_hat = np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
        return inv_rms * (dx_hat - (x_hat / self.dim) * sum_dx_hat_x_hat)`
  },
  adamw_scratch: {
    module: '03',
    title: "AdamW Decoupled Weight Decay",
    language: "NumPy",
    path: "python/module03_generalization_and_normalization/dropout_and_weight_decay.py",
    description: "Loshchilov & Hutter 2019 decoupled weight decay preventing variance-dilution of L2 regularization.",
    code: `def step_adamw(self, grads):
    self.t += 1
    for name, param in self.params.items():
        grad = grads[name]
        # Keep moments clean of weight decay!
        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m[name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[name] / (1 - self.beta2 ** self.t)

        # 1. Apply decoupled weight decay directly
        param -= self.lr * self.weight_decay * param
        # 2. Apply moment update
        param -= (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat`
  },
  pytorch_rmsnorm_block: {
    module: '03',
    title: "Modern PyTorch Pre-LN RMSNorm Transformer Block",
    language: "PyTorch 2.x",
    path: "python/module03_generalization_and_normalization/pytorch_sota_norm.py",
    description: "LLaMA 3 / Mistral architecture with FP32 normalization casting and uninterrupted identity residual highway.",
    code: `class PreLNTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = RMSNormPyTorch(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = RMSNormPyTorch(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: Residual stream is an unhindered identity highway
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x`
  },

  // Module 04 Snippets
  conv2d_scratch: {
    module: '04',
    title: "Conv2D from Scratch via im2col (NumPy)",
    language: "NumPy",
    path: "python/module04_spatial_vision_and_cnns/conv2d_scratch.py",
    description: "LeCun et al. 1998 2D convolution with stride, padding, dilation, and analytical col2im backward pass.",
    code: `class Conv2dScratch:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = np.zeros(out_channels)
        self.stride = stride
        self.padding = padding

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, Cin, Hin, Win = x.shape
        kh, kw = self.weight.shape[2:]
        Hout = (Hin + 2 * self.padding - kh) // self.stride + 1
        Wout = (Win + 2 * self.padding - kw) // self.stride + 1

        # im2col matrix transformation
        cols = self._im2col(x, kh, kw, Hout, Wout)
        out_mat = self.weight.reshape(self.out_channels, -1) @ cols + self.bias[:, None]
        return out_mat.reshape(self.out_channels, Hout, Wout, N).transpose(3, 0, 1, 2)`
  },
  receptive_field_calc: {
    module: '04',
    title: "Exact Receptive Field Calculator",
    language: "Python",
    path: "python/module04_spatial_vision_and_cnns/receptive_field.py",
    description: "Layer-by-layer spatial tracking of receptive field, jump (effective stride), and center offsets.",
    code: `def calculate_receptive_field(layers, input_size=224):
    current_size = input_size
    current_jump = 1
    current_rf = 1

    for layer in layers:
        k, s, p, d = layer.kernel_size, layer.stride, layer.padding, layer.dilation
        out_size = math.floor((current_size + 2*p - d*(k - 1) - 1) / s) + 1
        rf_out = current_rf + (k - 1) * d * current_jump
        jump_out = current_jump * s

        print(f"Layer {layer.name}: OutSize={out_size}, Jump={jump_out}, RF={rf_out}px")
        current_size, current_jump, current_rf = out_size, jump_out, rf_out`
  },
  convnext_pytorch: {
    module: '04',
    title: "ConvNeXt Block (A ConvNet for the 2020s)",
    language: "PyTorch 2.x",
    path: "python/module04_spatial_vision_and_cnns/resnet_pytorch.py",
    description: "Liu et al. 2022 modern CNN block featuring 7x7 depthwise conv, inverted bottleneck, and LayerNorm.",
    code: `class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # 7x7 Depthwise
        self.norm = nn.GroupNorm(1, dim) # LayerNorm over channels
        self.pwconv1 = nn.Linear(dim, 4 * dim) # Inverted bottleneck 4x
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x
        x = self.norm(self.dwconv(x)).permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(x))).permute(0, 3, 1, 2)
        return input_tensor + x  # Residual identity shortcut`
  },

  // Module 07 Snippets
  diffusion_ddpm: {
    module: '07',
    title: "DDPM Variance Schedule & Reverse Step",
    language: "PyTorch 2.x",
    path: "python/module07_generative_diffusion_flow/diffusion_ddpm_scratch.py",
    description: "Ho et al. 2020 Denoising Diffusion Probabilistic Models: closed-form forward noise schedule and reverse Langevin sampling step.",
    code: `class DDPM2D:
    def __init__(self, model, schedule):
        self.model = model
        self.schedule = schedule

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.schedule.num_timesteps, (batch_size,))
        noise = torch.randn_like(x_0)
        x_t, _ = self.schedule.q_sample(x_0, t, noise)
        pred_noise = self.model(x_t, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample_step(self, x_t: torch.Tensor, t_int: int) -> torch.Tensor:
        t_tensor = torch.full((x_t.shape[0],), t_int, dtype=torch.long)
        eps_theta = self.model(x_t, t_tensor)
        beta_t = self.schedule.betas[t_int]
        alpha_t = self.schedule.alphas[t_int]
        sqrt_one_minus_alpha_bar = self.schedule.sqrt_one_minus_alphas_cumprod[t_int]

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / sqrt_one_minus_alpha_bar) * eps_theta
        )
        if t_int == 0:
            return mean
        sigma = torch.sqrt(self.schedule.posterior_variance[t_int])
        return mean + sigma * torch.randn_like(x_t)`
  },

  flow_matching: {
    module: '07',
    title: "Optimal Transport Flow Matching (OT-CFM)",
    language: "PyTorch 2.x",
    path: "python/module07_generative_diffusion_flow/flow_matching_toy.py",
    description: "Lipman et al. 2023: Straight deterministic probability paths and ODE sampling via Midpoint Runge-Kutta.",
    code: `class OptimalTransportFlowMatcher:
    def __init__(self, model, sigma_min: float = 1e-4):
        self.model = model
        self.sigma_min = sigma_min

    def compute_loss(self, x_1: torch.Tensor, x_0: torch.Tensor = None) -> torch.Tensor:
        if x_0 is None:
            x_0 = torch.randn_like(x_1)
        t = torch.rand((x_1.shape[0],)).view(-1, 1)
        # Optimal Transport linear interpolation path
        x_t = (1.0 - (1.0 - self.sigma_min) * t) * x_0 + t * x_1
        # Constant target velocity along straight trajectory
        target_velocity = x_1 - (1.0 - self.sigma_min) * x_0
        pred_velocity = self.model(x_t, t.squeeze(-1))
        return F.mse_loss(pred_velocity, target_velocity)`
  },

  // Module 08 Snippets
  moe_router: {
    module: '08',
    title: "Sparse MoE Top-K Router & Aux Load Balancing",
    language: "PyTorch 2.x",
    path: "python/module08_llm_reasoning_and_alignment/moe_routing_scratch.py",
    description: "DeepSeek / Mixtral Top-K sparse routing with auxiliary load-balancing loss preventing router collapse.",
    code: `class TopKRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2, aux_loss_weight: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)
        router_probs = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1) # Renormalize

        # Auxiliary Load Balancing Loss: alpha * N * sum(f_i * P_i)
        dispatch_mask = torch.zeros_like(router_probs)
        dispatch_mask.scatter_(1, topk_indices, 1.0)
        f_i = dispatch_mask.mean(dim=0)
        P_i = router_probs.mean(dim=0)
        aux_loss = self.aux_loss_weight * self.num_experts * torch.sum(f_i * P_i)
        return topk_weights, topk_indices, aux_loss`
  },

  grpo_and_dpo: {
    module: '08',
    title: "GRPO & DPO Analytical Objectives",
    language: "PyTorch 2.x",
    path: "python/module08_llm_reasoning_and_alignment/dpo_and_grpo_loss.py",
    description: "DeepSeek-R1 Group Relative Policy Optimization (critic-free) and Rafailov 2023 Direct Preference Optimization.",
    code: `class GRPOLoss(nn.Module):
    def __init__(self, clip_epsilon: float = 0.2, kl_weight: float = 0.04):
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.kl_weight = kl_weight

    def forward(self, policy_logp, old_logp, ref_logp, rewards):
        # 1. Group Relative Advantage Normalization
        mean_r = rewards.mean()
        std_r = rewards.std(unbiased=False) + 1e-6
        advantages = (rewards - mean_r) / std_r

        # 2. PPO Surrogate Clipped Objective
        ratio = torch.exp(policy_logp - old_logp)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_obj = torch.min(surr1, surr2).mean()

        # 3. Non-negative KL penalty (DeepSeek formulation)
        u = ref_logp - policy_logp
        kl_div = (torch.exp(u) - u - 1.0).mean()
        return -policy_obj + self.kl_weight * kl_div`
  },

  // Module 09 Snippets
  sparse_autoencoder: {
    module: '09',
    title: "Sparse Autoencoder (SAE) Monosemantic Dictionary",
    language: "PyTorch 2.x",
    path: "python/module09_mechanistic_interpretability/sparse_autoencoder_sae.py",
    description: "Anthropic 2023/2024: Decomposes polysemantic activation vectors in superposition into monosemantic feature directions.",
    code: `class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_sae: int, l1_coefficient: float = 0.04):
        super().__init__()
        self.w_enc = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.w_dec = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.l1_coefficient = l1_coefficient

    def forward(self, x: torch.Tensor):
        x_centered = x - self.b_dec
        f = F.relu(F.linear(x_centered, self.w_enc, self.b_enc))
        x_hat = F.linear(f, self.w_dec, self.b_dec)

        recon_loss = F.mse_loss(x_hat, x)
        l1_loss = self.l1_coefficient * torch.mean(torch.sum(torch.abs(f), dim=-1))
        return x_hat, f, recon_loss + l1_loss

    @torch.no_grad()
    def normalize_decoder_columns(self):
        norms = torch.norm(self.w_dec.data, p=2, dim=0, keepdim=True)
        self.w_dec.data = self.w_dec.data / torch.clamp(norms, min=1e-8)`
  },

  induction_circuit: {
    module: '09',
    title: "2-Layer Transformer Induction Head Circuit",
    language: "PyTorch 2.x",
    path: "python/module09_mechanistic_interpretability/circuit_induction_heads.py",
    description: "Anthropic 2021/2022: In-context learning mechanism where Layer 0 encodes predecessor tokens and Layer 1 induction head copies them.",
    code: `class TwoLayerToyTransformer(nn.Module):
    def __init__(self, vocab_size: int = 32, d_model: int = 32, num_heads: int = 2):
        super().__init__()
        self.w_e = nn.Embedding(vocab_size, d_model)
        self.layer0 = MultiHeadAttentionLayer(d_model, num_heads) # Previous-token head
        self.layer1 = MultiHeadAttentionLayer(d_model, num_heads) # Induction head
        self.w_u = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor):
        resid = self.w_e(tokens)
        causal_mask = torch.tril(torch.ones(tokens.shape[1], tokens.shape[1]))
        out0, attn0 = self.layer0(resid, causal_mask)
        resid = resid + out0
        out1, attn1 = self.layer1(resid, causal_mask) # Spikes attention to token B!
        resid = resid + out1
        return self.w_u(resid), attn0, attn1`
  }
};

export default function CodeViewer({ currentModule = '00' }) {
  const filteredKeys = Object.keys(ALL_SNIPPETS).filter(
    (k) => ALL_SNIPPETS[k].module === currentModule
  );
  const availableKeys = filteredKeys.length > 0 ? filteredKeys : Object.keys(ALL_SNIPPETS);

  const [activeTab, setActiveTab] = useState(availableKeys[0]);
  const [copied, setCopied] = useState(false);

  React.useEffect(() => {
    setActiveTab(availableKeys[0]);
  }, [currentModule]);

  const activeSnippet = ALL_SNIPPETS[activeTab] || ALL_SNIPPETS[availableKeys[0]];

  const handleCopy = () => {
    navigator.clipboard.writeText(activeSnippet.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl overflow-hidden backdrop-blur-sm shadow-2xl">
      {/* Top Tabs */}
      <div className="flex items-center justify-between px-6 pt-5 pb-3 border-b border-slate-800 bg-slate-950/60">
        <div className="flex items-center gap-2 overflow-x-auto scrollbar-none">
          {availableKeys.map((key) => {
            const snip = ALL_SNIPPETS[key];
            const isActive = activeTab === key;
            return (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition flex items-center gap-2 ${
                  isActive
                    ? 'bg-slate-800 text-cyan-300 border border-slate-700 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                }`}
              >
                <span className="text-[9px] px-1.5 py-0.2 rounded font-mono bg-cyan-500/20 text-cyan-300">
                  {snip.language}
                </span>
                <span>{snip.title.split(' (')[0]}</span>
              </button>
            );
          })}
        </div>

        {/* Copy Button */}
        <button
          onClick={handleCopy}
          className="flex-shrink-0 flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-mono text-slate-300 transition"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
          <span>{copied ? 'Copied!' : 'Copy Code'}</span>
        </button>
      </div>

      {/* Snippet Header & Metadata */}
      <div className="px-6 py-4 bg-slate-950/30 border-b border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
        <div>
          <div className="font-bold text-white text-sm">{activeSnippet.title}</div>
          <div className="text-slate-400 mt-0.5">{activeSnippet.description}</div>
        </div>
        <div className="flex items-center gap-2 font-mono text-[11px] text-slate-400 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800 self-start sm:self-auto">
          <Terminal className="w-3.5 h-3.5 text-cyan-400" />
          <span>{activeSnippet.path}</span>
        </div>
      </div>

      {/* Code Block */}
      <div className="p-6 bg-[#060910] overflow-x-auto">
        <pre className="font-mono text-xs text-slate-200 leading-relaxed">
          <code>{activeSnippet.code}</code>
        </pre>
      </div>
    </div>
  );
}
