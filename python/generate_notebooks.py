"""
Script to generate clean, rich Jupyter Notebooks for Module 03 and Module 04.
"""

import json
import os

def create_notebook_03():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🧠 Module 03: Generalization, Regularization & Normalization (2014 - 2019)\n",
                    "### *From Dropout and the BatchNorm Myth to Modern SOTA RMSNorm and AdamW*\n",
                    "\n",
                    "---\n",
                    "\n",
                    "## 📜 Historical Context & The Three Great Crises\n",
                    "Deep neural networks possess millions (and now trillions) of free parameters. With this immense capacity comes three existential challenges:\n",
                    "1. **The Overfitting Crisis (Co-adaptation)**: Neurons learn complex mutual dependencies on training data that fail miserably on test data.\n",
                    "2. **The Internal Covariate Shift Myth vs Loss Surface Smoothing**: Why does Batch Normalization actually work? (Santurkar et al. 2018 debunking Ioffe & Szegedy 2015).\n",
                    "3. **The Adaptive Regularization Breakdown**: Why standard L2 regularization fails with Adam, necessitating Loshchilov & Hutter's 2019 **AdamW** (Decoupled Weight Decay).\n",
                    "\n",
                    "In this notebook, we journey from foundational principles to the modern normalization stack powering **LLaMA 3**, **Mistral**, and modern Transformers."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. The Overfitting Crisis & Dropout (Srivastava et al., 2014)\n",
                    "\n",
                    "### 🎻 The Musician Ensemble Analogy:\n",
                    "> If an orchestra always practices together, the second violinist may stop playing the difficult passages, relying on the first violinist to cover them (**co-adaptation**).\n",
                    "> But if the conductor randomly mutes 50% of the musicians on each measure during rehearsal, every single player is forced to master the score independently. At concert time (inference), all musicians play together, creating an immensely robust ensemble.\n",
                    "\n",
                    "### Inverted Dropout Formulation:\n",
                    "During training, sample a Bernoulli mask $m \\sim \\text{Bernoulli}(1 - p)$ and scale activations by $\\frac{1}{1-p}$:\n",
                    "$$y = \\frac{x \\odot m}{1 - p}$$\n",
                    "At inference time, dropout is disabled: $y = x$. No scaling or computational overhead is required at test time!"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "class InvertedDropout:\n",
                    "    def __init__(self, p: float = 0.5):\n",
                    "        self.p = p\n",
                    "        self.keep_prob = 1.0 - p\n",
                    "        self.training = True\n",
                    "        self.mask = None\n",
                    "        \n",
                    "    def forward(self, x: np.ndarray) -> np.ndarray:\n",
                    "        if not self.training or self.p == 0.0:\n",
                    "            return x\n",
                    "        self.mask = (np.random.rand(*x.shape) < self.keep_prob).astype(np.float64)\n",
                    "        return (x * self.mask) / self.keep_prob\n",
                    "\n",
                    "# Verify Expectation Preservation E[y] == E[x]\n",
                    "np.random.seed(42)\n",
                    "x = np.ones((1000, 50))\n",
                    "dropout = InvertedDropout(p=0.4)\n",
                    "\n",
                    "y_train = dropout.forward(x)\n",
                    "dropout.training = False\n",
                    "y_test = dropout.forward(x)\n",
                    "\n",
                    "print(f\"Input mean:            {np.mean(x):.4f}\")\n",
                    "print(f\"Training output mean:  {np.mean(y_train):.4f} (Expectation preserved!)\")\n",
                    "print(f\"Dropout rate active:   {np.mean(y_train == 0.0):.4f} (target p=0.40)\")\n",
                    "print(f\"Inference output mean: {np.mean(y_test):.4f} (Zero-overhead identity!)\")\n",
                    "\n",
                    "plt.figure(figsize=(10, 3))\n",
                    "plt.subplot(1, 2, 1)\n",
                    "plt.title('Training Activations (p=0.4 Sparsified)')\n",
                    "plt.imshow(y_train[:20, :20], cmap='viridis')\n",
                    "plt.colorbar()\n",
                    "plt.subplot(1, 2, 2)\n",
                    "plt.title('Inference Activations (Dense Identity)')\n",
                    "plt.imshow(y_test[:20, :20], cmap='viridis')\n",
                    "plt.colorbar()\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Decoupled Weight Decay vs L2 Regularization (Loshchilov & Hutter, 2019)\n",
                    "\n",
                    "Why did standard $L_2$ regularization fail when deep learning moved from SGD to adaptive optimizers like Adam?\n",
                    "\n",
                    "- **In SGD**, adding $\\frac{1}{2}\\lambda \\|w\\|^2$ to the loss yields:\n",
                    "  $$w_{t+1} = w_t - \\eta (\\nabla L(w_t) + \\lambda w_t) = w_t(1 - \\eta \\lambda) - \\eta \\nabla L(w_t)$$\n",
                    "  *Here, $L_2$ regularization and weight decay are identical.*\n",
                    "\n",
                    "- **In Adam**, $L_2$ modifies the gradient estimate: $g_t = \\nabla L(w_t) + \\lambda w_t$. This gradient is fed into the variance estimate $v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$:\n",
                    "  $$w_{t+1} = w_t - \\frac{\\eta}{\\sqrt{v_t} + \\epsilon} m_t$$\n",
                    "  *Catastrophe*: Parameters that receive frequent, large gradients have large $v_t$, which **divides and shrinks** the weight decay penalty! The parameters that need the most regularization end up receiving the least.\n",
                    "\n",
                    "- **AdamW (Decoupled Weight Decay)** removes $\\lambda w_t$ from the moments entirely, applying it directly:\n",
                    "  $$w_{t+1} = w_t - \\eta \\lambda w_t - \\frac{\\eta}{\\sqrt{v_t} + \\epsilon} m_t$$"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Simulation: Tracking parameter decay under disparate gradient scales\n",
                    "w_adam_l2 = np.array([2.0, 2.0])\n",
                    "w_adam_w  = np.array([2.0, 2.0])\n",
                    "\n",
                    "# Dimension 0 has huge gradients; Dimension 1 has sparse, small gradients\n",
                    "grads = np.array([50.0, 0.05])\n",
                    "lr = 0.01\n",
                    "decay = 0.1\n",
                    "\n",
                    "# Adam moments\n",
                    "m_l2, v_l2 = np.zeros(2), np.zeros(2)\n",
                    "m_w, v_w = np.zeros(2), np.zeros(2)\n",
                    "\n",
                    "trajectory_l2 = []\n",
                    "trajectory_w = []\n",
                    "\n",
                    "for t in range(1, 100):\n",
                    "    # 1. Adam with L2\n",
                    "    g_l2 = grads + decay * w_adam_l2\n",
                    "    m_l2 = 0.9 * m_l2 + 0.1 * g_l2\n",
                    "    v_l2 = 0.999 * v_l2 + 0.001 * (g_l2**2)\n",
                    "    m_hat_l2 = m_l2 / (1 - 0.9**t)\n",
                    "    v_hat_l2 = v_l2 / (1 - 0.999**t)\n",
                    "    w_adam_l2 -= (lr / (np.sqrt(v_hat_l2) + 1e-8)) * m_hat_l2\n",
                    "    \n",
                    "    # 2. AdamW (Decoupled)\n",
                    "    g_w = grads\n",
                    "    m_w = 0.9 * m_w + 0.1 * g_w\n",
                    "    v_w = 0.999 * v_w + 0.001 * (g_w**2)\n",
                    "    m_hat_w = m_w / (1 - 0.9**t)\n",
                    "    v_hat_w = v_w / (1 - 0.999**t)\n",
                    "    w_adam_w -= lr * decay * w_adam_w + (lr / (np.sqrt(v_hat_w) + 1e-8)) * m_hat_w\n",
                    "    \n",
                    "    trajectory_l2.append(w_adam_l2.copy())\n",
                    "    trajectory_w.append(w_adam_w.copy())\n",
                    "\n",
                    "trajectory_l2 = np.array(trajectory_l2)\n",
                    "trajectory_w = np.array(trajectory_w)\n",
                    "\n",
                    "print(f\"Final Adam + L2 weight (dim 0 with high grad): {trajectory_l2[-1, 0]:.4f}\")\n",
                    "print(f\"Final AdamW weight     (dim 0 with high grad): {trajectory_w[-1, 0]:.4f}\")\n",
                    "print(\"Notice how AdamW strictly regularizes weights without moment distortion.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. The Normalization Revolution (BatchNorm vs LayerNorm vs RMSNorm)\n",
                    "\n",
                    "### 🏭 The Assembly Line Quality Control Analogy:\n",
                    "If station 10 shifts components by +5mm, downstream station 50 drifts completely off calibration. Normalization locks the distribution into a clean coordinate space.\n",
                    "\n",
                    "| Normalization | Normalizes Across | Batch Size Dependency | Primary Domain |\n",
                    "| :--- | :--- | :--- | :--- |\n",
                    "| **BatchNorm** (2015) | Mini-batch $(B)$ for each feature channel | High (fails if $B=1$) | Computer Vision (CNNs) |\n",
                    "| **LayerNorm** (2016) | Feature channels $(D)$ for each sample | Zero (works for $B=1$) | Recurrent Nets & Transformers |\n",
                    "| **RMSNorm** (2019) | Scale only (no mean centering) | Zero (works for $B=1$) | Modern LLMs (LLaMA 3, Mistral) |\n",
                    "\n",
                    "### The Santurkar et al. (2018) Revelation:\n",
                    "Ioffe & Szegedy initially claimed BatchNorm accelerates training by reducing *Internal Covariate Shift (ICS)*. Santurkar et al. showed that even if you intentionally inject synthetic covariate shift into BatchNorm layers, the network trains just as fast! BatchNorm's real superpower is **smoothing the loss landscape**, making the gradient Lipschitz continuous and suppressing sharp ravine walls."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "\n",
                    "# SOTA RMSNorm PyTorch Implementation (LLaMA 3 Style)\n",
                    "class RMSNorm(nn.Module):\n",
                    "    def __init__(self, dim: int, eps: float = 1e-6):\n",
                    "        super().__init__()\n",
                    "        self.eps = eps\n",
                    "        self.weight = nn.Parameter(torch.ones(dim))\n",
                    "\n",
                    "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n",
                    "        # Cast to float32 internally to prevent BF16/FP16 overflow\n",
                    "        variance = x.float().pow(2).mean(-1, keepdim=True)\n",
                    "        x_norm = x * torch.rsqrt(variance + self.eps)\n",
                    "        return (x_norm * self.weight).type_as(x)\n",
                    "\n",
                    "# Distribution Comparison\n",
                    "torch.manual_seed(42)\n",
                    "raw_activations = torch.randn(64, 512) * 5.0 + 12.0  # severely shifted and scaled\n",
                    "\n",
                    "bn = nn.BatchNorm1d(512)\n",
                    "ln = nn.LayerNorm(512)\n",
                    "rms = RMSNorm(512)\n",
                    "\n",
                    "with torch.no_grad():\n",
                    "    act_bn = bn(raw_activations).flatten().numpy()\n",
                    "    act_ln = ln(raw_activations).flatten().numpy()\n",
                    "    act_rms = rms(raw_activations).flatten().numpy()\n",
                    "    act_raw = raw_activations.flatten().numpy()\n",
                    "\n",
                    "plt.figure(figsize=(12, 4))\n",
                    "plt.hist(act_raw, bins=50, alpha=0.4, color='red', label=f'Raw (mean={act_raw.mean():.1f}, std={act_raw.std():.1f})')\n",
                    "plt.hist(act_bn, bins=50, alpha=0.5, color='blue', label=f'BatchNorm (mean={act_bn.mean():.2f}, std={act_bn.std():.2f})')\n",
                    "plt.hist(act_ln, bins=50, alpha=0.5, color='green', label=f'LayerNorm (mean={act_ln.mean():.2f}, std={act_ln.std():.2f})')\n",
                    "plt.hist(act_rms, bins=50, alpha=0.5, color='purple', label=f'RMSNorm (std={act_rms.std():.2f})')\n",
                    "plt.title('Live Distribution Transformation: Raw vs BatchNorm vs LayerNorm vs RMSNorm')\n",
                    "plt.xlabel('Activation Value')\n",
                    "plt.ylabel('Frequency')\n",
                    "plt.legend()\n",
                    "plt.grid(True, alpha=0.2)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Pre-LN vs Post-LN Architecture in Transformers\n",
                    "\n",
                    "In the original 2017 Transformer (Vaswani et al.), **Post-LN** was used:\n",
                    "$$x = \\text{LayerNorm}(x + \\text{Sublayer}(x))$$\n",
                    "The normalization sits directly on the residual stream. As depth increases, gradients flowing backward through the normalization get progressively scaled down or degraded, requiring a delicate learning rate warmup schedule.\n",
                    "\n",
                    "In **Pre-LN** (used in GPT-2, GPT-3, LLaMA, Mistral):\n",
                    "$$x = x + \\text{Sublayer}(\\text{RMSNorm}(x))$$\n",
                    "The residual stream $x$ is an unhindered **identity highway**, ensuring that gradients from the final loss can flow back directly to layer 1 without distortion!"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys, os\n",
                    "sys.path.insert(0, os.path.abspath('../python/module03_generalization_and_normalization'))\n",
                    "from pytorch_sota_norm import compare_pre_vs_post_ln_gradients\n",
                    "\n",
                    "depths = [4, 8, 16, 24]\n",
                    "pre_grads, post_grads = [], []\n",
                    "\n",
                    "for d in depths:\n",
                    "    g_pre, g_post = compare_pre_vs_post_ln_gradients(depth=d)\n",
                    "    pre_grads.append(g_pre)\n",
                    "    post_grads.append(g_post)\n",
                    "\n",
                    "plt.figure(figsize=(8, 4))\n",
                    "plt.plot(depths, pre_grads, marker='o', linewidth=2, color='#06b6d4', label='Pre-LN (RMSNorm Highway)')\n",
                    "plt.plot(depths, post_grads, marker='s', linewidth=2, color='#f43f5e', label='Post-LN (Vaswani 2017)')\n",
                    "plt.title('Gradient Norm at Layer 1 vs Network Depth')\n",
                    "plt.xlabel('Number of Transformer Layers')\n",
                    "plt.ylabel('Input Gradient Norm ||dL/dx||')\n",
                    "plt.legend()\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "print('Key Finding: Pre-LN preserves stable, well-behaved gradients across deep stacks,')\n",
                    "print('eliminating training collapse and enabling 100+ layer Transformer scaling!')"
                ]
            }
        ],
        "metadata": {
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open("notebooks/03_generalization_and_normalization.ipynb", "w") as f:
        json.dump(notebook, f, indent=1)
    print("Successfully generated notebooks/03_generalization_and_normalization.ipynb")


def create_notebook_04():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 👁️ Module 04: Spatial Vision & Convolutions (CNNs) (1998 - 2022)\n",
                    "### *From LeNet-5 & Receptive Fields to the ResNet Revolution & ConvNeXt*\n",
                    "\n",
                    "---\n",
                    "\n",
                    "## 📜 Historical Context & The Three Spatial Frontiers\n",
                    "In 1998, **Yann LeCun** and colleagues published the landmark paper introducing **LeNet-5**, demonstrating that machines could read handwritten digits with superhuman accuracy.\n",
                    "\n",
                    "Why did dense neural networks fail on images? Why was the Convolutional Neural Network (CNN) necessary?\n",
                    "1. **The Curse of Dimensionality & Spatial Blindness**: An MLP treats pixel (0,0) and pixel (0,1) as having no more relationship than pixel (0,0) and pixel (224,224).\n",
                    "2. **The Receptive Field & Spatial Hierarchy**: How local features (edges, corners) compose into complex semantics (eyes, wheels, faces).\n",
                    "3. **The 2015 Deep Degradation Crisis**: Kaiming He et al. discovered that stacking 50+ plain layers causes accuracy to collapse — resolved by **ResNet's Identity Shortcut Connection**.\n",
                    "4. **ConvNeXt (2022)**: Modernizing CNNs with Vision Transformer principles."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Spatial Locality & The Stenciled Flashlight Analogy\n",
                    "\n",
                    "### 🔦 The Metaphor:\n",
                    "> Imagine examining a massive painting in a pitch-black gallery. An MLP turns on a colossal floodlight connecting every microscopic pigment particle to every receptor in your eye simultaneously — requiring hundreds of millions of parameters.\n",
                    "> A Convolutional Layer instead hands you a **stenciled flashlight** (the kernel). You systematically sweep it across the canvas (stride), scanning for localized visual patterns regardless of where they sit in space (**translation equivariance**).\n",
                    "\n",
                    "### Mathematical Definition of 2D Discrete Convolution:\n",
                    "$$S(i, j) = (I * K)(i, j) = \\sum_{m} \\sum_{n} I(i + m, j + n) K(m, n) + b$$"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "# Create a synthetic 16x16 test pattern with vertical and horizontal edges\n",
                    "image = np.zeros((16, 16))\n",
                    "image[4:12, 4:12] = 1.0\n",
                    "image[7:9, 7:9] = 0.5\n",
                    "\n",
                    "# Define classic spatial kernels\n",
                    "sobel_vertical = np.array([\n",
                    "    [-1, 0, 1],\n",
                    "    [-2, 0, 2],\n",
                    "    [-1, 0, 1]\n",
                    "], dtype=np.float64)\n",
                    "\n",
                    "sobel_horizontal = np.array([\n",
                    "    [-1, -2, -1],\n",
                    "    [ 0,  0,  0],\n",
                    "    [ 1,  2,  1]\n",
                    "], dtype=np.float64)\n",
                    "\n",
                    "sharpen = np.array([\n",
                    "    [ 0, -1,  0],\n",
                    "    [-1,  5, -1],\n",
                    "    [ 0, -1,  0]\n",
                    "], dtype=np.float64)\n",
                    "\n",
                    "def apply_kernel_2d(img, kernel):\n",
                    "    kh, kw = kernel.shape\n",
                    "    h, w = img.shape\n",
                    "    out = np.zeros((h - kh + 1, w - kw + 1))\n",
                    "    for i in range(out.shape[0]):\n",
                    "        for j in range(out.shape[1]):\n",
                    "            out[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)\n",
                    "    return out\n",
                    "\n",
                    "out_v = apply_kernel_2d(image, sobel_vertical)\n",
                    "out_h = apply_kernel_2d(image, sobel_horizontal)\n",
                    "out_s = apply_kernel_2d(image, sharpen)\n",
                    "\n",
                    "fig, axes = plt.subplots(1, 4, figsize=(14, 3))\n",
                    "axes[0].imshow(image, cmap='magma'); axes[0].set_title('Original Image (16x16)')\n",
                    "axes[1].imshow(out_v, cmap='coolwarm'); axes[1].set_title('Sobel Vertical Edges')\n",
                    "axes[2].imshow(out_h, cmap='coolwarm'); axes[2].set_title('Sobel Horizontal Edges')\n",
                    "axes[3].imshow(out_s, cmap='viridis'); axes[3].set_title('Sharpen Filter')\n",
                    "for ax in axes: ax.axis('off')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. The Receptive Field Arithmetic\n",
                    "\n",
                    "The **Receptive Field (RF)** of a unit in layer $l$ is the region of the original input canvas that influences its activation.\n",
                    "\n",
                    "### Exact Recurrence Formulas:\n",
                    "For layer $l$ with kernel size $k_l$, stride $s_l$, padding $p_l$, and dilation $d_l$:\n",
                    "1. **Jump**: $j_l = j_{l-1} \\cdot s_l$\n",
                    "2. **Receptive Field**: $r_l = r_{l-1} + (k_l - 1) \\cdot d_l \\cdot j_{l-1}$\n",
                    "3. **Center Coordinate**: $\\text{start}_l = \\text{start}_{l-1} + \\left(\\frac{k_l - 1}{2} \\cdot d_l - p_l\\right) \\cdot j_{l-1}$\n",
                    "\n",
                    "### The VGG Insight (Simonyan & Zisserman, 2014):\n",
                    "Why stack two $3 \\times 3$ convolutions instead of one $5 \\times 5$ convolution?\n",
                    "- Both cover an identical receptive field of $5 \\times 5$.\n",
                    "- But two $3 \\times 3$ filters use $2 \\times (3^2 \\cdot C^2) = 18 C^2$ parameters vs $5^2 \\cdot C^2 = 25 C^2$ (a **28% reduction** in parameters) and introduce **two non-linear activation functions** instead of one!"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys, os\n",
                    "sys.path.insert(0, os.path.abspath('../python/module04_spatial_vision_and_cnns'))\n",
                    "from receptive_field import build_lenet5, build_vgg16_early_stage\n",
                    "\n",
                    "print(\"=== LeNet-5 (1998) Receptive Field Tracking ===\")\n",
                    "print(build_lenet5().summary_table())\n",
                    "\n",
                    "print(\"\\n=== VGG-16 Early Stage Receptive Field Tracking ===\")\n",
                    "print(build_vgg16_early_stage().summary_table())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. The 2015 Deep Degradation Problem & The ResNet Highway (He et al., 2015)\n",
                    "\n",
                    "Before ResNet, researchers assumed deeper networks always performed better until overfitting set in. But in 2015, Kaiming He et al. discovered an astonishing paradox:\n",
                    "> A 56-layer plain network suffered higher **training error** than a 20-layer plain network!\n",
                    "\n",
                    "Because training error was higher, this was NOT overfitting. And because BatchNorm was present, gradients were not vanishing. The cause was **Optimization Degradation**: SGD simply cannot optimize deep stacks of non-linear transformations to learn the identity function $\\mathcal{H}(x) = x$.\n",
                    "\n",
                    "### The Residual Solution:\n",
                    "$$\\mathcal{H}(x) = \\mathcal{F}(x) + x$$\n",
                    "By adding the skip connection $x$, the gradient flows backward as:\n",
                    "$$\\frac{\\partial \\mathcal{E}}{\\partial x_l} = \\frac{\\partial \\mathcal{E}}{\\partial x_L} \\left( \\mathbf{I} + \\frac{\\partial}{\\partial x_l} \\sum_{i=l}^{L-1} \\mathcal{F}(x_i) \\right)$$\n",
                    "The identity term $\\mathbf{I}$ guarantees that gradient signal flows directly from the loss to the first layer, completely unaffected by vanishing weights!"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from resnet_pytorch import run_degradation_experiment\n",
                    "\n",
                    "print(\"Running Deep Degradation Experiment (30-layer PlainNet vs 30-layer ResNet)...\\n\")\n",
                    "results = run_degradation_experiment(num_blocks=14, channels=16, steps=25)\n",
                    "\n",
                    "plt.figure(figsize=(12, 4))\n",
                    "plt.subplot(1, 2, 1)\n",
                    "plt.plot(results['plain_losses'], label='30-layer PlainNet (No Skips)', color='#f43f5e', linewidth=2)\n",
                    "plt.plot(results['res_losses'], label='30-layer ResNet (With Skips)', color='#10b981', linewidth=2)\n",
                    "plt.title('Training Loss Progression')\n",
                    "plt.xlabel('Optimization Steps')\n",
                    "plt.ylabel('Cross Entropy Loss')\n",
                    "plt.legend()\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "\n",
                    "plt.subplot(1, 2, 2)\n",
                    "plt.plot(results['plain_grad_norms'], label='PlainNet Layer 1 Gradient Norm', color='#f43f5e', linewidth=2)\n",
                    "plt.plot(results['res_grad_norms'], label='ResNet Layer 1 Gradient Norm', color='#10b981', linewidth=2)\n",
                    "plt.title('First Layer Gradient Norm ||dL/dw_1||')\n",
                    "plt.xlabel('Optimization Steps')\n",
                    "plt.ylabel('Gradient Norm')\n",
                    "plt.legend()\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "print(f\"Final PlainNet Loss: {results['plain_final_loss']:.4f}\")\n",
                    "print(f\"Final ResNet Loss:   {results['res_final_loss']:.4f} (Dramatically superior optimization!)\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. ConvNeXt: Modernizing CNNs for the 2020s (Liu et al., 2022)\n",
                    "\n",
                    "In 2020, Vision Transformers (ViT) threatened to render CNNs obsolete. In 2022, Liu et al. re-engineered ResNet using modern architectural choices borrowed from Transformers:\n",
                    "\n",
                    "1. **Macro design**: Changed stage compute ratio to 1:1:3:1 (like Swin Transformer).\n",
                    "2. **Depthwise Separable Convolutions**: Using $7 \\times 7$ depthwise convs (matching the larger window size of ViT patches).\n",
                    "3. **Inverted Bottleneck**: Channel expansion $D \\to 4D \\to D$.\n",
                    "4. **LayerNorm instead of BatchNorm**: Standardizing on feature normalization.\n",
                    "5. **Fewer Activations & Normalizations**: Only 1 GELU and 1 LayerNorm per block."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "from resnet_pytorch import ConvNeXtBlock, ClassicResidualBlock\n",
                    "\n",
                    "x = torch.randn(4, 64, 32, 32)\n",
                    "resnet_block = ClassicResidualBlock(channels=64)\n",
                    "convnext_block = ConvNeXtBlock(dim=64)\n",
                    "\n",
                    "out_res = resnet_block(x)\n",
                    "out_next = convnext_block(x)\n",
                    "\n",
                    "print(\"Input Tensor shape:     \", x.shape)\n",
                    "print(\"ResNet Block output:    \", out_res.shape)\n",
                    "print(\"ConvNeXt Block output:  \", out_next.shape)\n",
                    "print(\"\\nConvNeXt parameters:\")\n",
                    "for name, p in convnext_block.named_parameters():\n",
                    "    print(f\"  {name:<20} {tuple(p.shape)}\")\n",
                    "\n",
                    "print(\"\\nSuccess! Both Module 03 and Module 04 demonstrate the deep evolutionary path of modern deep vision.\")"
                ]
            }
        ],
        "metadata": {
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open("notebooks/04_spatial_vision_and_cnns.ipynb", "w") as f:
        json.dump(notebook, f, indent=1)
    print("Successfully generated notebooks/04_spatial_vision_and_cnns.ipynb")


if __name__ == "__main__":
    create_notebook_03()
    create_notebook_04()
