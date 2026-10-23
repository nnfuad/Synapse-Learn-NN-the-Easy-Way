"""
Module 04: Spatial Vision & Convolutions (CNNs)
Tier 2: Production SOTA Residual Architectures and The Degradation Problem Resolution

Historical Context & Citations:
1. The Deep Degradation Crisis:
   - He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Deep Residual Learning for Image Recognition."
     CVPR 2016 (Best Paper Award).
   - Core Discovery: When deeper networks start converging, a degradation problem is exposed:
     with network depth increasing, accuracy gets saturated and degrades rapidly.
     Unexpectedly, this degradation is NOT caused by vanishing/exploding gradients
     (verified by normalized initializations and BatchNorm), but by optimization difficulty:
     stochastic gradient descent struggles to optimize an unconstrained stack of non-linear
     layers to learn even a trivial identity mapping H(x) = x.
   - The Residual Solution: Parameterize H(x) = F(x) + x. Driving residual weights F(x) -> 0
     is effortless, establishing an indestructible identity gradient highway.

2. Pre-activation ResNet:
   - He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Identity Mappings in Deep Residual Networks."
     ECCV 2016.
   - Places BN and ReLU *before* the convolution, leaving the skip connection 100% clean.

3. ConvNeXt (A ConvNet for the 2020s):
   - Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022).
     "A ConvNet for the 2020s." CVPR 2022.
   - Modernizes ResNet with Vision Transformer (ViT) designs:
     1. Inverted bottleneck (dim -> 4 * dim -> dim).
     2. 7x7 Depthwise convolution.
     3. LayerNorm instead of BatchNorm.
     4. Fewer activations (GELU) and normalizations.
"""

from typing import Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicResidualBlock(nn.Module):
    """
    Classic Post-Activation Residual Block (He et al., 2015).
    
    Path:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN (+ x) -> ReLU
    """

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # Shortcut projection if spatial or channel dimension changes
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + identity)
        return out


class PreActivationResidualBlock(nn.Module):
    """
    Pre-Activation Residual Block (He et al., 2016).
    
    Path:
        x -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv3x3 (+ x)
    
    The identity addition is completely unobstructed, allowing direct gradient flow.
    """

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1:
            self.shortcut = nn.Conv2d(channels, channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + identity


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block (Liu et al., 2022).
    
    Features:
    1. 7x7 Depthwise Convolution (groups = channels).
    2. Channels-first LayerNorm (or permuted LayerNorm).
    3. Pointwise 1x1 Conv with 4x hidden expansion (inverted bottleneck).
    4. GELU activation.
    5. Pointwise 1x1 Conv back to channels.
    6. Residual shortcut connection.
    """

    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        # 7x7 Depthwise Conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # LayerNorm across channel dimension
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)  # GroupNorm(1, C) == LayerNorm over (C, H, W)
        # Pointwise 1x1 convs (inverted bottleneck)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # Learnable layer scale parameter
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x
        x = self.dwconv(x)
        x = self.norm(x)
        # Permute to (N, H, W, C) for linear layers
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # Permute back to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return input_tensor + x


class PlainBlock(nn.Module):
    """
    Plain Convolutional Block with NO skip connection.
    Used for verifying the degradation problem against ResNet.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out)


def run_degradation_experiment(
    num_blocks: int = 15,
    channels: int = 16,
    batch_size: int = 8,
    img_size: int = 16,
    steps: int = 20,
) -> Dict[str, Any]:
    """
    Demonstrates the degradation problem:
    Trains a 30-layer PlainNet vs a 30-layer ResNet on synthetic data.
    Logs gradient norms at layer 1 and loss progression.
    """
    torch.manual_seed(42)

    # Build PlainNet
    plain_net = nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1),
        *[PlainBlock(channels) for _ in range(num_blocks)],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(channels, 2),
    )

    # Build ResNet
    res_net = nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1),
        *[ClassicResidualBlock(channels) for _ in range(num_blocks)],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(channels, 2),
    )

    opt_plain = torch.optim.SGD(plain_net.parameters(), lr=0.05, momentum=0.9)
    opt_res = torch.optim.SGD(res_net.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Synthetic dataset
    X = torch.randn(batch_size, channels, img_size, img_size)
    y = torch.randint(0, 2, (batch_size,))

    plain_losses = []
    res_losses = []
    plain_grad_norms = []
    res_grad_norms = []

    for _ in range(steps):
        # PlainNet Step
        opt_plain.zero_grad()
        out_p = plain_net(X)
        loss_p = criterion(out_p, y)
        loss_p.backward()
        # Gradient norm at the first conv layer
        p_grad_norm = float(plain_net[0].weight.grad.norm().item())
        opt_plain.step()

        # ResNet Step
        opt_res.zero_grad()
        out_r = res_net(X)
        loss_r = criterion(out_r, y)
        loss_r.backward()
        r_grad_norm = float(res_net[0].weight.grad.norm().item())
        opt_res.step()

        plain_losses.append(loss_p.item())
        res_losses.append(loss_r.item())
        plain_grad_norms.append(p_grad_norm)
        res_grad_norms.append(r_grad_norm)

    return {
        "plain_losses": plain_losses,
        "res_losses": res_losses,
        "plain_grad_norms": plain_grad_norms,
        "res_grad_norms": res_grad_norms,
        "res_final_loss": res_losses[-1],
        "plain_final_loss": plain_losses[-1],
    }
