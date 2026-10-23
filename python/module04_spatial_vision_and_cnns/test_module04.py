"""
Pytest Verification Suite for Module 04: Spatial Vision & Convolutions (CNNs)

Verifies:
1. Pure NumPy Conv2dScratch forward and analytical backward against torch.nn.Conv2d (1e-5).
2. Multiple configurations: stride, padding, dilation, and multi-channel tensors.
3. Pure NumPy MaxPool2dScratch forward and backward against torch.nn.MaxPool2d (1e-5).
4. Receptive field mathematical tracking on LeNet-5 and VGG stacks.
5. Residual blocks (Classic, Pre-activation, ConvNeXt) forward execution.
6. Degradation experiment execution.
"""

import os
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure local imports work reliably
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conv2d_scratch import Conv2dScratch, MaxPool2dScratch
from receptive_field import (
    calculate_receptive_field,
    LayerSpec,
    build_lenet5,
    build_vgg16_early_stage,
)
from resnet_pytorch import (
    ClassicResidualBlock,
    PreActivationResidualBlock,
    ConvNeXtBlock,
    run_degradation_experiment,
)


@pytest.mark.parametrize(
    "stride,padding,dilation",
    [
        (1, 0, 1),
        (2, 1, 1),
        (1, 2, 2),
    ],
)
def test_conv2d_scratch_vs_pytorch(stride, padding, dilation):
    torch.manual_seed(42)
    np.random.seed(42)

    N, Cin, Cout = 2, 3, 4
    Hin, Win = 7, 7
    kernel_size = 3

    # Generate synthetic input
    x_np = np.random.randn(N, Cin, Hin, Win).astype(np.float64)
    x_torch = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)

    # Initialize scratch conv and PyTorch conv
    conv_scratch = Conv2dScratch(
        in_channels=Cin,
        out_channels=Cout,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
    )

    conv_torch = nn.Conv2d(
        in_channels=Cin,
        out_channels=Cout,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
        dtype=torch.float64,
    )

    # Align weights and biases
    with torch.no_grad():
        conv_torch.weight.copy_(torch.tensor(conv_scratch.weight))
        conv_torch.bias.copy_(torch.tensor(conv_scratch.bias))

    # 1. Forward Pass
    out_scratch = conv_scratch.forward(x_np)
    out_torch = conv_torch(x_torch)

    np.testing.assert_allclose(
        out_scratch,
        out_torch.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Conv2D forward mismatch for s={stride}, p={padding}, d={dilation}",
    )

    # 2. Backward Pass
    grad_output = np.random.randn(*out_scratch.shape).astype(np.float64)
    out_torch.backward(torch.tensor(grad_output, dtype=torch.float64))
    dx_scratch = conv_scratch.backward(grad_output)

    # Verify input gradient dL/dx
    np.testing.assert_allclose(
        dx_scratch,
        x_torch.grad.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Conv2D dL/dx backward mismatch for s={stride}, p={padding}, d={dilation}",
    )

    # Verify weight gradient dL/dW
    np.testing.assert_allclose(
        conv_scratch.grad_weight,
        conv_torch.weight.grad.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Conv2D dL/dW backward mismatch for s={stride}, p={padding}, d={dilation}",
    )

    # Verify bias gradient dL/db
    np.testing.assert_allclose(
        conv_scratch.grad_bias,
        conv_torch.bias.grad.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Conv2D dL/db backward mismatch for s={stride}, p={padding}, d={dilation}",
    )


def test_maxpool2d_scratch_vs_pytorch():
    torch.manual_seed(42)
    np.random.seed(42)

    N, C, H, W = 2, 3, 6, 6
    kernel_size = 2
    stride = 2

    x_np = np.random.randn(N, C, H, W).astype(np.float64)
    x_torch = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)

    pool_scratch = MaxPool2dScratch(kernel_size=kernel_size, stride=stride)
    pool_torch = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    # Forward
    out_scratch = pool_scratch.forward(x_np)
    out_torch = pool_torch(x_torch)

    np.testing.assert_allclose(
        out_scratch,
        out_torch.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="MaxPool2d forward mismatch",
    )

    # Backward
    grad_output = np.random.randn(*out_scratch.shape).astype(np.float64)
    out_torch.backward(torch.tensor(grad_output, dtype=torch.float64))
    dx_scratch = pool_scratch.backward(grad_output)

    np.testing.assert_allclose(
        dx_scratch,
        x_torch.grad.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="MaxPool2d backward mismatch",
    )


def test_receptive_field_math():
    # 1. Two stacked 3x3 convs: RF should be 5
    layers_vgg = [
        LayerSpec("c1", kernel_size=3, stride=1, padding=1),
        LayerSpec("c2", kernel_size=3, stride=1, padding=1),
    ]
    metrics = calculate_receptive_field(layers_vgg, input_size=28)
    assert metrics[0].receptive_field == 3
    assert metrics[1].receptive_field == 5
    assert metrics[1].jump == 1

    # 2. Strided Conv (s=2) followed by s=1: RF should be 3, then 7
    layers_strided = [
        LayerSpec("c1", kernel_size=3, stride=2, padding=1),
        LayerSpec("c2", kernel_size=3, stride=1, padding=1),
    ]
    metrics_s = calculate_receptive_field(layers_strided, input_size=28)
    assert metrics_s[0].jump == 2
    assert metrics_s[0].receptive_field == 3
    assert metrics_s[1].receptive_field == 7  # 3 + (3 - 1) * 2 = 7

    # 3. LeNet-5 full network check
    lenet = build_lenet5()
    m_lenet = lenet.compute()
    assert m_lenet[-1].receptive_field == 32
    assert m_lenet[-1].out_size == 1


def test_residual_blocks():
    x = torch.randn(2, 16, 16, 16)

    # Classic ResNet Block
    classic_block = ClassicResidualBlock(channels=16)
    out_c = classic_block(x)
    assert out_c.shape == x.shape

    # Pre-Activation ResNet Block
    preact_block = PreActivationResidualBlock(channels=16)
    out_p = preact_block(x)
    assert out_p.shape == x.shape

    # ConvNeXt Block
    convnext_block = ConvNeXtBlock(dim=16)
    out_cn = convnext_block(x)
    assert out_cn.shape == x.shape


def test_degradation_experiment():
    results = run_degradation_experiment(num_blocks=4, channels=8, batch_size=4, img_size=8, steps=5)
    assert len(results["plain_losses"]) == 5
    assert len(results["res_losses"]) == 5
    assert results["res_grad_norms"][-1] > 0.0
