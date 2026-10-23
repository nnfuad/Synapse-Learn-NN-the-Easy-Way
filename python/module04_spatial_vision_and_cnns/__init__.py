"""
Module 04: Spatial Vision & Convolutions (CNNs)
Project Synapse: The Graphical Neural Network Explainer

From LeNet-5 (LeCun et al. 1998) to ResNet (He et al. 2015)
and ConvNeXt (Liu et al. 2022).
"""

from .conv2d_scratch import Conv2dScratch, MaxPool2dScratch
from .receptive_field import ReceptiveFieldTracker, calculate_receptive_field
from .resnet_pytorch import (
    ClassicResidualBlock,
    PreActivationResidualBlock,
    ConvNeXtBlock,
    run_degradation_experiment,
)

__all__ = [
    "Conv2dScratch",
    "MaxPool2dScratch",
    "ReceptiveFieldTracker",
    "calculate_receptive_field",
    "ClassicResidualBlock",
    "PreActivationResidualBlock",
    "ConvNeXtBlock",
    "run_degradation_experiment",
]
