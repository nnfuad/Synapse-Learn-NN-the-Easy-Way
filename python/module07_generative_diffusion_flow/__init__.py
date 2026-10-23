"""
Module 07: Generative Paradigms (Diffusion & Flow Matching)
"""

from .diffusion_ddpm_scratch import (
    DiffusionNoiseSchedule,
    SinusoidalTimeEmbedding,
    DenoisingMLP2D,
    DDPM2D,
    generate_toy_data,
)
from .flow_matching_toy import (
    VelocityMLP2D,
    OptimalTransportFlowMatcher,
)

__all__ = [
    "DiffusionNoiseSchedule",
    "SinusoidalTimeEmbedding",
    "DenoisingMLP2D",
    "DDPM2D",
    "generate_toy_data",
    "VelocityMLP2D",
    "OptimalTransportFlowMatcher",
]
