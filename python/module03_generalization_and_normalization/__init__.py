"""
Module 03: Generalization, Regularization & Normalization
Project Synapse: The Graphical Neural Network Explainer

From Dropout (Srivastava et al. 2014) and BatchNorm (Ioffe & Szegedy 2015)
to LayerNorm (Ba et al. 2016), Decoupled Weight Decay / AdamW (Loshchilov & Hutter 2019),
and SOTA RMSNorm (Zhang & Sennrich 2019).
"""

from .normalization_scratch import BatchNorm1dScratch, LayerNormScratch, RMSNormScratch
from .dropout_and_weight_decay import InvertedDropoutScratch, DecoupledWeightDecaySGD, AdamWScratch
from .pytorch_sota_norm import RMSNormPyTorch, PreLNTransformerBlock

__all__ = [
    "BatchNorm1dScratch",
    "LayerNormScratch",
    "RMSNormScratch",
    "InvertedDropoutScratch",
    "DecoupledWeightDecaySGD",
    "AdamWScratch",
    "RMSNormPyTorch",
    "PreLNTransformerBlock",
]
