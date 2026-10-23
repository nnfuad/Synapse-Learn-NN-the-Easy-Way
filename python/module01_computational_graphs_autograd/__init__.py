"""
Module 01: Computational Graphs, Autograd & The Reverse Chain Rule
Pedagogical and reference implementations of automatic differentiation.
"""

from .autograd_engine import Value, Module, Neuron, Layer, MLP
from .pytorch_autograd_comparison import CustomSigmoid, CustomGELU, compare_scalar_graph

__all__ = [
    "Value",
    "Module",
    "Neuron",
    "Layer",
    "MLP",
    "CustomSigmoid",
    "CustomGELU",
    "compare_scalar_graph",
]
