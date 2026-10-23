"""
Unit Tests for Module 00: The Genesis & The Atomic Neuron
Verifies logical gates, perceptron convergence on linear data, XOR impossibility, and MLP resolution.
"""

import numpy as np
import torch
import pytest

from mcculloch_pitts import (
    McCullochPittsNeuron,
    build_and_gate,
    build_or_gate,
    build_not_gate,
)
from rosenblatt_perceptron import RosenblattPerceptron, PyTorchPerceptron
from mlp_solution_pytorch import XORResolutionNet


def test_mcculloch_pitts_and():
    and_gate = build_and_gate()
    assert and_gate.forward([0, 0]) == 0
    assert and_gate.forward([0, 1]) == 0
    assert and_gate.forward([1, 0]) == 0
    assert and_gate.forward([1, 1]) == 1


def test_mcculloch_pitts_or():
    or_gate = build_or_gate()
    assert or_gate.forward([0, 0]) == 0
    assert or_gate.forward([0, 1]) == 1
    assert or_gate.forward([1, 0]) == 1
    assert or_gate.forward([1, 1]) == 1


def test_mcculloch_pitts_not():
    not_gate = build_not_gate()
    assert not_gate(0) == 1
    assert not_gate(1) == 0


def test_rosenblatt_perceptron_convergence_and_gate():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1])
    perceptron = RosenblattPerceptron(n_inputs=2, learning_rate=0.1, max_epochs=30)
    converged = perceptron.fit(X, y)
    assert converged is True
    for xi, target in zip(X, y):
        assert perceptron.predict(xi) == target


def test_rosenblatt_perceptron_fails_on_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    perceptron = RosenblattPerceptron(n_inputs=2, learning_rate=0.1, max_epochs=25)
    converged = perceptron.fit(X, y)
    assert converged is False
    # Max accuracy for a line on XOR is 3 out of 4 (75%)
    preds = [perceptron.predict(xi) for xi in X]
    acc = sum(p == t for p, t in zip(preds, y)) / 4.0
    assert acc <= 0.75


def test_pytorch_mlp_solves_xor():
    torch.manual_seed(42)
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.], [1.], [0.]])
    model = XORResolutionNet(hidden_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    criterion = torch.nn.BCELoss()

    for _ in range(350):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = (model(X) > 0.5).float()
        assert torch.equal(preds, y)
