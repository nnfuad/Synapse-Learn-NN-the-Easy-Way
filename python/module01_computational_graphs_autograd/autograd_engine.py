"""
Module 01: Computational Graphs, Autograd & The Reverse Chain Rule
Part 1: Pure Scalar Autodiff Engine from Scratch (The "Micrograd" Foundation)

===============================================================================
HISTORICAL CONTEXT & FOUNDATIONAL PAPERS:
1. Seppo Linnainmaa (1970):
   "The representation of the cumulative rounding error of an algorithm as a
   Taylor expansion of the local rounding errors."
   Master's Thesis, University of Helsinki.
   -> Invented general reverse-mode automatic differentiation (Reverse AD) on
      computational graphs with O(V + E) complexity.

2. Paul J. Werbos (1974):
   "Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences."
   PhD Dissertation, Harvard University.
   -> First to propose applying reverse automatic differentiation to multi-stage
      neural networks.

3. David E. Rumelhart, Geoffrey E. Hinton, & Ronald J. Williams (1986):
   "Learning representations by back-propagating errors."
   Nature, Vol. 323, pp. 533–536.
   -> Popularized backpropagation, showing how intermediate hidden representations
      empower neural nets to solve non-linearly separable problems and end the
      17-year First AI Winter.

===============================================================================
THE PHYSICAL ANALOGY: The Water Pipe Network with Reverse Pressure Gauges
Imagine a branching water pipeline:
- Forward Pass:
  Water flows from input tanks (x, w, b) through valves and mixing junctions
  (+, *, non-linear pumps). It finally collects in a single reservoir: the Loss (L).
  Each junction computes its local throughput value based on incoming flow.

- Backward Pass:
  We apply a unit surge of hydraulic test pressure at the Loss reservoir (dL/dL = 1.0).
  This pressure pulse travels BACKWARDS upstream through the pipes.
  At each junction, the gauge sensitivity (local partial derivative dy/dx) regulates
  how much pressure transmits to the upstream pipes.
  If an upstream pipe branches into MULTIPLE downstream junctions, the backpressures
  ADD TOGETHER (Superposition / Chain Rule with multiple paths).
===============================================================================
"""

from __future__ import annotations
import math
import random
from typing import Set, List, Tuple, Union, Optional, Dict, Any


class Value:
    """
    A scalar Value node in a dynamic Directed Acyclic Graph (DAG) for automatic differentiation.
    Stores a float scalar, its gradient, reference to parent nodes, and the backward function.
    """

    def __init__(
        self,
        data: float,
        _children: Tuple[Value, ...] = (),
        _op: str = "",
        label: str = "",
    ):
        self.data: float = float(data)
        self.grad: float = 0.0
        self._backward: callable = lambda: None
        self._prev: Set[Value] = set(_children)
        self._op: str = _op
        self.label: str = label

    def __repr__(self) -> str:
        label_str = f", label='{self.label}'" if self.label else ""
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f}, op='{self._op}'{label_str})"

    # -------------------------------------------------------------------------
    # Core Elementary Arithmetic Operations
    # -------------------------------------------------------------------------

    def __add__(self, other: Union[Value, float, int]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(x + y)/dx = 1, d(x + y)/dy = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Union[Value, float, int]) -> Value:
        return self + other

    def __mul__(self, other: Union[Value, float, int]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(x * y)/dx = y, d(x * y)/dy = x
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Union[Value, float, int]) -> Value:
        return self * other

    def __pow__(self, other: Union[int, float]) -> Value:
        assert isinstance(
            other, (int, float)
        ), f"Only int/float powers supported, got {type(other)}"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            # d(x^c)/dx = c * x^(c - 1)
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Union[Value, float, int]) -> Value:
        return self + (-other)

    def __rsub__(self, other: Union[Value, float, int]) -> Value:
        return Value(other) + (-self)

    def __truediv__(self, other: Union[Value, float, int]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * (other**-1)

    def __rtruediv__(self, other: Union[Value, float, int]) -> Value:
        return Value(other) * (self**-1)

    # -------------------------------------------------------------------------
    # Non-linear Activation Functions
    # -------------------------------------------------------------------------

    def relu(self) -> Value:
        """Rectified Linear Unit: f(x) = max(0, x)."""
        out = Value(self.data if self.data > 0 else 0.0, (self,), "ReLU")

        def _backward():
            # d(ReLU(x))/dx = 1 if x > 0 else 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        """
        Sigmoid activation function:
        sigma(x) = 1 / (1 + exp(-x))
        d(sigma(x))/dx = sigma(x) * (1 - sigma(x))
        """
        x = self.data
        # Clip x to prevent overflow in math.exp
        x_clamped = max(-60.0, min(60.0, x))
        s = 1.0 / (1.0 + math.exp(-x_clamped))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        """
        Hyperbolic tangent activation function:
        tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        d(tanh(x))/dx = 1 - tanh(x)^2
        """
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        """Exponential function: e^x. d(e^x)/dx = e^x."""
        x = self.data
        x_clamped = max(-60.0, min(60.0, x))
        e = math.exp(x_clamped)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        """Natural logarithm: ln(x). d(ln(x))/dx = 1 / x."""
        assert self.data > 0, f"Cannot take log of non-positive number: {self.data}"
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Reverse-Mode Automatic Differentiation Execution (Linnainmaa 1970)
    # -------------------------------------------------------------------------

    def backward(self) -> None:
        """
        Compute gradients of this scalar with respect to all graph ancestors.
        Executes reverse-mode autodiff via topological sort in O(V + E) time.
        """
        topo: List[Value] = []
        visited: Set[Value] = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed the output gradient: dL / dL = 1.0
        self.grad = 1.0

        # Traverse DAG in reverse topological order (from output back to inputs)
        for node in reversed(topo):
            node._backward()

    # -------------------------------------------------------------------------
    # Computational Graph Trace Export for Pedagogical Visualizers
    # -------------------------------------------------------------------------

    def to_graph_dict(self) -> Dict[str, Any]:
        """
        Serialize the DAG connected to this node into a JSON-serializable structure
        with node labels, forward values, gradients, operations, and edges.
        """
        nodes = []
        edges = []
        visited = set()

        def trace(v: Value):
            if v not in visited:
                visited.add(v)
                node_id = f"node_{id(v)}"
                nodes.append({
                    "id": node_id,
                    "label": v.label or (v._op if v._op else f"{v.data:.2f}"),
                    "data": float(v.data),
                    "grad": float(v.grad),
                    "op": v._op,
                })
                for child in v._prev:
                    child_id = f"node_{id(child)}"
                    edges.append({
                        "source": child_id,
                        "target": node_id,
                    })
                    trace(child)

        trace(self)
        return {"nodes": nodes, "edges": edges}


# =============================================================================
# Neural Network Modules Built Atop the Value Class
# =============================================================================

class Module:
    """Base class for all neural network modules."""

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> List[Value]:
        return []


class Neuron(Module):
    """
    A single biological/artificial neuron computing:
    y = activation(sum(w_i * x_i) + b)
    """

    def __init__(self, nin: int, nonlin: str = "tanh"):
        # Initialize weights with standard Xavier-like scaling
        scale = 1.0 / math.sqrt(nin) if nin > 0 else 1.0
        self.w = [Value(random.uniform(-scale, scale), label=f"w_{i}") for i in range(nin)]
        self.b = Value(0.0, label="b")
        self.nonlin = nonlin

    def __call__(self, x: List[Union[Value, float]]) -> Value:
        assert len(x) == len(self.w), f"Expected {len(self.w)} inputs, got {len(x)}"
        # Linear aggregation: act = w1*x1 + w2*x2 + ... + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.nonlin == "relu":
            return act.relu()
        elif self.nonlin == "sigmoid":
            return act.sigmoid()
        elif self.nonlin == "tanh":
            return act.tanh()
        elif self.nonlin == "linear" or self.nonlin is None:
            return act
        else:
            raise ValueError(f"Unknown non-linearity: {self.nonlin}")

    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer(Module):
    """A layer of independent neurons."""

    def __init__(self, nin: int, nout: int, nonlin: str = "tanh"):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x: List[Union[Value, float]]) -> Union[Value, List[Value]]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> List[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """
    Multi-Layer Perceptron (MLP) constructed purely with scalar `Value` objects.
    """

    def __init__(self, nin: int, nouts: List[int], nonlin: str = "tanh"):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            # Use linear activation on final output layer if desired, or specified nonlin
            layer_nonlin = nonlin if i < len(nouts) - 1 else "sigmoid"
            self.layers.append(Layer(sz[i], sz[i + 1], nonlin=layer_nonlin))

    def __call__(self, x: List[Union[Value, float]]) -> Union[Value, List[Value]]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]


if __name__ == "__main__":
    print("=" * 70)
    print("Module 01: Pure Scratch Autograd Engine Demo")
    print("=" * 70)

    # 1. Test basic arithmetic expression with shared variable (multiple paths)
    x = Value(2.0, label="x")
    y = Value(3.0, label="y")
    z = Value(-4.0, label="z")

    # Expression: L = (x * y + z) * (x + y)
    q = (x * y + z); q.label = "q"
    r = (x + y); r.label = "r"
    L = q * r; L.label = "L"

    L.backward()

    print(f"Forward Result: L = {L.data:.4f} (Expected: (2*3 - 4)*(2+3) = 2*5 = 10)")
    print(f"dL/dL = {L.grad:.4f} (Expected: 1.0)")
    print(f"dL/dx = {x.grad:.4f} (Analytical: y*(x+y) + (x*y+z)*1 = 3*5 + 2*1 = 17.0)")
    print(f"dL/dy = {y.grad:.4f} (Analytical: x*(x+y) + (x*y+z)*1 = 2*5 + 2*1 = 12.0)")
    print(f"dL/dz = {z.grad:.4f} (Analytical: 1*(x+y) = 5.0)")

    # 2. Test Neuron with Sigmoid
    print("\nNeuron forward and backward:")
    n = Neuron(nin=2, nonlin="sigmoid")
    out = n([Value(1.5), Value(-0.5)])
    out.backward()
    print(f"Neuron output: {out.data:.4f}")
    print(f"Neuron weights grad: {[w.grad for w in n.w]}")
    print("Autograd engine executed successfully.")
