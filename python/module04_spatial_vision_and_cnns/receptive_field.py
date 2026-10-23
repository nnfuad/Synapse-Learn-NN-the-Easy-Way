"""
Module 04: Spatial Vision & Convolutions (CNNs)
Exact Mathematical Receptive Field Calculator for Deep CNN Stacks

Mathematical Formulation:
For a 1D/2D layer with kernel k, stride s, padding p, dilation d:
1. Feature Map Size:
   n_out = floor((n_in + 2*p - d*(k - 1) - 1) / s) + 1
2. Jump (cumulative effective stride in input space):
   j_out = j_in * s
3. Receptive Field:
   r_out = r_in + (k - 1) * d * j_in
4. Center Position (offset of first receptive field center):
   start_out = start_in + ((k - 1) * d / 2 - p) * j_in

Citations:
- Araujo et al. (2019). "Computing Receptive Fields of Convolutional Networks." Distill.
- Simonyan & Zisserman (2014). "Very Deep Convolutional Networks for Large-Scale
  Image Recognition" (VGG-16: stacking two 3x3 convs matches a 5x5 receptive field
  with 28% fewer parameters).
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math


@dataclass
class LayerSpec:
    name: str
    kernel_size: int
    stride: int = 1
    padding: int = 0
    dilation: int = 1


@dataclass
class LayerMetrics:
    layer_name: str
    out_size: int
    jump: int
    receptive_field: int
    start_offset: float


def calculate_receptive_field(
    layers: List[LayerSpec],
    input_size: int = 224,
) -> List[LayerMetrics]:
    """
    Computes exact receptive field and spatial propagation metrics across a layer stack.
    """
    metrics = []

    current_size = input_size
    current_jump = 1
    current_rf = 1
    current_start = 0.5  # center of pixel 0 is at coordinate 0.5

    for layer in layers:
        k = layer.kernel_size
        s = layer.stride
        p = layer.padding
        d = layer.dilation

        # 1. Output spatial dimension
        out_size = math.floor((current_size + 2 * p - d * (k - 1) - 1) / s) + 1

        # 2. Receptive Field expansion
        rf_out = current_rf + (k - 1) * d * current_jump

        # 3. Start center offset
        start_out = current_start + (((k - 1) * d / 2.0) - p) * current_jump

        # 4. Jump expansion
        jump_out = current_jump * s

        metrics.append(
            LayerMetrics(
                layer_name=layer.name,
                out_size=out_size,
                jump=jump_out,
                receptive_field=rf_out,
                start_offset=start_out,
            )
        )

        current_size = out_size
        current_jump = jump_out
        current_rf = rf_out
        current_start = start_out

    return metrics


class ReceptiveFieldTracker:
    """
    Interactive Tracker and Analyzer for Deep CNN Stacks.
    """

    def __init__(self, input_size: int = 224):
        self.input_size = input_size
        self.layers: List[LayerSpec] = []

    def add_conv(
        self,
        name: str,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> "ReceptiveFieldTracker":
        self.layers.append(
            LayerSpec(name, kernel_size, stride, padding, dilation)
        )
        return self

    def add_pool(
        self,
        name: str,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
    ) -> "ReceptiveFieldTracker":
        self.layers.append(
            LayerSpec(name, kernel_size, stride, padding, dilation=1)
        )
        return self

    def compute(self) -> List[LayerMetrics]:
        return calculate_receptive_field(self.layers, self.input_size)

    def summary_table(self) -> str:
        metrics = self.compute()
        header = f"{'Layer':<20} | {'Out Size':<10} | {'Jump':<8} | {'RF (pixels)':<12} | {'Center Offset':<14}"
        separator = "-" * len(header)
        rows = [header, separator]

        for m in metrics:
            rows.append(
                f"{m.layer_name:<20} | {m.out_size:<10} | {m.jump:<8} | {m.receptive_field:<12} | {m.start_offset:<14.2f}"
            )
        return "\n".join(rows)


def build_lenet5() -> ReceptiveFieldTracker:
    """LeNet-5 (LeCun et al. 1998) on 32x32 input."""
    tracker = ReceptiveFieldTracker(input_size=32)
    tracker.add_conv("C1_Conv5x5", kernel_size=5, stride=1, padding=0)  # out: 28, rf: 5
    tracker.add_pool("S2_AvgPool2x2", kernel_size=2, stride=2, padding=0)  # out: 14, rf: 6
    tracker.add_conv("C3_Conv5x5", kernel_size=5, stride=1, padding=0)  # out: 10, rf: 14
    tracker.add_pool("S4_AvgPool2x2", kernel_size=2, stride=2, padding=0)  # out: 5, rf: 16
    tracker.add_conv("C5_Conv5x5", kernel_size=5, stride=1, padding=0)  # out: 1, rf: 32
    return tracker


def build_vgg16_early_stage() -> ReceptiveFieldTracker:
    """Early VGG-16 demonstrating two 3x3 convs replacing a 5x5 conv."""
    tracker = ReceptiveFieldTracker(input_size=224)
    tracker.add_conv("Conv1_1_3x3", kernel_size=3, stride=1, padding=1)
    tracker.add_conv("Conv1_2_3x3", kernel_size=3, stride=1, padding=1)
    tracker.add_pool("Pool1_2x2", kernel_size=2, stride=2, padding=0)
    tracker.add_conv("Conv2_1_3x3", kernel_size=3, stride=1, padding=1)
    tracker.add_conv("Conv2_2_3x3", kernel_size=3, stride=1, padding=1)
    return tracker


if __name__ == "__main__":
    print("=== LeNet-5 (1998) Receptive Field Evolution ===")
    print(build_lenet5().summary_table())
    print("\n=== VGG-16 Early Stage Receptive Field Evolution ===")
    print(build_vgg16_early_stage().summary_table())
