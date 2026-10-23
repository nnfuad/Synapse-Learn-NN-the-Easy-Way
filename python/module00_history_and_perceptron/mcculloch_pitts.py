"""
Module 00: The Genesis & The Atomic Neuron
Part 1: The 1943 McCulloch-Pitts Artificial Neuron

===============================================================================
HISTORICAL CONTEXT:
In 1943, neurophysiologist Warren McCulloch and 18-year-old self-taught logician
Walter Pitts published "A Logical Calculus of the Ideas Immanent in Nervous Activity".
This was the first mathematical model of a biological neuron in human history.

KEY INSIGHT:
McCulloch and Pitts proved that networks of simplified binary neurons could compute
ANY logical function (AND, OR, NOT) and thus perform any calculation that a
Turing machine could execute.

BIOLOGICAL ANALOGY:
- Dendrites receive electrochemical signals.
- Some signals are EXCITATORY (they push the cell toward firing).
- Some signals are INHIBITORY (they instantly veto firing, like an emergency brake).
- The Axon Hillock sums the charge; if it meets or exceeds a threshold θ,
  the neuron fires an "action potential" (1); otherwise it stays silent (0).
===============================================================================
"""

from typing import List, Callable


class McCullochPittsNeuron:
    """
    The 1943 McCulloch-Pitts Neuron Model.
    
    Attributes:
        threshold (int): The threshold of excitation θ required to fire.
        weights (List[int]): Fixed weights for excitatory inputs (typically all 1).
    """

    def __init__(self, threshold: int, num_excitatory_inputs: int):
        self.threshold = threshold
        # In the original 1943 model, all excitatory connections share unit weight
        self.weights = [1] * num_excitatory_inputs

    def forward(self, excitatory_inputs: List[int], inhibitory_inputs: List[int] = None) -> int:
        """
        Compute the neuron's firing state.
        
        Rule:
        1. If ANY inhibitory input is active (1), the neuron CANNOT fire (output = 0).
        2. Otherwise, if sum(excitatory_inputs) >= threshold, output = 1.
        3. Else, output = 0.
        """
        if inhibitory_inputs is None:
            inhibitory_inputs = []

        # Absolute inhibition rule: any active inhibitory input vetoes the neuron
        if any(inh == 1 for inh in inhibitory_inputs):
            return 0

        # Summation of excitatory inputs
        total_excitation = sum(
            w * x for w, x in zip(self.weights, excitatory_inputs)
        )

        return 1 if total_excitation >= self.threshold else 0


# ============================================================================
# Pedagogical Factory: Building Fundamental Logic Gates from 1943 Neurons
# ============================================================================

def build_and_gate() -> McCullochPittsNeuron:
    """
    AND Gate: Both inputs must be 1 to fire.
    Inputs: 2 excitatory inputs.
    Threshold: 2 (since 1 + 1 = 2 >= 2; 1 + 0 = 1 < 2).
    """
    return McCullochPittsNeuron(threshold=2, num_excitatory_inputs=2)


def build_or_gate() -> McCullochPittsNeuron:
    """
    OR Gate: At least one input must be 1 to fire.
    Inputs: 2 excitatory inputs.
    Threshold: 1 (since 1 + 0 = 1 >= 1; 0 + 0 = 0 < 1).
    """
    return McCullochPittsNeuron(threshold=1, num_excitatory_inputs=2)


def build_not_gate() -> Callable[[int], int]:
    """
    NOT Gate: Inverts input.
    Input: 1 signal acting as an INHIBITORY wire to a neuron with threshold 0.
    If input is 0 -> inhibition is inactive -> fires (1).
    If input is 1 -> inhibition active -> vetoed (0).
    """
    neuron = McCullochPittsNeuron(threshold=0, num_excitatory_inputs=0)
    return lambda x: neuron.forward(excitatory_inputs=[], inhibitory_inputs=[x])


# ============================================================================
# Demonstration & Verification
# ============================================================================

def demonstrate_mcculloch_pitts():
    print("=" * 70)
    print("🧠 1943 McCULLOCH-PITTS LOGIC GATES IN ACTION")
    print("=" * 70)

    and_gate = build_and_gate()
    or_gate = build_or_gate()
    not_gate = build_not_gate()

    print("\n[1] AND GATE (Threshold = 2):")
    print(" x1 | x2 | Output | Expected")
    print("----+----+--------+---------")
    for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        out = and_gate.forward([x1, x2])
        print(f"  {x1} |  {x2} |   {out}    |    {x1 and x2}")

    print("\n[2] OR GATE (Threshold = 1):")
    print(" x1 | x2 | Output | Expected")
    print("----+----+--------+---------")
    for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        out = or_gate.forward([x1, x2])
        print(f"  {x1} |  {x2} |   {out}    |    {x1 or x2}")

    print("\n[3] NOT GATE (Inhibitory Veto):")
    print(" x  | Output | Expected")
    print("----+--------+---------")
    for x in [0, 1]:
        out = not_gate(x)
        expected = 1 if x == 0 else 0
        print(f" {x}  |   {out}    |    {expected}")

    print("\n" + "=" * 70)
    print("LIMITATION THAT TRIGGERED THE NEXT LEAP:")
    print("Notice that McCulloch-Pitts neurons cannot LEARN. Their weights and")
    print("thresholds were hand-designed by a human logician.")
    print("This led Frank Rosenblatt in 1957 to invent the PERCEPTRON...")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_mcculloch_pitts()
