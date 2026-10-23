"""
Module 00: The Genesis & The Atomic Neuron
Part 3: The 1969 XOR Crisis & The First AI Winter

===============================================================================
HISTORICAL CONTEXT:
By the late 1960s, hype around the Perceptron was deafening. The media claimed
machines were months away from reading minds and achieving human consciousness.

In 1969, Marvin Minsky (co-founder of the MIT AI Lab) and Seymour Papert published
their landmark book: "Perceptrons: An Introduction to Computational Geometry".
In it, they proved mathematically that a single-layer perceptron CANNOT compute
the simple Exclusive-OR (XOR) function.

THE SHOCKWAVE:
Because XOR is one of the most basic binary logic operations, proving that
perceptrons couldn't solve it—and speculating that multi-layer networks might be
intractable to train—shattered funding from DARPA and other agencies.
This triggered the devastating "First AI Winter" (lasting roughly 1970–1982).

THE GEOMETRIC & ALGEBRAIC PROOF:
Let inputs be x1, x2 ∈ {0, 1}.
Target XOR outputs:
  (0, 0) -> 0
  (0, 1) -> 1
  (1, 0) -> 1
  (1, 1) -> 0

For a perceptron to succeed, there must exist weights w1, w2 and bias b such that:
  [Eq 1] For (0, 0):  w1*(0) + w2*(0) + b < 0   ===>  b < 0
  [Eq 2] For (0, 1):  w1*(0) + w2*(1) + b >= 0  ===>  w2 + b >= 0
  [Eq 3] For (1, 0):  w1*(1) + w2*(0) + b >= 0  ===>  w1 + b >= 0
  [Eq 4] For (1, 1):  w1*(1) + w2*(1) + b < 0   ===>  w1 + w2 + b < 0

THE IMPOSSIBILITY CONTRADICTION:
Add Eq 2 and Eq 3 together:
       (w1 + b) + (w2 + b) >= 0
       w1 + w2 + 2b >= 0
       w1 + w2 + b >= -b

Since Eq 1 established that b < 0, it follows that -b > 0 (strictly positive).
Therefore:
       w1 + w2 + b >= -b > 0
       ===> w1 + w2 + b > 0  (Strictly positive!)

Notice this directly contradicts Eq 4, which demands w1 + w2 + b < 0!
A number cannot be both strictly positive and strictly negative.
Q.E.D. It is mathematically IMPOSSIBLE for a single linear boundary to solve XOR!
===============================================================================
"""

import numpy as np
import os
import sys

# Support running directly or as package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from rosenblatt_perceptron import RosenblattPerceptron


def verify_xor_impossibility():
    print("=" * 75)
    print("❄️ THE 1969 XOR CRISIS: PROVING THE SINGLE PERCEPTRON'S FATAL FLAW")
    print("=" * 75)

    print("\n[Step 1] Algebraic Proof of Contradiction:")
    print("  1. From (0,0) -> 0:  b < 0")
    print("  2. From (1,0) -> 1:  w1 + b >= 0")
    print("  3. From (0,1) -> 1:  w2 + b >= 0")
    print("  4. From (1,1) -> 0:  w1 + w2 + b < 0")
    print("  Adding (2) + (3):     w1 + w2 + 2b >= 0  ==>  w1 + w2 + b >= -b > 0")
    print("  Contradiction:       w1 + w2 + b must be > 0 AND < 0 simultaneously!")
    print("  Conclusion:          No such straight line exists in Euclidean space.\n")

    print("[Step 2] Empirical Demonstration: Attempting to train Rosenblatt Perceptron on XOR")
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    y = np.array([0, 1, 1, 0])

    perceptron = RosenblattPerceptron(n_inputs=2, learning_rate=0.1, max_epochs=50)
    converged = perceptron.fit(X, y)

    print(f"\nTraining finished: Converged = {converged}")
    print("Testing predictions on XOR:")
    print(" x1 | x2 | Target | Predicted | Status")
    print("----+----+--------+-----------+-------")
    errors = 0
    for xi, target in zip(X, y):
        pred = perceptron.predict(xi)
        ok = (pred == target)
        if not ok:
            errors += 1
        print(f" {xi[0]:.0f}  | {xi[1]:.0f}  |   {target}    |     {pred}     | {'✅' if ok else '❌ FAIL'}")

    print(f"\nFinal Accuracy: {(4 - errors)/4 * 100:.1f}% (Maximum possible for a line is 75%)")
    print("=" * 75)
    print("HOW WAS THIS CRISIS FINALLY RESOLVED?")
    print("Minsky & Papert showed that one line cannot separate XOR.")
    print("The solution required TWO lines working together: A HIDDEN LAYER.")
    print("XOR can be decomposed into:  XOR(x1, x2) = (x1 OR x2) AND NOT(x1 AND x2)")
    print("This leads directly to the Multilayer Perceptron (MLP) and Backpropagation!")
    print("=" * 75)


if __name__ == "__main__":
    verify_xor_impossibility()
